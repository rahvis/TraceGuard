"""Metadata-only instrumentation for the paper's observable surface."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .shield_runtime import ShieldRuntime
from .types import EventCallback, ObservableTrace, ProviderResponse, TraceStep


@dataclass(frozen=True, slots=True)
class RuntimeViolation:
    step_type: str
    code: str

    def to_dict(self) -> dict[str, str]:
        return {"step_type": self.step_type, "code": self.code}


class TraceRecorder:
    """Record only ordered step type, timing, and both wire byte counts.

    The model is outside the confidential boundary, so a host watching the
    guest's NIC sees each round-trip in both directions.  Recording only one of
    them would understate the observable surface.
    """

    def __init__(
        self,
        *,
        run_id: str,
        shield: ShieldRuntime,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.run_id = run_id
        self.shield = shield
        self._callback = event_callback
        self._started = time.perf_counter()
        self._steps: list[TraceStep] = []
        self._violations: list[RuntimeViolation] = []
        self._input_tokens = 0
        self._output_tokens = 0

    def _emit(self, event: dict[str, object]) -> None:
        if self._callback is None:
            return
        safe = {
            "run_id": self.run_id,
            "sequence": len(self._steps),
            **event,
        }
        try:
            self._callback(safe)
        except Exception:
            # UI/event consumers must never alter the scientific execution.
            return

    def observe(
        self, step_type: str, operation: Callable[[], ProviderResponse]
    ) -> ProviderResponse:
        index = len(self._steps)
        self._emit({"event_type": "step_started", "step_type": step_type})
        try:
            execution = self.shield.execute(operation)
        except Exception:
            self._emit({"event_type": "step_failed", "step_type": step_type, "status": "failed"})
            raise

        if self.shield.full_padding:
            wall_time = (index + 1) * self.shield.settings.step_deadline_s
        else:
            wall_time = time.perf_counter() - self._started
        step = TraceStep(
            index=index,
            step_type=step_type,
            wall_time_s=wall_time,
            duration_s=execution.observable_duration_s,
            egress_bytes=execution.observable_egress_bytes,
            ingress_bytes=execution.observable_ingress_bytes,
        )
        self._steps.append(step)
        self._input_tokens += execution.response.input_tokens
        self._output_tokens += execution.response.output_tokens
        if execution.violation:
            self._violations.append(RuntimeViolation(step_type=step_type, code=execution.violation))
        self._emit(
            {
                "event_type": "step_completed",
                "step_type": step_type,
                "index": index,
                "duration_s": round(step.duration_s, 6),
                "wall_time_s": round(step.wall_time_s, 6),
                "egress_bytes": step.egress_bytes,
                "ingress_bytes": step.ingress_bytes,
                "status": "fail_closed" if execution.fail_closed else "completed",
            }
        )
        return execution.response

    @property
    def trace(self) -> ObservableTrace:
        return ObservableTrace(tuple(self._steps))

    @property
    def violations(self) -> tuple[RuntimeViolation, ...]:
        return tuple(self._violations)

    @property
    def fail_closed(self) -> bool:
        return bool(self._violations)

    @property
    def token_budget_used(self) -> int:
        return self._input_tokens + self._output_tokens

    def emit_run_event(self, event_type: str, *, status: str) -> None:
        self._emit({"event_type": event_type, "status": status})
