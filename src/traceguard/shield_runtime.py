"""Enforcing runtime for structure/timing/size trace defenses.

The full-pad condition makes three of the four host-observable coordinates public
constants, and it is precise about the fourth:

* **structure** -- the canonical plan is executed, so the step sequence and its
  length are data-independent;
* **timing** -- every step is held to the public deadline by an actual sleep;
* **egress** (request bytes) -- the request body is padded to the public ceiling
  before transmission, in an inert transport field;
* **ingress** (response bytes) -- *not* bounded.  The provider chooses how many
  bytes to return and they cross the boundary before the guest regains control,
  so no in-guest mechanism can make this a constant.  It is recorded as measured
  in every condition and the residual leakage it carries is reported.

Closing the fourth coordinate requires the model inside the confidential boundary
or constant-rate shaping at the transport, neither of which this module provides.

This module also does not implement a TEE, ORAM, output DP, or remote
attestation; those remain deployment assumptions and are stated as such in every
receipt.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from .config import Settings
from .providers import EgressCeilingExceeded
from .types import Condition, ProviderResponse


@dataclass(frozen=True, slots=True)
class StepExecution:
    response: ProviderResponse
    observable_duration_s: float
    observable_egress_bytes: int
    # The response direction is recorded as measured in every condition.  The
    # guest does not choose how many bytes the provider sends back, so no
    # in-guest mechanism can make this a constant; pretending otherwise would
    # make the padded release look data-independent when it is not.
    observable_ingress_bytes: int = 0
    fail_closed: bool = False
    violation: str | None = None


class ShieldRuntime:
    """Apply the selected trace condition at the model egress boundary."""

    def __init__(self, settings: Settings, condition: Condition) -> None:
        self.settings = settings
        self.condition = condition

    @property
    def fixed_structure(self) -> bool:
        return self.condition in {Condition.STRUCTURE_ONLY, Condition.FULL_PAD}

    @property
    def full_padding(self) -> bool:
        return self.condition is Condition.FULL_PAD

    @property
    def pad_request_to_bytes(self) -> int | None:
        """Public egress ceiling, or ``None`` when the request is not padded.

        Threaded down into ``ProviderRequest`` so the padding happens where the
        bytes are actually serialized.  Only the full-pad condition bounds it.
        """

        return self.settings.egress_ceiling_bytes if self.full_padding else None

    def execute(self, operation: Callable[[], ProviderResponse]) -> StepExecution:
        if not self.full_padding:
            started = time.perf_counter()
            response = operation()
            duration = time.perf_counter() - started
            return StepExecution(
                response=response,
                observable_duration_s=duration,
                observable_egress_bytes=response.request_bytes,
                observable_ingress_bytes=response.response_bytes,
            )
        return self._execute_full_pad(operation)

    def _execute_full_pad(self, operation: Callable[[], ProviderResponse]) -> StepExecution:
        deadline = self.settings.step_deadline_s
        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="traceguard-step")
        future = executor.submit(operation)
        response: ProviderResponse | None = None
        violation: str | None = None
        try:
            response = future.result(timeout=deadline)
        except FutureTimeoutError:
            violation = "deadline_overrun"
            future.cancel()
        except EgressCeilingExceeded:
            # A request that will not fit under the public ceiling: distinct from
            # a transport failure, and the one case where the size mechanism
            # actually binds.  Kept as its own code so the two are never pooled.
            violation = "egress_overrun"
        except Exception:
            # A provider exception is converted to the same fixed release.  Its
            # message is never retained because SDK transports may echo prompts.
            violation = "provider_failure"
        finally:
            # Do not wait for a timed-out network call before releasing the public
            # deadline value.  The official provider also has its own transport timeout.
            executor.shutdown(wait=False, cancel_futures=True)

        elapsed = time.perf_counter() - started
        if elapsed > deadline and violation is None:
            violation = "deadline_overrun"

        # Hold the wall clock to the public deadline.  This is the one padding
        # that is genuinely enforced on the observable: the host sees the step
        # take exactly `deadline`, whether the call returned early or not.
        remaining = deadline - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)

        if violation is not None or response is None:
            response = ProviderResponse(
                text="",
                response_bytes=0,
                request_bytes=self.settings.egress_ceiling_bytes,
                input_tokens=0,
                output_tokens=0,
                stop_reason="fail_closed",
            )
        return StepExecution(
            response=response,
            observable_duration_s=deadline,
            # Egress is a true constant: the provider layer padded the request
            # body to this ceiling before transmitting it.
            observable_egress_bytes=self.settings.egress_ceiling_bytes,
            # Ingress is NOT constant and is not claimed to be.  The provider
            # chose this many bytes and they have already crossed the boundary by
            # the time control returns here; overwriting the field with the
            # ceiling would record a constant the host never saw.
            observable_ingress_bytes=response.response_bytes,
            fail_closed=violation is not None,
            violation=violation,
        )
