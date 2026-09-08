"""TraceGuard public research SDK."""

from .config import Settings
from .corpus import CorpusCatalog
from .ledger import HashChainLedger
from .providers import AzureOpenAIProvider, FixtureProvider, LLMProvider
from .receipt import ReceiptEnvelope, ReceiptSigner
from .sdk import TraceGuardSDK
from .types import (
    Condition,
    MedicalCase,
    MedicalDocument,
    ObservableTrace,
    RunMetrics,
    RunResult,
    TraceStep,
)
from .verifier import ReceiptVerifier, VerificationResult, verify_ledger, verify_receipt

__all__ = [
    "AzureOpenAIProvider",
    "Condition",
    "CorpusCatalog",
    "FixtureProvider",
    "HashChainLedger",
    "LLMProvider",
    "MedicalCase",
    "MedicalDocument",
    "ObservableTrace",
    "ReceiptEnvelope",
    "ReceiptSigner",
    "ReceiptVerifier",
    "RunMetrics",
    "RunResult",
    "Settings",
    "TraceGuardSDK",
    "TraceStep",
    "VerificationResult",
    "verify_ledger",
    "verify_receipt",
]

__version__ = "0.1.0"
