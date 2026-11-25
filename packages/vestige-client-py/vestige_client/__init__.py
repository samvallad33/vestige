"""Zero-config Python SDK for Vestige Causal Proof Engine."""

from .client import AsyncVestigeClient, ProofReceipt, VestigeClient, VestigeError

__all__ = [
    "AsyncVestigeClient",
    "ProofReceipt",
    "VestigeClient",
    "VestigeError",
]
__version__ = "7.0.0"
