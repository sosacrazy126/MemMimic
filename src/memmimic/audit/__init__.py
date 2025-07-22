"""
MemMimic v2.0 Immutable Audit Logging System

Provides enterprise-grade cryptographic audit logging with:
- Immutable audit trails with hash chain integrity
- SHA-256 cryptographic verification
- Tamper detection and real-time alerts
- Performance-optimized logging with <1ms overhead
"""

from .immutable_logger import ImmutableAuditLogger, AuditEntry
from .cryptographic_verifier import CryptographicVerifier, HashChainVerifier
from .tamper_detector import TamperDetector, TamperAlert
from .audit_trail_manager import AuditTrailManager, AuditQuery, AuditQueryResult
from .security_metrics import SecurityMetrics, AuditMetrics

__all__ = [
    'ImmutableAuditLogger',
    'AuditEntry',
    'CryptographicVerifier',
    'HashChainVerifier',
    'TamperDetector',
    'TamperAlert',
    'AuditTrailManager',
    'AuditQuery',
    'AuditQueryResult',
    'SecurityMetrics',
    'AuditMetrics'
]

__version__ = "2.0.0"