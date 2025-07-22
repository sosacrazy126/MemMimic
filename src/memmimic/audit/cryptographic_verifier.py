"""
Cryptographic Verifier - SHA-256 hash verification and integrity checking.

Provides cryptographic verification services for the immutable audit logging system
with real-time tamper detection and hash chain validation.
"""

import hashlib
import hmac
import json
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from ..errors import get_error_logger, handle_errors
from ..errors.exceptions import MemoryStorageError

logger = get_error_logger(__name__)


class VerificationStatus(Enum):
    """Verification status codes."""
    VALID = "valid"
    INVALID = "invalid"
    TAMPERED = "tampered"
    MISSING = "missing"
    ERROR = "error"


@dataclass
class VerificationResult:
    """Result of cryptographic verification."""
    status: VerificationStatus
    entry_id: str
    verification_time_ms: float
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_valid(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.VALID
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'status': self.status.value,
            'entry_id': self.entry_id,
            'verification_time_ms': self.verification_time_ms,
            'details': self.details,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'is_valid': self.is_valid()
        }


@dataclass
class HashChainVerificationResult:
    """Result of hash chain verification."""
    valid: bool
    total_entries: int
    verified_entries: int
    broken_links: List[Dict[str, Any]]
    verification_time_ms: float
    start_position: int = 0
    end_position: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def integrity_percentage(self) -> float:
        """Calculate integrity percentage."""
        if self.total_entries == 0:
            return 100.0
        return ((self.total_entries - len(self.broken_links)) / self.total_entries) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'valid': self.valid,
            'total_entries': self.total_entries,
            'verified_entries': self.verified_entries,
            'broken_links_count': len(self.broken_links),
            'broken_links': self.broken_links,
            'verification_time_ms': self.verification_time_ms,
            'integrity_percentage': self.integrity_percentage(),
            'start_position': self.start_position,
            'end_position': self.end_position,
            'timestamp': self.timestamp.isoformat()
        }


class CryptographicVerifier:
    """
    Cryptographic verifier for audit entries and hash chains.
    
    Features:
    - SHA-256 hash verification with salting
    - HMAC-based authentication
    - Real-time tamper detection
    - Performance-optimized verification (<1ms)
    - Batch verification for chain validation
    """
    
    def __init__(self, verification_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cryptographic verifier.
        
        Args:
            verification_key: Secret key for HMAC operations
            config: Additional configuration options
        """
        self.verification_key = verification_key.encode('utf-8')
        self.config = config or {}
        
        # Performance tracking
        self._metrics = {
            'verifications_performed': 0,
            'tamper_detections': 0,
            'avg_verification_time_ms': 0.0,
            'chain_verifications': 0
        }
        
        # Thread safety
        self._metrics_lock = threading.Lock()
        
        logger.info("CryptographicVerifier initialized")
    
    @handle_errors(MemoryStorageError)
    def verify_entry_hash(
        self,
        entry_data: Dict[str, Any],
        expected_hash: str,
        salt: str,
        entry_id: str
    ) -> VerificationResult:
        """
        Verify SHA-256 hash of audit entry data.
        
        Args:
            entry_data: Canonical entry data (without entry_hash)
            expected_hash: Expected hash value
            salt: Salt used in original hash calculation
            entry_id: Entry ID for tracking
            
        Returns:
            VerificationResult with status and details
        """
        start_time = time.perf_counter()
        
        try:
            # Calculate hash using same method as original
            calculated_hash = self._calculate_entry_hash(entry_data, salt)
            
            # Compare hashes
            is_valid = hmac.compare_digest(calculated_hash, expected_hash)
            
            verification_time = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            with self._metrics_lock:
                self._metrics['verifications_performed'] += 1
                if not is_valid:
                    self._metrics['tamper_detections'] += 1
                self._update_avg_verification_time(verification_time)
            
            status = VerificationStatus.VALID if is_valid else VerificationStatus.TAMPERED
            details = "Hash verification successful" if is_valid else "Hash mismatch detected - potential tampering"
            
            result = VerificationResult(
                status=status,
                entry_id=entry_id,
                verification_time_ms=verification_time,
                details=details,
                metadata={
                    'calculated_hash': calculated_hash,
                    'expected_hash': expected_hash,
                    'salt': salt
                }
            )
            
            if not is_valid:
                logger.warning(f"Hash verification failed for entry {entry_id}: {details}")
            else:
                logger.debug(f"Hash verification passed for entry {entry_id} ({verification_time:.2f}ms)")
            
            return result
            
        except Exception as e:
            verification_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Hash verification error for entry {entry_id}: {e}")
            
            return VerificationResult(
                status=VerificationStatus.ERROR,
                entry_id=entry_id,
                verification_time_ms=verification_time,
                details=f"Verification error: {e}",
                metadata={'error': str(e)}
            )
    
    def _calculate_entry_hash(self, entry_data: Dict[str, Any], salt: str) -> str:
        """Calculate SHA-256 hash for entry data with salt and verification key."""
        # Create canonical JSON representation
        canonical_json = json.dumps(entry_data, sort_keys=True, separators=(',', ':'))
        
        # Combine with salt and verification key
        hash_input = f"{canonical_json}:{salt}:{self.verification_key.decode('utf-8')}"
        
        # Calculate SHA-256 hash
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    def verify_hmac_signature(
        self,
        data: Union[str, bytes],
        signature: str,
        entry_id: str
    ) -> VerificationResult:
        """
        Verify HMAC signature for additional authentication.
        
        Args:
            data: Data to verify
            signature: HMAC signature to verify against
            entry_id: Entry ID for tracking
            
        Returns:
            VerificationResult with status and details
        """
        start_time = time.perf_counter()
        
        try:
            # Ensure data is bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Calculate HMAC signature
            calculated_signature = hmac.new(
                self.verification_key,
                data,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            is_valid = hmac.compare_digest(calculated_signature, signature)
            
            verification_time = (time.perf_counter() - start_time) * 1000
            
            status = VerificationStatus.VALID if is_valid else VerificationStatus.TAMPERED
            details = "HMAC verification successful" if is_valid else "HMAC signature mismatch"
            
            result = VerificationResult(
                status=status,
                entry_id=entry_id,
                verification_time_ms=verification_time,
                details=details,
                metadata={
                    'calculated_signature': calculated_signature,
                    'provided_signature': signature
                }
            )
            
            if not is_valid:
                logger.warning(f"HMAC verification failed for entry {entry_id}")
                
            return result
            
        except Exception as e:
            verification_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"HMAC verification error for entry {entry_id}: {e}")
            
            return VerificationResult(
                status=VerificationStatus.ERROR,
                entry_id=entry_id,
                verification_time_ms=verification_time,
                details=f"HMAC verification error: {e}",
                metadata={'error': str(e)}
            )
    
    def _update_avg_verification_time(self, verification_time: float) -> None:
        """Update average verification time metric."""
        current_avg = self._metrics['avg_verification_time_ms']
        count = self._metrics['verifications_performed']
        
        # Running average calculation
        self._metrics['avg_verification_time_ms'] = ((current_avg * (count - 1)) + verification_time) / count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get verification performance metrics."""
        with self._metrics_lock:
            return {
                **self._metrics,
                'tamper_detection_rate': (
                    self._metrics['tamper_detections'] / max(1, self._metrics['verifications_performed']) * 100
                )
            }


class HashChainVerifier:
    """
    Hash chain verifier for audit trail integrity.
    
    Verifies the cryptographic hash chain that links audit entries
    to detect tampering and ensure temporal ordering integrity.
    """
    
    def __init__(self, cryptographic_verifier: CryptographicVerifier):
        """
        Initialize hash chain verifier.
        
        Args:
            cryptographic_verifier: Cryptographic verifier instance
        """
        self.crypto_verifier = cryptographic_verifier
        
        # Performance tracking
        self._chain_metrics = {
            'chains_verified': 0,
            'avg_chain_verification_time_ms': 0.0,
            'total_entries_verified': 0,
            'chain_breaks_detected': 0
        }
        
        self._metrics_lock = threading.Lock()
        
        logger.info("HashChainVerifier initialized")
    
    @handle_errors(MemoryStorageError)
    def verify_hash_chain(
        self,
        entries: List[Dict[str, Any]],
        start_position: int = 0,
        end_position: Optional[int] = None
    ) -> HashChainVerificationResult:
        """
        Verify integrity of hash chain across multiple entries.
        
        Args:
            entries: List of audit entries in chronological order
            start_position: Starting position in chain
            end_position: Ending position (None for end of chain)
            
        Returns:
            HashChainVerificationResult with detailed analysis
        """
        start_time = time.perf_counter()
        
        if not entries:
            return HashChainVerificationResult(
                valid=True,
                total_entries=0,
                verified_entries=0,
                broken_links=[],
                verification_time_ms=0.0,
                start_position=start_position
            )
        
        try:
            # Determine verification range
            end_pos = end_position or len(entries)
            entries_to_verify = entries[start_position:end_pos]
            
            broken_links = []
            verified_count = 0
            
            # Verify each link in the chain
            for i, entry in enumerate(entries_to_verify):
                current_position = start_position + i
                
                # Verify individual entry hash
                entry_verification = self._verify_entry_in_chain(entry)
                
                if not entry_verification.is_valid():
                    broken_links.append({
                        'position': current_position,
                        'entry_id': entry.get('entry_id', 'unknown'),
                        'type': 'hash_verification_failed',
                        'details': entry_verification.details,
                        'verification_result': entry_verification.to_dict()
                    })
                
                # Verify chain link (previous hash reference)
                if i > 0 or start_position > 0:
                    chain_link_valid = self._verify_chain_link(
                        current_entry=entry,
                        previous_entry=entries[current_position - 1] if current_position > 0 else None,
                        position=current_position
                    )
                    
                    if not chain_link_valid['valid']:
                        broken_links.append(chain_link_valid)
                
                verified_count += 1
            
            verification_time = (time.perf_counter() - start_time) * 1000
            is_chain_valid = len(broken_links) == 0
            
            # Update metrics
            with self._metrics_lock:
                self._chain_metrics['chains_verified'] += 1
                self._chain_metrics['total_entries_verified'] += verified_count
                if not is_chain_valid:
                    self._chain_metrics['chain_breaks_detected'] += len(broken_links)
                self._update_avg_chain_verification_time(verification_time)
            
            result = HashChainVerificationResult(
                valid=is_chain_valid,
                total_entries=len(entries_to_verify),
                verified_entries=verified_count,
                broken_links=broken_links,
                verification_time_ms=verification_time,
                start_position=start_position,
                end_position=end_pos
            )
            
            logger.info(
                f"Hash chain verification complete: {verified_count} entries, "
                f"{len(broken_links)} breaks, {verification_time:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            verification_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Hash chain verification error: {e}")
            
            return HashChainVerificationResult(
                valid=False,
                total_entries=len(entries),
                verified_entries=0,
                broken_links=[{
                    'type': 'verification_error',
                    'details': f"Chain verification failed: {e}",
                    'error': str(e)
                }],
                verification_time_ms=verification_time,
                start_position=start_position
            )
    
    def _verify_entry_in_chain(self, entry: Dict[str, Any]) -> VerificationResult:
        """Verify individual entry within chain context."""
        entry_id = entry.get('entry_id', 'unknown')
        expected_hash = entry.get('entry_hash')
        salt = entry.get('salt', '')
        
        if not expected_hash:
            return VerificationResult(
                status=VerificationStatus.MISSING,
                entry_id=entry_id,
                verification_time_ms=0.0,
                details="Missing entry hash"
            )
        
        # Create canonical data (exclude hash for verification)
        canonical_data = {k: v for k, v in entry.items() if k != 'entry_hash'}
        
        return self.crypto_verifier.verify_entry_hash(
            entry_data=canonical_data,
            expected_hash=expected_hash,
            salt=salt,
            entry_id=entry_id
        )
    
    def _verify_chain_link(
        self,
        current_entry: Dict[str, Any],
        previous_entry: Optional[Dict[str, Any]],
        position: int
    ) -> Dict[str, Any]:
        """Verify hash chain link between consecutive entries."""
        current_id = current_entry.get('entry_id', 'unknown')
        current_previous_hash = current_entry.get('previous_hash')
        
        if previous_entry is None:
            # First entry in chain - previous_hash should be None
            if current_previous_hash is not None:
                return {
                    'position': position,
                    'entry_id': current_id,
                    'type': 'invalid_first_entry',
                    'valid': False,
                    'details': f"First entry has non-null previous_hash: {current_previous_hash}",
                    'expected_previous': None,
                    'actual_previous': current_previous_hash
                }
        else:
            # Subsequent entry - verify chain link
            expected_previous_hash = previous_entry.get('entry_hash')
            
            if current_previous_hash != expected_previous_hash:
                return {
                    'position': position,
                    'entry_id': current_id,
                    'type': 'broken_chain_link',
                    'valid': False,
                    'details': f"Chain link broken at position {position}",
                    'expected_previous': expected_previous_hash,
                    'actual_previous': current_previous_hash,
                    'previous_entry_id': previous_entry.get('entry_id', 'unknown')
                }
        
        # Chain link is valid
        return {
            'position': position,
            'entry_id': current_id,
            'type': 'valid_chain_link',
            'valid': True,
            'details': "Chain link verified"
        }
    
    def verify_temporal_ordering(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify temporal ordering of entries in hash chain.
        
        Args:
            entries: List of audit entries to verify
            
        Returns:
            Dict with ordering verification results
        """
        start_time = time.perf_counter()
        
        try:
            ordering_violations = []
            
            for i in range(1, len(entries)):
                current_entry = entries[i]
                previous_entry = entries[i - 1]
                
                # Parse timestamps
                current_time = datetime.fromisoformat(current_entry.get('timestamp', ''))
                previous_time = datetime.fromisoformat(previous_entry.get('timestamp', ''))
                
                # Check temporal ordering
                if current_time < previous_time:
                    ordering_violations.append({
                        'position': i,
                        'entry_id': current_entry.get('entry_id', 'unknown'),
                        'current_time': current_time.isoformat(),
                        'previous_time': previous_time.isoformat(),
                        'time_diff_seconds': (previous_time - current_time).total_seconds()
                    })
            
            verification_time = (time.perf_counter() - start_time) * 1000
            is_valid = len(ordering_violations) == 0
            
            return {
                'valid': is_valid,
                'total_entries': len(entries),
                'ordering_violations': ordering_violations,
                'verification_time_ms': verification_time
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'total_entries': len(entries),
                'ordering_violations': [],
                'verification_time_ms': 0.0
            }
    
    def _update_avg_chain_verification_time(self, verification_time: float) -> None:
        """Update average chain verification time metric."""
        current_avg = self._chain_metrics['avg_chain_verification_time_ms']
        count = self._chain_metrics['chains_verified']
        
        # Running average calculation
        self._chain_metrics['avg_chain_verification_time_ms'] = ((current_avg * (count - 1)) + verification_time) / count
    
    def get_chain_metrics(self) -> Dict[str, Any]:
        """Get hash chain verification metrics."""
        with self._metrics_lock:
            return {
                **self._chain_metrics,
                'avg_entries_per_verification': (
                    self._chain_metrics['total_entries_verified'] / max(1, self._chain_metrics['chains_verified'])
                ),
                'chain_break_rate': (
                    self._chain_metrics['chain_breaks_detected'] / max(1, self._chain_metrics['total_entries_verified']) * 100
                )
            }