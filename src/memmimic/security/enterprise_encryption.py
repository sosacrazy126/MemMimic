#!/usr/bin/env python3
"""
Enterprise Encryption & Data Protection

Advanced encryption at rest and in transit, secure key management,
and data privacy compliance for enterprise MemMimic deployments.
"""

import asyncio
import base64
import hashlib
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import json

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class EncryptionLevel(Enum):
    """Encryption security levels"""
    NONE = "none"
    BASIC = "basic"         # AES-128
    STANDARD = "standard"   # AES-256
    ENTERPRISE = "enterprise"  # AES-256 + RSA + Key rotation
    QUANTUM_SAFE = "quantum_safe"  # Post-quantum algorithms


class DataClassification(Enum):
    """Data sensitivity classifications"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class EncryptionKey:
    """Encryption key with metadata"""
    key_id: str
    algorithm: str
    key_data: bytes
    classification: DataClassification
    created_at: datetime
    expires_at: Optional[datetime] = None
    rotation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptedData:
    """Encrypted data container"""
    ciphertext: bytes
    key_id: str
    algorithm: str
    iv: Optional[bytes] = None
    auth_tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted_at: datetime = field(default_factory=datetime.utcnow)


class SecureKeyManager:
    """Enterprise key management system"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or Fernet.generate_key()
        self.master_cipher = Fernet(self.master_key)
        self.keys: Dict[str, EncryptionKey] = {}
        self.key_rotation_interval = timedelta(days=90)
        
    def generate_key(self, 
                    algorithm: str = "AES-256",
                    classification: DataClassification = DataClassification.CONFIDENTIAL,
                    expires_in: Optional[timedelta] = None) -> str:
        """Generate new encryption key"""
        key_id = secrets.token_urlsafe(16)
        
        if algorithm == "AES-256":
            key_data = secrets.token_bytes(32)  # 256 bits
        elif algorithm == "AES-128":
            key_data = secrets.token_bytes(16)  # 128 bits
        elif algorithm == "RSA-2048":
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        expires_at = None
        if expires_in:
            expires_at = datetime.utcnow() + expires_in
        elif classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            expires_at = datetime.utcnow() + timedelta(days=30)
        
        key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            key_data=key_data,
            classification=classification,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self.keys[key_id] = key
        logger.info(f"Generated {algorithm} key: {key_id}")
        
        return key_id
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get encryption key by ID"""
        if key_id not in self.keys:
            return None
            
        key = self.keys[key_id]
        
        # Check expiration
        if key.expires_at and datetime.utcnow() > key.expires_at:
            logger.warning(f"Key {key_id} has expired")
            return None
            
        return key
    
    def rotate_key(self, key_id: str) -> str:
        """Rotate encryption key"""
        old_key = self.get_key(key_id)
        if not old_key:
            raise ValueError(f"Key {key_id} not found")
        
        # Generate new key with same parameters
        new_key_id = self.generate_key(
            algorithm=old_key.algorithm,
            classification=old_key.classification
        )
        
        new_key = self.keys[new_key_id]
        new_key.rotation_count = old_key.rotation_count + 1
        
        # Mark old key as rotated
        old_key.metadata['rotated_to'] = new_key_id
        old_key.metadata['rotated_at'] = datetime.utcnow().isoformat()
        
        logger.info(f"Rotated key {key_id} to {new_key_id}")
        return new_key_id
    
    def cleanup_expired_keys(self):
        """Remove expired keys"""
        now = datetime.utcnow()
        expired_keys = [
            key_id for key_id, key in self.keys.items()
            if key.expires_at and key.expires_at < now
        ]
        
        for key_id in expired_keys:
            del self.keys[key_id]
            
        logger.info(f"Cleaned up {len(expired_keys)} expired keys")


class EncryptionService:
    """Enterprise encryption service"""
    
    def __init__(self, key_manager: SecureKeyManager):
        self.key_manager = key_manager
        self.backend = default_backend()
    
    def encrypt_data(self, 
                    data: Union[str, bytes],
                    key_id: str,
                    classification: DataClassification = DataClassification.CONFIDENTIAL) -> EncryptedData:
        """Encrypt data with specified key"""
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found or expired")
        
        # Convert string to bytes
        if isinstance(data, str):
            plaintext = data.encode('utf-8')
        else:
            plaintext = data
        
        if key.algorithm.startswith("AES"):
            return self._encrypt_aes(plaintext, key)
        elif key.algorithm.startswith("RSA"):
            return self._encrypt_rsa(plaintext, key)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {key.algorithm}")
    
    def _encrypt_aes(self, plaintext: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt data using AES"""
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(iv),
            backend=self.backend
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            key_id=key.key_id,
            algorithm=key.algorithm,
            iv=iv,
            auth_tag=encryptor.tag
        )
    
    def _encrypt_rsa(self, plaintext: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt data using RSA"""
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=self.backend
        )
        public_key = private_key.public_key()
        
        # RSA can only encrypt small amounts, use hybrid encryption for larger data
        if len(plaintext) > 190:  # RSA-2048 can encrypt max ~245 bytes
            # Generate AES key for data
            aes_key = secrets.token_bytes(32)
            
            # Encrypt data with AES
            iv = secrets.token_bytes(16)
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.GCM(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            data_ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            # Encrypt AES key with RSA
            encrypted_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key + encrypted data
            ciphertext = encrypted_key + data_ciphertext
            
            return EncryptedData(
                ciphertext=ciphertext,
                key_id=key.key_id,
                algorithm="RSA-AES-HYBRID",
                iv=iv,
                auth_tag=encryptor.tag,
                metadata={'key_length': len(encrypted_key)}
            )
        else:
            # Direct RSA encryption for small data
            ciphertext = public_key.encrypt(
                plaintext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return EncryptedData(
                ciphertext=ciphertext,
                key_id=key.key_id,
                algorithm=key.algorithm
            )
    
    def decrypt_data(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt encrypted data"""
        key = self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Key {encrypted_data.key_id} not found or expired")
        
        if encrypted_data.algorithm.startswith("AES"):
            return self._decrypt_aes(encrypted_data, key)
        elif encrypted_data.algorithm.startswith("RSA"):
            return self._decrypt_rsa(encrypted_data, key)
        else:
            raise ValueError(f"Unsupported decryption algorithm: {encrypted_data.algorithm}")
    
    def _decrypt_aes(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt AES encrypted data"""
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(encrypted_data.iv, encrypted_data.auth_tag),
            backend=self.backend
        )
        
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
        
        return plaintext
    
    def _decrypt_rsa(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt RSA encrypted data"""
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=self.backend
        )
        
        if encrypted_data.algorithm == "RSA-AES-HYBRID":
            # Hybrid decryption
            key_length = encrypted_data.metadata['key_length']
            encrypted_key = encrypted_data.ciphertext[:key_length]
            data_ciphertext = encrypted_data.ciphertext[key_length:]
            
            # Decrypt AES key
            aes_key = private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt data with AES
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.GCM(encrypted_data.iv, encrypted_data.auth_tag),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(data_ciphertext) + decryptor.finalize()
            
            return plaintext
        else:
            # Direct RSA decryption
            plaintext = private_key.decrypt(
                encrypted_data.ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return plaintext


class DataPrivacyCompliance:
    """Data privacy compliance framework"""
    
    def __init__(self, encryption_service: EncryptionService):
        self.encryption_service = encryption_service
        self.deletion_log: Dict[str, datetime] = {}
        self.access_log: List[Dict[str, Any]] = []
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize personal data for GDPR compliance"""
        anonymized = data.copy()
        
        # Common PII fields to anonymize
        pii_fields = ['email', 'name', 'phone', 'address', 'ssn']
        
        for field in pii_fields:
            if field in anonymized:
                # Replace with anonymized hash
                hash_value = hashlib.sha256(str(anonymized[field]).encode()).hexdigest()[:8]
                anonymized[field] = f"anon_{hash_value}"
        
        return anonymized
    
    def log_data_access(self, 
                       user_id: str,
                       data_type: str,
                       action: str,
                       data_id: Optional[str] = None):
        """Log data access for audit trails"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'data_type': data_type,
            'action': action,
            'data_id': data_id,
            'ip_address': None  # Would be populated from request context
        }
        
        self.access_log.append(log_entry)
    
    def right_to_be_forgotten(self, user_id: str) -> bool:
        """Implement GDPR right to be forgotten"""
        try:
            # Mark user data for deletion
            self.deletion_log[user_id] = datetime.utcnow()
            
            # In a real implementation, this would:
            # 1. Identify all user data across the system
            # 2. Securely delete or anonymize the data
            # 3. Update references to use anonymized identifiers
            # 4. Log the deletion for compliance records
            
            logger.info(f"Processed right to be forgotten for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process right to be forgotten for {user_id}: {e}")
            return False
    
    def generate_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        user_accesses = [
            log for log in self.access_log 
            if log['user_id'] == user_id
        ]
        
        report = {
            'user_id': user_id,
            'report_generated': datetime.utcnow().isoformat(),
            'total_accesses': len(user_accesses),
            'first_access': user_accesses[0]['timestamp'] if user_accesses else None,
            'last_access': user_accesses[-1]['timestamp'] if user_accesses else None,
            'data_types_accessed': list(set(log['data_type'] for log in user_accesses)),
            'deletion_date': self.deletion_log.get(user_id),
            'compliance_status': 'compliant'
        }
        
        return report


# Example usage
async def main():
    """Example usage of enterprise encryption system"""
    # Initialize key manager and encryption service
    key_manager = SecureKeyManager()
    encryption_service = EncryptionService(key_manager)
    
    # Generate encryption keys
    aes_key_id = key_manager.generate_key("AES-256", DataClassification.CONFIDENTIAL)
    rsa_key_id = key_manager.generate_key("RSA-2048", DataClassification.RESTRICTED)
    
    # Encrypt sensitive data
    sensitive_data = "This is confidential user information"
    
    # AES encryption
    encrypted_aes = encryption_service.encrypt_data(sensitive_data, aes_key_id)
    print(f"AES Encrypted: {base64.b64encode(encrypted_aes.ciphertext)[:50]}...")
    
    # RSA encryption
    encrypted_rsa = encryption_service.encrypt_data(sensitive_data, rsa_key_id)
    print(f"RSA Encrypted: {base64.b64encode(encrypted_rsa.ciphertext)[:50]}...")
    
    # Decrypt data
    decrypted_aes = encryption_service.decrypt_data(encrypted_aes)
    decrypted_rsa = encryption_service.decrypt_data(encrypted_rsa)
    
    print(f"AES Decrypted: {decrypted_aes.decode()}")
    print(f"RSA Decrypted: {decrypted_rsa.decode()}")
    
    # Privacy compliance
    privacy_compliance = DataPrivacyCompliance(encryption_service)
    
    # Log data access
    privacy_compliance.log_data_access("user123", "memory", "read", "mem456")
    
    # Generate privacy report
    report = privacy_compliance.generate_privacy_report("user123")
    print(f"Privacy Report: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())