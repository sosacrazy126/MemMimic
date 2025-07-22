"""
MemMimic Security Module

Comprehensive input validation and security framework to protect against
injection attacks and other security vulnerabilities.

Components:
- validation: Core input validation engine
- sanitization: Content sanitization utilities
- decorators: Security validation decorators
- schemas: Input validation schemas
- config: Security configuration management
- audit: Security event logging and monitoring
"""

from .validation import InputValidator, ValidationError, SecurityValidationError
from .sanitization import SecuritySanitizer, SanitizationResult
from .decorators import (
    validate_input, sanitize_output, rate_limit, audit_security,
    validate_memory_content, validate_tale_input, validate_query_input
)
from .schemas import (
    MemoryInputSchema, TaleInputSchema, QueryInputSchema,
    MCPRequestSchema, get_validation_schema
)
from .config import SecurityConfig, get_security_config
from .audit import SecurityAuditLogger, SecurityEvent

__all__ = [
    # Core validation
    'InputValidator', 'ValidationError', 'SecurityValidationError',
    
    # Sanitization
    'SecuritySanitizer', 'SanitizationResult',
    
    # Decorators
    'validate_input', 'sanitize_output', 'rate_limit', 'audit_security',
    'validate_memory_content', 'validate_tale_input', 'validate_query_input',
    
    # Schemas
    'MemoryInputSchema', 'TaleInputSchema', 'QueryInputSchema',
    'MCPRequestSchema', 'get_validation_schema',
    
    # Configuration
    'SecurityConfig', 'get_security_config',
    
    # Auditing
    'SecurityAuditLogger', 'SecurityEvent'
]

# Version and metadata
__version__ = "1.0.0"
__security_level__ = "HIGH"
__protection_coverage__ = [
    "SQL Injection Prevention",
    "XSS Protection", 
    "Path Traversal Prevention",
    "DoS Protection (Size Limits)",
    "Input Sanitization",
    "Malformed Request Protection",
    "Security Audit Logging"
]