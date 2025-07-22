"""
Core Input Validation Engine

Comprehensive input validation to prevent injection attacks and ensure data integrity.
Implements defense-in-depth validation with configurable security policies.
"""

import re
import json
import html
import unicodedata
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None, 
                 validation_type: str = None):
        self.field = field
        self.value = value
        self.validation_type = validation_type
        super().__init__(message)


class SecurityValidationError(ValidationError):
    """Exception for security-critical validation failures."""
    
    def __init__(self, message: str, field: str = None, threat_type: str = None,
                 severity: str = "HIGH"):
        self.threat_type = threat_type
        self.severity = severity
        super().__init__(message, field)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any = None
    errors: List[ValidationError] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class InputValidator:
    """
    Core input validation engine with comprehensive security checks.
    
    Provides multiple validation layers:
    1. Size and format validation
    2. Content sanitization
    3. Security pattern detection
    4. Business logic validation
    """
    
    # Dangerous patterns that could indicate attacks
    SQL_INJECTION_PATTERNS = [
        r"(?i)(\s*(union|select|insert|update|delete|drop|alter|create|exec|execute)\s+)",
        r"(?i)(\s*(or|and)\s+\w+\s*=\s*\w+)",
        r"(?i)(\s*;\s*(drop|delete|update|insert)\s+)",
        r"(?i)(\s*--\s*)",
        r"(?i)(\s*/\*.*?\*/\s*)",
        r"(?i)(\s*'\s*(or|and)\s+'\w+'\s*=\s*'\w+')",
    ]
    
    XSS_PATTERNS = [
        r"(?i)<script[^>]*>.*?</script>",
        r"(?i)<iframe[^>]*>.*?</iframe>",
        r"(?i)javascript:",
        r"(?i)on\w+\s*=",
        r"(?i)expression\s*\(",
        r"(?i)vbscript:",
        r"(?i)data:\s*text/html",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.(/|\\)",
        r"(^|/)\.\.($|/)",
        r"\\\\",
        r"/etc/passwd",
        r"/proc/",
        r"C:\\Windows",
        r"%2e%2e",
        r"%252e%252e",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r";\s*(ls|cat|rm|mv|cp|chmod|sudo|su|wget|curl)",
        r"\|{1,2}\s*(ls|cat|rm|mv|cp|chmod|sudo|su|wget|curl)",
        r"&&\s*(ls|cat|rm|mv|cp|chmod|sudo|su|wget|curl)",
        r"`.*`",
        r"\$\(.*\)",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the input validator.
        
        Args:
            config: Security configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Compile regex patterns for performance
        self._sql_patterns = [re.compile(p) for p in self.SQL_INJECTION_PATTERNS]
        self._xss_patterns = [re.compile(p) for p in self.XSS_PATTERNS]
        self._path_patterns = [re.compile(p) for p in self.PATH_TRAVERSAL_PATTERNS]
        self._cmd_patterns = [re.compile(p) for p in self.COMMAND_INJECTION_PATTERNS]
        
        logger.info("InputValidator initialized with security policies")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default security configuration."""
        return {
            'max_content_length': 100 * 1024,  # 100KB
            'max_request_size': 1024 * 1024,   # 1MB
            'max_query_length': 1000,
            'max_tale_name_length': 200,
            'max_category_length': 100,
            'allowed_tale_categories': [
                'claude/core', 'claude/contexts', 'claude/insights', 
                'claude/current', 'claude/archive', 'projects/', 'misc/'
            ],
            'enable_content_sanitization': True,
            'enable_path_validation': True,
            'enable_sql_injection_detection': True,
            'enable_xss_detection': True,
            'enable_command_injection_detection': True,
            'normalize_unicode': True,
            'strip_control_characters': True,
            'log_security_violations': True,
            'strict_mode': True
        }
    
    def validate_memory_content(self, content: str, memory_type: str = None) -> ValidationResult:
        """
        Validate memory content for storage.
        
        Args:
            content: Memory content to validate
            memory_type: Type of memory (interaction, reflection, milestone)
            
        Returns:
            ValidationResult with validation status and sanitized content
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Size validation
            if len(content) > self.config['max_content_length']:
                result.errors.append(ValidationError(
                    f"Content exceeds maximum length ({self.config['max_content_length']} characters)",
                    field="content",
                    validation_type="size_limit"
                ))
                result.is_valid = False
            
            # Content sanitization and security checks
            sanitized_content, security_issues = self._sanitize_and_validate_content(content)
            result.sanitized_value = sanitized_content
            
            # Add security violations as errors
            for issue in security_issues:
                result.errors.append(SecurityValidationError(
                    f"Security violation detected: {issue['type']} - {issue['description']}",
                    field="content",
                    threat_type=issue['type'],
                    severity=issue['severity']
                ))
                if issue['severity'] in ['HIGH', 'CRITICAL']:
                    result.is_valid = False
            
            # Memory type validation
            if memory_type and memory_type not in ['interaction', 'reflection', 'milestone']:
                result.warnings.append(f"Unknown memory type: {memory_type}")
            
            result.metadata['original_length'] = len(content)
            result.metadata['sanitized_length'] = len(sanitized_content)
            result.metadata['security_issues_found'] = len(security_issues)
            
        except Exception as e:
            logger.error(f"Memory content validation failed: {e}")
            result.errors.append(ValidationError(
                f"Validation failed: {str(e)}",
                field="content",
                validation_type="internal_error"
            ))
            result.is_valid = False
        
        return result
    
    def validate_tale_input(self, name: str, content: str, category: str = None, 
                          tags: List[str] = None) -> ValidationResult:
        """
        Validate tale input data.
        
        Args:
            name: Tale name
            content: Tale content
            category: Tale category
            tags: List of tags
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True)
        sanitized_data = {}
        
        try:
            # Validate tale name
            name_result = self._validate_tale_name(name)
            if not name_result.is_valid:
                result.errors.extend(name_result.errors)
                result.is_valid = False
            sanitized_data['name'] = name_result.sanitized_value
            
            # Validate content
            content_result = self.validate_memory_content(content)
            if not content_result.is_valid:
                result.errors.extend(content_result.errors)
                result.is_valid = False
            sanitized_data['content'] = content_result.sanitized_value
            
            # Validate category
            if category:
                category_result = self._validate_tale_category(category)
                if not category_result.is_valid:
                    result.errors.extend(category_result.errors)
                    result.is_valid = False
                sanitized_data['category'] = category_result.sanitized_value
            
            # Validate tags
            if tags:
                tags_result = self._validate_tale_tags(tags)
                if not tags_result.is_valid:
                    result.errors.extend(tags_result.errors)
                    result.is_valid = False
                sanitized_data['tags'] = tags_result.sanitized_value
            
            result.sanitized_value = sanitized_data
            
        except Exception as e:
            logger.error(f"Tale input validation failed: {e}")
            result.errors.append(ValidationError(
                f"Tale validation failed: {str(e)}",
                validation_type="internal_error"
            ))
            result.is_valid = False
        
        return result
    
    def validate_query_input(self, query: str, limit: int = None, 
                           filters: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate search query input.
        
        Args:
            query: Search query text
            limit: Result limit
            filters: Additional filters
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True)
        sanitized_data = {}
        
        try:
            # Query text validation
            if not query or not query.strip():
                result.errors.append(ValidationError(
                    "Query cannot be empty",
                    field="query",
                    validation_type="required"
                ))
                result.is_valid = False
                return result
            
            if len(query) > self.config['max_query_length']:
                result.errors.append(ValidationError(
                    f"Query exceeds maximum length ({self.config['max_query_length']} characters)",
                    field="query",
                    validation_type="size_limit"
                ))
                result.is_valid = False
            
            # Sanitize query and check for security issues
            sanitized_query, security_issues = self._sanitize_and_validate_content(query)
            sanitized_data['query'] = sanitized_query
            
            # Check for SQL injection in query
            for issue in security_issues:
                if issue['type'] == 'sql_injection':
                    result.errors.append(SecurityValidationError(
                        f"Potentially malicious query detected: {issue['description']}",
                        field="query",
                        threat_type="sql_injection",
                        severity="CRITICAL"
                    ))
                    result.is_valid = False
            
            # Limit validation
            if limit is not None:
                if not isinstance(limit, int) or limit <= 0 or limit > 1000:
                    result.errors.append(ValidationError(
                        "Limit must be a positive integer between 1 and 1000",
                        field="limit",
                        validation_type="range"
                    ))
                    result.is_valid = False
                else:
                    sanitized_data['limit'] = limit
            
            # Filters validation
            if filters:
                try:
                    # Ensure filters is a dict and validate its content
                    if not isinstance(filters, dict):
                        result.errors.append(ValidationError(
                            "Filters must be a dictionary",
                            field="filters",
                            validation_type="type"
                        ))
                        result.is_valid = False
                    else:
                        sanitized_filters = self._validate_filters(filters)
                        sanitized_data['filters'] = sanitized_filters
                except Exception as e:
                    result.errors.append(ValidationError(
                        f"Invalid filters: {str(e)}",
                        field="filters",
                        validation_type="format"
                    ))
                    result.is_valid = False
            
            result.sanitized_value = sanitized_data
            result.metadata['query_length'] = len(query)
            result.metadata['security_issues_found'] = len(security_issues)
            
        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            result.errors.append(ValidationError(
                f"Query validation failed: {str(e)}",
                validation_type="internal_error"
            ))
            result.is_valid = False
        
        return result
    
    def validate_json_input(self, json_str: str, max_size: int = None) -> ValidationResult:
        """
        Validate JSON input with size and structure checks.
        
        Args:
            json_str: JSON string to validate
            max_size: Maximum JSON size in bytes
            
        Returns:
            ValidationResult with parsed JSON if valid
        """
        result = ValidationResult(is_valid=True)
        max_size = max_size or self.config['max_request_size']
        
        try:
            # Size check
            if len(json_str.encode('utf-8')) > max_size:
                result.errors.append(ValidationError(
                    f"JSON input exceeds maximum size ({max_size} bytes)",
                    field="json",
                    validation_type="size_limit"
                ))
                result.is_valid = False
                return result
            
            # Parse JSON
            try:
                parsed_data = json.loads(json_str)
                result.sanitized_value = parsed_data
            except json.JSONDecodeError as e:
                result.errors.append(ValidationError(
                    f"Invalid JSON format: {str(e)}",
                    field="json",
                    validation_type="format"
                ))
                result.is_valid = False
                return result
            
            # Structure validation for deeply nested objects
            if self._has_excessive_nesting(parsed_data, max_depth=10):
                result.errors.append(SecurityValidationError(
                    "JSON structure has excessive nesting (possible DoS attack)",
                    field="json",
                    threat_type="dos_attack",
                    severity="HIGH"
                ))
                result.is_valid = False
            
            result.metadata['json_size'] = len(json_str.encode('utf-8'))
            result.metadata['parsed_type'] = type(parsed_data).__name__
            
        except Exception as e:
            logger.error(f"JSON validation failed: {e}")
            result.errors.append(ValidationError(
                f"JSON validation failed: {str(e)}",
                validation_type="internal_error"
            ))
            result.is_valid = False
        
        return result
    
    def _sanitize_and_validate_content(self, content: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Sanitize content and detect security issues.
        
        Returns:
            Tuple of (sanitized_content, security_issues)
        """
        sanitized = content
        security_issues = []
        
        # Unicode normalization
        if self.config['normalize_unicode']:
            sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Strip control characters
        if self.config['strip_control_characters']:
            sanitized = ''.join(char for char in sanitized 
                               if unicodedata.category(char) != 'Cc')
        
        # Detect SQL injection
        if self.config['enable_sql_injection_detection']:
            for pattern in self._sql_patterns:
                if pattern.search(sanitized):
                    security_issues.append({
                        'type': 'sql_injection',
                        'description': 'Potential SQL injection pattern detected',
                        'severity': 'CRITICAL'
                    })
                    break
        
        # Detect XSS
        if self.config['enable_xss_detection']:
            for pattern in self._xss_patterns:
                if pattern.search(sanitized):
                    security_issues.append({
                        'type': 'xss',
                        'description': 'Potential XSS pattern detected',
                        'severity': 'HIGH'
                    })
                    # Sanitize HTML if XSS detected
                    sanitized = html.escape(sanitized)
                    break
        
        # Detect command injection
        if self.config['enable_command_injection_detection']:
            for pattern in self._cmd_patterns:
                if pattern.search(sanitized):
                    security_issues.append({
                        'type': 'command_injection',
                        'description': 'Potential command injection pattern detected',
                        'severity': 'HIGH'
                    })
                    break
        
        # Detect path traversal
        if self.config['enable_path_validation']:
            for pattern in self._path_patterns:
                if pattern.search(sanitized):
                    security_issues.append({
                        'type': 'path_traversal',
                        'description': 'Potential path traversal pattern detected',
                        'severity': 'HIGH'
                    })
                    break
        
        return sanitized, security_issues
    
    def _validate_tale_name(self, name: str) -> ValidationResult:
        """Validate tale name for filesystem safety."""
        result = ValidationResult(is_valid=True)
        
        if not name or not name.strip():
            result.errors.append(ValidationError(
                "Tale name cannot be empty",
                field="name",
                validation_type="required"
            ))
            result.is_valid = False
            return result
        
        if len(name) > self.config['max_tale_name_length']:
            result.errors.append(ValidationError(
                f"Tale name exceeds maximum length ({self.config['max_tale_name_length']} characters)",
                field="name",
                validation_type="size_limit"
            ))
            result.is_valid = False
        
        # Check for path traversal in name
        sanitized_name, security_issues = self._sanitize_and_validate_content(name)
        for issue in security_issues:
            if issue['type'] == 'path_traversal':
                result.errors.append(SecurityValidationError(
                    "Tale name contains potentially dangerous characters",
                    field="name",
                    threat_type="path_traversal",
                    severity="HIGH"
                ))
                result.is_valid = False
        
        # Sanitize for filesystem
        sanitized_name = re.sub(r'[<>:"|?*\\]', '_', sanitized_name)
        sanitized_name = re.sub(r'[^\w\s\-_.]', '', sanitized_name)
        sanitized_name = re.sub(r'\s+', '_', sanitized_name.strip()).lower()
        
        result.sanitized_value = sanitized_name
        return result
    
    def _validate_tale_category(self, category: str) -> ValidationResult:
        """Validate tale category."""
        result = ValidationResult(is_valid=True)
        
        if len(category) > self.config['max_category_length']:
            result.errors.append(ValidationError(
                f"Category exceeds maximum length ({self.config['max_category_length']} characters)",
                field="category",
                validation_type="size_limit"
            ))
            result.is_valid = False
        
        # Validate against allowed categories
        allowed_prefixes = self.config['allowed_tale_categories']
        if not any(category.startswith(prefix) for prefix in allowed_prefixes):
            result.warnings.append(f"Category '{category}' not in standard categories")
        
        # Check for path traversal
        sanitized_category, security_issues = self._sanitize_and_validate_content(category)
        for issue in security_issues:
            if issue['type'] == 'path_traversal':
                result.errors.append(SecurityValidationError(
                    "Category contains potentially dangerous path elements",
                    field="category",
                    threat_type="path_traversal",
                    severity="HIGH"
                ))
                result.is_valid = False
        
        # Sanitize category path
        sanitized_category = category.replace('\\', '/').strip('/')
        path_parts = [part for part in sanitized_category.split('/') if part and part != '..' and part != '.']
        sanitized_category = '/'.join(path_parts)
        
        result.sanitized_value = sanitized_category
        return result
    
    def _validate_tale_tags(self, tags: List[str]) -> ValidationResult:
        """Validate tale tags."""
        result = ValidationResult(is_valid=True)
        sanitized_tags = []
        
        for tag in tags:
            if not isinstance(tag, str):
                result.errors.append(ValidationError(
                    "Tags must be strings",
                    field="tags",
                    validation_type="type"
                ))
                result.is_valid = False
                continue
            
            # Sanitize tag
            sanitized_tag = re.sub(r'[^\w\-_]', '_', tag.strip())
            if sanitized_tag:
                sanitized_tags.append(sanitized_tag.lower())
        
        # Limit number of tags
        if len(sanitized_tags) > 10:
            result.warnings.append("Too many tags, consider reducing for better organization")
        
        result.sanitized_value = sanitized_tags
        return result
    
    def _validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize filter dictionary."""
        sanitized_filters = {}
        
        for key, value in filters.items():
            # Sanitize filter key
            if not isinstance(key, str) or not key.strip():
                continue
            
            sanitized_key = re.sub(r'[^\w_]', '_', key.strip())
            
            # Validate filter value
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, str):
                    sanitized_value, _ = self._sanitize_and_validate_content(value)
                    sanitized_filters[sanitized_key] = sanitized_value
                else:
                    sanitized_filters[sanitized_key] = value
            elif isinstance(value, list):
                # Sanitize list values
                sanitized_list = []
                for item in value:
                    if isinstance(item, str):
                        sanitized_item, _ = self._sanitize_and_validate_content(item)
                        sanitized_list.append(sanitized_item)
                    elif isinstance(item, (int, float, bool)):
                        sanitized_list.append(item)
                sanitized_filters[sanitized_key] = sanitized_list
        
        return sanitized_filters
    
    def _has_excessive_nesting(self, data: Any, current_depth: int = 0, max_depth: int = 10) -> bool:
        """Check if data structure has excessive nesting."""
        if current_depth > max_depth:
            return True
        
        if isinstance(data, dict):
            return any(self._has_excessive_nesting(v, current_depth + 1, max_depth) 
                      for v in data.values())
        elif isinstance(data, list):
            return any(self._has_excessive_nesting(item, current_depth + 1, max_depth) 
                      for item in data)
        
        return False


# Global validator instance
_global_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """Get the global input validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = InputValidator()
    return _global_validator


def initialize_validator(config: Optional[Dict[str, Any]] = None) -> InputValidator:
    """Initialize the global validator with custom configuration."""
    global _global_validator
    _global_validator = InputValidator(config)
    return _global_validator