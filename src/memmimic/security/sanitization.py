"""
Security Sanitization Utilities

Advanced content sanitization to prevent XSS, injection attacks, and other
security vulnerabilities while preserving legitimate content.
"""

import re
import html
import json
import urllib.parse
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SanitizationType(Enum):
    """Types of sanitization operations."""
    HTML_ESCAPE = "html_escape"
    HTML_STRIP = "html_strip"
    SQL_ESCAPE = "sql_escape"
    JSON_SAFE = "json_safe"
    URL_ENCODE = "url_encode"
    FILENAME_SAFE = "filename_safe"
    UNICODE_NORMALIZE = "unicode_normalize"
    CONTROL_CHARS_STRIP = "control_chars_strip"
    WHITESPACE_NORMALIZE = "whitespace_normalize"


@dataclass
class SanitizationResult:
    """Result of sanitization operation."""
    original_value: Any
    sanitized_value: Any
    operations_applied: List[str]
    security_issues_found: List[Dict[str, str]]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.operations_applied is None:
            self.operations_applied = []
        if self.security_issues_found is None:
            self.security_issues_found = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class SecuritySanitizer:
    """
    Advanced security sanitization engine.
    
    Provides comprehensive sanitization for different contexts:
    - HTML/XSS prevention
    - SQL injection prevention
    - Filename safety
    - Unicode normalization
    - Control character removal
    """
    
    # Dangerous HTML tags and attributes
    DANGEROUS_TAGS = [
        'script', 'iframe', 'object', 'embed', 'applet', 'link', 'style',
        'meta', 'form', 'input', 'button', 'textarea', 'select', 'option'
    ]
    
    DANGEROUS_ATTRIBUTES = [
        'onload', 'onunload', 'onclick', 'onmouseover', 'onmouseout',
        'onfocus', 'onblur', 'onchange', 'onsubmit', 'onreset',
        'onerror', 'onabort', 'onresize', 'onmove', 'ondragdrop'
    ]
    
    DANGEROUS_PROTOCOLS = [
        'javascript:', 'vbscript:', 'data:', 'mailto:', 'ftp:', 'file:'
    ]
    
    # SQL injection patterns for escaping
    SQL_SPECIAL_CHARS = ["'", '"', '\\', '\x00', '\n', '\r', '\x1a']
    
    # Filesystem unsafe characters
    FILESYSTEM_UNSAFE = r'[<>:"|?*\\\x00-\x1f\x7f-\x9f]'
    
    # Control characters (excluding common whitespace)
    CONTROL_CHARS_PATTERN = r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sanitizer with configuration.
        
        Args:
            config: Sanitization configuration
        """
        self.config = config or self._get_default_config()
        
        # Compile regex patterns for performance
        self._html_tag_pattern = re.compile(r'<[^>]+>', re.IGNORECASE)
        self._protocol_pattern = re.compile(
            '|'.join(re.escape(p) for p in self.DANGEROUS_PROTOCOLS), 
            re.IGNORECASE
        )
        self._filesystem_pattern = re.compile(self.FILESYSTEM_UNSAFE)
        self._control_chars_pattern = re.compile(self.CONTROL_CHARS_PATTERN)
        
        logger.info("SecuritySanitizer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default sanitization configuration."""
        return {
            'default_encoding': 'utf-8',
            'max_unicode_length': 1000000,  # 1MB
            'preserve_whitespace': True,
            'aggressive_html_stripping': False,
            'normalize_unicode': True,
            'strip_control_chars': True,
            'filename_replacement_char': '_',
            'sql_escape_quote_char': "''",  # Standard SQL escaping
            'log_sanitization_actions': True
        }
    
    def sanitize_memory_content(self, content: str, 
                              preserve_formatting: bool = True) -> SanitizationResult:
        """
        Sanitize memory content for safe storage and retrieval.
        
        Args:
            content: Content to sanitize
            preserve_formatting: Whether to preserve basic formatting
            
        Returns:
            SanitizationResult with sanitized content
        """
        result = SanitizationResult(
            original_value=content,
            sanitized_value=content,
            operations_applied=[],
            security_issues_found=[],
            warnings=[],
            metadata={'original_length': len(content)}
        )
        
        try:
            sanitized = content
            
            # Unicode normalization
            if self.config['normalize_unicode']:
                normalized = unicodedata.normalize('NFKC', sanitized)
                if normalized != sanitized:
                    result.operations_applied.append('unicode_normalize')
                    sanitized = normalized
            
            # Strip control characters (but preserve newlines/tabs if formatting preserved)
            if self.config['strip_control_chars']:
                if preserve_formatting:
                    # Keep \n, \r, \t but remove other control chars
                    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', sanitized)
                else:
                    cleaned = self._control_chars_pattern.sub('', sanitized)
                
                if cleaned != sanitized:
                    result.operations_applied.append('control_chars_strip')
                    sanitized = cleaned
            
            # Detect and handle potential XSS
            xss_issues = self._detect_xss_patterns(sanitized)
            if xss_issues:
                result.security_issues_found.extend(xss_issues)
                
                if preserve_formatting:
                    # Light HTML escaping - escape dangerous parts but preserve basic formatting
                    sanitized = self._selective_html_escape(sanitized)
                    result.operations_applied.append('selective_html_escape')
                else:
                    # Full HTML escaping
                    sanitized = html.escape(sanitized)
                    result.operations_applied.append('html_escape')
            
            # Detect potential SQL injection patterns
            sql_issues = self._detect_sql_patterns(sanitized)
            if sql_issues:
                result.security_issues_found.extend(sql_issues)
                # SQL escaping for content (double single quotes)
                if "'" in sanitized:
                    sanitized = sanitized.replace("'", "''")
                    result.operations_applied.append('sql_escape')
            
            # Normalize excessive whitespace
            if not preserve_formatting:
                normalized_ws = re.sub(r'\s+', ' ', sanitized).strip()
                if normalized_ws != sanitized:
                    result.operations_applied.append('whitespace_normalize')
                    sanitized = normalized_ws
            
            result.sanitized_value = sanitized
            result.metadata['sanitized_length'] = len(sanitized)
            result.metadata['size_change'] = len(sanitized) - len(content)
            
        except Exception as e:
            logger.error(f"Content sanitization failed: {e}")
            result.warnings.append(f"Sanitization error: {str(e)}")
            result.sanitized_value = content  # Return original on error
        
        return result
    
    def sanitize_filename(self, filename: str) -> SanitizationResult:
        """
        Sanitize filename for filesystem safety.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            SanitizationResult with safe filename
        """
        result = SanitizationResult(
            original_value=filename,
            sanitized_value=filename,
            operations_applied=[],
            security_issues_found=[],
            warnings=[],
            metadata={}
        )
        
        try:
            sanitized = filename
            
            # Remove/replace unsafe characters
            unsafe_replaced = self._filesystem_pattern.sub(
                self.config['filename_replacement_char'], sanitized
            )
            if unsafe_replaced != sanitized:
                result.operations_applied.append('unsafe_chars_replace')
                sanitized = unsafe_replaced
            
            # Normalize unicode
            if self.config['normalize_unicode']:
                normalized = unicodedata.normalize('NFKC', sanitized)
                if normalized != sanitized:
                    result.operations_applied.append('unicode_normalize')
                    sanitized = normalized
            
            # Remove control characters
            cleaned = self._control_chars_pattern.sub('', sanitized)
            if cleaned != sanitized:
                result.operations_applied.append('control_chars_strip')
                sanitized = cleaned
            
            # Trim and handle edge cases
            sanitized = sanitized.strip()
            
            # Check for reserved Windows filenames
            reserved_names = [
                'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
            ]
            
            if sanitized.upper().split('.')[0] in reserved_names:
                sanitized = f"safe_{sanitized}"
                result.operations_applied.append('reserved_name_prefix')
                result.warnings.append("Filename was a reserved system name")
            
            # Ensure filename is not empty or just dots
            if not sanitized or sanitized in ['.', '..']:
                sanitized = 'untitled'
                result.operations_applied.append('empty_name_replace')
                result.warnings.append("Filename was empty or invalid, replaced with 'untitled'")
            
            # Limit length (255 chars is typical filesystem limit)
            if len(sanitized.encode('utf-8')) > 255:
                # Truncate while preserving extension
                name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
                max_name_length = 255 - len(ext.encode('utf-8')) - (1 if ext else 0)
                
                truncated_name = name.encode('utf-8')[:max_name_length].decode('utf-8', errors='ignore')
                sanitized = f"{truncated_name}.{ext}" if ext else truncated_name
                
                result.operations_applied.append('length_truncate')
                result.warnings.append("Filename was too long and was truncated")
            
            result.sanitized_value = sanitized
            
        except Exception as e:
            logger.error(f"Filename sanitization failed: {e}")
            result.warnings.append(f"Sanitization error: {str(e)}")
            result.sanitized_value = 'safe_filename'  # Safe fallback
        
        return result
    
    def sanitize_json_input(self, json_data: Union[str, Dict, List]) -> SanitizationResult:
        """
        Sanitize JSON input for safe processing.
        
        Args:
            json_data: JSON data to sanitize (string or parsed)
            
        Returns:
            SanitizationResult with sanitized JSON
        """
        result = SanitizationResult(
            original_value=json_data,
            sanitized_value=json_data,
            operations_applied=[],
            security_issues_found=[],
            warnings=[],
            metadata={}
        )
        
        try:
            # Parse JSON if string
            if isinstance(json_data, str):
                try:
                    parsed_data = json.loads(json_data)
                    result.operations_applied.append('json_parse')
                except json.JSONDecodeError as e:
                    result.warnings.append(f"Invalid JSON: {str(e)}")
                    result.sanitized_value = {}
                    return result
            else:
                parsed_data = json_data
            
            # Recursively sanitize JSON structure
            sanitized_data = self._sanitize_json_structure(parsed_data)
            
            result.sanitized_value = sanitized_data
            result.operations_applied.append('recursive_sanitize')
            
        except Exception as e:
            logger.error(f"JSON sanitization failed: {e}")
            result.warnings.append(f"Sanitization error: {str(e)}")
            result.sanitized_value = {}  # Safe fallback
        
        return result
    
    def sanitize_sql_value(self, value: str) -> SanitizationResult:
        """
        Sanitize value for safe SQL usage.
        
        Args:
            value: Value to sanitize for SQL
            
        Returns:
            SanitizationResult with SQL-safe value
        """
        result = SanitizationResult(
            original_value=value,
            sanitized_value=value,
            operations_applied=[],
            security_issues_found=[],
            warnings=[],
            metadata={}
        )
        
        try:
            sanitized = value
            
            # Detect SQL injection patterns
            sql_issues = self._detect_sql_patterns(sanitized)
            if sql_issues:
                result.security_issues_found.extend(sql_issues)
            
            # Escape SQL special characters
            for char in self.SQL_SPECIAL_CHARS:
                if char in sanitized:
                    if char == "'":
                        sanitized = sanitized.replace("'", "''")
                    elif char == '"':
                        sanitized = sanitized.replace('"', '""')
                    elif char == '\\':
                        sanitized = sanitized.replace('\\', '\\\\')
                    elif char == '\x00':
                        sanitized = sanitized.replace('\x00', '')
                    elif char == '\n':
                        sanitized = sanitized.replace('\n', '\\n')
                    elif char == '\r':
                        sanitized = sanitized.replace('\r', '\\r')
                    elif char == '\x1a':
                        sanitized = sanitized.replace('\x1a', '\\Z')
                    
                    if char in self.SQL_SPECIAL_CHARS[:3]:  # Only log for main escape chars
                        result.operations_applied.append(f'sql_escape_{repr(char)}')
            
            result.sanitized_value = sanitized
            
        except Exception as e:
            logger.error(f"SQL sanitization failed: {e}")
            result.warnings.append(f"Sanitization error: {str(e)}")
            result.sanitized_value = value  # Return original on error
        
        return result
    
    def _detect_xss_patterns(self, content: str) -> List[Dict[str, str]]:
        """Detect potential XSS patterns in content."""
        issues = []
        
        # Check for script tags
        if re.search(r'<script[^>]*>.*?</script>', content, re.IGNORECASE | re.DOTALL):
            issues.append({
                'type': 'xss_script_tag',
                'description': 'Script tag detected in content',
                'severity': 'HIGH'
            })
        
        # Check for event handlers
        if re.search(r'on\w+\s*=', content, re.IGNORECASE):
            issues.append({
                'type': 'xss_event_handler',
                'description': 'Event handler attribute detected',
                'severity': 'MEDIUM'
            })
        
        # Check for dangerous protocols
        if self._protocol_pattern.search(content):
            issues.append({
                'type': 'xss_dangerous_protocol',
                'description': 'Dangerous protocol (javascript:, data:, etc.) detected',
                'severity': 'HIGH'
            })
        
        return issues
    
    def _detect_sql_patterns(self, content: str) -> List[Dict[str, str]]:
        """Detect potential SQL injection patterns."""
        issues = []
        
        # Common SQL injection patterns
        patterns = [
            (r"(?i)\s*(union|select|insert|update|delete|drop|alter)\s+", 'sql_command', 'HIGH'),
            (r"(?i)\s*(or|and)\s+\w+\s*=\s*\w+", 'sql_condition', 'MEDIUM'),
            (r"(?i)\s*;\s*(drop|delete|update)", 'sql_chaining', 'CRITICAL'),
            (r"(?i)\s*--\s*", 'sql_comment', 'LOW'),
            (r"(?i)\s*/\*.*?\*/", 'sql_comment_block', 'LOW')
        ]
        
        for pattern, threat_type, severity in patterns:
            if re.search(pattern, content):
                issues.append({
                    'type': threat_type,
                    'description': f'Potential SQL injection pattern detected: {threat_type}',
                    'severity': severity
                })
        
        return issues
    
    def _selective_html_escape(self, content: str) -> str:
        """
        Perform selective HTML escaping to preserve basic formatting.
        
        Escapes dangerous elements while preserving basic text formatting.
        """
        # Escape potentially dangerous characters but preserve some formatting
        content = content.replace('&', '&amp;')  # Must be first
        content = content.replace('<script', '&lt;script')
        content = content.replace('javascript:', 'javascript&#58;')
        content = content.replace('vbscript:', 'vbscript&#58;')
        content = content.replace('onload=', 'onload&#61;')
        content = content.replace('onerror=', 'onerror&#61;')
        content = content.replace('onclick=', 'onclick&#61;')
        
        # Escape other event handlers
        content = re.sub(r'(on\w+)=', r'\1&#61;', content, flags=re.IGNORECASE)
        
        return content
    
    def _sanitize_json_structure(self, data: Any, max_depth: int = 10, 
                               current_depth: int = 0) -> Any:
        """
        Recursively sanitize JSON structure.
        
        Args:
            data: Data to sanitize
            max_depth: Maximum nesting depth allowed
            current_depth: Current nesting depth
            
        Returns:
            Sanitized data structure
        """
        if current_depth > max_depth:
            return "[TRUNCATED: Max depth exceeded]"
        
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Sanitize key
                if isinstance(key, str):
                    sanitized_key = self._sanitize_json_string(key)
                    sanitized_value = self._sanitize_json_structure(
                        value, max_depth, current_depth + 1
                    )
                    sanitized[sanitized_key] = sanitized_value
                else:
                    # Convert non-string keys to strings
                    sanitized[str(key)] = self._sanitize_json_structure(
                        value, max_depth, current_depth + 1
                    )
            return sanitized
        
        elif isinstance(data, list):
            return [
                self._sanitize_json_structure(item, max_depth, current_depth + 1)
                for item in data
            ]
        
        elif isinstance(data, str):
            return self._sanitize_json_string(data)
        
        elif isinstance(data, (int, float, bool)) or data is None:
            return data
        
        else:
            # Convert unknown types to strings
            return str(data)
    
    def _sanitize_json_string(self, text: str) -> str:
        """Sanitize individual JSON string values."""
        # Basic sanitization for JSON strings
        sanitized = text
        
        # Remove control characters
        sanitized = self._control_chars_pattern.sub('', sanitized)
        
        # Truncate if too long
        if len(sanitized) > 10000:  # 10KB limit for individual strings
            sanitized = sanitized[:10000] + "...[TRUNCATED]"
        
        return sanitized


# Global sanitizer instance
_global_sanitizer: Optional[SecuritySanitizer] = None


def get_security_sanitizer() -> SecuritySanitizer:
    """Get the global security sanitizer instance."""
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = SecuritySanitizer()
    return _global_sanitizer


def sanitize_content(content: str, context: str = "memory") -> str:
    """
    Quick sanitization function for common use cases.
    
    Args:
        content: Content to sanitize
        context: Context for sanitization ("memory", "filename", "sql", "json")
        
    Returns:
        Sanitized content
    """
    sanitizer = get_security_sanitizer()
    
    if context == "memory":
        result = sanitizer.sanitize_memory_content(content)
    elif context == "filename":
        result = sanitizer.sanitize_filename(content)
    elif context == "sql":
        result = sanitizer.sanitize_sql_value(content)
    elif context == "json":
        result = sanitizer.sanitize_json_input(content)
    else:
        # Default to memory content sanitization
        result = sanitizer.sanitize_memory_content(content)
    
    return result.sanitized_value