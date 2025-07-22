"""
Secure credential management for MemMimic.

This module provides secure credential loading with validation,
secure defaults, and runtime checks to prevent credential exposure.
"""

import os
import warnings
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class CredentialSecurityError(Exception):
    """Raised when credential security requirements are not met."""
    pass


@dataclass
class SecureCredentials:
    """Secure credential container with validation."""
    
    # API Keys
    anthropic_api_key: Optional[str] = field(default=None, repr=False)
    perplexity_api_key: Optional[str] = field(default=None, repr=False)
    openai_api_key: Optional[str] = field(default=None, repr=False)
    google_api_key: Optional[str] = field(default=None, repr=False)
    mistral_api_key: Optional[str] = field(default=None, repr=False)
    xai_api_key: Optional[str] = field(default=None, repr=False)
    azure_openai_api_key: Optional[str] = field(default=None, repr=False)
    ollama_api_key: Optional[str] = field(default=None, repr=False)
    github_api_key: Optional[str] = field(default=None, repr=False)
    
    # Security settings
    enable_api_key_validation: bool = True
    secure_defaults: bool = True
    log_credential_usage: bool = False
    
    # Runtime validation flags
    _validated: bool = field(default=False, init=False, repr=False)
    _validation_errors: list = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        """Perform security validation after initialization."""
        self._validate_credentials()
        self._check_security_best_practices()
    
    def _validate_credentials(self) -> None:
        """Validate credential formats and detect security issues."""
        validation_patterns = {
            'anthropic_api_key': ('sk-ant-api03-', 'Anthropic API key'),
            'perplexity_api_key': ('pplx-', 'Perplexity API key'),
            'openai_api_key': ('sk-proj-', 'OpenAI API key'),
            'google_api_key': ('AIzaSy', 'Google API key'),
            'github_api_key': (('ghp_', 'github_pat_'), 'GitHub API key'),
        }
        
        for attr_name, (expected_prefix, description) in validation_patterns.items():
            value = getattr(self, attr_name)
            if value and not self._is_placeholder(value):
                if isinstance(expected_prefix, tuple):
                    if not any(value.startswith(prefix) for prefix in expected_prefix):
                        self._validation_errors.append(
                            f"{description} format appears invalid (should start with {' or '.join(expected_prefix)})"
                        )
                else:
                    if not value.startswith(expected_prefix):
                        self._validation_errors.append(
                            f"{description} format appears invalid (should start with {expected_prefix})"
                        )
                
                # Check for potentially exposed/test keys
                if self._looks_like_exposed_key(value):
                    self._validation_errors.append(
                        f"{description} appears to be exposed or a test key - verify security"
                    )
        
        self._validated = True
        
        if self._validation_errors and self.secure_defaults:
            warnings.warn(
                f"Credential validation warnings: {'; '.join(self._validation_errors)}",
                UserWarning
            )
    
    def _is_placeholder(self, value: str) -> bool:
        """Check if a value is a placeholder and not a real credential."""
        placeholder_indicators = [
            'your_', 'YOUR_', 'placeholder', 'PLACEHOLDER',
            'example', 'EXAMPLE', 'test_key', 'TEST_KEY',
            '_here', '_HERE', 'replace_with', 'REPLACE_WITH'
        ]
        return any(indicator in value for indicator in placeholder_indicators)
    
    def _looks_like_exposed_key(self, value: str) -> bool:
        """Check if key looks like it might be exposed/public."""
        # This is a simple heuristic - real security would use more sophisticated detection
        if len(value) < 20:  # Too short to be real
            return True
        if value.count('a') > len(value) * 0.3:  # Too many 'a's (common in test keys)
            return True
        return False
    
    def _check_security_best_practices(self) -> None:
        """Check for security best practices compliance."""
        if not self.enable_api_key_validation:
            warnings.warn(
                "API key validation is disabled - this reduces security",
                UserWarning
            )
        
        if self.log_credential_usage:
            warnings.warn(
                "Credential usage logging is enabled - ensure logs are secure",
                UserWarning
            )
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Securely retrieve API key for a provider."""
        key_mapping = {
            'anthropic': self.anthropic_api_key,
            'perplexity': self.perplexity_api_key,
            'openai': self.openai_api_key,
            'google': self.google_api_key,
            'mistral': self.mistral_api_key,
            'xai': self.xai_api_key,
            'azure_openai': self.azure_openai_api_key,
            'ollama': self.ollama_api_key,
            'github': self.github_api_key,
        }
        
        key = key_mapping.get(provider.lower())
        
        if key and not self._is_placeholder(key):
            if self.log_credential_usage:
                logger.info(f"Retrieved API key for provider: {provider}")
            return key
        
        if self.secure_defaults:
            logger.warning(f"No valid API key found for provider: {provider}")
        
        return None
    
    def validate_required_keys(self, required_providers: list) -> bool:
        """Validate that all required API keys are present and valid."""
        missing_keys = []
        
        for provider in required_providers:
            if not self.get_api_key(provider):
                missing_keys.append(provider)
        
        if missing_keys:
            if self.secure_defaults:
                raise CredentialSecurityError(
                    f"Missing required API keys for providers: {', '.join(missing_keys)}. "
                    f"Please set the appropriate environment variables."
                )
            else:
                logger.warning(f"Missing API keys for providers: {', '.join(missing_keys)}")
                return False
        
        return True
    
    @property
    def security_summary(self) -> Dict[str, Any]:
        """Return a security summary for auditing purposes."""
        configured_providers = []
        placeholder_providers = []
        
        for provider in ['anthropic', 'perplexity', 'openai', 'google', 
                        'mistral', 'xai', 'azure_openai', 'ollama', 'github']:
            key = self.get_api_key(provider)
            if key:
                configured_providers.append(provider)
            else:
                attr_name = f"{provider}_api_key"
                if provider == 'azure_openai':
                    attr_name = 'azure_openai_api_key'
                value = getattr(self, attr_name)
                if value and self._is_placeholder(value):
                    placeholder_providers.append(provider)
        
        return {
            'configured_providers': configured_providers,
            'placeholder_providers': placeholder_providers,
            'validation_enabled': self.enable_api_key_validation,
            'secure_defaults': self.secure_defaults,
            'validation_errors': self._validation_errors,
            'total_validation_errors': len(self._validation_errors)
        }


def load_secure_credentials(env_file: Optional[str] = None) -> SecureCredentials:
    """
    Load credentials securely from environment variables.
    
    Args:
        env_file: Optional path to .env file to load
        
    Returns:
        SecureCredentials instance with validated credentials
        
    Raises:
        CredentialSecurityError: If security requirements are not met
    """
    # Load .env file if specified
    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            _load_env_file(env_path)
        else:
            logger.warning(f"Environment file not found: {env_file}")
    
    # Load from environment variables with secure defaults
    credentials = SecureCredentials(
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        perplexity_api_key=os.getenv('PERPLEXITY_API_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        mistral_api_key=os.getenv('MISTRAL_API_KEY'),
        xai_api_key=os.getenv('XAI_API_KEY'),
        azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        ollama_api_key=os.getenv('OLLAMA_API_KEY'),
        github_api_key=os.getenv('GITHUB_API_KEY'),
        enable_api_key_validation=_parse_bool(os.getenv('ENABLE_API_KEY_VALIDATION', 'true')),
        secure_defaults=_parse_bool(os.getenv('SECURE_DEFAULTS', 'true')),
        log_credential_usage=_parse_bool(os.getenv('LOG_CREDENTIAL_USAGE', 'false')),
    )
    
    return credentials


def _load_env_file(env_path: Path) -> None:
    """Load environment variables from a .env file."""
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
    except Exception as e:
        logger.error(f"Failed to load environment file {env_path}: {e}")


def _parse_bool(value: str) -> bool:
    """Parse a string value to boolean."""
    return value.lower() in ('true', '1', 'yes', 'on')


# Global instance for the application
_global_credentials: Optional[SecureCredentials] = None


def get_credentials() -> SecureCredentials:
    """Get the global credentials instance, loading if necessary."""
    global _global_credentials
    if _global_credentials is None:
        _global_credentials = load_secure_credentials()
    return _global_credentials


def initialize_credentials(env_file: Optional[str] = None, 
                         required_providers: Optional[list] = None) -> SecureCredentials:
    """
    Initialize credentials with validation.
    
    Args:
        env_file: Optional path to .env file
        required_providers: List of required providers (will raise error if missing)
        
    Returns:
        Initialized and validated SecureCredentials instance
    """
    global _global_credentials
    _global_credentials = load_secure_credentials(env_file)
    
    if required_providers:
        _global_credentials.validate_required_keys(required_providers)
    
    logger.info("Credentials initialized successfully")
    security_summary = _global_credentials.security_summary
    logger.info(f"Security summary: {security_summary['total_validation_errors']} validation errors, "
               f"{len(security_summary['configured_providers'])} providers configured")
    
    return _global_credentials


def audit_credential_security() -> Dict[str, Any]:
    """Perform a security audit of current credential configuration."""
    credentials = get_credentials()
    summary = credentials.security_summary
    
    # Additional security checks
    env_file_exists = Path('.env').exists()
    env_template_exists = Path('.env.template').exists()
    gitignore_excludes_env = False
    
    gitignore_path = Path('.gitignore')
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
            gitignore_excludes_env = '.env' in gitignore_content
    
    audit_result = {
        **summary,
        'env_file_exists': env_file_exists,
        'env_template_exists': env_template_exists,
        'gitignore_excludes_env': gitignore_excludes_env,
        'security_score': _calculate_security_score(summary, env_file_exists, 
                                                   env_template_exists, gitignore_excludes_env),
        'recommendations': _generate_security_recommendations(summary, env_file_exists,
                                                            env_template_exists, gitignore_excludes_env)
    }
    
    return audit_result


def _calculate_security_score(summary: Dict, env_exists: bool, 
                            template_exists: bool, gitignore_ok: bool) -> int:
    """Calculate a security score from 0-100."""
    score = 100
    
    # Deduct points for issues
    if summary['total_validation_errors'] > 0:
        score -= min(50, summary['total_validation_errors'] * 10)
    
    if not summary['validation_enabled']:
        score -= 20
    
    if not summary['secure_defaults']:
        score -= 15
    
    if not gitignore_ok:
        score -= 25  # Very important
    
    if not template_exists:
        score -= 10
    
    if len(summary['placeholder_providers']) > 5:
        score -= 5  # Many unconfigured providers
    
    return max(0, score)


def _generate_security_recommendations(summary: Dict, env_exists: bool,
                                     template_exists: bool, gitignore_ok: bool) -> list:
    """Generate security recommendations based on audit results."""
    recommendations = []
    
    if summary['total_validation_errors'] > 0:
        recommendations.append("Fix API key validation errors")
    
    if not gitignore_ok:
        recommendations.append("Add .env to .gitignore to prevent credential exposure")
    
    if not template_exists:
        recommendations.append("Create .env.template file with placeholder values")
    
    if not summary['validation_enabled']:
        recommendations.append("Enable API key validation for better security")
    
    if not summary['secure_defaults']:
        recommendations.append("Enable secure defaults to enforce security policies")
    
    if len(summary['configured_providers']) == 0:
        recommendations.append("Configure at least one API provider")
    
    return recommendations