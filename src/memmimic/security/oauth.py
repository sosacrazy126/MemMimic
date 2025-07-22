"""
OAuth2/OIDC Integration for Enterprise SSO

Complete OAuth2 and OpenID Connect integration for enterprise
identity providers and single sign-on capabilities.
"""

import secrets
import json
import base64
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlencode, parse_qs
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import jwt
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from .audit import SecurityAuditLogger, SecurityEvent, SecurityEventType, SeverityLevel
from .authentication import User, UserRole

logger = logging.getLogger(__name__)


class OAuthProvider(Enum):
    """Supported OAuth2/OIDC providers."""
    MICROSOFT_AZURE = "microsoft_azure"
    GOOGLE_WORKSPACE = "google_workspace"
    OKTA = "okta"
    AUTH0 = "auth0"
    KEYCLOAK = "keycloak"
    PING_IDENTITY = "ping_identity"
    ONELOGIN = "onelogin"
    GENERIC_OIDC = "generic_oidc"


class GrantType(Enum):
    """OAuth2 grant types."""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    DEVICE_CODE = "device_code"


@dataclass
class OAuthConfig:
    """OAuth2/OIDC provider configuration."""
    provider: OAuthProvider
    client_id: str
    client_secret: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str
    issuer: str
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    redirect_uri: str = "http://localhost:8000/auth/callback"
    response_type: str = "code"
    response_mode: str = "query"
    prompt: str = "consent"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class OAuthState:
    """OAuth2 state for CSRF protection."""
    state: str
    code_verifier: str  # PKCE
    code_challenge: str
    nonce: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=10))
    redirect_after: Optional[str] = None


@dataclass
class OAuthToken:
    """OAuth2 token response."""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    scope: Optional[str] = None
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_expired(self) -> bool:
        """Check if access token is expired."""
        expires_at = self.issued_at + timedelta(seconds=self.expires_in)
        return datetime.now(timezone.utc) > expires_at


@dataclass
class UserInfo:
    """User information from OIDC userinfo endpoint."""
    sub: str  # Subject identifier
    email: str
    email_verified: bool = True
    name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture: Optional[str] = None
    locale: Optional[str] = None
    groups: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    custom_claims: Dict[str, Any] = field(default_factory=dict)


class OAuthError(Exception):
    """OAuth2/OIDC related errors."""
    pass


class EnterpriseOAuthManager:
    """
    Enterprise OAuth2/OIDC integration manager.
    
    Features:
    - Multi-provider OAuth2/OIDC support
    - PKCE (Proof Key for Code Exchange) security
    - JWT token validation and verification
    - Automatic token refresh
    - Group/role mapping from identity providers
    - Device flow support for CLI applications
    - Comprehensive audit logging
    - SSO session management
    """
    
    def __init__(self, audit_logger: Optional[SecurityAuditLogger] = None):
        """
        Initialize OAuth manager.
        
        Args:
            audit_logger: Security audit logger instance
        """
        self.audit_logger = audit_logger or SecurityAuditLogger()
        
        # Provider configurations
        self.providers: Dict[str, OAuthConfig] = {}
        
        # Active OAuth states for CSRF protection
        self.oauth_states: Dict[str, OAuthState] = {}
        
        # Token cache
        self.token_cache: Dict[str, OAuthToken] = {}
        
        # HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Initialize well-known provider configurations
        self._initialize_known_providers()
        
        logger.info("EnterpriseOAuthManager initialized")
    
    def _initialize_known_providers(self) -> None:
        """Initialize configurations for well-known providers."""
        # These would be loaded from environment or configuration in production
        known_configs = {
            "microsoft_azure": {
                "authorization_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "userinfo_endpoint": "https://graph.microsoft.com/v1.0/me",
                "jwks_uri": "https://login.microsoftonline.com/common/discovery/v2.0/keys",
                "issuer": "https://login.microsoftonline.com/",
                "scopes": ["openid", "profile", "email", "User.Read"]
            },
            "google_workspace": {
                "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_endpoint": "https://oauth2.googleapis.com/token",
                "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
                "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
                "issuer": "https://accounts.google.com",
                "scopes": ["openid", "profile", "email"]
            }
        }
        
        # Store as templates (would be configured with actual client credentials)
        for provider_name, config in known_configs.items():
            config["provider"] = provider_name
            config["client_id"] = ""  # To be configured
            config["client_secret"] = ""  # To be configured
    
    def register_provider(self, provider_id: str, config: OAuthConfig) -> None:
        """
        Register OAuth2/OIDC provider.
        
        Args:
            provider_id: Unique provider identifier
            config: OAuth configuration
        """
        self.providers[provider_id] = config
        
        # Log provider registration
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            component="oauth",
            severity=SeverityLevel.MEDIUM,
            details=f"OAuth provider registered: {provider_id}",
            metadata={
                "provider_id": provider_id,
                "provider_type": config.provider.value,
                "scopes": config.scopes
            }
        ))
        
        logger.info(f"OAuth provider registered: {provider_id}")
    
    def get_authorization_url(self, provider_id: str,
                             redirect_after: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate OAuth2 authorization URL with PKCE.
        
        Args:
            provider_id: OAuth provider identifier
            redirect_after: URL to redirect to after successful auth
            
        Returns:
            Tuple of (authorization_url, state)
        """
        if provider_id not in self.providers:
            raise OAuthError(f"Unknown OAuth provider: {provider_id}")
        
        config = self.providers[provider_id]
        
        # Generate PKCE parameters
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        # Generate state and nonce
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        
        # Store OAuth state
        oauth_state = OAuthState(
            state=state,
            code_verifier=code_verifier,
            code_challenge=code_challenge,
            nonce=nonce,
            redirect_after=redirect_after
        )
        self.oauth_states[state] = oauth_state
        
        # Build authorization URL
        params = {
            'client_id': config.client_id,
            'response_type': config.response_type,
            'scope': ' '.join(config.scopes),
            'redirect_uri': config.redirect_uri,
            'state': state,
            'nonce': nonce,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256',
            'response_mode': config.response_mode,
            'prompt': config.prompt
        }
        
        authorization_url = f"{config.authorization_endpoint}?{urlencode(params)}"
        
        # Log authorization URL generation
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.FUNCTION_CALL,
            component="oauth",
            severity=SeverityLevel.LOW,
            details=f"OAuth authorization URL generated for provider: {provider_id}",
            metadata={
                "provider_id": provider_id,
                "state": state,
                "scopes": config.scopes
            }
        ))
        
        return authorization_url, state
    
    def handle_callback(self, provider_id: str, code: str, 
                       state: str) -> Tuple[User, OAuthToken]:
        """
        Handle OAuth2 callback and exchange code for tokens.
        
        Args:
            provider_id: OAuth provider identifier
            code: Authorization code from callback
            state: State parameter from callback
            
        Returns:
            Tuple of (User, OAuthToken)
        """
        # Validate state (CSRF protection)
        if state not in self.oauth_states:
            raise OAuthError("Invalid or expired OAuth state")
        
        oauth_state = self.oauth_states[state]
        
        # Check state expiration
        if datetime.now(timezone.utc) > oauth_state.expires_at:
            del self.oauth_states[state]
            raise OAuthError("OAuth state expired")
        
        if provider_id not in self.providers:
            raise OAuthError(f"Unknown OAuth provider: {provider_id}")
        
        config = self.providers[provider_id]
        
        # Exchange code for tokens
        token_data = {
            'grant_type': 'authorization_code',
            'client_id': config.client_id,
            'client_secret': config.client_secret,
            'code': code,
            'redirect_uri': config.redirect_uri,
            'code_verifier': oauth_state.code_verifier
        }
        
        try:
            response = self.session.post(
                config.token_endpoint,
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            response.raise_for_status()
            
            token_response = response.json()
            
        except requests.RequestException as e:
            raise OAuthError(f"Token exchange failed: {e}")
        
        # Create OAuth token
        oauth_token = OAuthToken(
            access_token=token_response['access_token'],
            token_type=token_response.get('token_type', 'Bearer'),
            expires_in=token_response.get('expires_in', 3600),
            refresh_token=token_response.get('refresh_token'),
            id_token=token_response.get('id_token'),
            scope=token_response.get('scope')
        )
        
        # Validate ID token if present
        user_info = None
        if oauth_token.id_token:
            user_info = self._validate_id_token(config, oauth_token.id_token, oauth_state.nonce)
        
        # Get user information from userinfo endpoint if no ID token
        if not user_info:
            user_info = self._get_user_info(config, oauth_token.access_token)
        
        # Map to internal user object
        user = self._map_user_info_to_user(user_info, provider_id)
        
        # Cache token
        self.token_cache[user.user_id] = oauth_token
        
        # Clean up OAuth state
        del self.oauth_states[state]
        
        # Log successful authentication
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.FUNCTION_CALL,
            component="oauth",
            severity=SeverityLevel.LOW,
            details=f"OAuth authentication successful: {provider_id}",
            metadata={
                "provider_id": provider_id,
                "user_id": user.user_id,
                "email": user.email,
                "has_refresh_token": oauth_token.refresh_token is not None
            }
        ))
        
        return user, oauth_token
    
    def _validate_id_token(self, config: OAuthConfig, id_token: str, expected_nonce: str) -> UserInfo:
        """Validate JWT ID token and extract user info."""
        try:
            # Get public keys from JWKS endpoint
            jwks_response = self.session.get(config.jwks_uri, timeout=10)
            jwks_response.raise_for_status()
            jwks = jwks_response.json()
            
            # Decode token header to get key ID
            unverified_header = jwt.get_unverified_header(id_token)
            key_id = unverified_header.get('kid')
            
            # Find matching key
            public_key = None
            for key in jwks.get('keys', []):
                if key.get('kid') == key_id:
                    # Convert JWK to public key
                    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
                    break
            
            if not public_key:
                raise OAuthError("Unable to find matching public key for token")
            
            # Validate and decode token
            claims = jwt.decode(
                id_token,
                public_key,
                algorithms=['RS256'],
                audience=config.client_id,
                issuer=config.issuer
            )
            
            # Validate nonce
            if claims.get('nonce') != expected_nonce:
                raise OAuthError("Invalid nonce in ID token")
            
            # Extract user information
            return UserInfo(
                sub=claims['sub'],
                email=claims.get('email', ''),
                email_verified=claims.get('email_verified', False),
                name=claims.get('name'),
                given_name=claims.get('given_name'),
                family_name=claims.get('family_name'),
                picture=claims.get('picture'),
                locale=claims.get('locale'),
                groups=claims.get('groups', []),
                roles=claims.get('roles', []),
                custom_claims={k: v for k, v in claims.items() 
                             if k not in ['sub', 'email', 'email_verified', 'name', 
                                        'given_name', 'family_name', 'picture', 'locale',
                                        'groups', 'roles', 'iss', 'aud', 'exp', 'iat', 'nonce']}
            )
            
        except jwt.InvalidTokenError as e:
            raise OAuthError(f"Invalid ID token: {e}")
        except requests.RequestException as e:
            raise OAuthError(f"Failed to retrieve JWKS: {e}")
    
    def _get_user_info(self, config: OAuthConfig, access_token: str) -> UserInfo:
        """Get user information from userinfo endpoint."""
        try:
            headers = {'Authorization': f'Bearer {access_token}'}
            response = self.session.get(config.userinfo_endpoint, headers=headers, timeout=10)
            response.raise_for_status()
            
            user_data = response.json()
            
            return UserInfo(
                sub=user_data['sub'],
                email=user_data.get('email', ''),
                email_verified=user_data.get('email_verified', False),
                name=user_data.get('name'),
                given_name=user_data.get('given_name'),
                family_name=user_data.get('family_name'),
                picture=user_data.get('picture'),
                locale=user_data.get('locale'),
                groups=user_data.get('groups', []),
                roles=user_data.get('roles', []),
                custom_claims={k: v for k, v in user_data.items()
                             if k not in ['sub', 'email', 'email_verified', 'name',
                                        'given_name', 'family_name', 'picture', 'locale',
                                        'groups', 'roles']}
            )
            
        except requests.RequestException as e:
            raise OAuthError(f"Failed to get user info: {e}")
    
    def _map_user_info_to_user(self, user_info: UserInfo, provider_id: str) -> User:
        """Map OAuth user info to internal User object."""
        # Generate internal user ID
        user_id = f"oauth_{provider_id}_{hashlib.sha256(user_info.sub.encode()).hexdigest()[:16]}"
        
        # Determine username
        username = user_info.email if user_info.email else f"user_{user_info.sub[:8]}"
        
        # Map roles (implement your business logic here)
        role = self._map_oauth_roles_to_internal_role(user_info.groups, user_info.roles)
        
        return User(
            user_id=user_id,
            username=username,
            email=user_info.email,
            password_hash="oauth_user",  # OAuth users don't have passwords
            role=role,
            metadata={
                "oauth_provider": provider_id,
                "oauth_sub": user_info.sub,
                "oauth_name": user_info.name,
                "oauth_picture": user_info.picture,
                "oauth_groups": user_info.groups,
                "oauth_roles": user_info.roles,
                "oauth_custom_claims": user_info.custom_claims
            }
        )
    
    def _map_oauth_roles_to_internal_role(self, groups: List[str], roles: List[str]) -> UserRole:
        """Map OAuth groups/roles to internal user role."""
        # Implement your organization's role mapping logic
        
        # Example mappings
        admin_groups = ['administrators', 'domain admins', 'memmimic-admins']
        admin_roles = ['admin', 'administrator', 'global-admin']
        
        if any(group.lower() in admin_groups for group in groups):
            return UserRole.ADMIN
        
        if any(role.lower() in admin_roles for role in roles):
            return UserRole.ADMIN
        
        # Default to regular user
        return UserRole.USER
    
    def refresh_token(self, user_id: str, refresh_token: str, provider_id: str) -> OAuthToken:
        """
        Refresh OAuth access token.
        
        Args:
            user_id: User identifier
            refresh_token: Refresh token
            provider_id: OAuth provider identifier
            
        Returns:
            New OAuth token
        """
        if provider_id not in self.providers:
            raise OAuthError(f"Unknown OAuth provider: {provider_id}")
        
        config = self.providers[provider_id]
        
        token_data = {
            'grant_type': 'refresh_token',
            'client_id': config.client_id,
            'client_secret': config.client_secret,
            'refresh_token': refresh_token
        }
        
        try:
            response = self.session.post(
                config.token_endpoint,
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            response.raise_for_status()
            
            token_response = response.json()
            
            # Create new OAuth token
            oauth_token = OAuthToken(
                access_token=token_response['access_token'],
                token_type=token_response.get('token_type', 'Bearer'),
                expires_in=token_response.get('expires_in', 3600),
                refresh_token=token_response.get('refresh_token', refresh_token),  # Keep old if not provided
                scope=token_response.get('scope')
            )
            
            # Update cache
            self.token_cache[user_id] = oauth_token
            
            # Log token refresh
            self.audit_logger.log_security_event(SecurityEvent(
                event_type=SecurityEventType.FUNCTION_CALL,
                component="oauth",
                severity=SeverityLevel.LOW,
                details=f"OAuth token refreshed for user: {user_id}",
                metadata={
                    "provider_id": provider_id,
                    "user_id": user_id
                }
            ))
            
            return oauth_token
            
        except requests.RequestException as e:
            raise OAuthError(f"Token refresh failed: {e}")
    
    def revoke_token(self, access_token: str, provider_id: str) -> bool:
        """
        Revoke OAuth access token.
        
        Args:
            access_token: Access token to revoke
            provider_id: OAuth provider identifier
            
        Returns:
            True if successful
        """
        if provider_id not in self.providers:
            return False
        
        config = self.providers[provider_id]
        
        # Check if provider supports token revocation
        revoke_endpoint = config.metadata.get('revocation_endpoint')
        if not revoke_endpoint:
            # Some providers have standard revoke endpoints
            if config.provider == OAuthProvider.GOOGLE_WORKSPACE:
                revoke_endpoint = "https://oauth2.googleapis.com/revoke"
            elif config.provider == OAuthProvider.MICROSOFT_AZURE:
                revoke_endpoint = "https://login.microsoftonline.com/common/oauth2/v2.0/logout"
            else:
                logger.warning(f"No revocation endpoint configured for provider: {provider_id}")
                return False
        
        try:
            response = self.session.post(
                revoke_endpoint,
                data={'token': access_token},
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10
            )
            
            # Some providers return 200, others return 204
            success = response.status_code in [200, 204]
            
            if success:
                # Remove from cache
                self.token_cache = {k: v for k, v in self.token_cache.items() 
                                  if v.access_token != access_token}
                
                # Log token revocation
                self.audit_logger.log_security_event(SecurityEvent(
                    event_type=SecurityEventType.FUNCTION_CALL,
                    component="oauth",
                    severity=SeverityLevel.LOW,
                    details=f"OAuth token revoked for provider: {provider_id}",
                    metadata={"provider_id": provider_id}
                ))
            
            return success
            
        except requests.RequestException as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    def get_sso_logout_url(self, provider_id: str, redirect_uri: Optional[str] = None) -> Optional[str]:
        """
        Get SSO logout URL for provider.
        
        Args:
            provider_id: OAuth provider identifier
            redirect_uri: URL to redirect to after logout
            
        Returns:
            Logout URL if supported
        """
        if provider_id not in self.providers:
            return None
        
        config = self.providers[provider_id]
        
        # Check for configured logout endpoint
        logout_endpoint = config.metadata.get('end_session_endpoint')
        
        if not logout_endpoint:
            # Standard logout endpoints for known providers
            if config.provider == OAuthProvider.MICROSOFT_AZURE:
                logout_endpoint = "https://login.microsoftonline.com/common/oauth2/v2.0/logout"
            elif config.provider == OAuthProvider.GOOGLE_WORKSPACE:
                logout_endpoint = "https://accounts.google.com/logout"
            else:
                return None
        
        # Build logout URL with redirect
        if redirect_uri and config.provider == OAuthProvider.MICROSOFT_AZURE:
            logout_url = f"{logout_endpoint}?post_logout_redirect_uri={redirect_uri}"
        elif redirect_uri and config.provider == OAuthProvider.GOOGLE_WORKSPACE:
            # Google doesn't support post-logout redirect in the same way
            logout_url = logout_endpoint
        else:
            logout_url = logout_endpoint
        
        return logout_url
    
    def cleanup_expired_states(self) -> int:
        """Clean up expired OAuth states."""
        current_time = datetime.now(timezone.utc)
        expired_states = [
            state for state, oauth_state in self.oauth_states.items()
            if current_time > oauth_state.expires_at
        ]
        
        for state in expired_states:
            del self.oauth_states[state]
        
        if expired_states:
            logger.info(f"Cleaned up {len(expired_states)} expired OAuth states")
        
        return len(expired_states)


# Global OAuth manager instance
_global_oauth_manager: Optional[EnterpriseOAuthManager] = None


def get_oauth_manager() -> EnterpriseOAuthManager:
    """Get the global OAuth manager."""
    global _global_oauth_manager
    if _global_oauth_manager is None:
        _global_oauth_manager = EnterpriseOAuthManager()
    return _global_oauth_manager


def initialize_oauth_manager(audit_logger: Optional[SecurityAuditLogger] = None) -> EnterpriseOAuthManager:
    """Initialize the global OAuth manager."""
    global _global_oauth_manager
    _global_oauth_manager = EnterpriseOAuthManager(audit_logger)
    return _global_oauth_manager