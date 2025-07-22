#!/usr/bin/env python3
"""
Enterprise Authentication & Authorization System

Advanced multi-factor authentication, role-based access control,
and Single Sign-On integration for enterprise MemMimic deployments.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urlencode, urlparse
import jwt
import pyotp
import qrcode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)


class AuthenticationLevel(Enum):
    """Authentication security levels"""
    BASIC = "basic"           # Username/password only
    ENHANCED = "enhanced"     # + Session validation
    MFA = "mfa"              # + Multi-factor authentication
    SSO = "sso"              # + Single Sign-On
    ENTERPRISE = "enterprise" # + Advanced security features


class Permission(Enum):
    """System permissions for RBAC"""
    # Memory operations
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"
    MEMORY_ADMIN = "memory:admin"
    
    # Tale operations
    TALE_READ = "tale:read"
    TALE_WRITE = "tale:write"
    TALE_DELETE = "tale:delete"
    TALE_ADMIN = "tale:admin"
    
    # System operations
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_ADMIN = "system:admin"
    
    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write" 
    USER_ADMIN = "user:admin"
    
    # Security operations
    SECURITY_AUDIT = "security:audit"
    SECURITY_ADMIN = "security:admin"


@dataclass
class Role:
    """Role definition with permissions"""
    name: str
    permissions: Set[Permission]
    description: str = ""
    inherits: List[str] = field(default_factory=list)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in self.permissions


@dataclass
class User:
    """User account with authentication details"""
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    mfa_secret: Optional[str] = None
    mfa_enabled: bool = False
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    session_tokens: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuthSession:
    """Authentication session with security context"""
    session_id: str
    user_id: str
    auth_level: AuthenticationLevel
    permissions: Set[Permission]
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    mfa_verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiFactorAuth:
    """Multi-factor authentication system"""
    
    def __init__(self, app_name: str = "MemMimic"):
        self.app_name = app_name
    
    def generate_secret(self) -> str:
        """Generate TOTP secret for user"""
        return pyotp.random_base32()
    
    def get_qr_code(self, username: str, secret: str) -> bytes:
        """Generate QR code for TOTP setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username,
            issuer_name=self.app_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
        except Exception as e:
            logger.warning(f"TOTP verification failed: {e}")
            return False
    
    def generate_backup_codes(self, count: int = 8) -> List[str]:
        """Generate backup codes for MFA"""
        return [secrets.token_hex(4).upper() for _ in range(count)]


class RoleBasedAccessControl:
    """Role-Based Access Control system"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self._init_default_roles()
    
    def _init_default_roles(self):
        """Initialize default system roles"""
        # Guest role - minimal permissions
        self.add_role(Role(
            name="guest",
            permissions={Permission.MEMORY_READ, Permission.TALE_READ},
            description="Guest user with read-only access"
        ))
        
        # User role - standard user permissions  
        self.add_role(Role(
            name="user",
            permissions={
                Permission.MEMORY_READ, Permission.MEMORY_WRITE,
                Permission.TALE_READ, Permission.TALE_WRITE
            },
            description="Standard user with memory and tale access"
        ))
        
        # Manager role - extended permissions
        self.add_role(Role(
            name="manager", 
            permissions={
                Permission.MEMORY_READ, Permission.MEMORY_WRITE, Permission.MEMORY_DELETE,
                Permission.TALE_READ, Permission.TALE_WRITE, Permission.TALE_DELETE,
                Permission.USER_READ, Permission.SYSTEM_MONITOR
            },
            description="Manager with user oversight capabilities"
        ))
        
        # Administrator role - full permissions
        self.add_role(Role(
            name="admin",
            permissions=set(Permission),
            description="System administrator with full access"
        ))
    
    def add_role(self, role: Role):
        """Add new role to system"""
        self.roles[role.name] = role
        logger.info(f"Added role: {role.name}")
    
    def get_user_permissions(self, username: str) -> Set[Permission]:
        """Get all permissions for user"""
        if username not in self.users:
            return set()
        
        user = self.users[username]
        permissions = set()
        
        for role_name in user.roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                permissions.update(role.permissions)
                
                # Handle role inheritance
                for inherited_role in role.inherits:
                    if inherited_role in self.roles:
                        permissions.update(self.roles[inherited_role].permissions)
        
        return permissions
    
    def has_permission(self, username: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in self.get_user_permissions(username)


class EnterpriseAuthManager:
    """Enterprise authentication and authorization manager"""
    
    def __init__(self, 
                 jwt_secret: str,
                 session_timeout: int = 3600,
                 max_failed_attempts: int = 5,
                 lockout_duration: int = 1800):
        self.jwt_secret = jwt_secret
        self.session_timeout = session_timeout
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        
        self.rbac = RoleBasedAccessControl()
        self.mfa = MultiFactorAuth()
        self.sessions: Dict[str, AuthSession] = {}
        self.refresh_tokens: Dict[str, str] = {}  # refresh_token -> user_id
        
        # Password encryption
        self.cipher_suite = Fernet(Fernet.generate_key())
    
    def _hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return base64.b64encode(key).decode(), salt
    
    def _verify_password(self, password: str, password_hash: str, salt: bytes) -> bool:
        """Verify password against hash"""
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return base64.b64encode(key).decode() == password_hash
    
    def create_user(self, 
                   username: str,
                   email: str, 
                   password: str,
                   roles: List[str] = None) -> User:
        """Create new user account"""
        if username in self.rbac.users:
            raise ValueError(f"User {username} already exists")
        
        password_hash, salt = self._hash_password(password)
        
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or ["user"]
        )
        
        # Store salt in metadata
        user.metadata['salt'] = base64.b64encode(salt).decode()
        
        self.rbac.users[username] = user
        logger.info(f"Created user: {username}")
        
        return user
    
    def authenticate(self, 
                    username: str, 
                    password: str,
                    mfa_token: Optional[str] = None,
                    ip_address: Optional[str] = None) -> Optional[AuthSession]:
        """Authenticate user with optional MFA"""
        if username not in self.rbac.users:
            logger.warning(f"Authentication failed: user {username} not found")
            return None
        
        user = self.rbac.users[username]
        
        # Check if account is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            logger.warning(f"Authentication failed: account {username} is locked")
            return None
        
        # Verify password
        salt = base64.b64decode(user.metadata.get('salt', ''))
        if not self._verify_password(password, user.password_hash, salt):
            user.failed_attempts += 1
            
            if user.failed_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.utcnow() + timedelta(seconds=self.lockout_duration)
                logger.warning(f"Account {username} locked due to failed attempts")
            
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None
        
        # Verify MFA if enabled
        auth_level = AuthenticationLevel.ENHANCED
        mfa_verified = False
        
        if user.mfa_enabled:
            if not mfa_token:
                logger.warning(f"MFA token required for {username}")
                return None
            
            if not self.mfa.verify_totp(user.mfa_secret, mfa_token):
                logger.warning(f"MFA verification failed for {username}")
                return None
            
            auth_level = AuthenticationLevel.MFA
            mfa_verified = True
        
        # Reset failed attempts on successful authentication
        user.failed_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Create session
        session = self._create_session(user, auth_level, ip_address)
        session.mfa_verified = mfa_verified
        
        logger.info(f"User {username} authenticated successfully")
        return session
    
    def _create_session(self, 
                       user: User,
                       auth_level: AuthenticationLevel,
                       ip_address: Optional[str] = None) -> AuthSession:
        """Create new authentication session"""
        session_id = secrets.token_urlsafe(32)
        permissions = self.rbac.get_user_permissions(user.username)
        
        session = AuthSession(
            session_id=session_id,
            user_id=user.username,
            auth_level=auth_level,
            permissions=permissions,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=self.session_timeout),
            last_activity=datetime.utcnow(),
            ip_address=ip_address
        )
        
        self.sessions[session_id] = session
        user.session_tokens.add(session_id)
        
        return session
    
    def generate_jwt_token(self, session: AuthSession) -> str:
        """Generate JWT token for session"""
        payload = {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'auth_level': session.auth_level.value,
            'permissions': [p.value for p in session.permissions],
            'exp': session.expires_at.timestamp(),
            'iat': session.created_at.timestamp(),
            'mfa_verified': session.mfa_verified
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[AuthSession]:
        """Verify JWT token and return session"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Update last activity
            session.last_activity = datetime.utcnow()
            
            return session
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def enable_mfa(self, username: str) -> Tuple[str, bytes]:
        """Enable MFA for user and return secret + QR code"""
        if username not in self.rbac.users:
            raise ValueError(f"User {username} not found")
        
        user = self.rbac.users[username]
        secret = self.mfa.generate_secret()
        qr_code = self.mfa.get_qr_code(username, secret)
        
        user.mfa_secret = secret
        user.mfa_enabled = True
        
        logger.info(f"MFA enabled for user: {username}")
        return secret, qr_code
    
    def revoke_session(self, session_id: str):
        """Revoke authentication session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Remove from user's session tokens
            if session.user_id in self.rbac.users:
                user = self.rbac.users[session.user_id]
                user.session_tokens.discard(session_id)
            
            del self.sessions[session_id]
            logger.info(f"Revoked session: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.expires_at < now
        ]
        
        for session_id in expired_sessions:
            self.revoke_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# OAuth2/OIDC Integration placeholder
class SSOIntegration:
    """Single Sign-On integration for enterprise identity providers"""
    
    def __init__(self, provider_config: Dict[str, Any]):
        self.provider_config = provider_config
        self.supported_providers = ["google", "azure", "okta", "auth0"]
    
    def get_authorization_url(self, provider: str, redirect_uri: str) -> str:
        """Get OAuth2 authorization URL"""
        # Implementation would depend on specific provider
        return f"https://{provider}.com/oauth2/authorize?client_id=..."
    
    def exchange_code_for_token(self, code: str, provider: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        # Implementation would make HTTP request to provider
        return {"access_token": "...", "id_token": "..."}
    
    def verify_id_token(self, id_token: str, provider: str) -> Dict[str, Any]:
        """Verify and decode ID token from provider"""
        # Implementation would verify JWT signature and extract claims
        return {"sub": "user_id", "email": "user@example.com"}


# Example usage and integration
async def main():
    """Example usage of enterprise authentication system"""
    # Initialize authentication manager
    auth_manager = EnterpriseAuthManager(
        jwt_secret="your-secret-key-here",
        session_timeout=3600
    )
    
    # Create test user
    user = auth_manager.create_user(
        username="admin",
        email="admin@company.com", 
        password="secure_password_123",
        roles=["admin"]
    )
    
    # Enable MFA
    secret, qr_code = auth_manager.enable_mfa("admin")
    print(f"MFA Secret: {secret}")
    
    # Authenticate with MFA
    mfa_token = pyotp.TOTP(secret).now()
    session = auth_manager.authenticate(
        username="admin",
        password="secure_password_123",
        mfa_token=mfa_token
    )
    
    if session:
        print(f"Authentication successful! Session: {session.session_id}")
        print(f"Permissions: {[p.value for p in session.permissions]}")
        
        # Generate JWT token
        jwt_token = auth_manager.generate_jwt_token(session)
        print(f"JWT Token: {jwt_token}")
        
        # Verify JWT token
        verified_session = auth_manager.verify_jwt_token(jwt_token)
        print(f"Token verification: {'Success' if verified_session else 'Failed'}")


if __name__ == "__main__":
    asyncio.run(main())