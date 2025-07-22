"""
Enterprise Authentication & Authorization System

Comprehensive JWT-based authentication with Multi-Factor Authentication,
Role-Based Access Control, and enterprise security features.
"""

import secrets
import hashlib
import hmac
import time
import json
import pyotp
import qrcode
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import bcrypt

from .audit import SecurityAuditLogger, SecurityEvent, SecurityEventType, SeverityLevel

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for RBAC system."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


class Permission(Enum):
    """System permissions."""
    READ_MEMORIES = "read_memories"
    WRITE_MEMORIES = "write_memories"
    DELETE_MEMORIES = "delete_memories"
    READ_TALES = "read_tales"
    WRITE_TALES = "write_tales"
    DELETE_TALES = "delete_tales"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    SYSTEM_ADMIN = "system_admin"
    API_ACCESS = "api_access"


class MFAMethod(Enum):
    """Multi-factor authentication methods."""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"


@dataclass
class User:
    """User account with security metadata."""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    lock_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_methods: List[MFAMethod] = field(default_factory=list)
    mfa_secret: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)
    session_timeout: int = 3600  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'failed_login_attempts': self.failed_login_attempts,
            'account_locked': self.account_locked,
            'lock_until': self.lock_until.isoformat() if self.lock_until else None,
            'mfa_enabled': self.mfa_enabled,
            'mfa_methods': [method.value for method in self.mfa_methods],
            'session_timeout': self.session_timeout,
            'metadata': self.metadata
        }


@dataclass
class APIKey:
    """API key for programmatic access."""
    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[Permission]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    rate_limit: Optional[int] = None
    ip_whitelist: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session with security tracking."""
    session_id: str
    user_id: str
    jwt_token: str
    refresh_token: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    mfa_verified: bool = False
    security_flags: Dict[str, bool] = field(default_factory=dict)


class AuthenticationError(Exception):
    """Authentication-related errors."""
    pass


class AuthorizationError(Exception):
    """Authorization-related errors."""
    pass


class MFAError(Exception):
    """Multi-factor authentication errors."""
    pass


class EnterpriseAuthenticationManager:
    """
    Enterprise-grade authentication and authorization system.
    
    Features:
    - JWT-based authentication with refresh tokens
    - Multi-factor authentication (TOTP, SMS, hardware tokens)
    - Role-based access control (RBAC)
    - API key management with automatic rotation
    - Session management with security tracking
    - Account lockout protection
    - Comprehensive audit logging
    """
    
    def __init__(self, db_path: str = "memmimic_auth.db", 
                 jwt_secret: Optional[str] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """
        Initialize authentication manager.
        
        Args:
            db_path: Path to authentication database
            jwt_secret: JWT signing secret (generated if not provided)
            audit_logger: Security audit logger instance
        """
        self.db_path = Path(db_path)
        self.audit_logger = audit_logger or SecurityAuditLogger()
        
        # Initialize JWT configuration
        self.jwt_secret = jwt_secret or self._generate_jwt_secret()
        self.jwt_algorithm = "HS256"
        self.jwt_expire_hours = 1
        self.refresh_expire_days = 30
        
        # Initialize encryption for sensitive data
        self.encryption_key = self._generate_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Role-based permissions
        self.role_permissions = self._initialize_role_permissions()
        
        # Account lockout settings
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        
        # Initialize database
        self._initialize_database()
        
        # Create default admin user if none exists
        self._create_default_admin()
        
        logger.info("EnterpriseAuthenticationManager initialized")
    
    def _generate_jwt_secret(self) -> str:
        """Generate secure JWT secret."""
        return secrets.token_urlsafe(64)
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data."""
        return Fernet.generate_key()
    
    def _initialize_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Initialize role-based permissions."""
        return {
            UserRole.ADMIN: list(Permission),  # All permissions
            UserRole.USER: [
                Permission.READ_MEMORIES,
                Permission.WRITE_MEMORIES,
                Permission.DELETE_MEMORIES,
                Permission.READ_TALES,
                Permission.WRITE_TALES,
                Permission.DELETE_TALES,
                Permission.API_ACCESS
            ],
            UserRole.VIEWER: [
                Permission.READ_MEMORIES,
                Permission.READ_TALES
            ],
            UserRole.API_CLIENT: [
                Permission.API_ACCESS,
                Permission.READ_MEMORIES,
                Permission.WRITE_MEMORIES,
                Permission.READ_TALES,
                Permission.WRITE_TALES
            ]
        }
    
    def _initialize_database(self) -> None:
        """Initialize authentication database."""
        with self._get_db_connection() as conn:
            # Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    failed_login_attempts INTEGER DEFAULT 0,
                    account_locked BOOLEAN DEFAULT FALSE,
                    lock_until TIMESTAMP,
                    mfa_enabled BOOLEAN DEFAULT FALSE,
                    mfa_methods TEXT,
                    mfa_secret_encrypted TEXT,
                    backup_codes_encrypted TEXT,
                    session_timeout INTEGER DEFAULT 3600,
                    metadata TEXT
                )
            ''')
            
            # Sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    jwt_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    mfa_verified BOOLEAN DEFAULT FALSE,
                    security_flags TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # API Keys table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    rate_limit INTEGER,
                    ip_whitelist TEXT,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions (expires_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys (user_id)')
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper error handling."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _create_default_admin(self) -> None:
        """Create default admin user if none exists."""
        try:
            # Check if any admin users exist
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM users WHERE role = ?",
                    (UserRole.ADMIN.value,)
                )
                admin_count = cursor.fetchone()['count']
                
                if admin_count == 0:
                    # Create default admin
                    admin_user = User(
                        user_id=secrets.token_urlsafe(16),
                        username="admin",
                        email="admin@memmimic.local",
                        password_hash=self._hash_password("admin123!"),
                        role=UserRole.ADMIN
                    )
                    
                    self._store_user(admin_user)
                    
                    logger.warning(
                        "Default admin user created (username: admin, password: admin123!). "
                        "Please change the password immediately!"
                    )
                    
                    # Log security event
                    self.audit_logger.log_security_event(SecurityEvent(
                        event_type=SecurityEventType.CONFIGURATION_CHANGE,
                        component="authentication",
                        severity=SeverityLevel.HIGH,
                        details="Default admin user created",
                        metadata={"username": "admin"}
                    ))
        except Exception as e:
            logger.error(f"Failed to create default admin: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode('utf-8')).decode('utf-8')
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')
    
    def _store_user(self, user: User) -> None:
        """Store user in database."""
        with self._get_db_connection() as conn:
            # Encrypt sensitive data
            mfa_secret_encrypted = self._encrypt_data(user.mfa_secret) if user.mfa_secret else None
            backup_codes_encrypted = self._encrypt_data(json.dumps(user.backup_codes)) if user.backup_codes else None
            
            conn.execute('''
                INSERT OR REPLACE INTO users (
                    user_id, username, email, password_hash, role, is_active,
                    created_at, updated_at, last_login, failed_login_attempts,
                    account_locked, lock_until, mfa_enabled, mfa_methods,
                    mfa_secret_encrypted, backup_codes_encrypted, session_timeout, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.user_id, user.username, user.email, user.password_hash,
                user.role.value, user.is_active, user.created_at.isoformat(),
                user.updated_at.isoformat(),
                user.last_login.isoformat() if user.last_login else None,
                user.failed_login_attempts, user.account_locked,
                user.lock_until.isoformat() if user.lock_until else None,
                user.mfa_enabled, json.dumps([method.value for method in user.mfa_methods]),
                mfa_secret_encrypted, backup_codes_encrypted,
                user.session_timeout, json.dumps(user.metadata)
            ))
            conn.commit()
    
    def _load_user_by_username(self, username: str) -> Optional[User]:
        """Load user by username from database."""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE username = ? AND is_active = TRUE",
                (username,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Decrypt sensitive data
            mfa_secret = None
            if row['mfa_secret_encrypted']:
                mfa_secret = self._decrypt_data(row['mfa_secret_encrypted'])
            
            backup_codes = []
            if row['backup_codes_encrypted']:
                backup_codes = json.loads(self._decrypt_data(row['backup_codes_encrypted']))
            
            return User(
                user_id=row['user_id'],
                username=row['username'],
                email=row['email'],
                password_hash=row['password_hash'],
                role=UserRole(row['role']),
                is_active=bool(row['is_active']),
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(timezone.utc),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.now(timezone.utc),
                last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None,
                failed_login_attempts=row['failed_login_attempts'] or 0,
                account_locked=bool(row['account_locked']),
                lock_until=datetime.fromisoformat(row['lock_until']) if row['lock_until'] else None,
                mfa_enabled=bool(row['mfa_enabled']),
                mfa_methods=[MFAMethod(method) for method in json.loads(row['mfa_methods'] or '[]')],
                mfa_secret=mfa_secret,
                backup_codes=backup_codes,
                session_timeout=row['session_timeout'] or 3600,
                metadata=json.loads(row['metadata'] or '{}')
            )
    
    def _load_user_by_id(self, user_id: str) -> Optional[User]:
        """Load user by ID from database."""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE user_id = ? AND is_active = TRUE",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Decrypt sensitive data
            mfa_secret = None
            if row['mfa_secret_encrypted']:
                mfa_secret = self._decrypt_data(row['mfa_secret_encrypted'])
            
            backup_codes = []
            if row['backup_codes_encrypted']:
                backup_codes = json.loads(self._decrypt_data(row['backup_codes_encrypted']))
            
            return User(
                user_id=row['user_id'],
                username=row['username'],
                email=row['email'],
                password_hash=row['password_hash'],
                role=UserRole(row['role']),
                is_active=bool(row['is_active']),
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(timezone.utc),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.now(timezone.utc),
                last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None,
                failed_login_attempts=row['failed_login_attempts'] or 0,
                account_locked=bool(row['account_locked']),
                lock_until=datetime.fromisoformat(row['lock_until']) if row['lock_until'] else None,
                mfa_enabled=bool(row['mfa_enabled']),
                mfa_methods=[MFAMethod(method) for method in json.loads(row['mfa_methods'] or '[]')],
                mfa_secret=mfa_secret,
                backup_codes=backup_codes,
                session_timeout=row['session_timeout'] or 3600,
                metadata=json.loads(row['metadata'] or '{}')
            )
    
    def _generate_jwt_token(self, user: User, session_id: str) -> str:
        """Generate JWT token for user."""
        now = datetime.now(timezone.utc)
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role.value,
            'session_id': session_id,
            'iat': now,
            'exp': now + timedelta(hours=self.jwt_expire_hours),
            'iss': 'memmimic-auth'
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _generate_refresh_token(self) -> str:
        """Generate secure refresh token."""
        return secrets.token_urlsafe(64)
    
    def create_user(self, username: str, email: str, password: str, 
                   role: UserRole = UserRole.USER) -> User:
        """
        Create new user account.
        
        Args:
            username: Unique username
            email: User email address
            password: Plain text password (will be hashed)
            role: User role for RBAC
            
        Returns:
            User object
            
        Raises:
            AuthenticationError: If user creation fails
        """
        # Validate password strength
        if len(password) < 8:
            raise AuthenticationError("Password must be at least 8 characters long")
        
        # Check if user already exists
        existing_user = self._load_user_by_username(username)
        if existing_user:
            raise AuthenticationError(f"Username {username} already exists")
        
        # Create user
        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            role=role
        )
        
        self._store_user(user)
        
        # Log security event
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            component="authentication",
            severity=SeverityLevel.MEDIUM,
            details=f"User account created: {username}",
            metadata={"username": username, "role": role.value}
        ))
        
        logger.info(f"User account created: {username}")
        return user
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: Optional[str] = None,
                         user_agent: Optional[str] = None) -> Tuple[User, Session]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (User, Session)
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Load user
        user = self._load_user_by_username(username)
        if not user:
            # Log failed attempt
            self.audit_logger.log_security_event(SecurityEvent(
                event_type=SecurityEventType.VALIDATION_FAILURE,
                component="authentication",
                severity=SeverityLevel.MEDIUM,
                details=f"Authentication failed: unknown username {username}",
                metadata={"username": username, "ip_address": ip_address}
            ))
            raise AuthenticationError("Invalid username or password")
        
        # Check account lockout
        if user.account_locked and user.lock_until:
            if datetime.now(timezone.utc) < user.lock_until:
                # Account still locked
                self.audit_logger.log_security_event(SecurityEvent(
                    event_type=SecurityEventType.ACCESS_DENIED,
                    component="authentication",
                    severity=SeverityLevel.HIGH,
                    details=f"Authentication blocked: account locked for {username}",
                    metadata={"username": username, "lock_until": user.lock_until.isoformat()}
                ))
                raise AuthenticationError("Account is locked due to too many failed attempts")
            else:
                # Lock expired, reset
                user.account_locked = False
                user.lock_until = None
                user.failed_login_attempts = 0
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failures
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.account_locked = True
                user.lock_until = datetime.now(timezone.utc) + timedelta(minutes=self.lockout_duration_minutes)
                
                self.audit_logger.log_security_event(SecurityEvent(
                    event_type=SecurityEventType.SECURITY_VIOLATION,
                    component="authentication",
                    severity=SeverityLevel.HIGH,
                    details=f"Account locked due to too many failed attempts: {username}",
                    metadata={"username": username, "failed_attempts": user.failed_login_attempts}
                ))
            
            user.updated_at = datetime.now(timezone.utc)
            self._store_user(user)
            
            # Log failed attempt
            self.audit_logger.log_security_event(SecurityEvent(
                event_type=SecurityEventType.VALIDATION_FAILURE,
                component="authentication",
                severity=SeverityLevel.MEDIUM,
                details=f"Authentication failed: invalid password for {username}",
                metadata={"username": username, "failed_attempts": user.failed_login_attempts}
            ))
            
            raise AuthenticationError("Invalid username or password")
        
        # Authentication successful - reset failed attempts
        user.failed_login_attempts = 0
        user.last_login = datetime.now(timezone.utc)
        user.updated_at = datetime.now(timezone.utc)
        self._store_user(user)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        jwt_token = self._generate_jwt_token(user, session_id)
        refresh_token = self._generate_refresh_token()
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            jwt_token=jwt_token,
            refresh_token=refresh_token,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=user.session_timeout),
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=not user.mfa_enabled  # If MFA disabled, consider verified
        )
        
        # Store session
        self._store_session(session)
        
        # Log successful authentication
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.FUNCTION_CALL,
            component="authentication",
            severity=SeverityLevel.LOW,
            details=f"User authenticated successfully: {username}",
            metadata={"username": username, "session_id": session_id, "ip_address": ip_address}
        ))
        
        return user, session
    
    def _store_session(self, session: Session) -> None:
        """Store session in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO sessions (
                    session_id, user_id, jwt_token, refresh_token,
                    created_at, expires_at, last_activity, ip_address,
                    user_agent, is_active, mfa_verified, security_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id, session.user_id, session.jwt_token,
                session.refresh_token, session.created_at.isoformat(),
                session.expires_at.isoformat(), session.last_activity.isoformat(),
                session.ip_address, session.user_agent, session.is_active,
                session.mfa_verified, json.dumps(session.security_flags)
            ))
            conn.commit()
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token and return payload.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            user: User object
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        role_permissions = self.role_permissions.get(user.role, [])
        return permission in role_permissions
    
    def require_permission(self, user: User, permission: Permission) -> None:
        """
        Require user to have specific permission.
        
        Args:
            user: User object
            permission: Required permission
            
        Raises:
            AuthorizationError: If user lacks permission
        """
        if not self.has_permission(user, permission):
            # Log authorization failure
            self.audit_logger.log_security_event(SecurityEvent(
                event_type=SecurityEventType.ACCESS_DENIED,
                component="authorization",
                severity=SeverityLevel.MEDIUM,
                details=f"Authorization denied: {user.username} lacks {permission.value}",
                metadata={"username": user.username, "required_permission": permission.value}
            ))
            
            raise AuthorizationError(f"Permission denied: {permission.value}")
    
    def setup_mfa_totp(self, user: User) -> Tuple[str, str]:
        """
        Setup TOTP multi-factor authentication for user.
        
        Args:
            user: User object
            
        Returns:
            Tuple of (secret, QR code URL)
        """
        # Generate secret key
        secret = pyotp.random_base32()
        
        # Update user with MFA settings
        user.mfa_secret = secret
        user.mfa_enabled = True
        if MFAMethod.TOTP not in user.mfa_methods:
            user.mfa_methods.append(MFAMethod.TOTP)
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        user.backup_codes = backup_codes
        
        user.updated_at = datetime.now(timezone.utc)
        self._store_user(user)
        
        # Generate QR code URL
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user.email,
            issuer_name="MemMimic"
        )
        
        # Log security event
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            component="authentication",
            severity=SeverityLevel.MEDIUM,
            details=f"MFA TOTP setup for user: {user.username}",
            metadata={"username": user.username, "mfa_method": "TOTP"}
        ))
        
        return secret, provisioning_uri
    
    def verify_mfa_totp(self, user: User, token: str) -> bool:
        """
        Verify TOTP token for user.
        
        Args:
            user: User object
            token: TOTP token
            
        Returns:
            True if token is valid
        """
        if not user.mfa_enabled or not user.mfa_secret:
            return False
        
        # Check TOTP token
        totp = pyotp.TOTP(user.mfa_secret)
        if totp.verify(token, valid_window=1):  # Allow 1 window tolerance
            return True
        
        # Check backup codes
        if token.upper() in user.backup_codes:
            # Remove used backup code
            user.backup_codes.remove(token.upper())
            user.updated_at = datetime.now(timezone.utc)
            self._store_user(user)
            
            # Log backup code usage
            self.audit_logger.log_security_event(SecurityEvent(
                event_type=SecurityEventType.FUNCTION_CALL,
                component="authentication",
                severity=SeverityLevel.MEDIUM,
                details=f"MFA backup code used: {user.username}",
                metadata={"username": user.username, "backup_codes_remaining": len(user.backup_codes)}
            ))
            
            return True
        
        return False
    
    def create_api_key(self, user: User, name: str, permissions: List[Permission],
                      expires_days: Optional[int] = None,
                      rate_limit: Optional[int] = None,
                      ip_whitelist: Optional[List[str]] = None) -> Tuple[str, APIKey]:
        """
        Create API key for user.
        
        Args:
            user: User object
            name: API key name/description
            permissions: List of permissions for the key
            expires_days: Days until expiration (None for no expiration)
            rate_limit: Rate limit for this key
            ip_whitelist: List of allowed IP addresses
            
        Returns:
            Tuple of (raw_key, APIKey object)
        """
        # Generate key
        raw_key = f"mk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Create API key object
        api_key = APIKey(
            key_id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            name=name,
            user_id=user.user_id,
            permissions=permissions,
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_days) if expires_days else None,
            rate_limit=rate_limit,
            ip_whitelist=ip_whitelist or [],
            metadata={"created_by": user.username}
        )
        
        # Store API key
        self._store_api_key(api_key)
        
        # Log security event
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            component="authentication",
            severity=SeverityLevel.MEDIUM,
            details=f"API key created: {name} for user {user.username}",
            metadata={"username": user.username, "api_key_name": name, "permissions": [p.value for p in permissions]}
        ))
        
        return raw_key, api_key
    
    def _store_api_key(self, api_key: APIKey) -> None:
        """Store API key in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO api_keys (
                    key_id, key_hash, name, user_id, permissions,
                    created_at, expires_at, last_used, is_active,
                    rate_limit, ip_whitelist, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                api_key.key_id, api_key.key_hash, api_key.name,
                api_key.user_id, json.dumps([p.value for p in api_key.permissions]),
                api_key.created_at.isoformat(),
                api_key.expires_at.isoformat() if api_key.expires_at else None,
                api_key.last_used.isoformat() if api_key.last_used else None,
                api_key.is_active, api_key.rate_limit,
                json.dumps(api_key.ip_whitelist), json.dumps(api_key.metadata)
            ))
            conn.commit()
    
    def verify_api_key(self, raw_key: str, ip_address: Optional[str] = None) -> Optional[Tuple[User, APIKey]]:
        """
        Verify API key and return associated user and key info.
        
        Args:
            raw_key: Raw API key
            ip_address: Client IP address
            
        Returns:
            Tuple of (User, APIKey) if valid, None otherwise
        """
        if not raw_key.startswith("mk_"):
            return None
        
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ? AND is_active = TRUE",
                (key_hash,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Check expiration
            if row['expires_at']:
                expires_at = datetime.fromisoformat(row['expires_at'])
                if datetime.now(timezone.utc) > expires_at:
                    return None
            
            # Load API key
            api_key = APIKey(
                key_id=row['key_id'],
                key_hash=row['key_hash'],
                name=row['name'],
                user_id=row['user_id'],
                permissions=[Permission(p) for p in json.loads(row['permissions'])],
                created_at=datetime.fromisoformat(row['created_at']),
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
                is_active=bool(row['is_active']),
                rate_limit=row['rate_limit'],
                ip_whitelist=json.loads(row['ip_whitelist'] or '[]'),
                metadata=json.loads(row['metadata'] or '{}')
            )
            
            # Check IP whitelist
            if api_key.ip_whitelist and ip_address:
                if ip_address not in api_key.ip_whitelist:
                    self.audit_logger.log_security_event(SecurityEvent(
                        event_type=SecurityEventType.ACCESS_DENIED,
                        component="authentication",
                        severity=SeverityLevel.HIGH,
                        details=f"API key access denied: IP not whitelisted",
                        metadata={"api_key_id": api_key.key_id, "ip_address": ip_address}
                    ))
                    return None
            
            # Load associated user
            user = self._load_user_by_id(api_key.user_id)
            if not user:
                return None
            
            # Update last used timestamp
            api_key.last_used = datetime.now(timezone.utc)
            self._update_api_key_last_used(api_key)
            
            return user, api_key
    
    def _update_api_key_last_used(self, api_key: APIKey) -> None:
        """Update API key last used timestamp."""
        with self._get_db_connection() as conn:
            conn.execute(
                "UPDATE api_keys SET last_used = ? WHERE key_id = ?",
                (api_key.last_used.isoformat(), api_key.key_id)
            )
            conn.commit()
    
    def rotate_api_key(self, user: User, key_id: str) -> Tuple[str, APIKey]:
        """
        Rotate API key (generate new key, deactivate old).
        
        Args:
            user: User object
            key_id: API key ID to rotate
            
        Returns:
            Tuple of (new_raw_key, new_APIKey)
        """
        # Load existing key
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM api_keys WHERE key_id = ? AND user_id = ? AND is_active = TRUE",
                (key_id, user.user_id)
            )
            row = cursor.fetchone()
            
            if not row:
                raise AuthenticationError("API key not found or not accessible")
            
            # Deactivate old key
            conn.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE key_id = ?",
                (key_id,)
            )
            
            # Create new key with same properties
            permissions = [Permission(p) for p in json.loads(row['permissions'])]
            expires_at = datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
            rate_limit = row['rate_limit']
            ip_whitelist = json.loads(row['ip_whitelist'] or '[]')
            
            return self.create_api_key(
                user, f"{row['name']} (rotated)",
                permissions,
                (expires_at - datetime.now(timezone.utc)).days if expires_at else None,
                rate_limit,
                ip_whitelist
            )
    
    def list_user_sessions(self, user: User) -> List[Session]:
        """List active sessions for user."""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND is_active = TRUE ORDER BY last_activity DESC",
                (user.user_id,)
            )
            
            sessions = []
            for row in cursor.fetchall():
                session = Session(
                    session_id=row['session_id'],
                    user_id=row['user_id'],
                    jwt_token=row['jwt_token'],
                    refresh_token=row['refresh_token'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    expires_at=datetime.fromisoformat(row['expires_at']),
                    last_activity=datetime.fromisoformat(row['last_activity']),
                    ip_address=row['ip_address'],
                    user_agent=row['user_agent'],
                    is_active=bool(row['is_active']),
                    mfa_verified=bool(row['mfa_verified']),
                    security_flags=json.loads(row['security_flags'] or '{}')
                )
                sessions.append(session)
            
            return sessions
    
    def revoke_session(self, user: User, session_id: str) -> bool:
        """Revoke user session."""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET is_active = FALSE WHERE session_id = ? AND user_id = ?",
                (session_id, user.user_id)
            )
            
            if cursor.rowcount > 0:
                conn.commit()
                
                # Log security event
                self.audit_logger.log_security_event(SecurityEvent(
                    event_type=SecurityEventType.FUNCTION_CALL,
                    component="authentication",
                    severity=SeverityLevel.LOW,
                    details=f"Session revoked: {session_id} for user {user.username}",
                    metadata={"username": user.username, "session_id": session_id}
                ))
                
                return True
            
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        now = datetime.now(timezone.utc)
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE expires_at < ? OR (is_active = FALSE AND last_activity < ?)",
                (now.isoformat(), (now - timedelta(days=7)).isoformat())
            )
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired sessions")
            
            return deleted_count


# Global authentication manager instance
_global_auth_manager: Optional[EnterpriseAuthenticationManager] = None


def get_auth_manager() -> EnterpriseAuthenticationManager:
    """Get the global authentication manager."""
    global _global_auth_manager
    if _global_auth_manager is None:
        _global_auth_manager = EnterpriseAuthenticationManager()
    return _global_auth_manager


def initialize_auth_manager(db_path: str = "memmimic_auth.db",
                           jwt_secret: Optional[str] = None,
                           audit_logger: Optional[SecurityAuditLogger] = None) -> EnterpriseAuthenticationManager:
    """Initialize the global authentication manager."""
    global _global_auth_manager
    _global_auth_manager = EnterpriseAuthenticationManager(db_path, jwt_secret, audit_logger)
    return _global_auth_manager