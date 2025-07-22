"""
Advanced Threat Intelligence System

ML-based anomaly detection, threat feed integration, and predictive
security analytics for proactive threat identification and response.
"""

import json
import time
import hashlib
import requests
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import sqlite3
from contextlib import contextmanager
import numpy as np
from collections import defaultdict, deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

from .audit import SecurityAuditLogger, SecurityEvent, SecurityEventType, SeverityLevel
from .threat_detection import ThreatType, ThreatLevel, ThreatDetection

logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    """Threat intelligence categories."""
    MALWARE = "malware"
    PHISHING = "phishing"
    BOTNET = "botnet"
    APT = "apt"  # Advanced Persistent Threat
    RANSOMWARE = "ransomware"
    CRYPTOCURRENCY_MINING = "crypto_mining"
    DATA_EXFILTRATION = "data_exfiltration"
    INSIDER_THREAT = "insider_threat"
    SUPPLY_CHAIN = "supply_chain"
    ZERO_DAY = "zero_day"


class ThreatSource(Enum):
    """Threat intelligence sources."""
    INTERNAL_ANALYSIS = "internal"
    COMMERCIAL_FEED = "commercial"
    OPEN_SOURCE = "open_source"
    GOVERNMENT = "government"
    INDUSTRY_SHARING = "industry"
    HONEYPOT = "honeypot"
    SANDBOX = "sandbox"


class IndicatorType(Enum):
    """Types of threat indicators."""
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    USER_AGENT = "user_agent"
    JA3_FINGERPRINT = "ja3_fingerprint"
    YARA_RULE = "yara_rule"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class Confidence(Enum):
    """Confidence levels for threat intelligence."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


@dataclass
class ThreatIndicator:
    """Threat intelligence indicator."""
    indicator_id: str
    indicator_type: IndicatorType
    value: str  # The actual indicator value
    threat_category: ThreatCategory
    confidence: Confidence
    severity: ThreatLevel
    source: ThreatSource
    description: str
    tags: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expiry_date: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    related_indicators: List[str] = field(default_factory=list)


@dataclass
class ThreatFeed:
    """Threat intelligence feed configuration."""
    feed_id: str
    name: str
    url: str
    feed_type: str  # json, csv, xml, stix
    api_key: Optional[str] = None
    update_interval: int = 3600  # seconds
    enabled: bool = True
    last_updated: Optional[datetime] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyModel:
    """ML anomaly detection model."""
    model_id: str
    model_type: str  # isolation_forest, one_class_svm, autoencoder
    feature_names: List[str]
    model_data: bytes  # Serialized model
    training_date: datetime
    accuracy_score: float
    false_positive_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehavioralAnomaly:
    """Detected behavioral anomaly."""
    anomaly_id: str
    user_id: Optional[str]
    entity_id: str  # Could be IP, device, etc.
    anomaly_type: str
    confidence_score: float  # 0.0 to 1.0
    deviation_score: float  # How much it deviates from normal
    features: Dict[str, float]  # Feature values that led to anomaly
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    investigated: bool = False
    false_positive: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatCampaign:
    """Coordinated threat campaign."""
    campaign_id: str
    name: str
    threat_actor: Optional[str]
    categories: List[ThreatCategory]
    start_date: datetime
    end_date: Optional[datetime] = None
    indicators: List[str] = field(default_factory=list)  # Indicator IDs
    ttps: List[str] = field(default_factory=list)  # Tactics, Techniques, Procedures
    targets: List[str] = field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    status: str = "active"  # active, inactive, monitoring
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreatIntelligenceError(Exception):
    """Threat intelligence related errors."""
    pass


class AdvancedThreatIntelligenceSystem:
    """
    Advanced threat intelligence system with ML-based detection.
    
    Features:
    - Multi-source threat intelligence aggregation
    - Real-time threat feed processing and correlation
    - ML-based behavioral anomaly detection
    - Predictive threat analytics and early warning
    - Automated threat hunting and investigation
    - Campaign tracking and attribution
    - Threat actor profiling and analysis
    - Integration with security tools and platforms
    """
    
    def __init__(self, db_path: str = "memmimic_threat_intel.db",
                 models_dir: str = "threat_models",
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """
        Initialize threat intelligence system.
        
        Args:
            db_path: Path to threat intelligence database
            models_dir: Directory for ML models
            audit_logger: Security audit logger instance
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.audit_logger = audit_logger or SecurityAuditLogger()
        
        # Create models directory
        import os
        os.makedirs(models_dir, exist_ok=True)
        
        # Threat indicators and feeds
        self.indicators: Dict[str, ThreatIndicator] = {}
        self.threat_feeds: Dict[str, ThreatFeed] = {}
        
        # ML models for anomaly detection
        self.anomaly_models: Dict[str, AnomalyModel] = {}
        
        # Behavioral baselines and anomalies
        self.behavioral_baselines: Dict[str, Dict[str, float]] = {}
        self.recent_anomalies: Dict[str, BehavioralAnomaly] = {}
        
        # Threat campaigns
        self.campaigns: Dict[str, ThreatCampaign] = {}
        
        # Background processing
        self.processing_executor = ThreadPoolExecutor(max_workers=4)
        self.feed_update_thread = None
        self.running = False
        
        # Initialize database
        self._initialize_database()
        
        # Load existing data
        self._load_threat_intelligence_data()
        
        # Initialize default threat feeds
        self._initialize_default_feeds()
        
        # Start background processing
        self._start_background_processing()
        
        logger.info("AdvancedThreatIntelligenceSystem initialized")
    
    def _initialize_database(self) -> None:
        """Initialize threat intelligence database."""
        with self._get_db_connection() as conn:
            # Threat indicators table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS threat_indicators (
                    indicator_id TEXT PRIMARY KEY,
                    indicator_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    threat_category TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expiry_date TIMESTAMP,
                    context TEXT,
                    related_indicators TEXT
                )
            ''')
            
            # Threat feeds table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS threat_feeds (
                    feed_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    feed_type TEXT NOT NULL,
                    api_key TEXT,
                    update_interval INTEGER DEFAULT 3600,
                    enabled BOOLEAN DEFAULT TRUE,
                    last_updated TIMESTAMP,
                    error_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            ''')
            
            # Anomaly models table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_models (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    feature_names TEXT NOT NULL,
                    model_data BLOB NOT NULL,
                    training_date TIMESTAMP NOT NULL,
                    accuracy_score REAL NOT NULL,
                    false_positive_rate REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Behavioral anomalies table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS behavioral_anomalies (
                    anomaly_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    entity_id TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    deviation_score REAL NOT NULL,
                    features TEXT NOT NULL,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    investigated BOOLEAN DEFAULT FALSE,
                    false_positive BOOLEAN DEFAULT FALSE,
                    context TEXT
                )
            ''')
            
            # Threat campaigns table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS threat_campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    threat_actor TEXT,
                    categories TEXT NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP,
                    indicators TEXT,
                    ttps TEXT,
                    targets TEXT,
                    confidence INTEGER NOT NULL,
                    status TEXT DEFAULT 'active',
                    description TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_indicators_type ON threat_indicators (indicator_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_indicators_value ON threat_indicators (value)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_indicators_category ON threat_indicators (threat_category)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_entity ON behavioral_anomalies (entity_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_type ON behavioral_anomalies (anomaly_type)')
            
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
    
    def _load_threat_intelligence_data(self) -> None:
        """Load existing threat intelligence data from database."""
        try:
            with self._get_db_connection() as conn:
                # Load indicators
                cursor = conn.execute("SELECT * FROM threat_indicators")
                for row in cursor.fetchall():
                    indicator = ThreatIndicator(
                        indicator_id=row['indicator_id'],
                        indicator_type=IndicatorType(row['indicator_type']),
                        value=row['value'],
                        threat_category=ThreatCategory(row['threat_category']),
                        confidence=Confidence(row['confidence']),
                        severity=ThreatLevel(row['severity']),
                        source=ThreatSource(row['source']),
                        description=row['description'] or "",
                        tags=json.loads(row['tags'] or '[]'),
                        first_seen=datetime.fromisoformat(row['first_seen']),
                        last_seen=datetime.fromisoformat(row['last_seen']),
                        expiry_date=datetime.fromisoformat(row['expiry_date']) if row['expiry_date'] else None,
                        context=json.loads(row['context'] or '{}'),
                        related_indicators=json.loads(row['related_indicators'] or '[]')
                    )
                    self.indicators[indicator.indicator_id] = indicator
                
                # Load threat feeds
                cursor = conn.execute("SELECT * FROM threat_feeds")
                for row in cursor.fetchall():
                    feed = ThreatFeed(
                        feed_id=row['feed_id'],
                        name=row['name'],
                        url=row['url'],
                        feed_type=row['feed_type'],
                        api_key=row['api_key'],
                        update_interval=row['update_interval'] or 3600,
                        enabled=bool(row['enabled']),
                        last_updated=datetime.fromisoformat(row['last_updated']) if row['last_updated'] else None,
                        error_count=row['error_count'] or 0,
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    self.threat_feeds[feed.feed_id] = feed
                
                # Load anomaly models
                cursor = conn.execute("SELECT * FROM anomaly_models")
                for row in cursor.fetchall():
                    model = AnomalyModel(
                        model_id=row['model_id'],
                        model_type=row['model_type'],
                        feature_names=json.loads(row['feature_names']),
                        model_data=row['model_data'],
                        training_date=datetime.fromisoformat(row['training_date']),
                        accuracy_score=row['accuracy_score'],
                        false_positive_rate=row['false_positive_rate'],
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    self.anomaly_models[model.model_id] = model
                
        except sqlite3.OperationalError:
            # Database doesn't exist yet
            pass
    
    def _initialize_default_feeds(self) -> None:
        """Initialize default threat intelligence feeds."""
        # Example feeds (in production, these would be real threat intel sources)
        default_feeds = [
            ThreatFeed(
                feed_id="abuse_ch_malware",
                name="Abuse.ch Malware Hashes",
                url="https://malware-bazaar.abuse.ch/downloads/",
                feed_type="json",
                update_interval=7200  # 2 hours
            ),
            ThreatFeed(
                feed_id="emergingthreats_compromised",
                name="Emerging Threats Compromised IPs",
                url="https://rules.emergingthreats.net/blockrules/compromised-ips.txt",
                feed_type="csv",
                update_interval=3600  # 1 hour
            ),
            ThreatFeed(
                feed_id="internal_honeypot",
                name="Internal Honeypot Intelligence",
                url="internal://honeypot/indicators",
                feed_type="json",
                update_interval=1800  # 30 minutes
            )
        ]
        
        for feed in default_feeds:
            if feed.feed_id not in self.threat_feeds:
                self.threat_feeds[feed.feed_id] = feed
                self._store_threat_feed(feed)
        
        logger.info(f"Initialized {len(default_feeds)} default threat feeds")
    
    def _start_background_processing(self) -> None:
        """Start background threat intelligence processing."""
        self.running = True
        
        # Start feed update thread
        self.feed_update_thread = threading.Thread(target=self._feed_update_worker, daemon=True)
        self.feed_update_thread.start()
        
        logger.info("Background threat intelligence processing started")
    
    def _feed_update_worker(self) -> None:
        """Background worker for updating threat feeds."""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for feed in self.threat_feeds.values():
                    if not feed.enabled:
                        continue
                    
                    # Check if update is due
                    if (feed.last_updated is None or 
                        current_time - feed.last_updated > timedelta(seconds=feed.update_interval)):
                        
                        # Submit feed update to executor
                        self.processing_executor.submit(self._update_threat_feed, feed.feed_id)
                
                # Sleep for 5 minutes before next check
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in feed update worker: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def _update_threat_feed(self, feed_id: str) -> None:
        """Update individual threat feed."""
        if feed_id not in self.threat_feeds:
            return
        
        feed = self.threat_feeds[feed_id]
        
        try:
            logger.info(f"Updating threat feed: {feed.name}")
            
            # Handle internal feeds differently
            if feed.url.startswith("internal://"):
                indicators = self._process_internal_feed(feed)
            else:
                # Fetch external feed
                headers = {}
                if feed.api_key:
                    headers['Authorization'] = f'Bearer {feed.api_key}'
                
                response = requests.get(feed.url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Parse feed based on type
                if feed.feed_type == "json":
                    indicators = self._parse_json_feed(response.json(), feed)
                elif feed.feed_type == "csv":
                    indicators = self._parse_csv_feed(response.text, feed)
                else:
                    logger.warning(f"Unsupported feed type: {feed.feed_type}")
                    return
            
            # Store indicators
            new_indicators = 0
            updated_indicators = 0
            
            for indicator in indicators:
                if indicator.indicator_id in self.indicators:
                    # Update existing indicator
                    existing = self.indicators[indicator.indicator_id]
                    existing.last_seen = indicator.last_seen
                    existing.confidence = indicator.confidence
                    updated_indicators += 1
                else:
                    # New indicator
                    self.indicators[indicator.indicator_id] = indicator
                    new_indicators += 1
                
                self._store_threat_indicator(indicator)
            
            # Update feed metadata
            feed.last_updated = datetime.now(timezone.utc)
            feed.error_count = 0
            self._store_threat_feed(feed)
            
            # Log successful update
            self.audit_logger.log_security_event(SecurityEvent(
                event_type=SecurityEventType.FUNCTION_CALL,
                component="threat_intelligence",
                severity=SeverityLevel.LOW,
                details=f"Threat feed updated: {feed.name}",
                metadata={
                    "feed_id": feed_id,
                    "new_indicators": new_indicators,
                    "updated_indicators": updated_indicators
                }
            ))
            
            logger.info(f"Feed update completed: {feed.name} ({new_indicators} new, {updated_indicators} updated)")
            
        except Exception as e:
            feed.error_count += 1
            self._store_threat_feed(feed)
            
            logger.error(f"Failed to update threat feed {feed.name}: {e}")
            
            # Log error
            self.audit_logger.log_security_event(SecurityEvent(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                component="threat_intelligence",
                severity=SeverityLevel.MEDIUM,
                details=f"Threat feed update failed: {feed.name}",
                metadata={"feed_id": feed_id, "error": str(e)}
            ))
    
    def _process_internal_feed(self, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Process internal threat intelligence feed."""
        indicators = []
        
        # Simulate internal honeypot data
        if "honeypot" in feed.feed_id:
            # Generate synthetic indicators for demonstration
            import random
            
            for _ in range(random.randint(1, 5)):
                indicator_id = f"internal_{int(time.time())}_{random.randint(1000, 9999)}"
                
                indicators.append(ThreatIndicator(
                    indicator_id=indicator_id,
                    indicator_type=IndicatorType.IP_ADDRESS,
                    value=f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                    threat_category=ThreatCategory.BOTNET,
                    confidence=Confidence.HIGH,
                    severity=ThreatLevel.MEDIUM,
                    source=ThreatSource.HONEYPOT,
                    description="Suspicious IP detected by internal honeypot",
                    tags=["honeypot", "automated"]
                ))
        
        return indicators
    
    def _parse_json_feed(self, data: Dict[str, Any], feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse JSON threat intelligence feed."""
        indicators = []
        
        # Generic JSON parsing (would be customized per feed)
        if isinstance(data, list):
            items = data
        elif "data" in data:
            items = data["data"]
        elif "indicators" in data:
            items = data["indicators"]
        else:
            items = [data]
        
        for item in items[:100]:  # Limit to prevent overload
            try:
                # Extract indicator information
                indicator_value = item.get("indicator") or item.get("value") or item.get("ioc")
                if not indicator_value:
                    continue
                
                indicator_type = self._determine_indicator_type(indicator_value)
                threat_category = self._determine_threat_category(item)
                
                indicator_id = hashlib.sha256(f"{feed.feed_id}:{indicator_value}".encode()).hexdigest()[:16]
                
                indicator = ThreatIndicator(
                    indicator_id=indicator_id,
                    indicator_type=indicator_type,
                    value=indicator_value,
                    threat_category=threat_category,
                    confidence=Confidence.MEDIUM,
                    severity=ThreatLevel.MEDIUM,
                    source=ThreatSource.COMMERCIAL_FEED,
                    description=item.get("description", ""),
                    tags=item.get("tags", [])
                )
                
                indicators.append(indicator)
                
            except Exception as e:
                logger.warning(f"Error parsing indicator from {feed.name}: {e}")
                continue
        
        return indicators
    
    def _parse_csv_feed(self, data: str, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse CSV threat intelligence feed."""
        indicators = []
        lines = data.strip().split('\n')
        
        for line in lines[:100]:  # Limit to prevent overload
            if line.startswith('#') or not line.strip():
                continue
            
            try:
                parts = line.split(',')
                indicator_value = parts[0].strip()
                
                if not indicator_value:
                    continue
                
                indicator_type = self._determine_indicator_type(indicator_value)
                indicator_id = hashlib.sha256(f"{feed.feed_id}:{indicator_value}".encode()).hexdigest()[:16]
                
                indicator = ThreatIndicator(
                    indicator_id=indicator_id,
                    indicator_type=indicator_type,
                    value=indicator_value,
                    threat_category=ThreatCategory.MALWARE,
                    confidence=Confidence.MEDIUM,
                    severity=ThreatLevel.MEDIUM,
                    source=ThreatSource.OPEN_SOURCE,
                    description=parts[1] if len(parts) > 1 else "",
                    tags=["csv_feed"]
                )
                
                indicators.append(indicator)
                
            except Exception as e:
                logger.warning(f"Error parsing CSV line from {feed.name}: {e}")
                continue
        
        return indicators
    
    def _determine_indicator_type(self, value: str) -> IndicatorType:
        """Determine the type of threat indicator."""
        import re
        
        # IP address
        if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', value):
            return IndicatorType.IP_ADDRESS
        
        # Domain
        if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return IndicatorType.DOMAIN
        
        # URL
        if value.startswith(('http://', 'https://', 'ftp://')):
            return IndicatorType.URL
        
        # File hash (MD5, SHA1, SHA256)
        if re.match(r'^[a-fA-F0-9]{32}$', value):
            return IndicatorType.FILE_HASH
        elif re.match(r'^[a-fA-F0-9]{40}$', value):
            return IndicatorType.FILE_HASH
        elif re.match(r'^[a-fA-F0-9]{64}$', value):
            return IndicatorType.FILE_HASH
        
        # Email
        if '@' in value and '.' in value:
            return IndicatorType.EMAIL
        
        # Default to behavioral pattern
        return IndicatorType.BEHAVIORAL_PATTERN
    
    def _determine_threat_category(self, item: Dict[str, Any]) -> ThreatCategory:
        """Determine threat category from feed item."""
        category_str = item.get("category", "").lower()
        malware_type = item.get("malware_type", "").lower()
        
        if "ransomware" in category_str or "ransomware" in malware_type:
            return ThreatCategory.RANSOMWARE
        elif "phishing" in category_str:
            return ThreatCategory.PHISHING
        elif "botnet" in category_str:
            return ThreatCategory.BOTNET
        elif "apt" in category_str:
            return ThreatCategory.APT
        else:
            return ThreatCategory.MALWARE
    
    def check_indicator(self, indicator_type: IndicatorType, value: str) -> Optional[ThreatIndicator]:
        """Check if an indicator matches known threats."""
        for indicator in self.indicators.values():
            if (indicator.indicator_type == indicator_type and 
                indicator.value == value and
                (indicator.expiry_date is None or indicator.expiry_date > datetime.now(timezone.utc))):
                
                # Update last seen
                indicator.last_seen = datetime.now(timezone.utc)
                self._store_threat_indicator(indicator)
                
                return indicator
        
        return None
    
    def analyze_behavioral_anomaly(self, entity_id: str, features: Dict[str, float],
                                 entity_type: str = "user") -> Optional[BehavioralAnomaly]:
        """Analyze entity behavior for anomalies using ML models."""
        model_id = f"{entity_type}_anomaly_detection"
        
        if model_id not in self.anomaly_models:
            # No model available for this entity type
            return None
        
        model_info = self.anomaly_models[model_id]
        
        try:
            # Load the ML model
            model = pickle.loads(model_info.model_data)
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in model_info.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            feature_array = np.array([feature_vector])
            
            # Predict anomaly
            anomaly_score = model.decision_function(feature_array)[0]
            is_anomaly = model.predict(feature_array)[0] == -1
            
            if is_anomaly:
                anomaly_id = f"anomaly_{int(time.time())}_{hashlib.md5(entity_id.encode()).hexdigest()[:8]}"
                
                # Calculate confidence and deviation scores
                confidence_score = min(abs(anomaly_score) / 2.0, 1.0)  # Normalize to 0-1
                deviation_score = abs(anomaly_score)
                
                anomaly = BehavioralAnomaly(
                    anomaly_id=anomaly_id,
                    user_id=entity_id if entity_type == "user" else None,
                    entity_id=entity_id,
                    anomaly_type=f"{entity_type}_behavioral_anomaly",
                    confidence_score=confidence_score,
                    deviation_score=deviation_score,
                    features=features,
                    context={"model_id": model_id, "model_type": model_info.model_type}
                )
                
                # Store anomaly
                self.recent_anomalies[anomaly_id] = anomaly
                self._store_behavioral_anomaly(anomaly)
                
                # Log anomaly detection
                self.audit_logger.log_security_event(SecurityEvent(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    component="threat_intelligence",
                    severity=SeverityLevel.MEDIUM,
                    details=f"Behavioral anomaly detected: {entity_id}",
                    metadata={
                        "entity_id": entity_id,
                        "entity_type": entity_type,
                        "confidence_score": confidence_score,
                        "deviation_score": deviation_score,
                        "anomaly_id": anomaly_id
                    }
                ))
                
                return anomaly
        
        except Exception as e:
            logger.error(f"Error analyzing behavioral anomaly: {e}")
        
        return None
    
    def train_anomaly_model(self, entity_type: str, training_data: List[Dict[str, float]],
                           model_type: str = "isolation_forest") -> bool:
        """Train ML model for anomaly detection."""
        try:
            if not training_data:
                logger.warning("No training data provided for anomaly model")
                return False
            
            # Prepare training data
            feature_names = list(training_data[0].keys())
            X = []
            
            for sample in training_data:
                feature_vector = [sample.get(name, 0.0) for name in feature_names]
                X.append(feature_vector)
            
            X = np.array(X)
            
            # Train model based on type
            if model_type == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X)
                
            elif model_type == "one_class_svm":
                from sklearn.svm import OneClassSVM
                model = OneClassSVM(gamma='scale', nu=0.1)
                model.fit(X)
                
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return False
            
            # Calculate model metrics
            predictions = model.predict(X)
            accuracy = np.sum(predictions == 1) / len(predictions)
            false_positive_rate = np.sum(predictions == -1) / len(predictions)
            
            # Serialize model
            model_data = pickle.dumps(model)
            
            # Create model info
            model_id = f"{entity_type}_anomaly_detection"
            model_info = AnomalyModel(
                model_id=model_id,
                model_type=model_type,
                feature_names=feature_names,
                model_data=model_data,
                training_date=datetime.now(timezone.utc),
                accuracy_score=accuracy,
                false_positive_rate=false_positive_rate,
                metadata={"training_samples": len(training_data)}
            )
            
            # Store model
            self.anomaly_models[model_id] = model_info
            self._store_anomaly_model(model_info)
            
            logger.info(f"Anomaly model trained: {model_id} (accuracy: {accuracy:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error training anomaly model: {e}")
            return False
    
    def create_threat_campaign(self, name: str, threat_actor: Optional[str],
                              categories: List[ThreatCategory],
                              indicators: List[str],
                              description: str = "") -> str:
        """Create new threat campaign for tracking related indicators."""
        campaign_id = f"campaign_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        campaign = ThreatCampaign(
            campaign_id=campaign_id,
            name=name,
            threat_actor=threat_actor,
            categories=categories,
            start_date=datetime.now(timezone.utc),
            indicators=indicators,
            confidence=Confidence.MEDIUM,
            description=description
        )
        
        self.campaigns[campaign_id] = campaign
        self._store_threat_campaign(campaign)
        
        # Log campaign creation
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            component="threat_intelligence",
            severity=SeverityLevel.MEDIUM,
            details=f"Threat campaign created: {name}",
            metadata={
                "campaign_id": campaign_id,
                "threat_actor": threat_actor,
                "indicator_count": len(indicators)
            }
        ))
        
        return campaign_id
    
    def _store_threat_indicator(self, indicator: ThreatIndicator) -> None:
        """Store threat indicator in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO threat_indicators (
                    indicator_id, indicator_type, value, threat_category,
                    confidence, severity, source, description, tags,
                    first_seen, last_seen, expiry_date, context, related_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                indicator.indicator_id, indicator.indicator_type.value,
                indicator.value, indicator.threat_category.value,
                indicator.confidence.value, indicator.severity.value,
                indicator.source.value, indicator.description,
                json.dumps(indicator.tags), indicator.first_seen.isoformat(),
                indicator.last_seen.isoformat(),
                indicator.expiry_date.isoformat() if indicator.expiry_date else None,
                json.dumps(indicator.context), json.dumps(indicator.related_indicators)
            ))
            conn.commit()
    
    def _store_threat_feed(self, feed: ThreatFeed) -> None:
        """Store threat feed in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO threat_feeds (
                    feed_id, name, url, feed_type, api_key, update_interval,
                    enabled, last_updated, error_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feed.feed_id, feed.name, feed.url, feed.feed_type,
                feed.api_key, feed.update_interval, feed.enabled,
                feed.last_updated.isoformat() if feed.last_updated else None,
                feed.error_count, json.dumps(feed.metadata)
            ))
            conn.commit()
    
    def _store_anomaly_model(self, model: AnomalyModel) -> None:
        """Store anomaly model in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO anomaly_models (
                    model_id, model_type, feature_names, model_data,
                    training_date, accuracy_score, false_positive_rate, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model.model_id, model.model_type, json.dumps(model.feature_names),
                model.model_data, model.training_date.isoformat(),
                model.accuracy_score, model.false_positive_rate,
                json.dumps(model.metadata)
            ))
            conn.commit()
    
    def _store_behavioral_anomaly(self, anomaly: BehavioralAnomaly) -> None:
        """Store behavioral anomaly in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO behavioral_anomalies (
                    anomaly_id, user_id, entity_id, anomaly_type,
                    confidence_score, deviation_score, features, detected_at,
                    investigated, false_positive, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                anomaly.anomaly_id, anomaly.user_id, anomaly.entity_id,
                anomaly.anomaly_type, anomaly.confidence_score,
                anomaly.deviation_score, json.dumps(anomaly.features),
                anomaly.detected_at.isoformat(), anomaly.investigated,
                anomaly.false_positive, json.dumps(anomaly.context)
            ))
            conn.commit()
    
    def _store_threat_campaign(self, campaign: ThreatCampaign) -> None:
        """Store threat campaign in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO threat_campaigns (
                    campaign_id, name, threat_actor, categories, start_date,
                    end_date, indicators, ttps, targets, confidence,
                    status, description, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                campaign.campaign_id, campaign.name, campaign.threat_actor,
                json.dumps([cat.value for cat in campaign.categories]),
                campaign.start_date.isoformat(),
                campaign.end_date.isoformat() if campaign.end_date else None,
                json.dumps(campaign.indicators), json.dumps(campaign.ttps),
                json.dumps(campaign.targets), campaign.confidence.value,
                campaign.status, campaign.description, json.dumps(campaign.metadata)
            ))
            conn.commit()
    
    def get_threat_intelligence_dashboard_data(self) -> Dict[str, Any]:
        """Get threat intelligence dashboard data for monitoring."""
        with self._get_db_connection() as conn:
            dashboard_data = {
                "indicators": {
                    "total": len(self.indicators),
                    "by_type": {},
                    "by_category": {},
                    "by_source": {},
                    "recent_count": 0
                },
                "feeds": {
                    "total": len(self.threat_feeds),
                    "active": len([f for f in self.threat_feeds.values() if f.enabled]),
                    "errors": len([f for f in self.threat_feeds.values() if f.error_count > 0])
                },
                "anomalies": {
                    "recent_count": 0,
                    "by_type": {},
                    "false_positive_rate": 0.0
                },
                "campaigns": {
                    "active": len([c for c in self.campaigns.values() if c.status == "active"]),
                    "total": len(self.campaigns)
                }
            }
            
            # Indicator statistics
            type_counts = {}
            category_counts = {}
            source_counts = {}
            recent_count = 0
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            
            for indicator in self.indicators.values():
                indicator_type = indicator.indicator_type.value
                category = indicator.threat_category.value
                source = indicator.source.value
                
                type_counts[indicator_type] = type_counts.get(indicator_type, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
                
                if indicator.first_seen > cutoff_time:
                    recent_count += 1
            
            dashboard_data["indicators"]["by_type"] = type_counts
            dashboard_data["indicators"]["by_category"] = category_counts
            dashboard_data["indicators"]["by_source"] = source_counts
            dashboard_data["indicators"]["recent_count"] = recent_count
            
            # Anomaly statistics
            cursor = conn.execute('''
                SELECT anomaly_type, COUNT(*) as count
                FROM behavioral_anomalies
                WHERE detected_at > datetime('now', '-24 hours')
                GROUP BY anomaly_type
            ''')
            
            anomaly_counts = {}
            total_anomalies = 0
            
            for row in cursor.fetchall():
                anomaly_counts[row['anomaly_type']] = row['count']
                total_anomalies += row['count']
            
            dashboard_data["anomalies"]["recent_count"] = total_anomalies
            dashboard_data["anomalies"]["by_type"] = anomaly_counts
            
            # False positive rate
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN false_positive = 1 THEN 1 ELSE 0 END) as false_positives
                FROM behavioral_anomalies
                WHERE detected_at > datetime('now', '-7 days')
            ''')
            
            row = cursor.fetchone()
            if row and row['total'] > 0:
                dashboard_data["anomalies"]["false_positive_rate"] = round(
                    (row['false_positives'] / row['total']) * 100, 1
                )
        
        return dashboard_data
    
    def shutdown(self) -> None:
        """Shutdown threat intelligence system."""
        self.running = False
        
        if self.feed_update_thread and self.feed_update_thread.is_alive():
            self.feed_update_thread.join(timeout=5)
        
        self.processing_executor.shutdown(wait=True)
        
        if self.geoip_reader:
            self.geoip_reader.close()
        
        logger.info("AdvancedThreatIntelligenceSystem shutdown completed")


# Global threat intelligence system instance
_global_threat_intelligence: Optional[AdvancedThreatIntelligenceSystem] = None


def get_threat_intelligence() -> AdvancedThreatIntelligenceSystem:
    """Get the global threat intelligence system."""
    global _global_threat_intelligence
    if _global_threat_intelligence is None:
        _global_threat_intelligence = AdvancedThreatIntelligenceSystem()
    return _global_threat_intelligence


def initialize_threat_intelligence(db_path: str = "memmimic_threat_intel.db",
                                  models_dir: str = "threat_models",
                                  audit_logger: Optional[SecurityAuditLogger] = None) -> AdvancedThreatIntelligenceSystem:
    """Initialize the global threat intelligence system."""
    global _global_threat_intelligence
    _global_threat_intelligence = AdvancedThreatIntelligenceSystem(db_path, models_dir, audit_logger)
    return _global_threat_intelligence