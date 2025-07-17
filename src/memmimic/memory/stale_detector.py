#!/usr/bin/env python3
"""
Stale Memory Detector - Intelligent Memory Cleanup
Sophisticated detection and management of stale memories for active memory pool optimization
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

class MemoryStatus(Enum):
    """Memory lifecycle status"""
    ACTIVE = "active"
    STALE_CANDIDATE = "stale_candidate" 
    ARCHIVE_CANDIDATE = "archive_candidate"
    PRUNE_CANDIDATE = "prune_candidate"
    PROTECTED = "protected"

class StaleReason(Enum):
    """Reasons for marking memory as stale"""
    LONG_UNUSED = "long_unused"
    LOW_IMPORTANCE = "low_importance"
    SUPERSEDED = "superseded"
    OUTDATED_CONTENT = "outdated_content"
    REDUNDANT = "redundant"
    LOW_QUALITY = "low_quality"

@dataclass
class StaleDetectionConfig:
    """Configuration for stale memory detection"""
    # Time-based thresholds
    stale_threshold_days: int = 30
    archive_threshold_days: int = 90
    prune_threshold_days: int = 180
    
    # Importance-based thresholds
    min_active_importance: float = 0.3
    archive_importance_threshold: float = 0.2
    prune_importance_threshold: float = 0.1
    
    # Access-based thresholds
    min_access_frequency: float = 0.01  # Accesses per day
    stale_access_gap_days: int = 45
    
    # Quality-based thresholds
    min_confidence_threshold: float = 0.4
    max_content_age_days: int = 365
    
    # Protection rules
    protected_types: Set[str] = None
    protected_tags: Set[str] = None
    min_protection_importance: float = 0.8
    
    def __post_init__(self):
        if self.protected_types is None:
            self.protected_types = {'synthetic_wisdom', 'milestone', 'consciousness_evolution'}
        if self.protected_tags is None:
            self.protected_tags = {'critical', 'permanent', 'system_important'}

@dataclass
class StaleDetectionResult:
    """Result of stale memory detection"""
    memory_id: int
    current_status: MemoryStatus
    recommended_status: MemoryStatus
    stale_reasons: List[StaleReason]
    staleness_score: float
    protection_reasons: List[str]
    confidence: float
    
    def should_change_status(self) -> bool:
        """Check if status should be changed"""
        return (self.current_status != self.recommended_status and 
                self.confidence > 0.7 and 
                len(self.protection_reasons) == 0)

class StaleMemoryDetector:
    """
    Intelligent stale memory detection and lifecycle management
    
    This class implements sophisticated algorithms to identify memories that
    should be archived or pruned based on multiple factors including:
    - Access patterns and frequency
    - Importance scores and quality metrics
    - Content age and relevance
    - Memory type and protection rules
    - Superseding relationships
    """
    
    def __init__(self, db_path: str, config: Optional[StaleDetectionConfig] = None):
        self.db_path = db_path
        self.config = config or StaleDetectionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Detection metrics
        self._detection_runs = 0
        self._total_memories_processed = 0
        self._stale_memories_found = 0
        
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA encoding = 'UTF-8'")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def detect_stale_memories(self, batch_size: int = 100) -> List[StaleDetectionResult]:
        """
        Detect stale memories in the active pool
        
        Args:
            batch_size: Number of memories to process in each batch
            
        Returns:
            List of StaleDetectionResult objects
        """
        results = []
        self._detection_runs += 1
        
        with self._get_connection() as conn:
            # Get active memories for analysis
            cursor = conn.execute("""
                SELECT m.*, 
                       COALESCE(ap.recent_accesses, 0) as recent_accesses,
                       COALESCE(ap.avg_access_interval, 999) as avg_access_interval
                FROM memories_enhanced m
                LEFT JOIN (
                    SELECT memory_id, 
                           COUNT(*) as recent_accesses,
                           AVG(julianday('now') - julianday(access_time)) as avg_access_interval
                    FROM memory_access_patterns 
                    WHERE access_time > datetime('now', '-30 days')
                    GROUP BY memory_id
                ) ap ON m.id = ap.memory_id
                WHERE m.archive_status = 'active'
                ORDER BY m.last_access_time ASC
                LIMIT ?
            """, (batch_size,))
            
            memories = cursor.fetchall()
            self._total_memories_processed += len(memories)
            
            for memory in memories:
                result = self._analyze_memory_staleness(dict(memory))
                results.append(result)
                
                if result.recommended_status != MemoryStatus.ACTIVE:
                    self._stale_memories_found += 1
        
        self.logger.info(f"Stale detection completed: {len(results)} memories analyzed, "
                        f"{self._stale_memories_found} stale found")
        
        return results
    
    def apply_stale_detection_results(self, results: List[StaleDetectionResult], 
                                    dry_run: bool = False) -> Dict[str, int]:
        """
        Apply stale detection results to update memory statuses
        
        Args:
            results: List of detection results to apply
            dry_run: If True, only simulate changes without applying them
            
        Returns:
            Dictionary with counts of changes made
        """
        stats = {
            'archived': 0,
            'marked_for_pruning': 0,
            'protected': 0,
            'no_change': 0
        }
        
        if dry_run:
            # Just calculate what would happen
            for result in results:
                if result.should_change_status():
                    if result.recommended_status == MemoryStatus.ARCHIVE_CANDIDATE:
                        stats['archived'] += 1
                    elif result.recommended_status == MemoryStatus.PRUNE_CANDIDATE:
                        stats['marked_for_pruning'] += 1
                else:
                    if result.protection_reasons:
                        stats['protected'] += 1
                    else:
                        stats['no_change'] += 1
            return stats
        
        # Apply actual changes
        with self._get_connection() as conn:
            for result in results:
                if result.should_change_status():
                    # Map status to archive_status values
                    archive_status = self._map_status_to_db(result.recommended_status)
                    
                    conn.execute("""
                        UPDATE memories_enhanced 
                        SET archive_status = ?, 
                            metadata = json_set(COALESCE(metadata, '{}'), 
                                              '$.stale_detection', json(?))
                        WHERE id = ?
                    """, (
                        archive_status,
                        self._create_stale_metadata(result),
                        result.memory_id
                    ))
                    
                    # Update statistics
                    if result.recommended_status == MemoryStatus.ARCHIVE_CANDIDATE:
                        stats['archived'] += 1
                    elif result.recommended_status == MemoryStatus.PRUNE_CANDIDATE:
                        stats['marked_for_pruning'] += 1
                else:
                    if result.protection_reasons:
                        stats['protected'] += 1
                    else:
                        stats['no_change'] += 1
            
            conn.commit()
        
        self.logger.info(f"Stale detection applied: {stats}")
        return stats
    
    def get_stale_detection_summary(self) -> Dict[str, Any]:
        """Get summary of stale detection statistics"""
        with self._get_connection() as conn:
            # Get current status distribution
            cursor = conn.execute("""
                SELECT archive_status, COUNT(*) as count, AVG(importance_score) as avg_importance
                FROM memories_enhanced 
                GROUP BY archive_status
            """)
            status_distribution = {row['archive_status']: {
                'count': row['count'],
                'avg_importance': row['avg_importance']
            } for row in cursor.fetchall()}
            
            # Get stale candidate analysis
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_candidates,
                    AVG(importance_score) as avg_importance,
                    MIN(last_access_time) as oldest_access,
                    MAX(last_access_time) as newest_access
                FROM memories_enhanced 
                WHERE archive_status = 'active' 
                AND importance_score < ?
                AND last_access_time < datetime('now', '-{} days')
            """.format(self.config.stale_threshold_days), (self.config.min_active_importance,))
            
            stale_candidates = dict(cursor.fetchone())
            
            return {
                'detection_runs': self._detection_runs,
                'total_processed': self._total_memories_processed,
                'stale_found': self._stale_memories_found,
                'status_distribution': status_distribution,
                'stale_candidates': stale_candidates,
                'config': self.config.__dict__
            }
    
    def _analyze_memory_staleness(self, memory_data: Dict[str, Any]) -> StaleDetectionResult:
        """Analyze a single memory for staleness"""
        memory_id = memory_data['id']
        current_status = MemoryStatus(memory_data.get('archive_status', 'active'))
        
        # Calculate staleness factors
        staleness_factors = self._calculate_staleness_factors(memory_data)
        
        # Check protection status
        protection_reasons = self._check_protection_status(memory_data)
        
        # Determine recommended status
        recommended_status, stale_reasons = self._determine_recommended_status(
            memory_data, staleness_factors, protection_reasons
        )
        
        # Calculate overall staleness score
        staleness_score = self._calculate_staleness_score(staleness_factors)
        
        # Calculate confidence in recommendation
        confidence = self._calculate_recommendation_confidence(
            staleness_factors, protection_reasons, stale_reasons
        )
        
        return StaleDetectionResult(
            memory_id=memory_id,
            current_status=current_status,
            recommended_status=recommended_status,
            stale_reasons=stale_reasons,
            staleness_score=staleness_score,
            protection_reasons=protection_reasons,
            confidence=confidence
        )
    
    def _calculate_staleness_factors(self, memory_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various staleness factors for a memory"""
        now = datetime.now()
        
        # Time factors
        last_access = datetime.fromisoformat(memory_data.get('last_access_time', memory_data['created_at']))
        created_at = datetime.fromisoformat(memory_data['created_at'])
        
        days_since_access = (now - last_access).days
        days_since_creation = (now - created_at).days
        
        # Access pattern factors
        access_count = memory_data.get('access_count', 0)
        recent_accesses = memory_data.get('recent_accesses', 0)
        avg_access_interval = memory_data.get('avg_access_interval', 999)
        
        # Calculate access frequency (accesses per day)
        access_frequency = access_count / max(days_since_creation, 1)
        
        # Quality factors
        importance_score = memory_data.get('importance_score', 0.0)
        confidence = memory_data.get('confidence', 0.0)
        
        # Content factors
        content_length = len(memory_data.get('content', ''))
        
        return {
            'days_since_access': days_since_access,
            'days_since_creation': days_since_creation,
            'access_frequency': access_frequency,
            'recent_accesses': recent_accesses,
            'avg_access_interval': avg_access_interval,
            'importance_score': importance_score,
            'confidence': confidence,
            'content_length': content_length,
            'access_count': access_count
        }
    
    def _check_protection_status(self, memory_data: Dict[str, Any]) -> List[str]:
        """Check if memory is protected from archival/pruning"""
        protection_reasons = []
        
        # Check memory type protection
        memory_type = memory_data.get('type', '')
        if memory_type in self.config.protected_types:
            protection_reasons.append(f"Protected type: {memory_type}")
        
        # Check importance-based protection
        importance = memory_data.get('importance_score', 0.0)
        if importance >= self.config.min_protection_importance:
            protection_reasons.append(f"High importance: {importance:.3f}")
        
        # Check tag-based protection
        tags = memory_data.get('tags', [])
        if isinstance(tags, str):
            import json
            try:
                tags = json.loads(tags)
            except:
                tags = []
        
        for tag in tags:
            if tag in self.config.protected_tags:
                protection_reasons.append(f"Protected tag: {tag}")
        
        # Check metadata protection
        metadata = memory_data.get('metadata', {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        if metadata.get('permanent', False):
            protection_reasons.append("Marked as permanent")
        
        if metadata.get('system_important', False):
            protection_reasons.append("System important")
        
        return protection_reasons
    
    def _determine_recommended_status(self, memory_data: Dict[str, Any], 
                                    staleness_factors: Dict[str, float],
                                    protection_reasons: List[str]) -> Tuple[MemoryStatus, List[StaleReason]]:
        """Determine recommended status and reasons"""
        stale_reasons = []
        
        # If protected, keep active
        if protection_reasons:
            return MemoryStatus.ACTIVE, stale_reasons
        
        # Check for pruning conditions (most severe)
        if (staleness_factors['days_since_access'] > self.config.prune_threshold_days or
            staleness_factors['importance_score'] < self.config.prune_importance_threshold):
            
            if staleness_factors['days_since_access'] > self.config.prune_threshold_days:
                stale_reasons.append(StaleReason.LONG_UNUSED)
            if staleness_factors['importance_score'] < self.config.prune_importance_threshold:
                stale_reasons.append(StaleReason.LOW_IMPORTANCE)
                
            return MemoryStatus.PRUNE_CANDIDATE, stale_reasons
        
        # Check for archival conditions
        if (staleness_factors['days_since_access'] > self.config.archive_threshold_days or
            staleness_factors['importance_score'] < self.config.archive_importance_threshold or
            staleness_factors['access_frequency'] < self.config.min_access_frequency):
            
            if staleness_factors['days_since_access'] > self.config.archive_threshold_days:
                stale_reasons.append(StaleReason.LONG_UNUSED)
            if staleness_factors['importance_score'] < self.config.archive_importance_threshold:
                stale_reasons.append(StaleReason.LOW_IMPORTANCE)
            if staleness_factors['access_frequency'] < self.config.min_access_frequency:
                stale_reasons.append(StaleReason.LOW_QUALITY)
                
            return MemoryStatus.ARCHIVE_CANDIDATE, stale_reasons
        
        # Check for stale conditions (warning)
        if (staleness_factors['days_since_access'] > self.config.stale_threshold_days or
            staleness_factors['importance_score'] < self.config.min_active_importance):
            
            if staleness_factors['days_since_access'] > self.config.stale_threshold_days:
                stale_reasons.append(StaleReason.LONG_UNUSED)
            if staleness_factors['importance_score'] < self.config.min_active_importance:
                stale_reasons.append(StaleReason.LOW_IMPORTANCE)
                
            return MemoryStatus.STALE_CANDIDATE, stale_reasons
        
        # Memory is fine, keep active
        return MemoryStatus.ACTIVE, stale_reasons
    
    def _calculate_staleness_score(self, staleness_factors: Dict[str, float]) -> float:
        """Calculate overall staleness score (0 = fresh, 1 = very stale)"""
        # Time staleness (40%)
        days_factor = min(staleness_factors['days_since_access'] / self.config.prune_threshold_days, 1.0)
        time_staleness = days_factor * 0.4
        
        # Importance staleness (30%)
        importance_staleness = (1.0 - staleness_factors['importance_score']) * 0.3
        
        # Access frequency staleness (20%)
        freq_staleness = (1.0 - min(staleness_factors['access_frequency'] * 10, 1.0)) * 0.2
        
        # Quality staleness (10%)
        quality_staleness = (1.0 - staleness_factors['confidence']) * 0.1
        
        return min(time_staleness + importance_staleness + freq_staleness + quality_staleness, 1.0)
    
    def _calculate_recommendation_confidence(self, staleness_factors: Dict[str, float],
                                           protection_reasons: List[str],
                                           stale_reasons: List[StaleReason]) -> float:
        """Calculate confidence in the recommendation"""
        base_confidence = 0.5
        
        # Higher confidence for clear-cut cases
        if len(stale_reasons) >= 2:
            base_confidence += 0.3
        elif len(stale_reasons) == 1:
            base_confidence += 0.2
        
        # Higher confidence for extreme values
        if staleness_factors['days_since_access'] > self.config.archive_threshold_days * 2:
            base_confidence += 0.2
        
        if staleness_factors['importance_score'] < 0.1:
            base_confidence += 0.2
        
        # Lower confidence if there are protection reasons
        if protection_reasons:
            base_confidence -= 0.3
        
        # Adjust based on access patterns
        if staleness_factors['access_count'] > 10:
            base_confidence += 0.1  # More data to make decision
        
        return max(0.0, min(1.0, base_confidence))
    
    def _map_status_to_db(self, status: MemoryStatus) -> str:
        """Map MemoryStatus to database archive_status values"""
        mapping = {
            MemoryStatus.ACTIVE: 'active',
            MemoryStatus.STALE_CANDIDATE: 'active',  # Still active but flagged
            MemoryStatus.ARCHIVE_CANDIDATE: 'archived',
            MemoryStatus.PRUNE_CANDIDATE: 'prune_candidate',
            MemoryStatus.PROTECTED: 'active'
        }
        return mapping.get(status, 'active')
    
    def _create_stale_metadata(self, result: StaleDetectionResult) -> str:
        """Create metadata JSON for stale detection results"""
        import json
        metadata = {
            'detected_at': datetime.now().isoformat(),
            'staleness_score': result.staleness_score,
            'stale_reasons': [reason.value for reason in result.stale_reasons],
            'confidence': result.confidence,
            'detector_version': '1.0'
        }
        return json.dumps(metadata)

# Utility functions
def create_stale_detector(db_path: str, config: Optional[StaleDetectionConfig] = None) -> StaleMemoryDetector:
    """Create and configure a stale memory detector"""
    return StaleMemoryDetector(db_path, config)

def run_stale_detection_batch(db_path: str, batch_size: int = 100, 
                             dry_run: bool = True) -> Dict[str, Any]:
    """Run a batch of stale detection and return results"""
    detector = create_stale_detector(db_path)
    
    # Detect stale memories
    results = detector.detect_stale_memories(batch_size)
    
    # Apply results
    stats = detector.apply_stale_detection_results(results, dry_run)
    
    # Get summary
    summary = detector.get_stale_detection_summary()
    
    return {
        'detection_results': len(results),
        'application_stats': stats,
        'system_summary': summary,
        'dry_run': dry_run
    }

if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        dry_run = len(sys.argv) < 3 or sys.argv[2] != '--apply'
        
        print(f"Running stale detection on: {db_path}")
        print(f"Mode: {'DRY RUN' if dry_run else 'APPLY CHANGES'}")
        
        results = run_stale_detection_batch(db_path, batch_size=50, dry_run=dry_run)
        
        print(f"\nResults:")
        print(f"  Detection results: {results['detection_results']}")
        print(f"  Application stats: {results['application_stats']}")
        print(f"  Active memories: {results['system_summary']['status_distribution'].get('active', {}).get('count', 0)}")
        print(f"  Archived memories: {results['system_summary']['status_distribution'].get('archived', {}).get('count', 0)}")
        
    else:
        print("Usage: python stale_detector.py <database_path> [--apply]")
        print("  --apply: Actually apply changes (default is dry run)")