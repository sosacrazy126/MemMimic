"""
Memory Evolution Tracking System

Tracks how memories evolve, get used, accessed, and modified over time.
Provides comprehensive analytics on memory lifecycle and usage patterns.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path

from ..memory.storage.amms_storage import Memory
from ..errors.exceptions import MemMimicError


class MemoryEventType(Enum):
    """Types of memory evolution events"""
    CREATED = "created"
    ACCESSED = "accessed"
    MODIFIED = "modified"
    RECALLED = "recalled"
    LINKED = "linked"
    IMPORTANCE_CHANGED = "importance_changed"
    CXD_RECLASSIFIED = "cxd_reclassified"
    ARCHIVED = "archived"
    ACTIVATED = "activated"
    PRUNED = "pruned"


@dataclass
class MemoryEvent:
    """A single memory evolution event"""
    memory_id: str
    event_type: MemoryEventType
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    trigger: Optional[str] = None  # What triggered this event


@dataclass
class MemoryUsageStats:
    """Statistics about memory usage patterns"""
    total_accesses: int = 0
    unique_contexts: int = 0
    last_accessed: Optional[datetime] = None
    access_frequency: float = 0.0  # accesses per day
    recall_success_rate: float = 0.0
    modification_count: int = 0
    link_count: int = 0
    importance_changes: int = 0
    cxd_changes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'total_accesses': self.total_accesses,
            'unique_contexts': self.unique_contexts,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'access_frequency': self.access_frequency,
            'recall_success_rate': self.recall_success_rate,
            'modification_count': self.modification_count,
            'link_count': self.link_count,
            'importance_changes': self.importance_changes,
            'cxd_changes': self.cxd_changes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryUsageStats':
        """Create from dictionary"""
        stats = cls()
        stats.total_accesses = data.get('total_accesses', 0)
        stats.unique_contexts = data.get('unique_contexts', 0)
        if data.get('last_accessed'):
            stats.last_accessed = datetime.fromisoformat(data['last_accessed'])
        stats.access_frequency = data.get('access_frequency', 0.0)
        stats.recall_success_rate = data.get('recall_success_rate', 0.0)
        stats.modification_count = data.get('modification_count', 0)
        stats.link_count = data.get('link_count', 0)
        stats.importance_changes = data.get('importance_changes', 0)
        stats.cxd_changes = data.get('cxd_changes', 0)
        return stats


@dataclass
class MemoryEvolutionPattern:
    """Detected patterns in memory evolution"""
    pattern_id: str
    pattern_type: str  # e.g., "importance_decay", "access_clustering", "modification_burst"
    memory_ids: List[str]
    confidence: float  # 0.0 to 1.0
    description: str
    detected_at: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)


class MemoryEvolutionTracker:
    """
    Tracks memory evolution and usage patterns over time.
    
    Core capabilities:
    - Event tracking: Records all memory lifecycle events
    - Usage analytics: Calculates comprehensive usage statistics
    - Pattern detection: Identifies evolution patterns and trends
    - Performance metrics: Tracks memory system performance
    - Evolution scoring: Provides memory evolution quality scores
    """
    
    def __init__(self, db_path: str = "memory_evolution.db"):
        self.db_path = Path(db_path)
        self._memory_stats: Dict[str, MemoryUsageStats] = {}
        self._event_buffer: List[MemoryEvent] = []
        self._detected_patterns: List[MemoryEvolutionPattern] = []
        self._init_database()
    
    def _init_database(self):
        """Initialize the evolution tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    previous_state TEXT,
                    new_state TEXT,
                    trigger TEXT
                )
            ''')
            
            # Create indexes separately
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_events_memory_id ON memory_events(memory_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_events_event_type ON memory_events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_events_timestamp ON memory_events(timestamp)')
            
            # Usage statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_usage_stats (
                    memory_id TEXT PRIMARY KEY,
                    stats_json TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # Evolution patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evolution_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    memory_ids TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    parameters TEXT
                )
            ''')
            
            # Create indexes for patterns table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_evolution_patterns_pattern_type ON evolution_patterns(pattern_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_evolution_patterns_detected_at ON evolution_patterns(detected_at)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            raise MemMimicError(f"Failed to initialize evolution tracking database: {e}")
    
    async def track_event(self, 
                         memory_id: str, 
                         event_type: MemoryEventType,
                         context: Dict[str, Any] = None,
                         previous_state: Dict[str, Any] = None,
                         new_state: Dict[str, Any] = None,
                         trigger: str = None):
        """Track a memory evolution event"""
        event = MemoryEvent(
            memory_id=memory_id,
            event_type=event_type,
            timestamp=datetime.now(),
            context=context or {},
            previous_state=previous_state,
            new_state=new_state,
            trigger=trigger
        )
        
        # Add to buffer for batch processing
        self._event_buffer.append(event)
        
        # Update real-time stats
        await self._update_memory_stats(event)
        
        # Flush buffer if it gets too large
        if len(self._event_buffer) > 100:
            await self._flush_event_buffer()
    
    async def _update_memory_stats(self, event: MemoryEvent):
        """Update memory usage statistics in real-time"""
        memory_id = event.memory_id
        
        if memory_id not in self._memory_stats:
            self._memory_stats[memory_id] = MemoryUsageStats()
        
        stats = self._memory_stats[memory_id]
        
        # Update based on event type
        if event.event_type == MemoryEventType.ACCESSED:
            stats.total_accesses += 1
            stats.last_accessed = event.timestamp
            
            # Track unique contexts
            context_key = str(event.context.get('context_hash', ''))
            if context_key:
                stats.unique_contexts += 1
                
        elif event.event_type == MemoryEventType.RECALLED:
            stats.total_accesses += 1
            stats.last_accessed = event.timestamp
            
            # Update recall success rate
            success = event.context.get('recall_success', True)
            current_rate = stats.recall_success_rate
            total_recalls = event.context.get('total_recalls', 1)
            stats.recall_success_rate = (current_rate * (total_recalls - 1) + (1.0 if success else 0.0)) / total_recalls
            
        elif event.event_type == MemoryEventType.MODIFIED:
            stats.modification_count += 1
            
        elif event.event_type == MemoryEventType.LINKED:
            stats.link_count += 1
            
        elif event.event_type == MemoryEventType.IMPORTANCE_CHANGED:
            stats.importance_changes += 1
            
        elif event.event_type == MemoryEventType.CXD_RECLASSIFIED:
            stats.cxd_changes += 1
        
        # Calculate access frequency (accesses per day)
        if stats.last_accessed and stats.total_accesses > 0:
            # Find first access event
            first_access = await self._get_first_access_time(memory_id)
            if first_access:
                days_active = max((stats.last_accessed - first_access).days, 1)
                stats.access_frequency = stats.total_accesses / days_active
    
    async def _get_first_access_time(self, memory_id: str) -> Optional[datetime]:
        """Get the timestamp of the first access event for a memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp FROM memory_events 
                WHERE memory_id = ? AND event_type IN ('accessed', 'recalled', 'created')
                ORDER BY timestamp ASC LIMIT 1
            ''', (memory_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return datetime.fromisoformat(result[0])
            return None
            
        except Exception:
            return None
    
    async def _flush_event_buffer(self):
        """Flush the event buffer to database"""
        if not self._event_buffer:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert events
            events_data = []
            for event in self._event_buffer:
                events_data.append((
                    event.memory_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    json.dumps(event.context),
                    json.dumps(event.previous_state) if event.previous_state else None,
                    json.dumps(event.new_state) if event.new_state else None,
                    event.trigger
                ))
            
            cursor.executemany('''
                INSERT INTO memory_events 
                (memory_id, event_type, timestamp, context, previous_state, new_state, trigger)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', events_data)
            
            # Update usage stats
            for memory_id, stats in self._memory_stats.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO memory_usage_stats 
                    (memory_id, stats_json, last_updated)
                    VALUES (?, ?, ?)
                ''', (
                    memory_id,
                    json.dumps(stats.to_dict()),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            # Clear buffer
            self._event_buffer.clear()
            
        except Exception as e:
            raise MemMimicError(f"Failed to flush event buffer: {e}")
    
    async def get_memory_evolution_summary(self, memory_id: str) -> Dict[str, Any]:
        """Get comprehensive evolution summary for a memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get events
            cursor.execute('''
                SELECT event_type, timestamp, context, trigger
                FROM memory_events 
                WHERE memory_id = ?
                ORDER BY timestamp ASC
            ''', (memory_id,))
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    'event_type': row[0],
                    'timestamp': row[1],
                    'context': json.loads(row[2]) if row[2] else {},
                    'trigger': row[3]
                })
            
            # Get usage stats
            stats = self._memory_stats.get(memory_id, MemoryUsageStats())
            
            # Calculate evolution metrics
            evolution_score = await self._calculate_evolution_score(memory_id, events, stats)
            
            conn.close()
            
            return {
                'memory_id': memory_id,
                'total_events': len(events),
                'events': events,
                'usage_stats': stats.to_dict(),
                'evolution_score': evolution_score,
                'lifecycle_stage': self._determine_lifecycle_stage(events, stats),
                'health_indicators': await self._calculate_health_indicators(memory_id, events, stats)
            }
            
        except Exception as e:
            raise MemMimicError(f"Failed to get memory evolution summary: {e}")
    
    async def _calculate_evolution_score(self, 
                                       memory_id: str, 
                                       events: List[Dict], 
                                       stats: MemoryUsageStats) -> float:
        """Calculate overall evolution quality score (0.0 to 1.0)"""
        score = 0.0
        max_score = 5.0
        
        # Access frequency component (0-1)
        if stats.access_frequency > 0:
            access_score = min(stats.access_frequency / 10.0, 1.0)  # Normalize to 10 accesses/day max
            score += access_score
        
        # Recall success rate component (0-1)
        score += stats.recall_success_rate
        
        # Usage diversity component (0-1)
        if stats.unique_contexts > 0:
            diversity_score = min(stats.unique_contexts / 20.0, 1.0)  # Normalize to 20 contexts max
            score += diversity_score
        
        # Evolution activity component (0-1)
        total_evolution_events = stats.modification_count + stats.importance_changes + stats.cxd_changes
        if total_evolution_events > 0:
            evolution_score = min(total_evolution_events / 10.0, 1.0)  # Normalize to 10 changes max
            score += evolution_score
        
        # Longevity component (0-1)
        if len(events) > 0:
            first_event = datetime.fromisoformat(events[0]['timestamp'])
            age_days = (datetime.now() - first_event).days
            longevity_score = min(age_days / 365.0, 1.0)  # Normalize to 1 year max
            score += longevity_score
        
        return min(score / max_score, 1.0)
    
    def _determine_lifecycle_stage(self, events: List[Dict], stats: MemoryUsageStats) -> str:
        """Determine the current lifecycle stage of a memory"""
        if not events:
            return "dormant"
        
        last_event_time = datetime.fromisoformat(events[-1]['timestamp'])
        days_since_last_event = (datetime.now() - last_event_time).days
        
        if days_since_last_event > 30:
            return "archived"
        elif stats.access_frequency > 1.0:  # More than 1 access per day
            return "active"
        elif stats.access_frequency > 0.1:  # More than 1 access per 10 days
            return "moderate"
        elif stats.total_accesses > 0:
            return "low_activity"
        else:
            return "created"
    
    async def _calculate_health_indicators(self, 
                                         memory_id: str, 
                                         events: List[Dict], 
                                         stats: MemoryUsageStats) -> Dict[str, Any]:
        """Calculate memory health indicators"""
        indicators = {
            'usage_health': 'good',
            'evolution_health': 'good',
            'access_pattern': 'normal',
            'warnings': []
        }
        
        # Usage health
        if stats.total_accesses == 0:
            indicators['usage_health'] = 'poor'
            indicators['warnings'].append('Memory has never been accessed')
        elif stats.access_frequency < 0.01:  # Less than 1 access per 100 days
            indicators['usage_health'] = 'fair'
            indicators['warnings'].append('Very low access frequency')
        
        # Evolution health
        if stats.modification_count == 0 and len(events) > 10:
            indicators['evolution_health'] = 'fair'
            indicators['warnings'].append('Memory has not evolved despite high activity')
        
        # Access pattern analysis
        if len(events) >= 5:
            recent_events = [e for e in events[-10:] if e['event_type'] in ['accessed', 'recalled']]
            if len(recent_events) >= 3:
                # Check for clustering (all accesses within short time period)
                timestamps = [datetime.fromisoformat(e['timestamp']) for e in recent_events]
                time_span = (max(timestamps) - min(timestamps)).total_seconds()
                if time_span < 3600:  # All within 1 hour
                    indicators['access_pattern'] = 'clustered'
                elif time_span > 86400 * 7:  # Spread over more than a week
                    indicators['access_pattern'] = 'distributed'
        
        return indicators
    
    async def detect_evolution_patterns(self, lookback_days: int = 30) -> List[MemoryEvolutionPattern]:
        """Detect patterns in memory evolution across the system"""
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent events
            cursor.execute('''
                SELECT memory_id, event_type, timestamp, context
                FROM memory_events 
                WHERE timestamp > ?
                ORDER BY timestamp ASC
            ''', (cutoff_time.isoformat(),))
            
            events_by_memory = {}
            for row in cursor.fetchall():
                memory_id = row[0]
                if memory_id not in events_by_memory:
                    events_by_memory[memory_id] = []
                events_by_memory[memory_id].append({
                    'event_type': row[1],
                    'timestamp': datetime.fromisoformat(row[2]),
                    'context': json.loads(row[3]) if row[3] else {}
                })
            
            conn.close()
            
            patterns = []
            
            # Pattern 1: Importance decay
            decay_memories = await self._detect_importance_decay_pattern(events_by_memory)
            if decay_memories:
                patterns.append(MemoryEvolutionPattern(
                    pattern_id=f"importance_decay_{int(time.time())}",
                    pattern_type="importance_decay",
                    memory_ids=decay_memories,
                    confidence=0.8,
                    description=f"Detected importance decay in {len(decay_memories)} memories",
                    detected_at=datetime.now(),
                    parameters={'lookback_days': lookback_days}
                ))
            
            # Pattern 2: Access clustering
            clustered_memories = await self._detect_access_clustering_pattern(events_by_memory)
            if clustered_memories:
                patterns.append(MemoryEvolutionPattern(
                    pattern_id=f"access_clustering_{int(time.time())}",
                    pattern_type="access_clustering",
                    memory_ids=clustered_memories,
                    confidence=0.7,
                    description=f"Detected access clustering in {len(clustered_memories)} memories",
                    detected_at=datetime.now(),
                    parameters={'lookback_days': lookback_days}
                ))
            
            # Pattern 3: Modification bursts
            burst_memories = await self._detect_modification_burst_pattern(events_by_memory)
            if burst_memories:
                patterns.append(MemoryEvolutionPattern(
                    pattern_id=f"modification_burst_{int(time.time())}",
                    pattern_type="modification_burst",
                    memory_ids=burst_memories,
                    confidence=0.9,
                    description=f"Detected modification bursts in {len(burst_memories)} memories",
                    detected_at=datetime.now(),
                    parameters={'lookback_days': lookback_days}
                ))
            
            # Store detected patterns
            await self._store_patterns(patterns)
            self._detected_patterns.extend(patterns)
            
            return patterns
            
        except Exception as e:
            raise MemMimicError(f"Failed to detect evolution patterns: {e}")
    
    async def _detect_importance_decay_pattern(self, events_by_memory: Dict) -> List[str]:
        """Detect memories showing importance decay pattern"""
        decay_memories = []
        
        for memory_id, events in events_by_memory.items():
            importance_events = [e for e in events if e['event_type'] == 'importance_changed']
            if len(importance_events) >= 2:
                # Check if importance is generally decreasing
                importance_values = []
                for event in importance_events:
                    new_importance = event['context'].get('new_importance')
                    if new_importance is not None:
                        importance_values.append(new_importance)
                
                if len(importance_values) >= 2:
                    # Simple check: is the trend downward?
                    avg_first_half = sum(importance_values[:len(importance_values)//2]) / (len(importance_values)//2)
                    avg_second_half = sum(importance_values[len(importance_values)//2:]) / (len(importance_values) - len(importance_values)//2)
                    
                    if avg_second_half < avg_first_half * 0.8:  # 20% decrease
                        decay_memories.append(memory_id)
        
        return decay_memories
    
    async def _detect_access_clustering_pattern(self, events_by_memory: Dict) -> List[str]:
        """Detect memories showing access clustering pattern"""
        clustered_memories = []
        
        for memory_id, events in events_by_memory.items():
            access_events = [e for e in events if e['event_type'] in ['accessed', 'recalled']]
            if len(access_events) >= 3:
                # Check for clustering (multiple accesses in short time periods)
                clusters = []
                current_cluster = [access_events[0]]
                
                for i in range(1, len(access_events)):
                    time_diff = (access_events[i]['timestamp'] - current_cluster[-1]['timestamp']).total_seconds()
                    if time_diff <= 3600:  # Within 1 hour
                        current_cluster.append(access_events[i])
                    else:
                        if len(current_cluster) >= 3:
                            clusters.append(current_cluster)
                        current_cluster = [access_events[i]]
                
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                
                if len(clusters) >= 2:  # Multiple clusters
                    clustered_memories.append(memory_id)
        
        return clustered_memories
    
    async def _detect_modification_burst_pattern(self, events_by_memory: Dict) -> List[str]:
        """Detect memories showing modification burst pattern"""
        burst_memories = []
        
        for memory_id, events in events_by_memory.items():
            mod_events = [e for e in events if e['event_type'] == 'modified']
            if len(mod_events) >= 3:
                # Check for bursts (multiple modifications in short time)
                for i in range(len(mod_events) - 2):
                    time_span = (mod_events[i+2]['timestamp'] - mod_events[i]['timestamp']).total_seconds()
                    if time_span <= 3600:  # 3 modifications within 1 hour
                        burst_memories.append(memory_id)
                        break
        
        return burst_memories
    
    async def _store_patterns(self, patterns: List[MemoryEvolutionPattern]):
        """Store detected patterns in database"""
        if not patterns:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for pattern in patterns:
                cursor.execute('''
                    INSERT OR REPLACE INTO evolution_patterns
                    (pattern_id, pattern_type, memory_ids, confidence, description, detected_at, parameters)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    json.dumps(pattern.memory_ids),
                    pattern.confidence,
                    pattern.description,
                    pattern.detected_at.isoformat(),
                    json.dumps(pattern.parameters)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            raise MemMimicError(f"Failed to store evolution patterns: {e}")
    
    async def get_system_evolution_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive system-wide evolution report"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Event statistics
            cursor.execute('''
                SELECT event_type, COUNT(*) 
                FROM memory_events 
                WHERE timestamp > ?
                GROUP BY event_type
            ''', (cutoff_time.isoformat(),))
            
            event_stats = dict(cursor.fetchall())
            
            # Memory activity distribution
            cursor.execute('''
                SELECT memory_id, COUNT(*) as event_count
                FROM memory_events 
                WHERE timestamp > ?
                GROUP BY memory_id
                ORDER BY event_count DESC
                LIMIT 20
            ''', (cutoff_time.isoformat(),))
            
            top_active_memories = dict(cursor.fetchall())
            
            # Recent patterns
            cursor.execute('''
                SELECT pattern_type, COUNT(*), AVG(confidence)
                FROM evolution_patterns
                WHERE detected_at > ?
                GROUP BY pattern_type
            ''', (cutoff_time.isoformat(),))
            
            pattern_stats = {}
            for row in cursor.fetchall():
                pattern_stats[row[0]] = {
                    'count': row[1],
                    'avg_confidence': row[2]
                }
            
            conn.close()
            
            # Calculate system health metrics
            total_events = sum(event_stats.values())
            active_memories = len(top_active_memories)
            
            return {
                'report_period_days': days,
                'generated_at': datetime.now().isoformat(),
                'event_statistics': event_stats,
                'total_events': total_events,
                'active_memories': active_memories,
                'top_active_memories': top_active_memories,
                'detected_patterns': pattern_stats,
                'system_health': {
                    'activity_level': 'high' if total_events > 100 else 'moderate' if total_events > 20 else 'low',
                    'evolution_rate': event_stats.get('modified', 0) / max(total_events, 1),
                    'recall_activity': event_stats.get('recalled', 0) / max(total_events, 1),
                    'pattern_diversity': len(pattern_stats)
                }
            }
            
        except Exception as e:
            raise MemMimicError(f"Failed to generate system evolution report: {e}")
    
    async def cleanup_old_events(self, days_to_keep: int = 90):
        """Clean up old evolution events to maintain performance"""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old events
            cursor.execute('''
                DELETE FROM memory_events 
                WHERE timestamp < ?
            ''', (cutoff_time.isoformat(),))
            
            deleted_events = cursor.rowcount
            
            # Delete old patterns
            cursor.execute('''
                DELETE FROM evolution_patterns 
                WHERE detected_at < ?
            ''', (cutoff_time.isoformat(),))
            
            deleted_patterns = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            return {
                'deleted_events': deleted_events,
                'deleted_patterns': deleted_patterns,
                'cutoff_date': cutoff_time.isoformat()
            }
            
        except Exception as e:
            raise MemMimicError(f"Failed to cleanup old events: {e}")
    
    async def close(self):
        """Close the evolution tracker and flush any remaining data"""
        await self._flush_event_buffer()