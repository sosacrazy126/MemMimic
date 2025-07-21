#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Persistent Queue - SQLite-based memory review queue
Stores pending memories awaiting quality approval with full persistence
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .quality_types import MemoryQualityResult


class PersistentMemoryQueue:
    """
    Persistent memory review queue using SQLite
    
    Stores memories awaiting quality approval with full persistence
    across application restarts and sessions
    """
    
    def __init__(self, db_path: str = "memory_queue.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the queue database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_review_queue (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        quality_result_json TEXT NOT NULL,
                        queued_at TEXT NOT NULL,
                        status TEXT DEFAULT 'pending_review',
                        reviewer_note TEXT,
                        processed_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memory_review_queue(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_queued_at ON memory_review_queue(queued_at)")
                
                conn.commit()
                self.logger.info(f"PersistentMemoryQueue initialized - {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize queue database: {e}")
            raise
    
    def add_to_queue(
        self, 
        content: str, 
        memory_type: str, 
        quality_result: MemoryQualityResult
    ) -> str:
        """Add memory to review queue"""
        try:
            queue_id = f"pending_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            now = datetime.now().isoformat()
            
            # Serialize quality result
            quality_data = {
                "approved": quality_result.approved,
                "reason": quality_result.reason,
                "confidence": quality_result.confidence,
                "auto_decision": quality_result.auto_decision,
                "suggested_content": quality_result.suggested_content,
                "timestamp": quality_result.timestamp.isoformat()
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO memory_review_queue 
                    (id, content, memory_type, quality_result_json, queued_at, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    queue_id,
                    content,
                    memory_type,
                    json.dumps(quality_data),
                    now,
                    now,
                    now
                ))
                conn.commit()
            
            self.logger.info(f"Added memory to review queue: {queue_id}")
            return queue_id
            
        except Exception as e:
            self.logger.error(f"Failed to add memory to queue: {e}")
            raise
    
    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get all memories awaiting review"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM memory_review_queue 
                    WHERE status = 'pending_review'
                    ORDER BY queued_at ASC
                """)
                
                reviews = []
                for row in cursor.fetchall():
                    quality_data = json.loads(row['quality_result_json'])
                    
                    # Reconstruct quality result
                    quality_result = MemoryQualityResult(
                        approved=quality_data['approved'],
                        reason=quality_data['reason'],
                        confidence=quality_data['confidence'],
                        auto_decision=quality_data['auto_decision'],
                        suggested_content=quality_data.get('suggested_content')
                    )
                    
                    reviews.append({
                        "id": row['id'],
                        "content": row['content'],
                        "memory_type": row['memory_type'],
                        "quality_result": quality_result,
                        "queued_at": datetime.fromisoformat(row['queued_at']),
                        "status": row['status']
                    })
                
                return reviews
                
        except Exception as e:
            self.logger.error(f"Failed to get pending reviews: {e}")
            return []
    
    def approve_memory(self, queue_id: str, reviewer_note: str = "") -> bool:
        """Approve a queued memory"""
        try:
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE memory_review_queue 
                    SET status = 'approved', reviewer_note = ?, processed_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'pending_review'
                """, (reviewer_note, now, now, queue_id))
                
                if cursor.rowcount == 0:
                    self.logger.warning(f"No pending memory found with ID: {queue_id}")
                    return False
                
                conn.commit()
                self.logger.info(f"Approved memory: {queue_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to approve memory {queue_id}: {e}")
            return False
    
    def reject_memory(self, queue_id: str, reason: str = "") -> bool:
        """Reject a queued memory"""
        try:
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE memory_review_queue 
                    SET status = 'rejected', reviewer_note = ?, processed_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'pending_review'
                """, (reason, now, now, queue_id))
                
                if cursor.rowcount == 0:
                    self.logger.warning(f"No pending memory found with ID: {queue_id}")
                    return False
                
                conn.commit()
                self.logger.info(f"Rejected memory: {queue_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to reject memory {queue_id}: {e}")
            return False
    
    def cleanup_old_entries(self, days: int = 30) -> int:
        """Clean up old processed entries"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM memory_review_queue 
                    WHERE status IN ('approved', 'rejected') 
                    AND processed_at < ?
                """, (cutoff_str,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old queue entries")
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old entries: {e}")
            return 0
    
    def get_memory_details(self, queue_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a queued memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM memory_review_queue WHERE id = ?
                """, (queue_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                quality_data = json.loads(row['quality_result_json'])
                quality_result = MemoryQualityResult(
                    approved=quality_data['approved'],
                    reason=quality_data['reason'],
                    confidence=quality_data['confidence'],
                    auto_decision=quality_data['auto_decision'],
                    suggested_content=quality_data.get('suggested_content')
                )
                
                return {
                    "id": row['id'],
                    "content": row['content'],
                    "memory_type": row['memory_type'],
                    "quality_result": quality_result,
                    "queued_at": datetime.fromisoformat(row['queued_at']),
                    "status": row['status'],
                    "reviewer_note": row['reviewer_note'],
                    "processed_at": datetime.fromisoformat(row['processed_at']) if row['processed_at'] else None
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get memory details {queue_id}: {e}")
            return None
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about the review queue"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM memory_review_queue 
                    GROUP BY status
                """)
                
                stats = {}
                for row in cursor.fetchall():
                    stats[row[0]] = row[1]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get queue stats: {e}")
            return {}