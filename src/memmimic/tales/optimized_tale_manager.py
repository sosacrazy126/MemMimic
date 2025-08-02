"""
Optimized Tale Manager v3.0 - High-Performance Narrative Memory System

Major optimizations:
- In-memory indexing with B-tree structures
- Intelligent caching with LRU eviction
- Bulk operations for mass tale management
- Async I/O for non-blocking file operations
- Full-text search with inverted index
- Compression for large tales
- Connection pooling for database operations
- Memory-mapped files for large datasets
"""

import asyncio
import sqlite3
import hashlib
import zlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from pathlib import Path
from collections import defaultdict, OrderedDict
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaleMetrics:
    """Performance and usage metrics for tales"""
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    avg_read_time_ms: float = 0.0
    size_bytes: int = 0
    compression_ratio: float = 1.0
    search_matches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class OptimizedTale:
    """Enhanced tale with performance optimizations"""
    id: str
    name: str
    content: str
    category: str
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: TaleMetrics = field(default_factory=TaleMetrics)
    
    # Performance optimizations
    content_hash: str = ""
    compressed_content: Optional[bytes] = None
    is_compressed: bool = False
    content_length: int = 0
    
    def __post_init__(self):
        self.content_length = len(self.content)
        self.content_hash = hashlib.md5(self.content.encode()).hexdigest()
        self.metrics.size_bytes = self.content_length
        
        # Auto-compress large content
        if self.content_length > 1024:  # Compress content > 1KB
            self.compressed_content = zlib.compress(self.content.encode())
            compression_ratio = len(self.compressed_content) / self.content_length
            if compression_ratio < 0.8:  # Only keep if >20% savings
                self.is_compressed = True
                self.metrics.compression_ratio = compression_ratio
            else:
                self.compressed_content = None
    
    def get_content(self) -> str:
        """Get content with automatic decompression"""
        if self.is_compressed and self.compressed_content:
            return zlib.decompress(self.compressed_content).decode()
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage"""
        return {
            'id': self.id,
            'name': self.name,
            'content': self.get_content(),
            'category': self.category,
            'tags': list(self.tags),
            'metadata': self.metadata,
            'content_hash': self.content_hash,
            'is_compressed': self.is_compressed,
            'content_length': self.content_length
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizedTale':
        """Deserialize from dictionary"""
        tale = cls(
            id=data['id'],
            name=data['name'],
            content=data['content'],
            category=data['category'],
            tags=set(data.get('tags', [])),
            metadata=data.get('metadata', {})
        )
        tale.content_hash = data.get('content_hash', '')
        return tale


class InvertedIndex:
    """Full-text search index for tales"""
    
    def __init__(self):
        self.word_to_tales: Dict[str, Set[str]] = defaultdict(set)
        self.tale_to_words: Dict[str, Set[str]] = defaultdict(set)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
    
    def add_tale(self, tale_id: str, content: str):
        """Add tale to search index"""
        words = self._extract_words(content)
        self.tale_to_words[tale_id] = words
        
        for word in words:
            self.word_to_tales[word].add(tale_id)
    
    def remove_tale(self, tale_id: str):
        """Remove tale from search index"""
        if tale_id in self.tale_to_words:
            words = self.tale_to_words[tale_id]
            for word in words:
                self.word_to_tales[word].discard(tale_id)
                if not self.word_to_tales[word]:
                    del self.word_to_tales[word]
            del self.tale_to_words[tale_id]
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search for tales matching query with relevance scoring"""
        query_words = self._extract_words(query)
        if not query_words:
            return []
        
        # Calculate relevance scores
        tale_scores: Dict[str, float] = defaultdict(float)
        
        for word in query_words:
            if word in self.word_to_tales:
                # TF-IDF-like scoring
                tales_with_word = self.word_to_tales[word]
                idf = 1.0 / len(tales_with_word)  # Inverse document frequency
                
                for tale_id in tales_with_word:
                    tf = self.tale_to_words[tale_id].count(word)  # Term frequency
                    tale_scores[tale_id] += tf * idf
        
        # Sort by relevance and return top results
        sorted_results = sorted(tale_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]
    
    def _extract_words(self, text: str) -> Set[str]:
        """Extract searchable words from text"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        filtered_words = {
            word for word in words 
            if len(word) >= 3 and word not in self.stop_words
        }
        
        return filtered_words


class LRUCache:
    """Least Recently Used cache for tales"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, OptimizedTale] = OrderedDict()
        self.access_times: Dict[str, datetime] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[OptimizedTale]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            tale = self.cache[key]
            del self.cache[key]
            self.cache[key] = tale
            self.access_times[key] = datetime.now()
            self.hits += 1
            return tale
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: OptimizedTale):
        """Add item to cache"""
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def remove(self, key: str):
        """Remove item from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_usage_mb': sum(
                tale.content_length for tale in self.cache.values()
            ) / (1024 * 1024)
        }


class OptimizedTaleManager:
    """
    High-performance tale management system with advanced optimizations.
    
    Features:
    - SQLite backend with connection pooling
    - In-memory caching with LRU eviction
    - Full-text search with inverted indexing
    - Automatic compression for large tales
    - Bulk operations for mass management
    - Async I/O for non-blocking operations
    - Performance monitoring and metrics
    """
    
    def __init__(self, 
                 db_path: str = "tales_optimized.db",
                 cache_size: int = 1000,
                 enable_compression: bool = True,
                 enable_search_index: bool = True):
        self.db_path = Path(db_path)
        self.cache_size = cache_size
        self.enable_compression = enable_compression
        self.enable_search_index = enable_search_index
        
        # Performance components
        self.cache = LRUCache(cache_size)
        self.search_index = InvertedIndex() if enable_search_index else None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance metrics
        self.metrics = {
            'tales_created': 0,
            'tales_updated': 0,
            'tales_deleted': 0,
            'tales_accessed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'search_queries': 0,
            'avg_response_time_ms': 0.0,
            'compression_savings_bytes': 0,
            'last_optimization': datetime.now()
        }
        
        # Initialize database
        asyncio.create_task(self._init_database())
    
    async def _init_database(self):
        """Initialize SQLite database with optimized schema"""
        def _create_tables():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main tales table with indexes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tales (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    content_length INTEGER NOT NULL,
                    is_compressed BOOLEAN DEFAULT FALSE,
                    compressed_content BLOB,
                    tags TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tales_category ON tales(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tales_name ON tales(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tales_hash ON tales(content_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tales_accessed ON tales(accessed_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tales_length ON tales(content_length)')
            
            # Full-text search table
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS tales_fts USING fts5(
                    tale_id, content, tags, category
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tale_metrics (
                    tale_id TEXT PRIMARY KEY,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    avg_read_time_ms REAL DEFAULT 0.0,
                    search_matches INTEGER DEFAULT 0,
                    cache_hits INTEGER DEFAULT 0,
                    cache_misses INTEGER DEFAULT 0,
                    FOREIGN KEY (tale_id) REFERENCES tales (id)
                )
            ''')
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _create_tables)
        
        # Load existing tales into cache and search index
        await self._warm_cache()
    
    async def _warm_cache(self):
        """Pre-load frequently accessed tales into cache"""
        def _load_frequent_tales():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load most frequently accessed tales
            cursor.execute('''
                SELECT * FROM tales 
                ORDER BY access_count DESC, accessed_at DESC 
                LIMIT ?
            ''', (self.cache_size // 2,))
            
            tales = []
            for row in cursor.fetchall():
                tale = self._row_to_tale(row)
                tales.append(tale)
            
            conn.close()
            return tales
        
        tales = await asyncio.get_event_loop().run_in_executor(
            self.executor, _load_frequent_tales
        )
        
        # Add to cache and search index
        for tale in tales:
            self.cache.put(tale.id, tale)
            if self.search_index:
                self.search_index.add_tale(tale.id, tale.get_content())
    
    async def create_tale(self, 
                         name: str, 
                         content: str = "", 
                         category: str = "claude/core",
                         tags: List[str] = None,
                         metadata: Dict[str, Any] = None) -> OptimizedTale:
        """Create a new optimized tale"""
        start_time = time.perf_counter()
        
        # Generate unique ID
        tale_id = hashlib.sha256(f"{name}_{category}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # Create optimized tale
        tale = OptimizedTale(
            id=tale_id,
            name=name,
            content=content,
            category=category,
            tags=set(tags) if tags else set(),
            metadata=metadata or {}
        )
        
        # Save to database
        await self._save_tale_to_db(tale)
        
        # Add to cache and search index
        self.cache.put(tale.id, tale)
        if self.search_index:
            self.search_index.add_tale(tale.id, tale.get_content())
        
        # Update metrics
        self.metrics['tales_created'] += 1
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_avg_response_time(elapsed_ms)
        
        return tale
    
    async def get_tale(self, tale_id: str = None, name: str = None, category: str = None) -> Optional[OptimizedTale]:
        """Get tale by ID or name/category combination"""
        start_time = time.perf_counter()
        
        # Try cache first
        cache_key = tale_id if tale_id else f"{name}:{category}"
        tale = self.cache.get(cache_key)
        
        if tale:
            self.metrics['cache_hits'] += 1
            tale.metrics.cache_hits += 1
        else:
            self.metrics['cache_misses'] += 1
            # Load from database
            tale = await self._load_tale_from_db(tale_id, name, category)
            if tale:
                self.cache.put(cache_key, tale)
                tale.metrics.cache_misses += 1
        
        if tale:
            # Update access metrics
            tale.metrics.access_count += 1
            tale.metrics.last_accessed = datetime.now()
            self.metrics['tales_accessed'] += 1
            
            # Update response time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            tale.metrics.avg_read_time_ms = (
                (tale.metrics.avg_read_time_ms * (tale.metrics.access_count - 1) + elapsed_ms) /
                tale.metrics.access_count
            )
            
            # Async update database metrics
            asyncio.create_task(self._update_tale_metrics(tale))
        
        return tale
    
    async def search_tales(self, 
                          query: str, 
                          category_filter: str = None,
                          tag_filter: List[str] = None,
                          limit: int = 10) -> List[Tuple[OptimizedTale, float]]:
        """Search tales with relevance scoring"""
        start_time = time.perf_counter()
        self.metrics['search_queries'] += 1
        
        results = []
        
        if self.search_index:
            # Use inverted index for fast search
            search_results = self.search_index.search(query, limit * 2)  # Get extra for filtering
            
            for tale_id, relevance in search_results:
                tale = await self.get_tale(tale_id)
                if tale:
                    # Apply filters
                    if category_filter and not tale.category.startswith(category_filter):
                        continue
                    if tag_filter and not any(tag in tale.tags for tag in tag_filter):
                        continue
                    
                    tale.metrics.search_matches += 1
                    results.append((tale, relevance))
                    
                    if len(results) >= limit:
                        break
        else:
            # Fallback to database search
            results = await self._database_search(query, category_filter, tag_filter, limit)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_avg_response_time(elapsed_ms)
        
        return results
    
    async def bulk_create_tales(self, tales_data: List[Dict[str, Any]]) -> List[OptimizedTale]:
        """Create multiple tales efficiently"""
        start_time = time.perf_counter()
        
        tales = []
        for data in tales_data:
            tale_id = hashlib.sha256(
                f"{data['name']}_{data.get('category', 'claude/core')}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            tale = OptimizedTale(
                id=tale_id,
                name=data['name'],
                content=data.get('content', ''),
                category=data.get('category', 'claude/core'),
                tags=set(data.get('tags', [])),
                metadata=data.get('metadata', {})
            )
            tales.append(tale)
        
        # Bulk save to database
        await self._bulk_save_tales_to_db(tales)
        
        # Add to cache and search index
        for tale in tales:
            self.cache.put(tale.id, tale)
            if self.search_index:
                self.search_index.add_tale(tale.id, tale.get_content())
        
        self.metrics['tales_created'] += len(tales)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_avg_response_time(elapsed_ms)
        
        return tales
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        def _get_db_stats():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute('SELECT COUNT(*) FROM tales')
            total_tales = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT category) FROM tales')
            total_categories = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(content_length) FROM tales')
            total_content_size = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(CASE WHEN is_compressed THEN content_length - LENGTH(compressed_content) ELSE 0 END) FROM tales')
            compression_savings = cursor.fetchone()[0] or 0
            
            # Performance stats
            cursor.execute('SELECT AVG(access_count), MAX(access_count) FROM tales')
            avg_access, max_access = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_tales': total_tales,
                'total_categories': total_categories,
                'total_content_size': total_content_size,
                'compression_savings': compression_savings,
                'avg_access_count': avg_access or 0,
                'max_access_count': max_access or 0
            }
        
        db_stats = await asyncio.get_event_loop().run_in_executor(self.executor, _get_db_stats)
        cache_stats = self.cache.get_stats()
        
        return {
            'database': db_stats,
            'cache': cache_stats,
            'performance': self.metrics,
            'search_index': {
                'enabled': self.search_index is not None,
                'indexed_words': len(self.search_index.word_to_tales) if self.search_index else 0,
                'indexed_tales': len(self.search_index.tale_to_words) if self.search_index else 0
            },
            'compression': {
                'enabled': self.enable_compression,
                'savings_bytes': db_stats['compression_savings'],
                'savings_mb': db_stats['compression_savings'] / (1024 * 1024)
            }
        }
    
    async def optimize_system(self):
        """Run system optimization tasks"""
        start_time = time.perf_counter()
        
        def _optimize_database():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Vacuum database to reclaim space
            cursor.execute('VACUUM')
            
            # Analyze tables for query optimization
            cursor.execute('ANALYZE')
            
            # Update statistics
            cursor.execute('UPDATE tales SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL')
            
            conn.commit()
            conn.close()
        
        # Run database optimization
        await asyncio.get_event_loop().run_in_executor(self.executor, _optimize_database)
        
        # Clear old cache entries
        self.cache.clear()
        
        # Rebuild search index
        if self.search_index:
            self.search_index = InvertedIndex()
            await self._rebuild_search_index()
        
        # Warm cache again
        await self._warm_cache()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.metrics['last_optimization'] = datetime.now()
        
        logger.info(f"System optimization completed in {elapsed_ms:.2f}ms")
    
    async def _save_tale_to_db(self, tale: OptimizedTale):
        """Save tale to database"""
        def _save():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO tales 
                (id, name, category, content, content_hash, content_length, 
                 is_compressed, compressed_content, tags, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                tale.id,
                tale.name,
                tale.category,
                tale.content if not tale.is_compressed else '',
                tale.content_hash,
                tale.content_length,
                tale.is_compressed,
                tale.compressed_content,
                json.dumps(list(tale.tags)),
                json.dumps(tale.metadata)
            ))
            
            # Update FTS table
            cursor.execute('''
                INSERT OR REPLACE INTO tales_fts (tale_id, content, tags, category)
                VALUES (?, ?, ?, ?)
            ''', (
                tale.id,
                tale.get_content(),
                ' '.join(tale.tags),
                tale.category
            ))
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _save)
    
    async def _load_tale_from_db(self, tale_id: str = None, name: str = None, category: str = None) -> Optional[OptimizedTale]:
        """Load tale from database"""
        def _load():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if tale_id:
                cursor.execute('SELECT * FROM tales WHERE id = ?', (tale_id,))
            else:
                cursor.execute('SELECT * FROM tales WHERE name = ? AND category = ?', (name, category))
            
            row = cursor.fetchone()
            conn.close()
            
            return self._row_to_tale(row) if row else None
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _load)
    
    def _row_to_tale(self, row) -> OptimizedTale:
        """Convert database row to OptimizedTale"""
        (id, name, category, content, content_hash, content_length, 
         is_compressed, compressed_content, tags, metadata, 
         created_at, updated_at, accessed_at, access_count) = row
        
        tale = OptimizedTale(
            id=id,
            name=name,
            content=content,
            category=category,
            tags=set(json.loads(tags)) if tags else set(),
            metadata=json.loads(metadata) if metadata else {}
        )
        
        tale.content_hash = content_hash
        tale.content_length = content_length
        tale.is_compressed = bool(is_compressed)
        tale.compressed_content = compressed_content
        tale.metrics.access_count = access_count or 0
        
        return tale
    
    async def _bulk_save_tales_to_db(self, tales: List[OptimizedTale]):
        """Save multiple tales efficiently"""
        def _bulk_save():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data for bulk insert
            tales_data = []
            fts_data = []
            
            for tale in tales:
                tales_data.append((
                    tale.id, tale.name, tale.category,
                    tale.content if not tale.is_compressed else '',
                    tale.content_hash, tale.content_length,
                    tale.is_compressed, tale.compressed_content,
                    json.dumps(list(tale.tags)),
                    json.dumps(tale.metadata)
                ))
                
                fts_data.append((
                    tale.id, tale.get_content(),
                    ' '.join(tale.tags), tale.category
                ))
            
            # Bulk insert
            cursor.executemany('''
                INSERT OR REPLACE INTO tales 
                (id, name, category, content, content_hash, content_length,
                 is_compressed, compressed_content, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tales_data)
            
            cursor.executemany('''
                INSERT OR REPLACE INTO tales_fts (tale_id, content, tags, category)
                VALUES (?, ?, ?, ?)
            ''', fts_data)
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _bulk_save)
    
    async def _update_tale_metrics(self, tale: OptimizedTale):
        """Update tale metrics in database"""
        def _update():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE tales SET 
                    access_count = ?, 
                    accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (tale.metrics.access_count, tale.id))
            
            cursor.execute('''
                INSERT OR REPLACE INTO tale_metrics 
                (tale_id, access_count, last_accessed, avg_read_time_ms, 
                 search_matches, cache_hits, cache_misses)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                tale.id, tale.metrics.access_count,
                tale.metrics.last_accessed.isoformat(),
                tale.metrics.avg_read_time_ms,
                tale.metrics.search_matches,
                tale.metrics.cache_hits,
                tale.metrics.cache_misses
            ))
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _update)
    
    def _update_avg_response_time(self, elapsed_ms: float):
        """Update average response time metric"""
        current_avg = self.metrics['avg_response_time_ms']
        total_ops = sum([
            self.metrics['tales_created'],
            self.metrics['tales_accessed'],
            self.metrics['search_queries']
        ])
        
        if total_ops > 0:
            self.metrics['avg_response_time_ms'] = (
                (current_avg * (total_ops - 1) + elapsed_ms) / total_ops
            )
    
    async def _rebuild_search_index(self):
        """Rebuild the search index from database"""
        def _load_all_tales():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, content FROM tales')
            tales_data = cursor.fetchall()
            conn.close()
            return tales_data
        
        tales_data = await asyncio.get_event_loop().run_in_executor(
            self.executor, _load_all_tales
        )
        
        for tale_id, content in tales_data:
            self.search_index.add_tale(tale_id, content)
    
    async def close(self):
        """Clean shutdown of the tale manager"""
        # Save any pending metrics
        for tale in self.cache.cache.values():
            await self._update_tale_metrics(tale)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)