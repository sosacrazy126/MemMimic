# memmimic/memory.py - With intelligent search engine
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List


class Memory:
    """A unit of memory with validation and type safety"""

    def __init__(
        self, content: str, memory_type: str = "interaction", confidence: float = 0.8
    ) -> None:
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")
        if len(content) > 10000:
            raise ValueError("Memory content cannot exceed 10000 characters")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

        self.content = content.strip()
        self.type = memory_type
        self.memory_type = memory_type  # Alias for compatibility
        self.confidence = float(confidence)
        self.created_at = datetime.now().isoformat()
        self.access_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary representation"""
        return {
            "content": self.content,
            "type": self.type,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "access_count": self.access_count,
        }


class MemoryStore:
    """SQLite-based memory store with intelligent search and proper connection management"""

    def __init__(self, db_path: str = "memories.db") -> None:
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()

        # Semantic expansions for intelligent search
        self.semantic_expansions = {
            # Philosophical concepts
            "uncertainty": ["certainty", "doubt", "honesty", "admit", "principle"],
            "philosophy": ["principle", "wisdom", "approach", "belief"],
            "transparency": ["process", "reasoning", "visible", "clear"],
            # Technical concepts
            "architecture": ["component", "structure", "design", "system"],
            "search": ["find", "recall", "memory", "relevant"],
            "reflection": ["analysis", "pattern", "insight", "meta"],
            # Project concepts
            "memmimic": ["memory", "persistent", "assistant", "context"],
            "collaborator": ["team", "leader", "project", "human", "claude"],
            "milestone": ["achievement", "completed", "phase"],
            # Development concepts
            "simplicity": ["complex", "simple", "elegant", "minimal"],
            "context": ["memory", "preserve", "continuity", "remember"],
            "learning": ["evolution", "improvement", "pattern", "knowledge"],
        }

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

    def _init_db(self) -> None:
        """Create table and indices if they don't exist"""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8,
                    created_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """
            )
            # Add indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content ON memories(content)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_confidence ON memories(confidence)"
            )
            conn.commit()

    def add(self, memory: Memory) -> int:
        """Save a memory with proper validation and error handling"""
        if not isinstance(memory, Memory):
            raise TypeError("Expected Memory instance")

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO memories (content, type, confidence, created_at, access_count) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        memory.content,
                        memory.type,
                        memory.confidence,
                        memory.created_at,
                        memory.access_count,
                    ),
                )
                conn.commit()
                memory_id = cursor.lastrowid
                self.logger.debug(f"Added memory with ID: {memory_id}")
                return memory_id
        except sqlite3.Error as e:
            self.logger.error(f"Failed to add memory: {e}")
            raise

    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Intelligent search engine that finds relevant memories
        including synthetic ones using semantic expansion and ranking

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of Memory objects sorted by relevance

        Raises:
            ValueError: If query is empty or limit is invalid
        """
        if not query or not query.strip():
            return []
        if limit <= 0:
            raise ValueError("Limit must be positive")
        if limit > 100:
            raise ValueError("Limit cannot exceed 100")
        query_lower = query.lower()
        search_results = []

        # STEP 1: Direct content search
        direct_results = self._search_content(query_lower, limit * 2)
        search_results.extend(direct_results)

        # STEP 2: Semantic expansion search
        expanded_results = self._search_semantic_expansion(query_lower, limit * 2)
        search_results.extend(expanded_results)

        # STEP 3: Type-based search for specific queries
        type_results = self._search_by_query_type(query_lower, limit)
        search_results.extend(type_results)

        # STEP 4: Deduplicate and rank
        unique_results = self._deduplicate_memories(search_results)
        ranked_results = self._rank_memories(unique_results, query_lower)

        return ranked_results[:limit]

    def _search_content(self, query: str, limit: int) -> List[Memory]:
        """Direct content search with improved error handling"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM memories 
                       WHERE LOWER(content) LIKE ? 
                       ORDER BY confidence DESC, created_at DESC 
                       LIMIT ?""",
                    (f"%{query}%", limit),
                )
                return [self._row_to_memory(row) for row in cursor]
        except sqlite3.Error as e:
            self.logger.error(f"Content search failed: {e}")
            return []

    def _search_semantic_expansion(self, query: str, limit: int) -> List[Memory]:
        """Search using semantic expansions with improved error handling"""
        results = []

        try:
            # Find expansions for words in the query
            query_words = [word for word in query.split() if len(word) > 3]

            with self._get_connection() as conn:
                for word in query_words:
                    if word in self.semantic_expansions:
                        for expansion in self.semantic_expansions[word]:
                            cursor = conn.execute(
                                """SELECT * FROM memories 
                                   WHERE LOWER(content) LIKE ? 
                                   ORDER BY confidence DESC 
                                   LIMIT ?""",
                                (f"%{expansion}%", 3),  # Few results per expansion
                            )
                            results.extend([self._row_to_memory(row) for row in cursor])
        except sqlite3.Error as e:
            self.logger.error(f"Semantic expansion search failed: {e}")

        return results

    def _search_by_query_type(self, query: str, limit: int) -> List[Memory]:
        """Specialized search by query type with improved error handling"""
        results = []

        try:
            with self._get_connection() as conn:
                # Philosophical/principle queries -> prioritize synthetic_wisdom
                if any(
                    word in query
                    for word in ["philosophy", "principle", "how", "why", "approach"]
                ):
                    cursor = conn.execute(
                        """SELECT * FROM memories 
                           WHERE type = 'synthetic_wisdom'
                           ORDER BY confidence DESC 
                           LIMIT ?""",
                        (limit,),
                    )
                    results.extend([self._row_to_memory(row) for row in cursor])

                # Technical queries -> prioritize synthetic_technical
                if any(
                    word in query
                    for word in [
                        "architecture",
                        "system",
                        "component",
                        "technical",
                        "implementation",
                    ]
                ):
                    cursor = conn.execute(
                        """SELECT * FROM memories 
                           WHERE type = 'synthetic_technical'
                           ORDER BY confidence DESC 
                           LIMIT ?""",
                        (limit,),
                    )
                    results.extend([self._row_to_memory(row) for row in cursor])

                # Project queries -> prioritize synthetic_history and project_info
                if any(
                    word in query
                    for word in ["project", "collaborator", "history", "origin", "team"]
                ):
                    cursor = conn.execute(
                        """SELECT * FROM memories 
                           WHERE type IN ('synthetic_history', 'project_info')
                           ORDER BY confidence DESC 
                           LIMIT ?""",
                        (limit,),
                    )
                    results.extend([self._row_to_memory(row) for row in cursor])

                # Status queries -> prioritize milestone and reflection
                if any(
                    word in query
                    for word in [
                        "status",
                        "progress",
                        "milestone",
                        "achievement",
                        "analysis",
                    ]
                ):
                    cursor = conn.execute(
                        """SELECT * FROM memories 
                           WHERE type IN ('milestone', 'reflection')
                           ORDER BY created_at DESC 
                           LIMIT ?""",
                        (limit,),
                    )
                    results.extend([self._row_to_memory(row) for row in cursor])
        except sqlite3.Error as e:
            self.logger.error(f"Query type search failed: {e}")

        return results

    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """Remove duplicate memories using content prefix as deduplication key"""
        seen_content = set()
        unique_memories = []

        for memory in memories:
            # Use first 50 characters as deduplication key
            key = memory.content[:50].lower()
            if key not in seen_content:
                seen_content.add(key)
                unique_memories.append(memory)

        return unique_memories

    def _rank_memories(self, memories: List[Memory], query: str) -> List[Memory]:
        """Rank memories by relevance with improved scoring algorithm"""
        scored_memories = []

        for memory in memories:
            score = 0
            content_lower = memory.content.lower()

            # Factor 1: Relevancia directa (coincidencias de palabras)
            query_words = query.split()
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    score += 10

            # Factor 2: Tipo de memoria (sintéticas tienen prioridad)
            if memory.type.startswith("synthetic"):
                score += 20
            elif memory.type in ["milestone", "reflection"]:
                score += 15
            elif memory.type == "project_info":
                score += 10

            # Factor 3: Confianza de la memoria
            score += memory.confidence * 10

            # Factor 4: Penalize very recent memories (avoid loops)
            try:
                mem_time = datetime.fromisoformat(memory.created_at)
                now = datetime.now()
                hours_diff = (now - mem_time).total_seconds() / 3600
                if hours_diff < 0.1:  # Less than 6 minutes
                    score -= 50
            except ValueError as e:
                self.logger.warning(f"Invalid datetime format in memory: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error in time calculation: {e}")

            scored_memories.append((memory, score))

        # Sort by score descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return [memory for memory, score in scored_memories]

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert SQLite row to Memory object with validation"""
        try:
            mem = Memory(row["content"], row["type"], row["confidence"])
            mem.created_at = row["created_at"]
            mem.access_count = row["access_count"]
            mem.id = row["id"]  # Add ID for pattern analysis
            return mem
        except (KeyError, ValueError) as e:
            self.logger.error(f"Failed to convert row to memory: {e}")
            raise

    def get_recent(self, hours: int = 24) -> List[Memory]:
        """Get recent memories for reflection with proper time filtering"""
        if hours <= 0:
            raise ValueError("Hours must be positive")

        try:
            with self._get_connection() as conn:
                # Calculate cutoff time
                from datetime import timedelta

                cutoff_time = datetime.now() - timedelta(hours=hours)
                cutoff_iso = cutoff_time.isoformat()

                cursor = conn.execute(
                    """SELECT * FROM memories 
                       WHERE created_at > ?
                       ORDER BY created_at DESC 
                       LIMIT 20""",
                    (cutoff_iso,),
                )

                return [self._row_to_memory(row) for row in cursor]
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get recent memories: {e}")
            return []

    def get_all(self) -> List[Memory]:
        """Get all memories for status and analysis with pagination support"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM memories 
                       ORDER BY created_at DESC"""
                )

                return [self._row_to_memory(row) for row in cursor]
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get all memories: {e}")
            return []

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        # No persistent connection to close
