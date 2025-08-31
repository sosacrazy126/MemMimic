# memory.py - With intelligent search engine
# TODO: [REFACTOR] Extract semantic expansions to configuration file
import json
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional, Dict

class Memory:
    """A unit of memory with optional embedding cache"""
    def __init__(self, content: str, memory_type: str = "interaction", confidence: float = 0.8, id: Optional[int] = None):
        self.id = id  # Add ID field for database operations
        self.content = content
        self.type = memory_type
        self.memory_type = memory_type  # Alias for compatibility
        self.confidence = confidence
        self.created_at = datetime.now().isoformat()
        self.access_count = 0
        self._embedding = None  # Cache for embedding vector
        self._embedding_model = None  # Track which model generated the embedding
        
    def get_embedding(self, model_name: str = None):
        """Get cached embedding or None if not cached or model differs."""
        if self._embedding is not None and self._embedding_model == model_name:
            return self._embedding
        return None
    
    def set_embedding(self, embedding, model_name: str):
        """Cache an embedding vector with its model name."""
        self._embedding = embedding
        self._embedding_model = model_name
    
    def clear_embedding_cache(self):
        """Clear the cached embedding."""
        self._embedding = None
        self._embedding_model = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "has_embedding": self._embedding is not None
        }

class MemoryStore:
    """SQLite with intelligent search and thread safety"""
    def __init__(self, db_path: str = "memories.db"):
        # Use check_same_thread=False for thread safety with proper locking
        self.db_path = db_path
        self.lock = threading.Lock()  # Thread safety lock
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA encoding = 'UTF-8'")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
        # Expansiones semánticas para búsqueda inteligente
        self.semantic_expansions = {
            # Conceptos filosóficos
            "incertidumbre": ["certeza", "duda", "honestidad", "admitir", "principio"],
            "filosofía": ["principio", "sabiduría", "enfoque", "creencia"],
            "transparencia": ["proceso", "razonamiento", "visible", "claro"],
            
            # Conceptos técnicos
            "arquitectura": ["componente", "estructura", "diseño", "sistema"],
            "búsqueda": ["encontrar", "recall", "memoria", "relevante"],
            "reflexión": ["análisis", "patrón", "insight", "meta"],
            
            # Conceptos de proyecto
            "clay": ["memoria", "persistente", "asistente", "contexto"],
            "colaborador": ["equipo", "líder", "proyecto", "humano", "claude"],
            "hito": ["milestone", "logro", "completado", "fase"],
            
            # Conceptos de desarrollo
            "simplicidad": ["complejo", "simple", "elegante", "mínimo"],
            "contexto": ["memoria", "preservar", "continuidad", "recordar"],
            "aprendizaje": ["evolución", "mejora", "patrón", "conocimiento"]
        }
        
    def _init_db(self):
        """Create table if it doesn't exist and add indexes for performance"""
        # Create main table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                created_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Add indexes for performance
        # Index on type for filtering by memory type
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type 
            ON memories(type)
        """)
        
        # Index on created_at for temporal queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at 
            ON memories(created_at DESC)
        """)
        
        # Index on confidence for ranking
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_confidence 
            ON memories(confidence DESC)
        """)
        
        # Composite index for common query pattern
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type_confidence 
            ON memories(type, confidence DESC)
        """)
        
        self.conn.commit()
    
    def _load_semantic_expansions(self) -> Dict:
        """Load semantic expansions from YAML configuration file."""
        # Try to find the config file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'config', 'semantic_expansions.yaml'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'semantic_expansions.yaml'),
            'src/memmimic/config/semantic_expansions.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        
                    # Flatten the nested structure into the format expected by search
                    expansions = {}
                    for category in config.values():
                        for term, related in category.items():
                            expansions[term] = related
                    
                    return expansions
                except Exception as e:
                    print(f"Warning: Could not load semantic expansions from {path}: {e}")
        
        # Fallback to default expansions if config not found
        return {
            "uncertainty": ["certainty", "doubt", "honesty", "admit", "principle"],
            "philosophy": ["principle", "wisdom", "approach", "belief"],
            "transparency": ["process", "reasoning", "visible", "clear"],
            "architecture": ["component", "structure", "design", "system"],
            "search": ["find", "recall", "memory", "relevant"],
            "reflection": ["analysis", "pattern", "insight", "meta"],
            "clay": ["memory", "persistent", "assistant", "context"],
            "collaborator": ["team", "leader", "project", "human", "claude"],
            "milestone": ["milestone", "achievement", "completed", "phase"],
            "simplicity": ["complex", "simple", "elegant", "minimal"],
            "context": ["memory", "preserve", "continuity", "remember"],
            "learning": ["evolution", "improvement", "pattern", "knowledge"]
        }
        
    def add(self, memory: Memory) -> int:
        """Save a memory with thread safety"""
        with self.lock:
            cursor = self.conn.execute(
                """INSERT INTO memories (content, type, confidence, created_at, access_count) 
                   VALUES (?, ?, ?, ?, ?)""",
                (memory.content, memory.type, memory.confidence, memory.created_at, memory.access_count)
            )
            self.conn.commit()
            return cursor.lastrowid
        
    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Intelligent search engine that finds relevant memories
        including synthetic ones using semantic expansion and ranking
        """
        query_lower = query.lower()
        search_results = []
        
        # PASO 1: Búsqueda directa en contenido
        direct_results = self._search_content(query_lower, limit * 2)
        search_results.extend(direct_results)
        
        # PASO 2: Búsqueda por expansión semántica
        expanded_results = self._search_semantic_expansion(query_lower, limit * 2)
        search_results.extend(expanded_results)
        
        # PASO 3: Búsqueda por tipo si es consulta específica
        type_results = self._search_by_query_type(query_lower, limit)
        search_results.extend(type_results)
        
        # PASO 4: Deduplicar y rankear
        unique_results = self._deduplicate_memories(search_results)
        ranked_results = self._rank_memories(unique_results, query_lower)
        
        return ranked_results[:limit]
    
    def _search_content(self, query: str, limit: int) -> List[Memory]:
        """Búsqueda directa en contenido"""
        cursor = self.conn.execute(
            """SELECT * FROM memories 
               WHERE LOWER(content) LIKE ? 
               ORDER BY confidence DESC, created_at DESC 
               LIMIT ?""",
            (f"%{query}%", limit)
        )
        
        return [self._row_to_memory(row) for row in cursor]
    
    def _search_semantic_expansion(self, query: str, limit: int) -> List[Memory]:
        """Búsqueda usando expansiones semánticas"""
        results = []
        
        # Buscar expansiones para palabras en el query
        query_words = [word for word in query.split() if len(word) > 3]
        
        for word in query_words:
            if word in self.semantic_expansions:
                for expansion in self.semantic_expansions[word]:
                    cursor = self.conn.execute(
                        """SELECT * FROM memories 
                           WHERE LOWER(content) LIKE ? 
                           ORDER BY confidence DESC 
                           LIMIT ?""",
                        (f"%{expansion}%", 3)  # Pocos resultados por expansión
                    )
                    results.extend([self._row_to_memory(row) for row in cursor])
        
        return results
    
    def _search_by_query_type(self, query: str, limit: int) -> List[Memory]:
        """Búsqueda especializada según tipo de consulta"""
        results = []
        
        # Consultas filosóficas/de principios -> priorizar synthetic_wisdom
        if any(word in query for word in ["filosofía", "principio", "cómo", "por qué", "enfoque"]):
            cursor = self.conn.execute(
                """SELECT * FROM memories 
                   WHERE type = 'synthetic_wisdom'
                   ORDER BY confidence DESC 
                   LIMIT ?""",
                (limit,)
            )
            results.extend([self._row_to_memory(row) for row in cursor])
        
        # Consultas técnicas -> priorizar synthetic_technical
        if any(word in query for word in ["arquitectura", "sistema", "componente", "técnico", "implementación"]):
            cursor = self.conn.execute(
                """SELECT * FROM memories 
                   WHERE type = 'synthetic_technical'
                   ORDER BY confidence DESC 
                   LIMIT ?""",
                (limit,)
            )
            results.extend([self._row_to_memory(row) for row in cursor])
        
        # Consultas de proyecto -> priorizar synthetic_history y project_info
        if any(word in query for word in ["proyecto", "colaborador", "historia", "origen", "equipo"]):
            cursor = self.conn.execute(
                """SELECT * FROM memories 
                   WHERE type IN ('synthetic_history', 'project_info')
                   ORDER BY confidence DESC 
                   LIMIT ?""",
                (limit,)
            )
            results.extend([self._row_to_memory(row) for row in cursor])
        
        # Consultas de estado -> priorizar milestone y reflection
        if any(word in query for word in ["estado", "progreso", "hito", "logro", "análisis"]):
            cursor = self.conn.execute(
                """SELECT * FROM memories 
                   WHERE type IN ('milestone', 'reflection')
                   ORDER BY created_at DESC 
                   LIMIT ?""",
                (limit,)
            )
            results.extend([self._row_to_memory(row) for row in cursor])
        
        return results
    
    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """Eliminar memorias duplicadas"""
        seen_content = set()
        unique_memories = []
        
        for memory in memories:
            # Usar primeras 50 caracteres como key de deduplicación
            key = memory.content[:50].lower()
            if key not in seen_content:
                seen_content.add(key)
                unique_memories.append(memory)
        
        return unique_memories
    
    def _rank_memories(self, memories: List[Memory], query: str) -> List[Memory]:
        """Rankear memorias por relevancia"""
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
            if memory.type.startswith('synthetic'):
                score += 20
            elif memory.type in ['milestone', 'reflection']:
                score += 15
            elif memory.type == 'project_info':
                score += 10
            
            # Factor 3: Confianza de la memoria
            score += memory.confidence * 10
            
            # Factor 4: Penalizar memorias muy recientes (evitar loops)
            try:
                mem_time = datetime.fromisoformat(memory.created_at)
                now = datetime.now()
                hours_diff = (now - mem_time).total_seconds() / 3600
                if hours_diff < 0.1:  # Menos de 6 minutos
                    score -= 50
            except:
                pass
            
            scored_memories.append((memory, score))
        
        # Ordenar por score descendente
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, score in scored_memories]
    
    def _row_to_memory(self, row) -> Memory:
        """Convertir row de SQLite a Memory"""
        mem = Memory(row["content"], row["type"], row["confidence"])
        mem.created_at = row["created_at"]
        mem.access_count = row["access_count"]
        mem.id = row["id"]  # Agregar ID para análisis de patrones
        return mem
    
    def get_recent(self, hours: int = 24) -> List[Memory]:
        """Obtener memorias recientes para reflexión"""
        cursor = self.conn.execute(
            """SELECT * FROM memories 
               ORDER BY created_at DESC 
               LIMIT 20"""
        )
        
        memories = []
        for row in cursor:
            mem = self._row_to_memory(row)
            memories.append(mem)
            
        return memories
    
    def get_all(self) -> List[Memory]:
        """Get all memories for status and analysis"""
        cursor = self.conn.execute(
            """SELECT * FROM memories 
               ORDER BY created_at DESC"""
        )
        
        memories = []
        for row in cursor:
            mem = self._row_to_memory(row)
            memories.append(mem)
            
        return memories
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID with thread safety."""
        with self.lock:
            try:
                cursor = self.conn.execute(
                    "DELETE FROM memories WHERE id = ?",
                    (memory_id,)
                )
                self.conn.commit()
                return cursor.rowcount > 0  # True if a row was deleted
            except Exception as e:
                print(f"Error deleting memory {memory_id}: {e}")
                return False
    
    def update(self, memory_id: int, content: str = None, confidence: float = None) -> bool:
        """Update a memory's content or confidence."""
        with self.lock:
            try:
                updates = []
                params = []
                
                if content is not None:
                    updates.append("content = ?")
                    params.append(content)
                
                if confidence is not None:
                    updates.append("confidence = ?")
                    params.append(confidence)
                
                if not updates:
                    return False  # Nothing to update
                
                params.append(memory_id)
                query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
                
                cursor = self.conn.execute(query, params)
                self.conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error updating memory {memory_id}: {e}")
                return False
