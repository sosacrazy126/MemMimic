# memmimic/memory.py - With intelligent search engine
import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict

class Memory:
    """A unit of memory"""
    def __init__(self, content: str, memory_type: str = "interaction", confidence: float = 0.8):
        self.content = content
        self.type = memory_type
        self.memory_type = memory_type  # Alias para compatibilidad
        self.confidence = confidence
        self.created_at = datetime.now().isoformat()
        self.access_count = 0
        
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "type": self.type,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "access_count": self.access_count
        }

class MemoryStore:
    """SQLite with intelligent search"""
    def __init__(self, db_path: str = "memories.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA encoding = 'UTF-8'")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
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
            "learning": ["evolution", "improvement", "pattern", "knowledge"]
        }
        
    def _init_db(self):
        """Create table if it doesn't exist"""
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
        self.conn.commit()
        
    def add(self, memory: Memory) -> int:
        """Save a memory"""
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
        
        # STEP 1: Direct content search
        direct_results = self._search_content(query_lower, limit * 2)
        search_results.extend(direct_results)
        
        # STEP 2: Semantic expansion search
        expanded_results = self._search_semantic_expansion(query_lower, limit * 2)
        search_results.extend(expanded_results)
        
        # STEP 3: Type-based search for specific queries
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
        """Obtener todas las memorias para status y análisis"""
        cursor = self.conn.execute(
            """SELECT * FROM memories 
               ORDER BY created_at DESC"""
        )
        
        memories = []
        for row in cursor:
            mem = self._row_to_memory(row)
            memories.append(mem)
            
        return memories
