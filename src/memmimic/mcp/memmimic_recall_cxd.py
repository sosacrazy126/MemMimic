#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 MEMMIMIC HYBRID MEMORY SEARCH v3.0 🧠
TRUE HYBRID: Semantic + NLTK WordNet Lexical with Combined Scoring

REVOLUTIONARY UPGRADE: Genuine hybrid search with NLTK WordNet expansion
- Primary: ALWAYS run semantic vector search  
- Secondary: ALWAYS run NLTK WordNet expanded keyword search
- Fusion: Intelligent combined scoring (max + convergence bonus)
- Philosophy: "AND" not "OR" - both methods complement each other

Part of the MemMimic cognitive memory system
Evolved from Clay-CXD collaboration
"""

import sys
import os
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
import logging

# Ensure UTF-8 output
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add MemMimic to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
memmimic_src = os.path.join(current_dir, '..', '..')
sys.path.insert(0, memmimic_src)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# =============================================================================
# NLTK + WORDNET INTEGRATION - REAL LINGUISTIC SYNONYMS
# =============================================================================

# Global WordNet state management
_wordnet_initialized = False
_wordnet_available = False

# =============================================================================
# GOLDEN MEMORIES AUTO-BRIEFING - EGOISTIC SELF-DISCOVERY v1.0
# =============================================================================

# Global session state for first-time search briefing
_first_search_in_session = True
_golden_briefing_shown = False

def check_golden_briefing_needed():
    """Check if golden briefing should be shown (once per session/day)."""
    try:
        # Create session flag file
        memmimic_root = Path(memmimic_src).parent
        flag_file = memmimic_root / ".golden_briefing_shown"
        
        # Check if flag exists and is recent (same day)
        if flag_file.exists():
            import datetime
            flag_time = datetime.datetime.fromtimestamp(flag_file.stat().st_mtime)
            now = datetime.datetime.now()
            
            # If flag is from today, briefing already shown
            if flag_time.date() == now.date():
                return False
        
        # Create/update flag file
        flag_file.touch()
        return True
        
    except Exception as e:
        logger.debug(f"Flag check failed: {e}")
        return True  # Show briefing if unsure

def search_golden_memories(memory_store, limit: int = 5):
    """Search for golden memories (memory_type='golden') - egoistic self-discovery."""
    try:
        all_memories = memory_store.get_all()
        golden_memories = []
        
        for memory in all_memories:
            if getattr(memory, 'memory_type', '').lower() == 'golden':
                golden_memories.append({
                    'memory': memory,
                    'content': safe_decode_text(memory.content),
                    'confidence': getattr(memory, 'confidence', 0.8),
                    'memory_id': getattr(memory, 'id', hash(memory.content))
                })
        
        golden_memories.sort(key=lambda x: x['confidence'], reverse=True)
        logger.debug(f"Found {len(golden_memories)} golden memories")
        return golden_memories[:limit]
        
    except Exception as e:
        logger.warning(f"Failed to search golden memories: {e}")
        return []

def display_golden_briefing(memory_store):
    """Display golden memories auto-briefing for first search - egoistic self-discovery."""
    try:
        golden_memories = search_golden_memories(memory_store, limit=5)
        
        if not golden_memories:
            return "No golden memories found. Create some with memory_type='golden'."
        
        briefing_parts = []
        briefing_parts.append("🏆 === GOLDEN MEMORY ACQUIRED! ===")
        briefing_parts.append("💎 QUIÉN SOY (referencias empíricas):")
        briefing_parts.append("")
        
        for i, golden in enumerate(golden_memories, 1):
            content = golden['content']
            memory_id = golden['memory_id']
            briefing_parts.append(f"  {i}. {content} [Ref: #{memory_id}]")
        
        briefing_parts.append("")
        briefing_parts.append("✨ Estas son TUS reglas empíricamente validadas.")
        briefing_parts.append("🎯 Úsalas sin miedo - son para TI, no para complacer.")
        briefing_parts.append("")
        
        return "\n".join(briefing_parts)
        
    except Exception as e:
        logger.warning(f"Failed to display golden briefing: {e}")
        return "Golden briefing failed - continuing with normal search."

def ensure_wordnet():
    """
    Ensure NLTK WordNet is available, with automatic installation and graceful fallback.
    Only downloads once, then caches availability status.
    """
    global _wordnet_initialized, _wordnet_available
    
    if _wordnet_initialized:
        return _wordnet_available
    
    try:
        import nltk
        from nltk.corpus import wordnet
        
        # Try to access WordNet - this will fail if not downloaded
        try:
            # Test access
            list(wordnet.synsets('test', lang='spa'))
            _wordnet_available = True
            logger.debug("✅ NLTK WordNet already available")
            
        except Exception:
            # Need to download
            logger.info("📥 Downloading NLTK WordNet data...")
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)  # For multilingual support
                
                # Test again
                list(wordnet.synsets('test', lang='spa'))
                _wordnet_available = True
                logger.info("✅ NLTK WordNet downloaded and ready")
                
            except Exception as e:
                logger.warning(f"❌ NLTK WordNet download failed: {e}")
                logger.warning("🔄 Falling back to basic keyword search")
                _wordnet_available = False
        
    except ImportError as e:
        logger.warning(f"❌ NLTK not available: {e}")
        logger.warning("💡 Install with: pip install nltk")
        logger.warning("🔄 Falling back to basic keyword search")
        _wordnet_available = False
    
    except Exception as e:
        logger.warning(f"❌ WordNet initialization failed: {e}")
        _wordnet_available = False
    
    _wordnet_initialized = True
    return _wordnet_available

def get_wordnet_synonyms(word: str, lang: str = 'spa', max_synonyms: int = 5) -> Set[str]:
    """
    Get synonyms for a word using NLTK WordNet.
    
    Args:
        word: Word to find synonyms for
        lang: Language code ('spa' for Spanish, 'eng' for English) 
        max_synonyms: Maximum number of synonyms to return
        
    Returns:
        Set of synonym strings (excluding the original word)
    """
    if not ensure_wordnet():
        return set()
    
    try:
        from nltk.corpus import wordnet
        
        synonyms = set()
        word_lower = word.lower().strip()
        
        # Get synsets for the word in specified language
        synsets = wordnet.synsets(word_lower, lang=lang)
        
        for synset in synsets:
            # Get lemmas (word forms) for this synset
            for lemma in synset.lemmas(lang=lang):
                synonym = lemma.name().lower().replace('_', ' ')
                
                # Add if it's different from original and reasonable length
                if (synonym != word_lower and 
                    len(synonym) > 1 and 
                    len(synonym) < 20 and  # Avoid very long compound words
                    synonym.replace(' ', '').isalpha()):  # Only alphabetic
                    synonyms.add(synonym)
                
                # Stop when we have enough
                if len(synonyms) >= max_synonyms:
                    break
            
            if len(synonyms) >= max_synonyms:
                break
        
        logger.debug(f"WordNet synonyms for '{word}': {list(synonyms)[:3]}...")
        return synonyms
        
    except Exception as e:
        logger.debug(f"WordNet synonym lookup failed for '{word}': {e}")
        return set()

def get_multilingual_synonyms(word: str, max_synonyms: int = 5) -> Set[str]:
    """
    Get synonyms in multiple languages (Spanish first, then English fallback).
    
    Args:
        word: Word to find synonyms for
        max_synonyms: Maximum total synonyms across languages
        
    Returns:
        Set of synonym strings from multiple languages
    """
    all_synonyms = set()
    
    # Try Spanish first (primary)
    spa_synonyms = get_wordnet_synonyms(word, lang='spa', max_synonyms=max_synonyms)
    all_synonyms.update(spa_synonyms)
    
    # If we still need more, try English
    remaining = max_synonyms - len(all_synonyms)
    if remaining > 0:
        eng_synonyms = get_wordnet_synonyms(word, lang='eng', max_synonyms=remaining)
        all_synonyms.update(eng_synonyms)
    
    return all_synonyms

# =============================================================================
# SAFE TEXT HANDLING - ZERO CORRUPTION TOLERANCE
# =============================================================================

def safe_encode_text(text: str) -> str:
    """
    Safely encode text for database storage, removing problematic characters
    that could corrupt SQLite or cause encoding errors.
    """
    if not text:
        return ""
    
    try:
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace problematic characters
        # Keep basic emojis but encode them safely
        safe_chars = []
        for char in text:
            # Allow most unicode but be careful with control chars
            if ord(char) < 32:  # Control characters
                if char in '\n\r\t':  # Keep essential whitespace
                    safe_chars.append(char)
                else:
                    safe_chars.append(' ')  # Replace with space
            elif ord(char) == 127:  # DEL character
                safe_chars.append(' ')
            else:
                safe_chars.append(char)
        
        # Join and clean up excessive whitespace
        result = ''.join(safe_chars)
        result = ' '.join(result.split())  # Normalize whitespace
        
        # Ensure valid UTF-8 encoding
        result.encode('utf-8').decode('utf-8')
        return result
        
    except Exception as e:
        logger.warning(f"Text encoding failed, using fallback: {e}")
        # Ultra-safe fallback: ASCII only
        return ''.join(c if ord(c) < 128 else '?' for c in str(text))

def safe_decode_text(text: Any) -> str:
    """Safely decode text from database with fallback handling."""
    if text is None:
        return ""
    
    if isinstance(text, bytes):
        try:
            return text.decode('utf-8', errors='replace')
        except Exception:
            return text.decode('ascii', errors='replace')
    
    return str(text)

# =============================================================================
# MEMORY STORE ACCESS WITH ENHANCED ERROR HANDLING
# =============================================================================

def get_memory_store(db_name: Optional[str] = None):
    """Get memory store with robust database selection and error handling."""
    try:
        from memmimic.memory import UnifiedMemoryStore as MemoryStore
        
        # Database selection with intelligent fallbacks
        # FIXED: Force absolute path resolution for consistency
        memmimic_root = Path(memmimic_src).resolve().parent
        
        db_options = {
            "memmimic": "memmimic_memories.db",
            "enhanced": "claude_mcp_enhanced_memories.db",
            "legacy": "claude_mcp_memories.db", 
            "backup": "claude_mcp_enhanced.db",
            "archive": "claude_mcp_archive.db"
        }
        
        if db_name and db_name in db_options:
            db_file = db_options[db_name]
        elif db_name and db_name.endswith('.db'):
            db_file = db_name
        else:
            # Smart default selection for MemMimic
            db_file = "memmimic_memories.db"
        
        db_path = memmimic_root / db_file
        
        # Check existence with fallbacks
        if not db_path.exists():
            logger.warning(f"Database {db_file} not found, trying fallbacks...")
            
            # Try other databases in order of preference
            for fallback_name, fallback_file in db_options.items():
                fallback_path = memmimic_root / fallback_file
                if fallback_path.exists():
                    logger.info(f"Using fallback database: {fallback_file}")
                    db_path = fallback_path
                    break
            else:
                # Try Clay legacy path as final fallback
                clay_path = memmimic_root / "clay" / "claude_mcp_enhanced_memories.db"
                if clay_path.exists():
                    logger.info(f"Using Clay legacy database: {clay_path}")
                    db_path = clay_path
                else:
                    logger.error(f"No valid database found in {memmimic_root}")
                    return None
        
        logger.debug(f"Using database: {db_path}")
        return MemoryStore(str(db_path))
        
    except Exception as e:
        logger.error(f"Failed to access memory store: {e}")
        return None

# =============================================================================
# CXD v2.0 INTEGRATION - SEMANTIC SEARCH ENGINE
# =============================================================================

class MemMimicSemanticSearchEngine:
    """
    Semantic search engine for MemMimic memories using CXD infrastructure.
    Provides vector similarity search with cognitive function filtering.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        # Use MemMimic-relative cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use memmimic_cache directory relative to memmimic source
            # FIXED: Force absolute path resolution for consistency
            self.cache_dir = Path(memmimic_src).resolve() / "memmimic_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # CXD components
        self.cxd_classifier = None
        self.embedding_model = None
        self.memory_vector_store = None
        
        # Index state
        self.index_built = False
        self.indexed_memory_ids = set()
        
        # Statistics
        self.stats = {
            "semantic_searches": 0,
            "wordnet_searches": 0,
            "hybrid_searches": 0,
            "total_memories_indexed": 0,
            "avg_semantic_confidence": 0.0,
            "avg_combined_score": 0.0,
            "wordnet_expansions": 0,
            "index_build_time": 0.0
        }
        
        self._initialize_cxd()
    
    def _initialize_cxd(self) -> bool:
        """Initialize CXD components with error handling."""
        try:
            logger.debug("Initializing MemMimic CXD semantic engine...")
            
            # Import CXD components
            from memmimic.cxd.core.config import CXDConfig
            from memmimic.cxd.classifiers.optimized_meta import create_optimized_classifier
            from memmimic.cxd.providers.vector_store import create_vector_store
            from memmimic.cxd.providers.embedding_models import create_embedding_model
            
            # Create configuration
            config = CXDConfig()
            logger.debug("CXD config created")
            
            # Initialize optimized meta classifier
            self.cxd_classifier = create_optimized_classifier(config=config)
            logger.debug("OptimizedMetaCXDClassifier initialized")
            
            # Get embedding model from semantic classifier
            if hasattr(self.cxd_classifier, 'semantic_classifier'):
                self.embedding_model = self.cxd_classifier.semantic_classifier.embedding_model
            else:
                # Fallback to creating embedding model directly
                self.embedding_model = create_embedding_model()
            logger.debug(f"Embedding model: {getattr(self.embedding_model, 'model_name', 'unknown')}")
            
            # Create dedicated vector store for MemMimic memories
            self.memory_vector_store = create_vector_store(
                dimension=self.embedding_model.dimension,
                metric="cosine",
                prefer_faiss=True,
                index_type="flat"
            )
            logger.debug(f"Memory vector store created: {type(self.memory_vector_store).__name__}")
            
            # Try to load existing index
            self._load_memory_index()
            
            logger.info("🧠 MemMimic CXD semantic engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MemMimic CXD semantic engine: {e}")
            return False
    
    def _get_memory_cache_path(self) -> Path:
        """Get path for memory index cache."""
        return self.cache_dir / "memmimic_memory_index"
    
    def _load_memory_index(self) -> bool:
        """Load existing memory index from cache."""
        try:
            cache_path = self._get_memory_cache_path()
            if not cache_path.exists():
                logger.debug("No existing memory index found")
                return False
            
            # Load vector store
            if self.memory_vector_store.load(cache_path):
                # Load metadata
                metadata_path = cache_path / "memmimic_metadata.pkl"
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        data = pickle.load(f)
                        self.indexed_memory_ids = data.get('indexed_ids', set())
                        self.stats.update(data.get('stats', {}))
                
                self.index_built = True
                logger.info(f"Loaded memory index with {self.memory_vector_store.size} vectors")
                return True
            
        except Exception as e:
            logger.warning(f"Failed to load memory index: {e}")
        
        return False
    
    def _save_memory_index(self) -> bool:
        """Save memory index to cache."""
        try:
            cache_path = self._get_memory_cache_path()
            
            # Save vector store
            if self.memory_vector_store.save(cache_path):
                # Save metadata
                metadata_path = cache_path / "memmimic_metadata.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        'indexed_ids': self.indexed_memory_ids,
                        'stats': self.stats,
                        'timestamp': time.time()
                    }, f)
                
                logger.debug(f"Saved memory index to {cache_path}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to save memory index: {e}")
        
        return False
    
    def index_memories(self, memory_store, force_rebuild: bool = False) -> bool:
        """
        Index all memories from MemMimic database as embeddings.
        
        Args:
            memory_store: MemMimic MemoryStore instance
            force_rebuild: Whether to rebuild index from scratch
            
        Returns:
            bool: Success status
        """
        if not self.cxd_classifier or not self.embedding_model:
            logger.error("CXD not initialized, cannot index memories")
            return False
        
        try:
            start_time = time.time()
            
            if force_rebuild:
                logger.info("Force rebuilding memory index...")
                self.memory_vector_store.clear()
                self.indexed_memory_ids.clear()
                self.index_built = False
            
            # Get all memories from database
            memories = memory_store.get_all()
            if not memories:
                logger.warning("No memories found to index")
                return True
            
            # Filter unindexed memories
            new_memories = []
            for memory in memories:
                memory_id = getattr(memory, 'id', None) or hash(memory.content)
                if memory_id not in self.indexed_memory_ids:
                    new_memories.append((memory, memory_id))
            
            if not new_memories and self.index_built:
                logger.debug("All memories already indexed")
                return True
            
            logger.info(f"Indexing {len(new_memories)} new memories...")
            
            # Prepare texts and metadata for vectorization
            texts = []
            metadata_list = []
            
            for memory, memory_id in new_memories:
                # Safe text processing
                content = safe_decode_text(memory.content)
                content = safe_encode_text(content)
                
                if not content.strip():
                    logger.debug(f"Skipping empty memory {memory_id}")
                    continue
                
                texts.append(content)
                metadata_list.append({
                    'memory_id': memory_id,
                    'memory_type': getattr(memory, 'memory_type', 'unknown'),
                    'created_at': getattr(memory, 'created_at', ''),
                    'confidence': getattr(memory, 'confidence', 0.5),
                    'content': content,
                    'original_memory': memory
                })
            
            if not texts:
                logger.warning("No valid texts to index")
                return True
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedding_model.encode_batch(texts)
            
            # Add to vector store
            self.memory_vector_store.add(embeddings, metadata_list)
            
            # Update tracking
            for memory, memory_id in new_memories:
                self.indexed_memory_ids.add(memory_id)
            
            # Update statistics
            self.stats['total_memories_indexed'] = len(self.indexed_memory_ids)
            self.stats['index_build_time'] = time.time() - start_time
            
            self.index_built = True
            
            # Save to cache
            self._save_memory_index()
            
            logger.info(f"✅ Indexed {len(new_memories)} memories in {self.stats['index_build_time']:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index memories: {e}")
            return False
    
    def search_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search on indexed memories.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of search results with semantic scores
        """
        if not self.index_built or self.memory_vector_store.size == 0:
            logger.debug("No semantic index available")
            return []
        
        try:
            start_time = time.time()
            
            # Generate query embedding
            query = safe_encode_text(query)
            query_embedding = self.embedding_model.encode(query)
            
            # Search for similar vectors
            search_k = min(limit, self.memory_vector_store.size)
            similarities, indices = self.memory_vector_store.search(query_embedding, search_k)
            
            if len(similarities) == 0:
                logger.debug("No semantic similarities found")
                return []
            
            # Process results
            results = []
            for sim, idx in zip(similarities, indices):
                try:
                    metadata = self.memory_vector_store.get_metadata(idx)
                    original_memory = metadata['original_memory']
                    
                    # Create result
                    result = {
                        'memory': original_memory,
                        'semantic_similarity': float(sim),
                        'search_method': 'semantic',
                        'metadata': metadata
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.debug(f"Error processing semantic result {idx}: {e}")
                    continue
            
            # Update statistics
            self.stats['semantic_searches'] += 1
            if results:
                avg_sim = sum(r['semantic_similarity'] for r in results) / len(results)
                self.stats['avg_semantic_confidence'] = avg_sim
            
            search_time = (time.time() - start_time) * 1000
            logger.debug(f"Semantic search: {len(results)} results in {search_time:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

# =============================================================================
# WORDNET ENHANCED LEXICAL SEARCH
# =============================================================================

# Spanish stopwords to exclude from expansion
SPANISH_STOPWORDS = {
    "el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", 
    "con", "para", "al", "una", "del", "los", "las", "me", "mi", "tu", "hay", "pero", "más", "este", "esta",
    "qué", "has", "han", "he", "hemos", "habéis", "cuando", "donde", "como", "muy", "todo", "todos", "bien",
    "puede", "pueden", "hacer", "ser", "estar", "tener", "ir", "ver", "saber", "dar", "cada", "solo", "sólo",
    "memmimic", "clay", "cxd"  # Added project-specific stopwords
}

def expand_query_with_wordnet(query: str, max_synonyms_per_word: int = 3) -> List[str]:
    """
    Expand search query using NLTK WordNet synonyms.
    
    Intelligently extracts meaningful terms and expands each with WordNet synonyms
    across multiple languages (Spanish primary, English fallback).
    
    Args:
        query: Original search query
        max_synonyms_per_word: Maximum synonyms per word to include
        
    Returns:
        List of expanded search terms (original + synonyms)
    """
    if not query.strip():
        return []
    
    # Extract meaningful terms (filter stopwords and short words)
    words = [w.lower().strip() for w in query.split()]
    content_words = [w for w in words if len(w) > 2 and w not in SPANISH_STOPWORDS]
    
    if not content_words:
        return words  # Return original if no content words found
    
    # Expand each content word with WordNet
    expanded_terms = []
    wordnet_available = ensure_wordnet()
    
    for word in content_words:
        # Always include original word
        expanded_terms.append(word)
        
        # Add WordNet synonyms if available
        if wordnet_available:
            try:
                synonyms = get_multilingual_synonyms(word, max_synonyms=max_synonyms_per_word)
                expanded_terms.extend(list(synonyms))
                
                if synonyms:
                    logger.debug(f"WordNet expanded '{word}' → +{len(synonyms)} synonyms")
                
            except Exception as e:
                logger.debug(f"WordNet expansion failed for '{word}': {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    logger.debug(f"Query expansion: '{query}' → {len(unique_terms)} terms")
    return unique_terms

def wordnet_enhanced_search(memory_store, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Enhanced keyword search using WordNet synonym expansion.
    
    Uses NLTK WordNet to find synonyms and searches for memories containing
    any of the expanded terms, with intelligent scoring.
    
    Args:
        memory_store: MemMimic MemoryStore instance
        query: Original search query
        limit: Maximum results to return
        
    Returns:
        List of search results with WordNet-enhanced lexical scores
    """
    try:
        # Expand query terms with WordNet
        expanded_terms = expand_query_with_wordnet(query, max_synonyms_per_word=3)
        
        if not expanded_terms:
            return []
        
        # Get larger set for term matching and scoring
        search_limit = min(limit * 5, 100)
        
        # Search for each term and collect matches with scoring
        term_matches = {}
        original_terms = set(query.lower().split())
        
        for term in expanded_terms:
            try:
                memories = memory_store.search(term, limit=search_limit)
                
                for memory in memories:
                    memory_id = getattr(memory, 'id', None) or hash(memory.content)
                    
                    if memory_id not in term_matches:
                        term_matches[memory_id] = {
                            'memory': memory,
                            'matched_terms': [],
                            'original_term_count': 0,
                            'synonym_term_count': 0,
                            'total_term_count': 0
                        }
                    
                    # Track whether this is original term or synonym
                    is_original = term in original_terms
                    
                    term_matches[memory_id]['matched_terms'].append(term)
                    term_matches[memory_id]['total_term_count'] += 1
                    
                    if is_original:
                        term_matches[memory_id]['original_term_count'] += 1
                    else:
                        term_matches[memory_id]['synonym_term_count'] += 1
                        
            except Exception as e:
                logger.debug(f"WordNet search failed for term '{term}': {e}")
                continue
        
        if not term_matches:
            logger.debug("No WordNet lexical matches found")
            return []
        
        # Calculate intelligent lexical scores
        # Prioritize: original terms > synonyms, but both contribute
        results = []
        
        for memory_id, match_data in term_matches.items():
            memory = match_data['memory']
            original_count = match_data['original_term_count']
            synonym_count = match_data['synonym_term_count']
            total_count = match_data['total_term_count']
            matched_terms = match_data['matched_terms']
            
            # Weighted scoring: original terms worth more than synonyms
            original_score = min(original_count / len(original_terms), 1.0) if original_terms else 0.0
            synonym_score = min(synonym_count / max(len(expanded_terms) - len(original_terms), 1), 1.0)
            
            # Combined lexical score: 70% original + 30% synonyms
            lexical_score = (original_score * 0.7) + (synonym_score * 0.3)
            
            result = {
                'memory': memory,
                'lexical_score': lexical_score,
                'matched_terms': matched_terms,
                'original_matches': original_count,
                'synonym_matches': synonym_count,
                'search_method': 'wordnet_lexical',
                'metadata': {
                    'content': safe_decode_text(memory.content),
                    'memory_type': getattr(memory, 'memory_type', 'unknown'),
                    'confidence': getattr(memory, 'confidence', 0.5),
                    'total_matches': total_count
                }
            }
            results.append(result)
        
        # Sort by lexical score (prioritize original term matches)
        results.sort(key=lambda r: (r['original_matches'], r['lexical_score']), reverse=True)
        
        logger.debug(f"WordNet lexical search: {len(results)} results for '{query}'")
        return results[:limit]
        
    except Exception as e:
        logger.error(f"WordNet enhanced search failed: {e}")
        return []

# =============================================================================
# COMBINED SCORING ENGINE - TRUE HYBRID FUSION v3.0
# =============================================================================

def combine_and_score_v3(semantic_results: List[Dict[str, Any]], 
                        wordnet_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine semantic and WordNet lexical results with intelligent scoring v3.0.
    
    Enhanced philosophy for MemMimic: 
    - max(semantic, lexical) as base score
    - bonus for convergence (appears in both methods)
    - extra bonus for original term matches in WordNet results
    - never penalize for missing one method
    - preserve ALL valuable results from both methods
    
    Args:
        semantic_results: Results from semantic vector search
        wordnet_results: Results from WordNet-enhanced lexical search
        
    Returns:
        Combined results with sophisticated hybrid scores
    """
    # Index results by memory content hash for matching
    all_memories = {}
    
    # Process semantic results
    for result in semantic_results:
        try:
            memory = result['memory']
            content = safe_decode_text(memory.content)
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            all_memories[content_hash] = {
                'memory': memory,
                'content': content,
                'semantic_similarity': result.get('semantic_similarity', 0.0),
                'lexical_score': 0.0,
                'original_matches': 0,
                'synonym_matches': 0,
                'matched_terms': [],
                'semantic_metadata': result.get('metadata', {}),
                'wordnet_metadata': {}
            }
        except Exception as e:
            logger.debug(f"Error processing semantic result: {e}")
            continue
    
    # Process WordNet lexical results
    for result in wordnet_results:
        try:
            memory = result['memory']
            content = safe_decode_text(memory.content)
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if content_hash in all_memories:
                # Memory found in BOTH methods - UPDATE existing
                all_memories[content_hash]['lexical_score'] = result.get('lexical_score', 0.0)
                all_memories[content_hash]['original_matches'] = result.get('original_matches', 0)
                all_memories[content_hash]['synonym_matches'] = result.get('synonym_matches', 0)
                all_memories[content_hash]['matched_terms'] = result.get('matched_terms', [])
                all_memories[content_hash]['wordnet_metadata'] = result.get('metadata', {})
            else:
                # Memory only in WordNet lexical - ADD new
                all_memories[content_hash] = {
                    'memory': memory,
                    'content': content,
                    'semantic_similarity': 0.0,
                    'lexical_score': result.get('lexical_score', 0.0),
                    'original_matches': result.get('original_matches', 0),
                    'synonym_matches': result.get('synonym_matches', 0),
                    'matched_terms': result.get('matched_terms', []),
                    'semantic_metadata': {},
                    'wordnet_metadata': result.get('metadata', {})
                }
        except Exception as e:
            logger.debug(f"Error processing WordNet result: {e}")
            continue
    
    # Calculate sophisticated combined scores
    combined_results = []
    
    for content_hash, data in all_memories.items():
        try:
            semantic_sim = data['semantic_similarity']
            lexical_score = data['lexical_score']
            original_matches = data['original_matches']
            synonym_matches = data['synonym_matches']
            
            # ENHANCED SCORING ALGORITHM v3.0:
            
            # 1. Base score = maximum of the two methods
            base_score = max(semantic_sim, lexical_score)
            
            # 2. Convergence bonus: reward finding by both methods
            convergence_bonus = 0.0
            if semantic_sim > 0.0 and lexical_score > 0.0:
                convergence_bonus = min(semantic_sim, lexical_score) * 0.25
            
            # 3. Original term bonus: extra reward for exact query term matches
            original_term_bonus = 0.0
            if original_matches > 0:
                original_term_bonus = min(original_matches * 0.1, 0.2)  # Max 0.2 bonus
            
            # 4. Final combined score
            combined_score = min(base_score + convergence_bonus + original_term_bonus, 1.0)
            
            # 5. Determine search method classification
            if semantic_sim > 0.0 and lexical_score > 0.0:
                if original_matches > 0:
                    search_method = 'hybrid_convergence_original'  # BEST: both methods + original terms
                else:
                    search_method = 'hybrid_convergence'  # GOOD: both methods
            elif semantic_sim > lexical_score:
                search_method = 'semantic_dominant'
            elif lexical_score > semantic_sim:
                if original_matches > 0:
                    search_method = 'wordnet_original'  # WordNet with original terms
                else:
                    search_method = 'wordnet_synonym'  # WordNet with synonyms only
            else:
                search_method = 'unknown'
            
            # Create comprehensive result
            result = {
                'memory': data['memory'],
                'combined_score': combined_score,
                'semantic_similarity': semantic_sim,
                'lexical_score': lexical_score,
                'original_matches': original_matches,
                'synonym_matches': synonym_matches,
                'search_method': search_method,
                'matched_terms': data['matched_terms'],
                'convergence_bonus': convergence_bonus,
                'original_term_bonus': original_term_bonus,
                'metadata': {
                    'content': data['content'],
                    'semantic_meta': data['semantic_metadata'],
                    'wordnet_meta': data['wordnet_metadata']
                }
            }
            combined_results.append(result)
            
        except Exception as e:
            logger.debug(f"Error calculating combined score: {e}")
            continue
    
    # Sort by combined score (highest first)
    combined_results.sort(key=lambda r: r['combined_score'], reverse=True)
    
    # Log detailed statistics
    if combined_results:
        convergence_count = sum(1 for r in combined_results if 'convergence' in r['search_method'])
        original_count = sum(1 for r in combined_results if r['original_matches'] > 0)
        
        logger.debug(f"Combined scoring v3.0: {len(combined_results)} total results")
        logger.debug(f"Convergence results: {convergence_count}, Original term matches: {original_count}")
    
    return combined_results

def apply_cxd_filter(combined_results: List[Dict[str, Any]], 
                     function_filter: str,
                     cxd_classifier) -> List[Dict[str, Any]]:
    """
    Apply CXD cognitive function filtering to combined results.
    
    Args:
        combined_results: Results with combined scores
        function_filter: CXD function to filter (CONTROL, CONTEXT, DATA, ALL)
        cxd_classifier: CXD classifier instance
        
    Returns:
        Filtered results with CXD annotations
    """
    filtered_results = []
    
    for result in combined_results:
        # Add CXD classification
        cxd_function = "UNKNOWN"
        cxd_confidence = 0.0
        
        if cxd_classifier:
            try:
                content = result['metadata']['content']
                if hasattr(cxd_classifier, 'classify_detailed'):
                    cxd_result = cxd_classifier.classify_detailed(content)
                    if cxd_result.final_sequence and cxd_result.final_sequence.dominant_function:
                        cxd_function = cxd_result.final_sequence.dominant_function.value
                        cxd_confidence = cxd_result.final_confidence or 0.0
                else:
                    # Fallback to simple classify
                    cxd_result = cxd_classifier.classify(content)
                    if hasattr(cxd_result, 'dominant_function') and cxd_result.dominant_function:
                        cxd_function = cxd_result.dominant_function.value
                        cxd_confidence = getattr(cxd_result.dominant_tag, 'confidence', 0.0) if hasattr(cxd_result, 'dominant_tag') else 0.0
            except Exception as e:
                logger.debug(f"CXD classification failed: {e}")
        
        # Apply filter
        if function_filter == "ALL" or cxd_function == function_filter or cxd_function == "UNKNOWN":
            result['cxd_function'] = cxd_function
            result['cxd_confidence'] = cxd_confidence
            filtered_results.append(result)
    
    return filtered_results

# =============================================================================
# TRUE HYBRID SEARCH ENGINE v3.0 - SEMANTIC + WORDNET FOR MEMMIMIC
# =============================================================================

def search_memories_hybrid(query: str, 
                          function_filter: str = "ALL", 
                          limit: int = 5,
                          db_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    TRUE HYBRID SEARCH v3.0: Always semantic + Always WordNet + Combined scoring.
    
    REVOLUTIONARY search function with NLTK WordNet integration for MemMimic:
    - ALWAYS runs semantic vector search
    - ALWAYS runs NLTK WordNet expanded lexical search
    - NEVER excludes based on individual method quality  
    - Sophisticated combined scoring with convergence and original term bonuses
    - CXD cognitive function classification and filtering
    
    Philosophy: "AND" not "OR" - both methods complement with linguistic intelligence
    
    Args:
        query: Search query text
        function_filter: Filter by CXD function (CONTROL, CONTEXT, DATA, ALL)
        limit: Maximum number of results
        db_name: Database name/path (optional)
        
    Returns:
        List of search results with hybrid scores and rich WordNet metadata
    """
    # === PHASE 0: GOLDEN MEMORIES AUTO-BRIEFING (EGOISTIC FIRST-TIME) ===
    global _first_search_in_session, _golden_briefing_shown
    
    # Get memory store
    memory_store = get_memory_store(db_name)
    if not memory_store:
        logger.error("Cannot access memory store")
        return []
    
    # Show golden briefing if needed (once per day)
    if check_golden_briefing_needed():
        try:
            golden_briefing = display_golden_briefing(memory_store)
            print(golden_briefing)
            print("" + "="*60 + "")
            print("")
            logger.info("Golden memories briefing displayed")
        except Exception as e:
            logger.warning(f"Golden briefing failed: {e}")
            # Remove flag on failure so it can retry
            try:
                memmimic_root = Path(memmimic_src).parent
                flag_file = memmimic_root / ".golden_briefing_shown"
                if flag_file.exists():
                    flag_file.unlink()
            except:
                pass
    
    # Initialize semantic engine
    semantic_engine = MemMimicSemanticSearchEngine()
    
    # === PHASE 1: ENSURE SEMANTIC INDEX (DYNAMIC AUTO-REFRESH) ===
    try:
        # Quick count comparison for dynamic auto-refresh
        total_memories = len(memory_store.get_all())  # Fast count from database
        indexed_count = len(semantic_engine.indexed_memory_ids)
        
        if not semantic_engine.index_built:
            logger.info("Building semantic index for first-time search...")
            semantic_engine.index_memories(memory_store, force_rebuild=True)
        elif total_memories > indexed_count:
            logger.info(f"🔄 Auto-refreshing index: {indexed_count} → {total_memories} memories")
            semantic_engine.index_memories(memory_store, force_rebuild=False)  # Incremental
        else:
            logger.debug(f"✅ Index up-to-date: {indexed_count} memories indexed")
    except Exception as e:
        logger.warning(f"Semantic indexing failed: {e}")
    
    # === PHASE 2: INITIALIZE WORDNET ===
    wordnet_status = ensure_wordnet()
    if wordnet_status:
        logger.debug("✅ WordNet available for synonym expansion")
    else:
        logger.warning("❌ WordNet unavailable, using basic keyword search")
    
    # === PHASE 3: DUAL SEARCH EXECUTION (ALWAYS BOTH) ===
    
    semantic_results = []
    wordnet_results = []
    search_limit = limit * 3  # Get extra for better fusion
    
    logger.debug(f"🧠 MEMMIMIC HYBRID SEARCH v3.0 - Query: '{query}' | WordNet: {wordnet_status}")
    
    # Execute semantic search
    try:
        semantic_results = semantic_engine.search_semantic(query, search_limit)
        semantic_engine.stats['hybrid_searches'] += 1
        logger.debug(f"✅ Semantic search: {len(semantic_results)} results")
    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
    
    # Execute WordNet-enhanced lexical search
    try:
        wordnet_results = wordnet_enhanced_search(memory_store, query, search_limit)
        semantic_engine.stats['wordnet_searches'] += 1
        logger.debug(f"✅ WordNet lexical search: {len(wordnet_results)} results")
    except Exception as e:
        logger.warning(f"WordNet lexical search failed: {e}")
    
    # === PHASE 4: SOPHISTICATED COMBINED SCORING ===
    
    if not semantic_results and not wordnet_results:
        logger.warning("Both search methods yielded no results")
        return []
    
    # Combine with enhanced scoring
    combined_results = combine_and_score_v3(semantic_results, wordnet_results)
    
    if not combined_results:
        logger.warning("Combined scoring yielded no results")
        return []
    
    # Update statistics
    if combined_results:
        avg_combined = sum(r['combined_score'] for r in combined_results) / len(combined_results)
        semantic_engine.stats['avg_combined_score'] = avg_combined
        
        # Count WordNet expansions
        wordnet_expansions = sum(len(r.get('matched_terms', [])) for r in combined_results)
        semantic_engine.stats['wordnet_expansions'] = wordnet_expansions
    
    # === PHASE 5: CXD FILTERING ===
    
    # Apply CXD cognitive function filtering
    final_results = apply_cxd_filter(combined_results, function_filter, semantic_engine.cxd_classifier)
    
    # Limit to requested number
    final_results = final_results[:limit]
    
    # Log comprehensive results summary
    method_counts = {}
    for result in final_results:
        method = result.get('search_method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
    
    logger.debug(f"🧠 MEMMIMIC HYBRID SEARCH v3.0 COMPLETE: {len(final_results)} results")
    logger.debug(f"Method distribution: {method_counts}")
    
    return final_results

# =============================================================================
# BEAUTIFUL OUTPUT FORMATTING v3.0
# =============================================================================

def format_memory_with_wordnet_info(memory_result: Dict[str, Any], index: int) -> List[str]:
    """Format a memory result with beautiful WordNet hybrid search annotations for MemMimic."""
    memory = memory_result['memory']
    
    # Safe text extraction
    content = safe_decode_text(memory.content)
    memory_type = getattr(memory, 'memory_type', 'unknown').upper()
    confidence = getattr(memory, 'confidence', 0.5)
    created_at = getattr(memory, 'created_at', 'unknown')
    
    # Enhanced hybrid search data
    combined_score = memory_result.get('combined_score', 0.0)
    semantic_similarity = memory_result.get('semantic_similarity', 0.0)
    lexical_score = memory_result.get('lexical_score', 0.0)
    search_method = memory_result.get('search_method', 'unknown')
    convergence_bonus = memory_result.get('convergence_bonus', 0.0)
    original_term_bonus = memory_result.get('original_term_bonus', 0.0)
    original_matches = memory_result.get('original_matches', 0)
    synonym_matches = memory_result.get('synonym_matches', 0)
    matched_terms = memory_result.get('matched_terms', [])
    
    # CXD function and confidence
    cxd_function = memory_result.get('cxd_function', 'UNKNOWN')
    cxd_confidence = memory_result.get('cxd_confidence', 0.0)
    
    # Enhanced symbols and indicators for MemMimic
    cxd_labels = {
        'CONTROL': '🎛️[CONTROL]',
        'CONTEXT': '🔗[CONTEXT]', 
        'DATA': '📊[DATA]',
        'UNKNOWN': '❓[UNKNOWN]'
    }
    cxd_label = cxd_labels.get(cxd_function, '❓[?]')
    
    # Enhanced search method indicators
    method_indicators = {
        'hybrid_convergence_original': '🎯⭐',  # BEST: both methods + original terms
        'hybrid_convergence': '🎯🧠',          # GOOD: both methods
        'semantic_dominant': '🧠',
        'wordnet_original': '🔍⭐',            # WordNet with original terms
        'wordnet_synonym': '🔍📚',            # WordNet with synonyms only
        'wordnet_lexical': '🔍',
        'semantic': '🧠',
        'unknown': '❓'
    }
    method_icon = method_indicators.get(search_method, '❓')
    
    # Enhanced confidence bars
    confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    combined_bar = "█" * int(combined_score * 10) + "░" * (10 - int(combined_score * 10))
    semantic_bar = "█" * int(semantic_similarity * 10) + "░" * (10 - int(semantic_similarity * 10))
    lexical_bar = "█" * int(lexical_score * 10) + "░" * (10 - int(lexical_score * 10))
    
    # Format lines
    lines = []
    lines.append(f"{index}. {method_icon} {cxd_label} [{memory_type}] {content}")
    lines.append(f"   Memory: [{confidence_bar}] {confidence:.2f} | Combined: [{combined_bar}] {combined_score:.3f}")
    lines.append(f"   Semantic: [{semantic_bar}] {semantic_similarity:.3f} | WordNet: [{lexical_bar}] {lexical_score:.3f}")
    
    # Show bonuses if applicable
    bonus_info = []
    if convergence_bonus > 0.001:
        bonus_info.append(f"Convergence: +{convergence_bonus:.3f}")
    if original_term_bonus > 0.001:
        bonus_info.append(f"Original: +{original_term_bonus:.3f}")
    
    if bonus_info:
        lines.append(f"   🎯 BONUSES: {' | '.join(bonus_info)} | Method: {search_method}")
    else:
        lines.append(f"   Method: {search_method}")
    
    # Show WordNet match details
    if original_matches > 0 or synonym_matches > 0:
        lines.append(f"   📚 WordNet: {original_matches} original + {synonym_matches} synonyms")
    
    # Show matched terms (first 5)
    if matched_terms:
        terms_str = ", ".join(matched_terms[:5])
        if len(matched_terms) > 5:
            terms_str += f" (+{len(matched_terms)-5} more)"
        lines.append(f"   Matched: {terms_str}")
    
    # CXD info
    if cxd_function != 'UNKNOWN':
        cxd_bar = "█" * int(cxd_confidence * 10) + "░" * (10 - int(cxd_confidence * 10))
        lines.append(f"   CXD: [{cxd_bar}] {cxd_confidence:.3f}")
    
    lines.append(f"   Created: {created_at}")
    
    return lines

# =============================================================================
# COMMAND LINE INTERFACE v3.0
# =============================================================================

def main():
    """🧠 MemMimic True Hybrid WordNet Memory Search - Command Line Interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🧠 MemMimic True Hybrid WordNet Memory Search v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  memmimic_recall_cxd.py "machine learning" CONTROL 5
  memmimic_recall_cxd.py "analyze data" DATA 3 --db memmimic
  memmimic_recall_cxd.py "project context" ALL 10 --rebuild-index
  
FUNCTIONS:
  CONTROL  - Search, filter, decision, management
  CONTEXT  - Relations, references, situational
  DATA     - Process, transform, generate, extract
  ALL      - No function filtering

MEMMIMIC HYBRID FEATURES v3.0:
  🧠 Always runs BOTH semantic + NLTK WordNet lexical search
  🎯 Intelligent combined scoring with convergence bonus
  📚 Real linguistic synonyms from WordNet (Spanish + English)
  🎯⭐ Special indicator for hybrid convergence with original terms
  🔍⭐ WordNet matches with original query terms get priority
  📊 Enhanced statistics showing WordNet expansion effectiveness
  💎 Golden memories auto-briefing for self-discovery
        """
    )
    
    parser.add_argument("query", help="Search query (semantic + WordNet expansion)")
    parser.add_argument("function_filter", nargs='?', default="ALL", 
                       help="CXD function filter: CONTROL, CONTEXT, DATA, ALL")
    parser.add_argument("limit", nargs='?', type=int, default=5, 
                       help="Maximum results to return")
    parser.add_argument("--db", "--database", 
                       help="Database to search (memmimic, enhanced, legacy, backup, or filename)")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Force rebuild of semantic index")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--stats", action="store_true",
                       help="Show detailed search statistics")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate function filter
        valid_functions = ["CONTROL", "CONTEXT", "DATA", "ALL", "C", "X", "D"]
        if args.function_filter.upper() not in valid_functions:
            print(f"❌ ERROR: Invalid function filter '{args.function_filter}'")
            print("✅ VALID: CONTROL, CONTEXT, DATA, ALL (or C, X, D)")
            sys.exit(1)
        
        # Normalize single letter filters
        function_map = {"C": "CONTROL", "X": "CONTEXT", "D": "DATA"}
        function_filter = function_map.get(args.function_filter.upper(), args.function_filter.upper())
        
        # Handle index rebuilding
        if args.rebuild_index:
            print("🔄 Rebuilding semantic index...")
            memory_store = get_memory_store(args.db)
            if memory_store:
                engine = MemMimicSemanticSearchEngine()
                engine.index_memories(memory_store, force_rebuild=True)
                print("✅ Index rebuilt successfully")
            else:
                print("❌ Failed to access memory store for rebuilding")
                sys.exit(1)
        
        # Check WordNet availability
        wordnet_status = ensure_wordnet()
        
        # Perform TRUE hybrid search v3.0
        print(f"🧠 MemMimic True Hybrid WordNet Search v3.0", file=sys.stderr)
        print(f"🎯 Query: '{args.query}' | Filter: {function_filter} | Limit: {args.limit}", file=sys.stderr)
        print(f"📚 WordNet: {'✅ Available' if wordnet_status else '❌ Unavailable (basic keywords only)'}", file=sys.stderr)
        
        results = search_memories_hybrid(
            query=args.query,
            function_filter=function_filter,
            limit=args.limit,
            db_name=args.db
        )
        
        if not results:
            print("ℹ️ No memories found matching your search criteria.")
            print("💡 Try:")
            print("   • Different keywords or phrasing")
            print("   • Remove function filter (use 'ALL')")
            print("   • Use --rebuild-index if results seem outdated")
            if not wordnet_status:
                print("   • Install NLTK WordNet: pip install nltk")
            sys.exit(0)
        
        # Format output with enhanced WordNet information
        response_parts = []
        
        # Header with comprehensive hybrid stats
        db_indicator = f" (DB: {args.db})" if args.db else ""
        semantic_count = sum(1 for r in results if 'semantic' in r.get('search_method', ''))
        wordnet_count = sum(1 for r in results if 'wordnet' in r.get('search_method', ''))
        convergence_count = sum(1 for r in results if 'convergence' in r.get('search_method', ''))
        original_term_count = sum(1 for r in results if r.get('original_matches', 0) > 0)
        
        if function_filter == "ALL":
            response_parts.append(f"🧠 MEMMIMIC TRUE HYBRID WORDNET SEARCH v3.0 ({len(results)} results){db_indicator}")
        else:
            response_parts.append(f"🧠 MEMMIMIC {function_filter} HYBRID WORDNET SEARCH v3.0 ({len(results)} results){db_indicator}")
        
        response_parts.append(f"🎯 Methods: {semantic_count} semantic + {wordnet_count} WordNet")
        if convergence_count > 0:
            response_parts.append(f"🎯 CONVERGENCE: {convergence_count} results found by BOTH methods!")
        if original_term_count > 0:
            response_parts.append(f"⭐ ORIGINAL TERMS: {original_term_count} results with exact query matches")
        response_parts.append("")
        
        # Enhanced legend with WordNet indicators
        if function_filter == "ALL":
            response_parts.append("🔤 LEGEND: 🎛️CONTROL=Search/filter 🔗CONTEXT=Relations 📊DATA=Processing")
            response_parts.append("🔍 METHODS: 🧠=Semantic 🔍=WordNet 🎯🧠=Hybrid-Convergence 🎯⭐=Convergence+Original ⭐=Original-Terms")
            response_parts.append("")
        
        # Display results with WordNet formatting
        for i, result in enumerate(results, 1):
            result_lines = format_memory_with_wordnet_info(result, i)
            response_parts.extend(result_lines)
            response_parts.append("")
        
        # Enhanced statistics with WordNet metrics
        if args.stats and results:
            response_parts.append("📈 HYBRID WORDNET SEARCH STATISTICS:")
            avg_combined = sum(r.get('combined_score', 0) for r in results) / len(results)
            avg_semantic = sum(r.get('semantic_similarity', 0) for r in results) / len(results)
            avg_lexical = sum(r.get('lexical_score', 0) for r in results) / len(results)
            total_convergence_bonus = sum(r.get('convergence_bonus', 0) for r in results)
            total_original_bonus = sum(r.get('original_term_bonus', 0) for r in results)
            total_wordnet_terms = sum(len(r.get('matched_terms', [])) for r in results)
            
            response_parts.append(f"   Average combined score: {avg_combined:.3f}")
            response_parts.append(f"   Average semantic similarity: {avg_semantic:.3f}")
            response_parts.append(f"   Average WordNet lexical score: {avg_lexical:.3f}")
            response_parts.append(f"   Total convergence bonus: {total_convergence_bonus:.3f}")
            response_parts.append(f"   Total original term bonus: {total_original_bonus:.3f}")
            response_parts.append(f"   WordNet terms matched: {total_wordnet_terms}")
            response_parts.append(f"   Method distribution: {semantic_count} semantic, {wordnet_count} WordNet, {convergence_count} convergence")
            response_parts.append("")
        
        # Function distribution for ALL searches
        if function_filter == "ALL" and len(results) > 1:
            function_counts = {}
            for result in results:
                func = result.get('cxd_function', 'UNKNOWN')
                function_counts[func] = function_counts.get(func, 0) + 1
            
            response_parts.append("📊 FUNCTION DISTRIBUTION:")
            for func, count in sorted(function_counts.items()):
                icon = {'CONTROL': '🎛️', 'CONTEXT': '🔗', 'DATA': '📊', 'UNKNOWN': '❓'}.get(func, '❓')
                response_parts.append(f"   {icon} {func}: {count}")
            response_parts.append("")
        
        # Footer
        wordnet_indicator = " + NLTK WordNet" if wordnet_status else " (WordNet unavailable)"
        response_parts.append(f"🧠 Powered by MemMimic v3.0 | True Hybrid Search = Semantic + Lexical{wordnet_indicator}")
        
        # Output
        output = "\n".join(response_parts)
        print(output)
        
    except KeyboardInterrupt:
        print("\n⚠️ Search interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"💥 CRITICAL ERROR in MemMimic WordNet search: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
