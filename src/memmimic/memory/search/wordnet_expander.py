#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WordNet Query Expansion Module

Handles NLTK WordNet integration for semantic query expansion.
Extracted from memmimic_recall_cxd.py for improved maintainability.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from ...utils.caching import lru_cached, cached_memory_operation

logger = logging.getLogger(__name__)

# Global WordNet state management
_wordnet_initialized = False
_wordnet_available = False


def ensure_wordnet():
    """Ensure NLTK WordNet is properly initialized."""
    global _wordnet_initialized, _wordnet_available
    
    if _wordnet_initialized:
        return _wordnet_available
    
    try:
        import nltk
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
            logger.debug("WordNet already available")
        except LookupError:
            logger.info("Downloading WordNet corpus...")
            nltk.download('wordnet', quiet=True)
        
        try:
            nltk.data.find('corpora/omw-1.4')
            logger.debug("OMW-1.4 already available")
        except LookupError:
            logger.info("Downloading OMW-1.4 corpus...")
            nltk.download('omw-1.4', quiet=True)
        
        # Test WordNet access
        from nltk.corpus import wordnet as wn
        test_synsets = wn.synsets('test')
        if test_synsets:
            _wordnet_available = True
            logger.info("WordNet initialization successful")
        else:
            logger.warning("WordNet available but no synsets found for 'test'")
            _wordnet_available = False
            
    except ImportError:
        logger.warning("NLTK not available - WordNet expansion disabled")
        _wordnet_available = False
    except Exception as e:
        logger.error(f"WordNet initialization failed: {e}")
        _wordnet_available = False
    
    _wordnet_initialized = True
    return _wordnet_available


class WordNetExpander:
    """
    WordNet-based query expansion for semantic search enhancement.
    
    Provides synonym expansion and linguistic analysis using NLTK WordNet.
    """
    
    def __init__(self):
        """Initialize WordNet expander."""
        self.wordnet_available = ensure_wordnet()
        if self.wordnet_available:
            from nltk.corpus import wordnet as wn
            self.wordnet = wn
    
    @lru_cached(maxsize=512)  # Cache synonym lookups
    def get_wordnet_synonyms(
        self,
        word: str,
        pos_filter: Optional[str] = None,
        max_synonyms: int = 5,
        include_lemmas: bool = True
    ) -> Set[str]:
        """
        Get WordNet synonyms for a word with comprehensive filtering.
        
        Args:
            word: Target word for synonym expansion
            pos_filter: Part-of-speech filter (n, v, a, r for noun, verb, adj, adv)
            max_synonyms: Maximum number of synonyms to return
            include_lemmas: Whether to include lemma names
            
        Returns:
            Set of synonyms found in WordNet
        """
        if not self.wordnet_available:
            return set()
        
        try:
            synonyms = set()
            word_clean = word.lower().strip()
            
            # Get synsets for the word
            synsets = self.wordnet.synsets(word_clean, pos=pos_filter)
            
            for synset in synsets[:max_synonyms]:  # Limit synsets processed
                # Add lemma names
                if include_lemmas:
                    for lemma in synset.lemmas():
                        lemma_name = lemma.name().replace('_', ' ')
                        if lemma_name != word_clean and len(lemma_name) > 2:
                            synonyms.add(lemma_name)
                
                # Add definition-based synonyms (extract key terms)
                definition = synset.definition().lower()
                def_words = re.findall(r'\b[a-z]{3,}\b', definition)
                for def_word in def_words[:3]:  # Limit definition words
                    if def_word not in ['the', 'and', 'that', 'with', 'for']:
                        synonyms.add(def_word)
                
                # Stop if we have enough synonyms
                if len(synonyms) >= max_synonyms:
                    break
            
            return synonyms
            
        except Exception as e:
            logger.error(f"WordNet synonym extraction failed for '{word}': {e}")
            return set()
    
    def get_multilingual_synonyms(self, word: str, max_synonyms: int = 5) -> Set[str]:
        """
        Get multilingual synonyms using WordNet's multilingual support.
        
        Args:
            word: Target word
            max_synonyms: Maximum synonyms to return
            
        Returns:
            Set of multilingual synonyms
        """
        if not self.wordnet_available:
            return set()
        
        try:
            synonyms = set()
            synsets = self.wordnet.synsets(word.lower())
            
            for synset in synsets[:3]:  # Limit synsets for performance
                # Get lemmas in different languages
                for lemma in synset.lemmas():
                    # English synonyms
                    name = lemma.name().replace('_', ' ')
                    if name != word.lower() and len(name) > 2:
                        synonyms.add(name)
                
                if len(synonyms) >= max_synonyms:
                    break
            
            return synonyms
            
        except Exception as e:
            logger.error(f"Multilingual synonym extraction failed: {e}")
            return set()
    
    @cached_memory_operation(ttl=1800)  # Cache query expansions for 30 minutes
    def expand_query(self, query: str, max_synonyms_per_word: int = 3) -> List[str]:
        """
        Expand a query using WordNet synonyms.
        
        Args:
            query: Original search query
            max_synonyms_per_word: Maximum synonyms per word
            
        Returns:
            List of expanded query variations
        """
        if not self.wordnet_available:
            return [query]  # Return original query if WordNet unavailable
        
        try:
            # Clean and tokenize query
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            if not words:
                return [query]
            
            expanded_queries = [query]  # Always include original
            
            # Process each significant word
            for word in words[:5]:  # Limit words processed for performance
                synonyms = self.get_wordnet_synonyms(
                    word, max_synonyms=max_synonyms_per_word
                )
                
                # Create variations by replacing word with synonyms
                for synonym in list(synonyms)[:max_synonyms_per_word]:
                    expanded_query = query.replace(word, synonym, 1)
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)
                
                # Limit total expansions
                if len(expanded_queries) >= 10:
                    break
            
            logger.debug(f"Expanded '{query}' into {len(expanded_queries)} variations")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def search_with_expansion(
        self,
        expanded_queries: List[str],
        memory_store,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search using expanded queries and aggregate results.
        
        Args:
            expanded_queries: List of query variations
            memory_store: Memory storage instance
            limit: Maximum results to return
            
        Returns:
            List of search results from WordNet expansion
        """
        try:
            all_results = []
            seen_content = set()
            
            for i, expanded_query in enumerate(expanded_queries[:5]):  # Limit queries
                try:
                    # Simple keyword search on the expanded query
                    results = memory_store.search_memories(expanded_query, limit=limit)
                    
                    for result in results:
                        content = result.content if hasattr(result, 'content') else str(result)
                        
                        # Avoid duplicates
                        if content not in seen_content:
                            seen_content.add(content)
                            
                            # Format result with WordNet metadata
                            formatted_result = {
                                "content": content,
                                "wordnet_score": max(0.1, 1.0 - (i * 0.2)),  # Decay score
                                "search_method": "wordnet",
                                "expansion_query": expanded_query,
                                "original_rank": len(all_results) + 1
                            }
                            
                            # Copy other attributes if available
                            if hasattr(result, '__dict__'):
                                for key, value in result.__dict__.items():
                                    if key not in formatted_result:
                                        formatted_result[key] = value
                            
                            all_results.append(formatted_result)
                            
                            if len(all_results) >= limit:
                                break
                    
                    if len(all_results) >= limit:
                        break
                        
                except Exception as e:
                    logger.error(f"Search failed for expanded query '{expanded_query}': {e}")
                    continue
            
            logger.debug(f"WordNet search found {len(all_results)} results")
            return all_results
            
        except Exception as e:
            logger.error(f"WordNet expansion search failed: {e}")
            return []


# Backward compatibility functions
def get_wordnet_synonyms(word: str, pos_filter: Optional[str] = None, 
                        max_synonyms: int = 5) -> Set[str]:
    """Backward compatibility wrapper."""
    expander = WordNetExpander()
    return expander.get_wordnet_synonyms(word, pos_filter, max_synonyms)


def expand_query_with_wordnet(query: str, max_synonyms_per_word: int = 3) -> List[str]:
    """Backward compatibility wrapper."""
    expander = WordNetExpander()
    return expander.expand_query(query, max_synonyms_per_word)