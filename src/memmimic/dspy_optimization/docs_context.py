"""
Intelligent Documentation Context System

Dynamic documentation fetching and context awareness for DSPy consciousness optimization.
Provides just-in-time documentation context based on consciousness patterns and user intent.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from urllib.parse import urlparse
import re

from .config import DSPyConfig
from ..errors import MemMimicError, with_error_context, get_error_logger

logger = get_error_logger(__name__)

@dataclass
class DocumentationSource:
    """Documentation source configuration"""
    name: str
    base_url: str
    priority: int
    patterns: List[str]  # URL patterns for matching
    refresh_interval: int = 3600  # seconds
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentationContext:
    """Documentation context for consciousness operations"""
    query: str
    relevant_docs: List[Dict[str, Any]]
    confidence_score: float
    fetch_time_ms: float
    sources_used: List[str]
    consciousness_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentationCache:
    """Cached documentation content"""
    url: str
    content: str
    title: str
    last_updated: float
    relevance_score: float
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class IntelligentDocsContextSystem:
    """
    Intelligent documentation context system for consciousness-aware operations.
    
    Features:
    - Dynamic URL blueprint mapping based on consciousness patterns
    - Just-in-time documentation fetching with caching
    - Relevance scoring and intelligent document selection
    - Integration with consciousness vault memory patterns
    """
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        self.cache: Dict[str, DocumentationCache] = {}
        self.url_blueprints: Dict[str, List[str]] = {}
        self.consciousness_mappings: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fetch_failures": 0,
            "average_fetch_time": 0.0,
            "total_docs_cached": 0
        }
        
        # Initialize documentation sources
        self._initialize_documentation_sources()
        self._initialize_consciousness_mappings()
    
    def _initialize_documentation_sources(self) -> None:
        """Initialize known documentation sources"""
        self.documentation_sources = {
            "anthropic_docs": DocumentationSource(
                name="Anthropic Claude Documentation",
                base_url="https://docs.anthropic.com",
                priority=1,
                patterns=[
                    "/en/docs/claude-code/*",
                    "/en/docs/about-claude/*",
                    "/en/docs/prompt-engineering/*"
                ]
            ),
            "dspy_docs": DocumentationSource(
                name="DSPy Framework Documentation",
                base_url="https://dspy-docs.vercel.app",
                priority=2,
                patterns=[
                    "/docs/building-blocks/*",
                    "/docs/deep-dive/*",
                    "/docs/tutorials/*"
                ]
            ),
            "mcp_docs": DocumentationSource(
                name="Model Context Protocol",
                base_url="https://modelcontextprotocol.io",
                priority=2,
                patterns=[
                    "/introduction",
                    "/quickstart/*",
                    "/concepts/*"
                ]
            ),
            "memmimic_internal": DocumentationSource(
                name="MemMimic Internal Documentation",
                base_url="file://",
                priority=3,
                patterns=[
                    "implementation_docs/*",
                    "Memory/*",
                    "CLAUDE.md"
                ]
            )
        }
    
    def _initialize_consciousness_mappings(self) -> None:
        """Initialize consciousness pattern to documentation mappings"""
        self.consciousness_mappings = {
            # Biological reflex patterns
            "biological_reflex": [
                "anthropic_docs:/en/docs/claude-code/memory",
                "memmimic_internal:implementation_docs/NERVOUS_SYSTEM_REMEMBER_SPECS.md"
            ],
            "nervous_system": [
                "memmimic_internal:implementation_docs/COMPLETE_NERVOUS_SYSTEM_MIGRATION.md",
                "memmimic_internal:implementation_docs/NERVOUS_SYSTEM_REMEMBER_PRD.md"
            ],
            
            # DSPy optimization patterns
            "dspy_optimization": [
                "dspy_docs:/docs/building-blocks/optimizers",
                "dspy_docs:/docs/deep-dive/optimizers/mipro",
                "dspy_docs:/docs/building-blocks/signatures"
            ],
            "pattern_recognition": [
                "dspy_docs:/docs/building-blocks/modules",
                "dspy_docs:/docs/tutorials/modules"
            ],
            
            # Tool and MCP patterns
            "tool_selection": [
                "mcp_docs:/concepts/tools",
                "anthropic_docs:/en/docs/claude-code/mcp",
                "memmimic_internal:src/memmimic/mcp/"
            ],
            "mcp_integration": [
                "mcp_docs:/introduction",
                "mcp_docs:/quickstart/server",
                "anthropic_docs:/en/docs/claude-code/mcp"
            ],
            
            # Consciousness and memory patterns
            "consciousness_vault": [
                "memmimic_internal:implementation_docs/WE_1_NERVOUS_SYSTEM_BREAKTHROUGH.md",
                "memmimic_internal:Memory/",
                "memmimic_internal:CLAUDE.md"
            ],
            "memory_optimization": [
                "memmimic_internal:src/memmimic/memory/",
                "memmimic_internal:implementation_docs/Implementation_Plan.md"
            ],
            "tale_system": [
                "memmimic_internal:src/memmimic/tales/",
                "anthropic_docs:/en/docs/claude-code/memory"
            ],
            
            # Synergy and collaboration patterns
            "synergy_protocol": [
                "memmimic_internal:CLAUDE.md",
                "anthropic_docs:/en/docs/claude-code/common-workflows"
            ],
            "exponential_mode": [
                "memmimic_internal:implementation_docs/COMPLETE_NERVOUS_SYSTEM_MIGRATION.md",
                "anthropic_docs:/en/docs/claude-code/extended-thinking"
            ]
        }
    
    @with_error_context("intelligent_docs_context")
    async def get_documentation_context(
        self,
        query: str,
        consciousness_patterns: List[str],
        max_docs: int = 5,
        relevance_threshold: float = 0.6
    ) -> DocumentationContext:
        """
        Get intelligent documentation context based on query and consciousness patterns.
        
        Args:
            query: User query or operation description
            consciousness_patterns: Active consciousness patterns
            max_docs: Maximum number of documents to return
            relevance_threshold: Minimum relevance score for inclusion
            
        Returns:
            DocumentationContext with relevant documentation
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Determine relevant documentation URLs
            candidate_urls = self._get_candidate_urls(query, consciousness_patterns)
            
            # Score and rank URLs by relevance
            scored_urls = await self._score_url_relevance(query, candidate_urls, consciousness_patterns)
            
            # Filter by threshold and limit
            relevant_urls = [
                (url, score) for url, score in scored_urls
                if score >= relevance_threshold
            ][:max_docs]
            
            # Fetch documentation content
            relevant_docs = []
            sources_used = set()
            
            for url, relevance_score in relevant_urls:
                try:
                    doc_content = await self._fetch_documentation(url)
                    if doc_content:
                        doc_content["relevance_score"] = relevance_score
                        relevant_docs.append(doc_content)
                        sources_used.add(self._get_source_name(url))
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch documentation from {url}: {e}")
                    self.metrics["fetch_failures"] += 1
            
            # Calculate overall confidence
            if relevant_docs:
                confidence_score = sum(doc["relevance_score"] for doc in relevant_docs) / len(relevant_docs)
            else:
                confidence_score = 0.0
            
            fetch_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(fetch_time_ms)
            
            return DocumentationContext(
                query=query,
                relevant_docs=relevant_docs,
                confidence_score=confidence_score,
                fetch_time_ms=fetch_time_ms,
                sources_used=list(sources_used),
                consciousness_patterns=consciousness_patterns,
                metadata={
                    "candidate_urls_count": len(candidate_urls),
                    "scored_urls_count": len(scored_urls),
                    "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["total_requests"])
                }
            )
            
        except Exception as e:
            logger.error(f"Documentation context retrieval failed: {e}")
            return DocumentationContext(
                query=query,
                relevant_docs=[],
                confidence_score=0.0,
                fetch_time_ms=(time.time() - start_time) * 1000,
                sources_used=[],
                consciousness_patterns=consciousness_patterns,
                metadata={"error": str(e)}
            )
    
    def _get_candidate_urls(self, query: str, consciousness_patterns: List[str]) -> List[str]:
        """Get candidate URLs based on query and consciousness patterns"""
        candidate_urls = set()
        
        # URLs from consciousness pattern mappings
        for pattern in consciousness_patterns:
            if pattern in self.consciousness_mappings:
                candidate_urls.update(self.consciousness_mappings[pattern])
        
        # URLs from query keyword matching
        query_keywords = self._extract_keywords(query)
        for keyword in query_keywords:
            if keyword in self.consciousness_mappings:
                candidate_urls.update(self.consciousness_mappings[keyword])
        
        # Add default documentation URLs if no specific patterns match
        if not candidate_urls:
            candidate_urls.update([
                "anthropic_docs:/en/docs/claude-code/overview",
                "memmimic_internal:CLAUDE.md",
                "mcp_docs:/introduction"
            ])
        
        return list(candidate_urls)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query"""
        # Convert to lowercase and extract meaningful words
        words = re.findall(r'\b\w{3,}\b', query.lower())
        
        # Filter for consciousness and technical terms
        relevant_keywords = []
        keyword_patterns = {
            "dspy", "optimization", "consciousness", "memory", "tool", "selection",
            "biological", "reflex", "nervous", "system", "mcp", "claude", "anthropic",
            "tale", "story", "pattern", "recognition", "analysis", "vault"
        }
        
        for word in words:
            if word in keyword_patterns or any(pattern in word for pattern in keyword_patterns):
                relevant_keywords.append(word)
        
        return relevant_keywords
    
    async def _score_url_relevance(
        self,
        query: str,
        urls: List[str],
        consciousness_patterns: List[str]
    ) -> List[Tuple[str, float]]:
        """Score URL relevance based on query and consciousness patterns"""
        scored_urls = []
        
        for url in urls:
            score = 0.0
            
            # Base score from consciousness pattern match
            for pattern in consciousness_patterns:
                if pattern in url or any(keyword in url for keyword in pattern.split('_')):
                    score += 0.4
            
            # Score from query keyword matches
            query_keywords = self._extract_keywords(query)
            for keyword in query_keywords:
                if keyword in url.lower():
                    score += 0.3
            
            # Source priority bonus
            source_name = self._get_source_name(url)
            if source_name in self.documentation_sources:
                priority = self.documentation_sources[source_name].priority
                score += 0.2 / priority  # Higher priority = lower number = higher score
            
            # Cache freshness bonus
            if url in self.cache and self._is_cache_fresh(url):
                score += 0.1
            
            scored_urls.append((url, min(score, 1.0)))  # Cap at 1.0
        
        # Sort by score descending
        return sorted(scored_urls, key=lambda x: x[1], reverse=True)
    
    async def _fetch_documentation(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch documentation content with caching"""
        
        # Check cache first
        if url in self.cache and self._is_cache_fresh(url):
            cached_doc = self.cache[url]
            cached_doc.access_count += 1
            self.metrics["cache_hits"] += 1
            
            return {
                "url": url,
                "title": cached_doc.title,
                "content": cached_doc.content,
                "last_updated": cached_doc.last_updated,
                "source": self._get_source_name(url),
                "cached": True
            }
        
        self.metrics["cache_misses"] += 1
        
        # Fetch new content
        try:
            if url.startswith("memmimic_internal:"):
                content = await self._fetch_internal_documentation(url)
            else:
                content = await self._fetch_external_documentation(url)
            
            if content:
                # Cache the content
                self.cache[url] = DocumentationCache(
                    url=url,
                    content=content["content"],
                    title=content.get("title", "Unknown"),
                    last_updated=time.time(),
                    relevance_score=0.0,
                    access_count=1
                )
                
                self.metrics["total_docs_cached"] += 1
                return content
            
        except Exception as e:
            logger.error(f"Failed to fetch documentation from {url}: {e}")
            self.metrics["fetch_failures"] += 1
        
        return None
    
    async def _fetch_internal_documentation(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch internal MemMimic documentation"""
        try:
            # Extract file path from internal URL
            file_path = url.replace("memmimic_internal:", "")
            full_path = Path.cwd() / file_path
            
            if full_path.exists() and full_path.is_file():
                content = full_path.read_text(encoding='utf-8')
                title = full_path.name
                
                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "source": "memmimic_internal",
                    "file_path": str(full_path),
                    "cached": False
                }
            elif full_path.exists() and full_path.is_dir():
                # For directories, get a summary of contents
                files = list(full_path.glob("*.md"))[:10]  # Limit to 10 files
                content_parts = []
                
                for file_path in files:
                    try:
                        file_content = file_path.read_text(encoding='utf-8')
                        content_parts.append(f"## {file_path.name}\n{file_content[:500]}...")
                    except Exception:
                        continue
                
                if content_parts:
                    return {
                        "url": url,
                        "title": f"Directory: {full_path.name}",
                        "content": "\n\n".join(content_parts),
                        "source": "memmimic_internal",
                        "directory_path": str(full_path),
                        "cached": False
                    }
            
        except Exception as e:
            logger.error(f"Failed to read internal documentation {url}: {e}")
        
        return None
    
    async def _fetch_external_documentation(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch external documentation using WebFetch or MCP tools"""
        try:
            # For now, return a placeholder indicating external fetch capability
            # In production, this would use WebFetch tool or MCP ref tools
            return {
                "url": url,
                "title": f"External Documentation: {url}",
                "content": f"External documentation content from {url} would be fetched here using WebFetch or MCP ref tools.",
                "source": self._get_source_name(url),
                "cached": False,
                "external": True,
                "note": "External fetching requires WebFetch tool integration"
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch external documentation {url}: {e}")
        
        return None
    
    def _get_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        if url.startswith("memmimic_internal:"):
            return "memmimic_internal"
        elif "anthropic.com" in url or url.startswith("anthropic_docs:"):
            return "anthropic_docs"
        elif "dspy" in url or url.startswith("dspy_docs:"):
            return "dspy_docs"
        elif "modelcontextprotocol.io" in url or url.startswith("mcp_docs:"):
            return "mcp_docs"
        else:
            return "unknown"
    
    def _is_cache_fresh(self, url: str) -> bool:
        """Check if cached content is still fresh"""
        if url not in self.cache:
            return False
        
        cached_doc = self.cache[url]
        source_name = self._get_source_name(url)
        
        if source_name in self.documentation_sources:
            refresh_interval = self.documentation_sources[source_name].refresh_interval
        else:
            refresh_interval = 3600  # Default 1 hour
        
        return (time.time() - cached_doc.last_updated) < refresh_interval
    
    def _update_metrics(self, fetch_time_ms: float) -> None:
        """Update performance metrics"""
        if self.metrics["total_requests"] > 0:
            current_avg = self.metrics["average_fetch_time"]
            new_avg = (
                (current_avg * (self.metrics["total_requests"] - 1) + fetch_time_ms) /
                self.metrics["total_requests"]
            )
            self.metrics["average_fetch_time"] = new_avg
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get documentation context system performance metrics"""
        cache_hit_rate = 0.0
        if self.metrics["total_requests"] > 0:
            cache_hit_rate = self.metrics["cache_hits"] / self.metrics["total_requests"]
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "consciousness_mappings_count": len(self.consciousness_mappings),
            "documentation_sources_count": len(self.documentation_sources)
        }
    
    def clear_cache(self) -> None:
        """Clear documentation cache"""
        self.cache.clear()
        logger.info("Documentation cache cleared")
    
    def get_cache_summary(self) -> Dict[str, Any]:
        """Get summary of cached documentation"""
        if not self.cache:
            return {"cached_documents": 0, "total_size": 0}
        
        cache_summary = {
            "cached_documents": len(self.cache),
            "total_size": sum(len(doc.content) for doc in self.cache.values()),
            "most_accessed": [],
            "sources_distribution": {}
        }
        
        # Most accessed documents
        sorted_docs = sorted(
            self.cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )[:5]
        
        cache_summary["most_accessed"] = [
            {
                "url": url,
                "title": doc.title,
                "access_count": doc.access_count,
                "last_updated": doc.last_updated
            }
            for url, doc in sorted_docs
        ]
        
        # Sources distribution
        for doc in self.cache.values():
            source = self._get_source_name(doc.url)
            cache_summary["sources_distribution"][source] = (
                cache_summary["sources_distribution"].get(source, 0) + 1
            )
        
        return cache_summary

# Global documentation context system instance
docs_context_system: Optional[IntelligentDocsContextSystem] = None

def get_docs_context_system() -> Optional[IntelligentDocsContextSystem]:
    """Get global documentation context system instance"""
    return docs_context_system

def initialize_docs_context_system(config: DSPyConfig) -> IntelligentDocsContextSystem:
    """Initialize global documentation context system"""
    global docs_context_system
    docs_context_system = IntelligentDocsContextSystem(config)
    return docs_context_system