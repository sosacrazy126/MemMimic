"""
TaleMemoryBinder - Narrative-Memory Fusion System

Creates a system that binds thematic content from the tales directory with the memory
system to enable story-driven memory patterns. This component bridges narrative
intelligence with memory storage for richer contextual understanding.

The binder extracts themes, patterns, and narrative structures from tales and uses
them to enhance memory classification, storage, and retrieval processes.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..errors import MemMimicError, with_error_context, get_error_logger
from ..memory.storage.amms_storage import create_amms_storage
from ..cxd.core.types import CXDFunction, CXDTag


@dataclass
class NarrativeTheme:
    """Represents a narrative theme extracted from tales"""
    theme_id: str
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    emotional_tone: str = "neutral"
    cognitive_patterns: List[str] = field(default_factory=list)
    cxd_affinity: Optional[CXDFunction] = None
    source_tales: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)


@dataclass
class NarrativeContext:
    """Context derived from narrative analysis"""
    primary_theme: Optional[NarrativeTheme] = None
    secondary_themes: List[NarrativeTheme] = field(default_factory=list)
    narrative_confidence: float = 0.0
    story_arc_position: str = "unknown"  # beginning, middle, end, standalone
    emotional_trajectory: str = "stable"  # rising, falling, stable, complex
    character_perspective: str = "neutral"  # first_person, third_person, omniscient
    temporal_context: str = "present"  # past, present, future, timeless


@dataclass
class BindingMetrics:
    """Metrics for narrative-memory binding operations"""
    themes_extracted: int = 0
    memories_enhanced: int = 0
    narrative_classifications: int = 0
    binding_operations: int = 0
    average_narrative_confidence: float = 0.0
    last_binding_time: float = field(default_factory=time.time)


class TaleMemoryBinder:
    """
    Narrative-Memory Fusion System
    
    Binds thematic content from tales with memory storage to enable story-driven
    narrative patterns and enhanced contextual understanding.
    """
    
    def __init__(self, tales_path: str = "tales", db_path: str = None):
        self.tales_path = Path(tales_path)
        self.db_path = db_path or "./src/memmimic/mcp/memmimic.db"
        self.logger = get_error_logger("tale_memory_binder")
        
        # Narrative intelligence
        self.narrative_themes: Dict[str, NarrativeTheme] = {}
        self.binding_metrics = BindingMetrics()
        
        # Memory integration
        self._storage = None
        self._initialized = False
        
        # Narrative analysis cache
        self._narrative_cache: Dict[str, NarrativeContext] = {}
        self._cache_ttl = 1800  # 30 minutes
        
        # Theme extraction patterns
        self._theme_patterns = {
            'awareness': [
                r'\b(?:awareness|sentience|cognition)\b',
                r'\b(?:mind|thought|thinking|mental)\b',
                r'\b(?:intelligence|AI|artificial)\b'
            ],
            'memory': [
                r'\b(?:memory|memories|remember|recall)\b',
                r'\b(?:forget|forgotten|amnesia)\b',
                r'\b(?:nostalgia|reminiscence|recollection)\b'
            ],
            'evolution': [
                r'\b(?:evolution|evolve|development|growth)\b',
                r'\b(?:adaptation|change|transformation)\b',
                r'\b(?:progress|advancement|improvement)\b'
            ],
            'relationship': [
                r'\b(?:relationship|connection|bond|link)\b',
                r'\b(?:friendship|love|trust|understanding)\b',
                r'\b(?:communication|dialogue|conversation)\b'
            ],
            'exploration': [
                r'\b(?:exploration|discovery|adventure|journey)\b',
                r'\b(?:curiosity|wonder|mystery|unknown)\b',
                r'\b(?:search|quest|investigation)\b'
            ],
            'creativity': [
                r'\b(?:creativity|creation|imagination|innovation)\b',
                r'\b(?:art|artistic|beauty|aesthetic)\b',
                r'\b(?:inspiration|vision|dream)\b'
            ]
        }
    
    async def initialize(self) -> None:
        """Initialize tale-memory binding system"""
        if self._initialized:
            return
            
        try:
            # Initialize storage
            self._storage = create_amms_storage(self.db_path)
            
            # Extract narrative themes from tales
            await self._extract_narrative_themes()
            
            # Initialize binding patterns
            await self._initialize_binding_patterns()
            
            self._initialized = True
            self.logger.info("Tale-memory binder initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tale-memory binder: {e}")
            raise MemMimicError(f"Tale-memory binder initialization failed: {e}")
    
    async def _extract_narrative_themes(self) -> None:
        """Extract narrative themes from tales directory"""
        if not self.tales_path.exists():
            self.logger.warning(f"Tales directory not found: {self.tales_path}")
            return
            
        with with_error_context(
            operation="extract_narrative_themes",
            component="tale_memory_binder"
        ):
            # Scan tales directory structure
            for category_dir in self.tales_path.iterdir():
                if category_dir.is_dir():
                    await self._process_tale_category(category_dir)
            
            self.binding_metrics.themes_extracted = len(self.narrative_themes)
            self.logger.info(f"Extracted {self.binding_metrics.themes_extracted} narrative themes")
    
    async def _process_tale_category(self, category_dir: Path) -> None:
        """Process a tale category directory"""
        try:
            category_name = category_dir.name
            
            # Process all tale files in category
            for tale_file in category_dir.rglob("*.md"):
                await self._analyze_tale_file(tale_file, category_name)
                
        except Exception as e:
            self.logger.error(f"Failed to process tale category {category_dir}: {e}")
    
    async def _analyze_tale_file(self, tale_file: Path, category: str) -> None:
        """Analyze a single tale file for narrative themes"""
        try:
            with open(tale_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract themes from content
            themes = await self._extract_themes_from_content(content, tale_file.name, category)
            
            # Add themes to collection
            for theme in themes:
                if theme.theme_id in self.narrative_themes:
                    # Merge with existing theme
                    existing = self.narrative_themes[theme.theme_id]
                    existing.source_tales.extend(theme.source_tales)
                    existing.keywords.extend(theme.keywords)
                    existing.keywords = list(set(existing.keywords))  # Remove duplicates
                else:
                    self.narrative_themes[theme.theme_id] = theme
                    
        except Exception as e:
            self.logger.error(f"Failed to analyze tale file {tale_file}: {e}")
    
    async def _extract_themes_from_content(self, content: str, tale_name: str, category: str) -> List[NarrativeTheme]:
        """Extract narrative themes from tale content"""
        themes = []
        
        # Analyze content for each theme pattern
        for theme_name, patterns in self._theme_patterns.items():
            theme_score = 0
            matched_keywords = []
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                theme_score += len(matches)
                matched_keywords.extend(matches)
            
            # Create theme if significant presence detected
            if theme_score >= 2:  # Threshold for theme significance
                theme_id = f"{category}_{theme_name}"
                
                # Determine CXD affinity
                cxd_affinity = self._determine_cxd_affinity(theme_name, content)
                
                # Analyze emotional tone
                emotional_tone = self._analyze_emotional_tone(content)
                
                theme = NarrativeTheme(
                    theme_id=theme_id,
                    name=f"{theme_name.title()} in {category}",
                    description=f"Narrative theme of {theme_name} found in {category} tales",
                    keywords=list(set(matched_keywords)),
                    emotional_tone=emotional_tone,
                    cxd_affinity=cxd_affinity,
                    source_tales=[tale_name]
                )
                
                themes.append(theme)
        
        return themes
    
    def _determine_cxd_affinity(self, theme_name: str, content: str) -> Optional[CXDFunction]:
        """Determine CXD function affinity for a theme"""
        # Map themes to CXD functions based on cognitive patterns
        theme_cxd_mapping = {
            'awareness': CXDFunction.CONTEXT,
            'memory': CXDFunction.DATA,
            'evolution': CXDFunction.CONTROL,
            'relationship': CXDFunction.CONTEXT,
            'exploration': CXDFunction.CONTROL,
            'creativity': CXDFunction.DATA
        }
        
        return theme_cxd_mapping.get(theme_name)
    
    def _analyze_emotional_tone(self, content: str) -> str:
        """Analyze emotional tone of content"""
        # Simple emotional tone analysis
        positive_words = ['joy', 'happy', 'love', 'hope', 'wonder', 'beautiful', 'amazing']
        negative_words = ['sad', 'fear', 'anger', 'loss', 'pain', 'difficult', 'struggle']
        neutral_words = ['analyze', 'consider', 'examine', 'process', 'understand']
        
        content_lower = content.lower()
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        neutral_count = sum(1 for word in neutral_words if word in content_lower)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            return "negative"
        else:
            return "neutral"
    
    async def _initialize_binding_patterns(self) -> None:
        """Initialize patterns for binding narratives to memories"""
        # This will be expanded with more sophisticated binding logic
        pass
    
    async def enhance_memory_with_narrative(self, content: str, memory_type: str) -> Dict[str, Any]:
        """Enhance memory storage with narrative context"""
        if not self._initialized:
            await self.initialize()
            
        with with_error_context(
            operation="enhance_memory_with_narrative",
            component="tale_memory_binder",
            metadata={"content_length": len(content), "memory_type": memory_type}
        ):
            try:
                # Analyze content for narrative context
                narrative_context = await self._analyze_narrative_context(content)
                
                # Generate narrative enhancement
                enhancement = {
                    "narrative_context": narrative_context,
                    "enhanced_cxd": await self._enhance_cxd_with_narrative(content, narrative_context),
                    "thematic_tags": self._generate_thematic_tags(narrative_context),
                    "story_position": narrative_context.story_arc_position,
                    "emotional_context": narrative_context.emotional_trajectory
                }
                
                self.binding_metrics.memories_enhanced += 1
                self.binding_metrics.binding_operations += 1
                
                return enhancement
                
            except Exception as e:
                self.logger.error(f"Failed to enhance memory with narrative: {e}")
                return {"narrative_enhancement": False, "error": str(e)}
    
    async def _analyze_narrative_context(self, content: str) -> NarrativeContext:
        """Analyze content for narrative context"""
        # Check cache first
        cache_key = hash(content) % 10000
        if cache_key in self._narrative_cache:
            return self._narrative_cache[cache_key]
        
        context = NarrativeContext()
        
        # Find matching themes
        matching_themes = []
        for theme in self.narrative_themes.values():
            theme_score = 0
            for keyword in theme.keywords:
                if keyword.lower() in content.lower():
                    theme_score += 1
            
            if theme_score > 0:
                matching_themes.append((theme, theme_score))
        
        # Sort by relevance
        matching_themes.sort(key=lambda x: x[1], reverse=True)
        
        if matching_themes:
            context.primary_theme = matching_themes[0][0]
            context.secondary_themes = [theme for theme, _ in matching_themes[1:3]]
            context.narrative_confidence = min(matching_themes[0][1] / 10.0, 1.0)
        
        # Analyze story structure
        context.story_arc_position = self._analyze_story_position(content)
        context.emotional_trajectory = self._analyze_emotional_trajectory(content)
        context.character_perspective = self._analyze_perspective(content)
        context.temporal_context = self._analyze_temporal_context(content)
        
        # Cache result
        self._narrative_cache[cache_key] = context
        
        return context
    
    def _analyze_story_position(self, content: str) -> str:
        """Analyze position in story arc"""
        beginning_indicators = ['start', 'begin', 'first', 'initial', 'introduction']
        middle_indicators = ['continue', 'develop', 'progress', 'during', 'meanwhile']
        end_indicators = ['conclude', 'final', 'end', 'result', 'outcome']
        
        content_lower = content.lower()
        
        beginning_score = sum(1 for word in beginning_indicators if word in content_lower)
        middle_score = sum(1 for word in middle_indicators if word in content_lower)
        end_score = sum(1 for word in end_indicators if word in content_lower)
        
        if beginning_score > middle_score and beginning_score > end_score:
            return "beginning"
        elif end_score > beginning_score and end_score > middle_score:
            return "end"
        elif middle_score > 0:
            return "middle"
        else:
            return "standalone"
    
    def _analyze_emotional_trajectory(self, content: str) -> str:
        """Analyze emotional trajectory of content"""
        # Simple analysis - could be enhanced with more sophisticated NLP
        if any(word in content.lower() for word in ['improve', 'better', 'grow', 'rise']):
            return "rising"
        elif any(word in content.lower() for word in ['decline', 'worse', 'fall', 'decrease']):
            return "falling"
        elif any(word in content.lower() for word in ['complex', 'mixed', 'varied', 'changing']):
            return "complex"
        else:
            return "stable"
    
    def _analyze_perspective(self, content: str) -> str:
        """Analyze narrative perspective"""
        first_person_indicators = content.count(' I ') + content.count(' me ') + content.count(' my ')
        third_person_indicators = content.count(' he ') + content.count(' she ') + content.count(' they ')
        
        if first_person_indicators > third_person_indicators:
            return "first_person"
        elif third_person_indicators > 0:
            return "third_person"
        else:
            return "neutral"
    
    def _analyze_temporal_context(self, content: str) -> str:
        """Analyze temporal context of content"""
        past_indicators = ['was', 'were', 'had', 'did', 'yesterday', 'before']
        future_indicators = ['will', 'shall', 'going to', 'tomorrow', 'next', 'future']
        present_indicators = ['is', 'are', 'am', 'now', 'today', 'currently']
        
        content_lower = content.lower()
        
        past_score = sum(1 for word in past_indicators if word in content_lower)
        future_score = sum(1 for word in future_indicators if word in content_lower)
        present_score = sum(1 for word in present_indicators if word in content_lower)
        
        if past_score > future_score and past_score > present_score:
            return "past"
        elif future_score > past_score and future_score > present_score:
            return "future"
        elif present_score > 0:
            return "present"
        else:
            return "timeless"
    
    async def _enhance_cxd_with_narrative(self, content: str, narrative_context: NarrativeContext) -> Optional[Dict[str, Any]]:
        """Enhance CXD classification with narrative context"""
        if not narrative_context.primary_theme:
            return None
        
        enhancement = {
            "narrative_cxd_affinity": narrative_context.primary_theme.cxd_affinity.value if narrative_context.primary_theme.cxd_affinity else None,
            "thematic_confidence": narrative_context.narrative_confidence,
            "emotional_modifier": narrative_context.emotional_trajectory,
            "perspective_context": narrative_context.character_perspective
        }
        
        return enhancement
    
    def _generate_thematic_tags(self, narrative_context: NarrativeContext) -> List[str]:
        """Generate thematic tags from narrative context"""
        tags = []
        
        if narrative_context.primary_theme:
            tags.append(f"theme:{narrative_context.primary_theme.name.lower().replace(' ', '_')}")
            tags.append(f"tone:{narrative_context.primary_theme.emotional_tone}")
        
        tags.append(f"arc:{narrative_context.story_arc_position}")
        tags.append(f"emotion:{narrative_context.emotional_trajectory}")
        tags.append(f"perspective:{narrative_context.character_perspective}")
        tags.append(f"time:{narrative_context.temporal_context}")
        
        return tags
    
    def get_narrative_themes(self) -> Dict[str, NarrativeTheme]:
        """Get all extracted narrative themes"""
        return self.narrative_themes
    
    def get_binding_metrics(self) -> BindingMetrics:
        """Get narrative-memory binding metrics"""
        return self.binding_metrics
    
    async def search_by_narrative_theme(self, theme_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by narrative theme"""
        if not self._initialized:
            await self.initialize()
        
        # This would integrate with the memory search system
        # For now, return placeholder
        return []
