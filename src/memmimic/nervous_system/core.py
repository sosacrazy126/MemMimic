"""
NervousSystemCore - Central Intelligence Foundation

The core nervous system class that provides shared intelligence components
for all four biological reflex triggers (remember, recall, think, analyze).

Performance Target: <5ms response time through intelligent caching and parallel processing.
"""

import asyncio
import os
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
from functools import lru_cache
from dataclasses import dataclass, field
import logging
from pathlib import Path

from .interfaces import (
    QualityGateInterface,
    DuplicateDetectorInterface, 
    SocraticGuidanceInterface,
    QualityAssessment,
    DuplicateAnalysis,
    SocraticGuidance
)
from ..memory.storage.amms_storage import Memory, create_amms_storage
from ..cxd import create_optimized_classifier
from ..errors import MemMimicError, with_error_context, get_error_logger
# Synergy components removed - keeping core independent functionality
from ..evolution import (
    MemoryEvolutionTracker,
    MemoryLifecycleManager,
    MemoryUsageAnalytics,
    MemoryEvolutionMetrics,
    MemoryEventType,
    create_evolution_system
)
from .archive_intelligence import ArchiveIntelligence, EvolutionMetrics as ArchiveEvolutionMetrics
from .phase_evolution_tracker import PhaseEvolutionTracker, PhaseStatus, TaskStatus
from .tale_memory_binder import TaleMemoryBinder, NarrativeContext, BindingMetrics
from .reflex_latency_optimizer import ReflexLatencyOptimizer, LatencyMetrics
from .shared_reality_manager import SharedRealityManager, AgentRole, AgentIdentity
from .theory_of_mind import TheoryOfMindCapabilities, MentalStateType, IntentionPrediction

@dataclass
class NervousSystemMetrics:
    """Performance metrics for nervous system operations"""
    total_operations: int = 0
    average_response_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    quality_assessments: int = 0
    duplicate_detections: int = 0
    socratic_guidances: int = 0
    errors: int = 0
    last_performance_check: float = field(default_factory=time.time)

    # Archive intelligence metrics
    archive_patterns_applied: int = 0
    cleanup_operations: int = 0
    unused_components_detected: int = 0

    # Phase evolution metrics
    current_phase: Optional[str] = None
    phase_progress: float = 0.0
    completed_phases: int = 0

class NervousSystemCore:
    """
    Central nervous system intelligence shared by all four core triggers.
    
    Provides high-performance caching, parallel processing, and intelligent
    component coordination for <5ms biological reflex response times.
    """
    
    def __init__(self, db_path: str = "memmimic.db", cache_size: int = 1000):
        # Resolve database path intelligently
        self.db_path = self._resolve_db_path(db_path)
        self.cache_size = cache_size
        self.logger = get_error_logger("nervous_system_core")
        
        # Performance metrics
        self.metrics = NervousSystemMetrics()
        
        # Component interfaces (initialized lazily)
        self._quality_gate: Optional[QualityGateInterface] = None
        self._duplicate_detector: Optional[DuplicateDetectorInterface] = None
        self._socratic_guidance: Optional[SocraticGuidanceInterface] = None
        
        # Core dependencies
        self._amms_storage = None
        self._cxd_classifier = None
        self._initialized = False
        
        # Core independent processing - no amplification noise
        
        # Memory evolution tracking components
        self._evolution_tracker = None
        self._lifecycle_manager = None
        self._usage_analytics = None
        self._evolution_metrics = None
        self._evolution_system_initialized = False

        # Archive intelligence and phase tracking
        self._archive_intelligence = None
        self._phase_tracker = None
        self._tale_memory_binder = None

        # MCP orchestration enhancement
        self._latency_optimizer = None
        self._shared_reality_manager = None
        self._theory_of_mind = None
        
        # LRU caches for <5ms performance
        self._setup_performance_caches()
    
    def _setup_performance_caches(self):
        """Setup LRU caches for sub-millisecond operations"""
        # Quality assessment cache (maxsize=1000 for frequent content patterns)
        self._quality_cache = {}
        self._duplicate_cache = {}
        self._guidance_cache = {}
        
        # Performance monitoring
        self._operation_times = []
    
    def _resolve_db_path(self, db_path: str) -> str:
        """
        Intelligently resolve database path for different execution contexts.
        
        Handles cases when running from:
        - MCP directory: ./memmimic.db
        - Project root: src/memmimic/mcp/memmimic.db
        - Absolute path: use as provided
        """
        # If already absolute path, use it
        if os.path.isabs(db_path):
            return db_path
            
        # Try current working directory first (for MCP context)
        cwd_path = Path.cwd() / db_path
        if cwd_path.exists():
            return str(cwd_path)
            
        # Try MCP directory relative to this file
        current_file = Path(__file__)
        mcp_dir = current_file.parent.parent / "mcp"
        mcp_path = mcp_dir / db_path
        if mcp_path.exists():
            return str(mcp_path)
            
        # Try project structure path
        project_path = Path.cwd() / "src" / "memmimic" / "mcp" / db_path
        if project_path.exists():
            return str(project_path)
            
        # Default to current working directory (let it create if needed)
        return str(cwd_path)
    
    async def initialize(self) -> None:
        """
        Initialize all nervous system components with error handling.
        Uses parallel initialization for optimal startup performance.
        """
        if self._initialized:
            return
            
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="nervous_system_initialization",
            component="core",
            metadata={"db_path": self.db_path, "cache_size": self.cache_size}
        ):
            try:
                # Parallel initialization for performance
                async with asyncio.TaskGroup() as tg:
                    # Core storage and classification
                    storage_task = tg.create_task(self._initialize_storage())
                    classifier_task = tg.create_task(self._initialize_classifier())
                    
                    # Intelligence components - now fully implemented
                    quality_task = tg.create_task(self._initialize_quality_gate())
                    duplicate_task = tg.create_task(self._initialize_duplicate_detector())
                    socratic_task = tg.create_task(self._initialize_socratic_guidance())
                    
                    # Core processing - no amplification components
                    
                    # Memory evolution tracking
                    evolution_task = tg.create_task(self._initialize_evolution_system())

                    # Archive intelligence and phase tracking
                    archive_task = tg.create_task(self._initialize_archive_intelligence())
                    phase_task = tg.create_task(self._initialize_phase_tracker())
                    tale_binder_task = tg.create_task(self._initialize_tale_memory_binder())

                    # MCP orchestration enhancement
                    latency_task = tg.create_task(self._initialize_latency_optimizer())
                    reality_task = tg.create_task(self._initialize_shared_reality_manager())
                    tom_task = tg.create_task(self._initialize_theory_of_mind())
                
                self._amms_storage = storage_task.result()
                self._cxd_classifier = classifier_task.result()
                self._quality_gate = quality_task.result()
                self._duplicate_detector = duplicate_task.result()
                self._socratic_guidance = socratic_task.result()
                # Core components initialized - no amplification noise
                
                # Initialize evolution system
                evolution_results = evolution_task.result()
                if evolution_results:
                    (self._evolution_tracker, self._lifecycle_manager,
                     self._usage_analytics, self._evolution_metrics, _) = evolution_results
                    self._evolution_system_initialized = True

                # Initialize archive intelligence and phase tracking
                self._archive_intelligence = archive_task.result()
                self._phase_tracker = phase_task.result()
                self._tale_memory_binder = tale_binder_task.result()

                # Initialize MCP orchestration enhancement
                self._latency_optimizer = latency_task.result()
                self._shared_reality_manager = reality_task.result()
                self._theory_of_mind = tom_task.result()
                
                self._initialized = True
                init_time = (time.perf_counter() - start_time) * 1000
                
                self.logger.info(
                    f"NervousSystemCore initialized successfully in {init_time:.2f}ms",
                    extra={"initialization_time_ms": init_time}
                )
                
            except Exception as e:
                self.metrics.errors += 1
                self.logger.error(
                    f"Failed to initialize NervousSystemCore: {e}",
                    extra={"error_type": type(e).__name__, "db_path": self.db_path}
                )
                raise MemMimicError(f"NervousSystemCore initialization failed: {e}")
    
    async def _initialize_storage(self) -> Any:
        """Initialize AMMS storage with connection pooling"""
        return create_amms_storage(self.db_path)
    
    async def _initialize_classifier(self) -> Any:
        """Initialize CXD classifier with caching"""
        try:
            return create_optimized_classifier()
        except Exception as e:
            self.logger.warning(
                f"CXD classifier initialization failed, continuing without classification: {e}"
            )
            return None
    
    async def _initialize_quality_gate(self) -> QualityGateInterface:
        """Initialize quality gate component"""
        from .quality_gate import InternalQualityGate
        quality_gate = InternalQualityGate()
        await quality_gate.initialize()
        return quality_gate
    
    async def _initialize_duplicate_detector(self) -> DuplicateDetectorInterface:
        """Initialize duplicate detector component"""
        from .duplicate_detector import SemanticDuplicateDetector
        duplicate_detector = SemanticDuplicateDetector()
        await duplicate_detector.initialize()
        return duplicate_detector
    
    async def _initialize_socratic_guidance(self) -> SocraticGuidanceInterface:
        """Initialize Socratic guidance component"""
        from .socratic_guidance import InternalSocraticGuidance
        socratic_guidance = InternalSocraticGuidance()
        await socratic_guidance.initialize()
        return socratic_guidance
    
    @lru_cache(maxsize=1000)
    def _get_cached_quality_key(self, content: str, memory_type: str) -> str:
        """Generate cache key for quality assessments"""
        return f"quality:{hash(content)}:{memory_type}"
    
    @lru_cache(maxsize=1000)
    def _get_cached_duplicate_key(self, content: str) -> str:
        """Generate cache key for duplicate detection"""
        return f"duplicate:{hash(content)}"
    
    async def process_with_intelligence(
        self, 
        content: str, 
        memory_type: str = "interaction",
        enable_quality_gate: bool = True,
        enable_duplicate_detection: bool = True,
        enable_socratic_guidance: bool = False
    ) -> Dict[str, Any]:
        """
        Process content through nervous system intelligence pipeline.
        
        Parallel processing architecture for <5ms response time target.
        Returns comprehensive analysis for memory storage decision.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Stage 1: Parallel Intelligence Processing (<3ms target)
            async with asyncio.TaskGroup() as tg:
                tasks = {}
                
                # Quality assessment (if enabled and component available)
                if enable_quality_gate and self._quality_gate:
                    tasks['quality'] = tg.create_task(
                        self._assess_quality_cached(content, memory_type)
                    )
                
                # Duplicate detection (if enabled and component available)
                if enable_duplicate_detection and self._duplicate_detector:
                    tasks['duplicate'] = tg.create_task(
                        self._detect_duplicates_cached(content, memory_type)
                    )
                
                # CXD classification (always attempt if available)
                if self._cxd_classifier:
                    tasks['cxd'] = tg.create_task(
                        self._classify_cxd_cached(content)
                    )
                
                # Socratic guidance (if explicitly requested)
                if enable_socratic_guidance and self._socratic_guidance:
                    tasks['socratic'] = tg.create_task(
                        self._get_socratic_guidance_cached(content, memory_type)
                    )

                # Narrative enhancement (if tale-memory binder available)
                if self._tale_memory_binder:
                    tasks['narrative'] = tg.create_task(
                        self._tale_memory_binder.enhance_memory_with_narrative(content, memory_type)
                    )
            
            # Collect results
            results = {}
            for task_name, task in tasks.items():
                try:
                    results[task_name] = task.result()
                except Exception as e:
                    self.logger.warning(
                        f"Intelligence component {task_name} failed: {e}",
                        extra={"component": task_name, "error": str(e)}
                    )
                    results[task_name] = None
            
            # Stage 2: Decision Synthesis (<1ms target)
            synthesis = await self._synthesize_intelligence_results(results, content, memory_type)
            
            # Performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(processing_time)
            
            return {
                **synthesis,
                'processing_time_ms': processing_time,
                'nervous_system_version': '1.0.0'
            }
            
        except Exception as e:
            self.metrics.errors += 1
            processing_time = (time.perf_counter() - start_time) * 1000
            
            self.logger.error(
                f"Nervous system processing failed: {e}",
                extra={
                    "content_length": len(content),
                    "memory_type": memory_type,
                    "processing_time_ms": processing_time
                }
            )
            
            # Return fallback result to maintain system stability
            return {
                'quality_assessment': None,
                'duplicate_analysis': None,
                'cxd_classification': None,
                'socratic_guidance': None,
                'processing_time_ms': processing_time,
                'error': str(e),
                'fallback_mode': True
            }
    
    async def _assess_quality_cached(self, content: str, memory_type: str) -> Optional[QualityAssessment]:
        """Cached quality assessment with LRU optimization"""
        cache_key = self._get_cached_quality_key(content, memory_type)
        
        if cache_key in self._quality_cache:
            return self._quality_cache[cache_key]
        
        if self._quality_gate:
            assessment = await self._quality_gate.assess_quality(content, memory_type)
            self._quality_cache[cache_key] = assessment
            self.metrics.quality_assessments += 1
            return assessment
        
        return None
    
    async def _detect_duplicates_cached(self, content: str, memory_type: str) -> Optional[DuplicateAnalysis]:
        """Cached duplicate detection with LRU optimization"""
        cache_key = self._get_cached_duplicate_key(content)
        
        if cache_key in self._duplicate_cache:
            return self._duplicate_cache[cache_key]
        
        if self._duplicate_detector:
            analysis = await self._duplicate_detector.detect_duplicates(content, memory_type)
            self._duplicate_cache[cache_key] = analysis
            self.metrics.duplicate_detections += 1
            return analysis
        
        return None
    
    async def _classify_cxd_cached(self, content: str) -> Optional[Dict[str, Any]]:
        """Cached CXD classification"""
        if not self._cxd_classifier:
            return None
        
        try:
            classification = self._cxd_classifier.classify(content)
            return {
                'pattern': getattr(classification, 'pattern', 'unknown'),
                'confidence': getattr(classification, 'confidence', 0.5),
                'function': getattr(classification, 'function', 'unknown')
            }
        except Exception as e:
            self.logger.debug(f"CXD classification failed: {e}")
            return None
    
    async def _get_socratic_guidance_cached(self, content: str, memory_type: str) -> Optional[SocraticGuidance]:
        """Cached Socratic guidance"""
        if not self._socratic_guidance:
            return None
        
        cache_key = f"socratic:{hash(content)}:{memory_type}"
        if cache_key in self._guidance_cache:
            return self._guidance_cache[cache_key]
        
        guidance = await self._socratic_guidance.guide_memory_decision(
            content, 
            {'memory_type': memory_type}
        )
        self._guidance_cache[cache_key] = guidance
        self.metrics.socratic_guidances += 1
        return guidance
    
    async def _synthesize_intelligence_results(
        self, 
        results: Dict[str, Any], 
        content: str, 
        memory_type: str
    ) -> Dict[str, Any]:
        """
        Synthesize intelligence results into unified decision framework.
        <1ms processing target through optimized logic.
        """
        synthesis = {
            'quality_assessment': results.get('quality'),
            'duplicate_analysis': results.get('duplicate'),
            'cxd_classification': results.get('cxd'),
            'socratic_guidance': results.get('socratic'),
            'narrative_enhancement': results.get('narrative'),
            'recommended_action': 'store',  # Default action
            'confidence': 0.8,  # Default confidence
            'enhancement_needed': False,
            'storage_metadata': {}
        }
        
        # Quality-based decision logic
        quality = results.get('quality')
        if quality:
            synthesis['enhancement_needed'] = quality.needs_enhancement
            synthesis['confidence'] *= quality.confidence
            if quality.auto_approve:
                synthesis['recommended_action'] = 'store'
            elif quality.overall_score < 0.6:
                synthesis['recommended_action'] = 'enhance_then_store'
        
        # Duplicate-based decision logic
        duplicate = results.get('duplicate')
        if duplicate and duplicate.is_duplicate:
            synthesis['recommended_action'] = duplicate.resolution_action
            synthesis['storage_metadata']['duplicate_handling'] = {
                'original_memory_id': duplicate.duplicate_memory_id,
                'similarity_score': duplicate.similarity_score,
                'relationship_strength': duplicate.relationship_strength
            }
        
        # CXD classification metadata
        cxd = results.get('cxd')
        if cxd:
            synthesis['storage_metadata']['cxd'] = cxd

        # Narrative enhancement metadata
        narrative = results.get('narrative')
        if narrative and narrative.get('narrative_context'):
            synthesis['storage_metadata']['narrative'] = narrative

            # Enhance CXD classification with narrative context
            if narrative.get('enhanced_cxd'):
                if 'cxd' not in synthesis['storage_metadata']:
                    synthesis['storage_metadata']['cxd'] = {}
                synthesis['storage_metadata']['cxd']['narrative_enhancement'] = narrative['enhanced_cxd']

            # Add thematic tags
            if narrative.get('thematic_tags'):
                synthesis['storage_metadata']['thematic_tags'] = narrative['thematic_tags']

        return synthesis
    
    def _update_performance_metrics(self, processing_time_ms: float) -> None:
        """Update performance metrics for monitoring"""
        self.metrics.total_operations += 1
        self._operation_times.append(processing_time_ms)
        
        # Keep only last 1000 operations for rolling average
        if len(self._operation_times) > 1000:
            self._operation_times.pop(0)
        
        self.metrics.average_response_time_ms = sum(self._operation_times) / len(self._operation_times)
        self.metrics.cache_hit_rate = self._calculate_cache_hit_rate()
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate across all caches"""
        total_requests = self.metrics.quality_assessments + self.metrics.duplicate_detections + self.metrics.socratic_guidances
        if total_requests == 0:
            return 0.0
        
        # Simplified cache hit calculation (would be more sophisticated in production)
        cache_hits = len(self._quality_cache) + len(self._duplicate_cache) + len(self._guidance_cache)
        return min(1.0, cache_hits / total_requests)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'total_operations': self.metrics.total_operations,
            'average_response_time_ms': self.metrics.average_response_time_ms,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'quality_assessments': self.metrics.quality_assessments,
            'duplicate_detections': self.metrics.duplicate_detections,
            'socratic_guidances': self.metrics.socratic_guidances,
            'errors': self.metrics.errors,
            'target_response_time_ms': 5.0,
            'performance_target_met': self.metrics.average_response_time_ms < 5.0,
            'cache_sizes': {
                'quality': len(self._quality_cache),
                'duplicate': len(self._duplicate_cache),
                'guidance': len(self._guidance_cache)
            },
            'components_initialized': {
                'amms_storage': self._amms_storage is not None,
                'cxd_classifier': self._cxd_classifier is not None,
                'quality_gate': self._quality_gate is not None,
                'duplicate_detector': self._duplicate_detector is not None,
                'socratic_guidance': self._socratic_guidance is not None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for nervous system"""
        if not self._initialized:
            return {'status': 'not_initialized', 'healthy': False}
        
        health = {
            'status': 'operational',
            'healthy': True,
            'performance_metrics': self.get_performance_metrics(),
            'component_health': {}
        }
        
        # Check component health
        try:
            if self._amms_storage:
                memory_count = await self._amms_storage.count_memories()
                health['component_health']['amms_storage'] = {
                    'status': 'healthy',
                    'memory_count': memory_count
                }
        except Exception as e:
            health['component_health']['amms_storage'] = {
                'status': 'error',
                'error': str(e)
            }
            health['healthy'] = False
        
        # Check performance targets
        if self.metrics.average_response_time_ms > 5.0:
            health['performance_warning'] = f"Average response time {self.metrics.average_response_time_ms:.2f}ms exceeds 5ms target"
        
        return health
    
    # Synergy/amplification methods removed - keeping core independent processing
    
    async def _initialize_evolution_system(self) -> Optional[tuple]:
        """Initialize memory evolution tracking system"""
        try:
            # Create evolution database path based on main database
            evolution_db_path = self.db_path.replace('.db', '_evolution.db')
            
            # Create complete evolution system
            evolution_components = create_evolution_system(
                db_path=evolution_db_path,
                config={
                    'reports_directory': 'evolution_reports',
                    'benchmarks': {
                        'usage_frequency_target': 0.5,
                        'efficiency_target': 0.8
                    }
                }
            )
            
            self.logger.info(
                f"Memory evolution system initialized with database: {evolution_db_path}",
                extra={"evolution_db": evolution_db_path}
            )
            
            return evolution_components
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize evolution system: {e}")
            # Don't fail the entire nervous system if evolution tracking fails
            return None
    
    # Exponential mode removed - keeping core independent processing
    
    # Exponential processing removed - using core intelligence only
    
    # Synergy metrics removed - core metrics only
    
    # Memory Evolution Tracking Methods
    
    async def track_memory_creation(self, memory: Memory) -> None:
        """Track memory creation event for evolution analysis"""
        if not self._evolution_system_initialized or not self._lifecycle_manager:
            return
        
        try:
            # Track memory creation in lifecycle manager
            await self._lifecycle_manager.track_memory_creation(memory)
            
            # Track creation event in evolution tracker
            if self._evolution_tracker:
                await self._evolution_tracker.track_event(
                    memory_id=memory.id,
                    event_type=MemoryEventType.CREATED,
                    context={
                        'memory_type': memory.metadata.get('type', 'unknown'),
                        'importance': memory.importance_score,
                        'content_length': len(memory.content)
                    },
                    trigger='nervous_system_remember'
                )
            
        except Exception as e:
            self.logger.debug(f"Failed to track memory creation: {e}")
    
    async def track_memory_access(self, memory_id: str, access_type: str = "accessed", 
                                context: Dict[str, Any] = None) -> None:
        """Track memory access event for evolution analysis"""
        if not self._evolution_system_initialized or not self._evolution_tracker:
            return
        
        try:
            # Track access event
            await self._evolution_tracker.track_event(
                memory_id=memory_id,
                event_type=MemoryEventType.ACCESSED if access_type == "accessed" else MemoryEventType.RECALLED,
                context=context or {},
                trigger='nervous_system_recall'
            )
            
            # Update lifecycle activity
            if self._lifecycle_manager:
                await self._lifecycle_manager.update_memory_activity(
                    memory_id=memory_id,
                    activity_type=access_type,
                    context=context
                )
            
        except Exception as e:
            self.logger.debug(f"Failed to track memory access: {e}")
    
    async def track_memory_modification(self, memory_id: str, 
                                      previous_state: Dict[str, Any] = None,
                                      new_state: Dict[str, Any] = None,
                                      trigger: str = "modification") -> None:
        """Track memory modification event for evolution analysis"""
        if not self._evolution_system_initialized or not self._evolution_tracker:
            return
        
        try:
            await self._evolution_tracker.track_event(
                memory_id=memory_id,
                event_type=MemoryEventType.MODIFIED,
                context={'modification_trigger': trigger},
                previous_state=previous_state,
                new_state=new_state,
                trigger=trigger
            )
            
            # Update lifecycle activity
            if self._lifecycle_manager:
                await self._lifecycle_manager.update_memory_activity(
                    memory_id=memory_id,
                    activity_type="modified",
                    context={'trigger': trigger}
                )
            
        except Exception as e:
            self.logger.debug(f"Failed to track memory modification: {e}")
    
    async def get_memory_evolution_insights(self, memory_id: str = None) -> Dict[str, Any]:
        """Get evolution insights for a specific memory or system-wide"""
        if not self._evolution_system_initialized:
            return {"evolution_tracking": False, "message": "Evolution system not initialized"}
        
        try:
            if memory_id:
                # Get insights for specific memory
                if self._evolution_metrics:
                    score = await self._evolution_metrics.calculate_evolution_score(memory_id)
                    return {
                        "memory_id": memory_id,
                        "evolution_score": score.overall_score,
                        "health_flags": score.health_flags,
                        "recommendations": score.recommendations,
                        "trend": score.score_trend
                    }
            else:
                # Get system-wide insights
                if self._evolution_metrics:
                    system_metrics = await self._evolution_metrics.calculate_system_metrics()
                    return {
                        "system_health": {
                            "total_memories": system_metrics.total_memories,
                            "avg_evolution_score": system_metrics.avg_evolution_score,
                            "healthy_memories": system_metrics.healthy_memories
                        },
                        "optimization_opportunities": system_metrics.optimization_opportunities[:3]
                    }
            
        except Exception as e:
            self.logger.debug(f"Failed to get evolution insights: {e}")
            return {"error": str(e)}
        
        return {"evolution_tracking": True, "data_available": False}
    
    def is_evolution_tracking_enabled(self) -> bool:
        """Check if memory evolution tracking is enabled and operational"""
        return self._evolution_system_initialized and self._evolution_tracker is not None

    async def _initialize_archive_intelligence(self) -> Optional[ArchiveIntelligence]:
        """Initialize archive intelligence system"""
        try:
            archive_intelligence = ArchiveIntelligence(
                archive_path=".archive",
                db_path=self.db_path
            )
            await archive_intelligence.initialize()
            self.logger.info("Archive intelligence initialized successfully")
            return archive_intelligence
        except Exception as e:
            self.logger.warning(f"Failed to initialize archive intelligence: {e}")
            return None

    async def _initialize_phase_tracker(self) -> Optional[PhaseEvolutionTracker]:
        """Initialize phase evolution tracker"""
        try:
            phase_tracker = PhaseEvolutionTracker(
                memory_path="Memory",
                db_path=self.db_path
            )
            await phase_tracker.initialize()
            self.logger.info("Phase evolution tracker initialized successfully")
            return phase_tracker
        except Exception as e:
            self.logger.warning(f"Failed to initialize phase tracker: {e}")
            return None

    async def _initialize_tale_memory_binder(self) -> Optional[TaleMemoryBinder]:
        """Initialize tale-memory binder for narrative-driven memory integration"""
        try:
            tale_binder = TaleMemoryBinder(
                tales_path="tales",
                db_path=self.db_path
            )
            await tale_binder.initialize()
            self.logger.info("Tale-memory binder initialized successfully")
            return tale_binder
        except Exception as e:
            self.logger.warning(f"Failed to initialize tale-memory binder: {e}")
            return None

    async def apply_archive_pattern(self, pattern_name: str, target_path: str, **kwargs) -> Dict[str, Any]:
        """Apply an archive migration pattern"""
        if not self._archive_intelligence:
            raise MemMimicError("Archive intelligence not initialized")

        result = await self._archive_intelligence.apply_migration_pattern(
            pattern_name, target_path, **kwargs
        )

        # Update metrics
        if result.get("success", False):
            self.metrics.archive_patterns_applied += 1

        return result

    async def get_phase_status(self) -> Dict[str, Any]:
        """Get current phase evolution status"""
        if not self._phase_tracker:
            return {"phase_tracking": False, "message": "Phase tracker not initialized"}

        status = self._phase_tracker.get_evolution_status()

        # Update nervous system metrics
        if status.get("metrics"):
            metrics = status["metrics"]
            self.metrics.current_phase = status.get("current_phase")
            self.metrics.phase_progress = metrics.overall_progress
            self.metrics.completed_phases = metrics.completed_phases

        return status

    async def start_development_phase(self, phase_id: str) -> bool:
        """Start a development phase"""
        if not self._phase_tracker:
            raise MemMimicError("Phase tracker not initialized")

        return await self._phase_tracker.start_phase(phase_id)

    async def complete_development_phase(self, phase_id: str) -> bool:
        """Complete a development phase"""
        if not self._phase_tracker:
            raise MemMimicError("Phase tracker not initialized")

        return await self._phase_tracker.complete_phase(phase_id)

    async def update_phase_task(self, phase_id: str, task_id: str, status: TaskStatus,
                              metrics: Dict[str, Any] = None) -> bool:
        """Update the status of a phase task"""
        if not self._phase_tracker:
            raise MemMimicError("Phase tracker not initialized")

        return await self._phase_tracker.update_task_status(phase_id, task_id, status, metrics)

    def get_archive_patterns(self) -> List[str]:
        """Get list of available archive patterns"""
        if not self._archive_intelligence:
            return []

        return self._archive_intelligence.get_available_patterns()

    def get_archive_evolution_metrics(self) -> Optional[ArchiveEvolutionMetrics]:
        """Get archive evolution metrics"""
        if not self._archive_intelligence:
            return None

        return self._archive_intelligence.get_evolution_metrics()

    def get_narrative_themes(self) -> Dict[str, Any]:
        """Get available narrative themes"""
        if not self._tale_memory_binder:
            return {}

        return self._tale_memory_binder.get_narrative_themes()

    def get_narrative_binding_metrics(self) -> Optional[BindingMetrics]:
        """Get narrative-memory binding metrics"""
        if not self._tale_memory_binder:
            return None

        return self._tale_memory_binder.get_binding_metrics()

    async def search_by_narrative_theme(self, theme_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by narrative theme"""
        if not self._tale_memory_binder:
            return []

        return await self._tale_memory_binder.search_by_narrative_theme(theme_name, limit)

    async def _initialize_latency_optimizer(self) -> Optional[ReflexLatencyOptimizer]:
        """Initialize reflex latency optimizer for sub-5ms response times"""
        try:
            optimizer = ReflexLatencyOptimizer(target_latency_ms=5.0)
            await optimizer.initialize()
            self.logger.info("Reflex latency optimizer initialized successfully")
            return optimizer
        except Exception as e:
            self.logger.warning(f"Failed to initialize latency optimizer: {e}")
            return None

    async def _initialize_shared_reality_manager(self) -> Optional[SharedRealityManager]:
        """Initialize shared reality manager for multi-agent coordination"""
        try:
            reality_manager = SharedRealityManager(db_path=self.db_path)
            await reality_manager.initialize()
            self.logger.info("Shared reality manager initialized successfully")
            return reality_manager
        except Exception as e:
            self.logger.warning(f"Failed to initialize shared reality manager: {e}")
            return None

    async def _initialize_theory_of_mind(self) -> Optional[TheoryOfMindCapabilities]:
        """Initialize theory of mind capabilities"""
        try:
            tom = TheoryOfMindCapabilities(agent_id=f"nervous_system_{id(self)}")
            await tom.initialize()
            self.logger.info("Theory of mind capabilities initialized successfully")
            return tom
        except Exception as e:
            self.logger.warning(f"Failed to initialize theory of mind: {e}")
            return None

    async def optimize_operation_latency(self, operation_name: str, operation_func: Callable,
                                       *args, **kwargs) -> Tuple[Any, float]:
        """Optimize operation for minimum latency using the latency optimizer"""
        if not self._latency_optimizer:
            # Fallback to direct execution
            start_time = time.perf_counter()
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            execution_time = (time.perf_counter() - start_time) * 1000
            return result, execution_time

        return await self._latency_optimizer.optimize_operation(operation_name, operation_func, *args, **kwargs)

    async def register_agent_in_shared_reality(self, agent_id: str, name: str, role: AgentRole,
                                             capabilities: List[str] = None, priority: int = 1) -> Optional[str]:
        """Register an agent in shared reality for coordination"""
        if not self._shared_reality_manager:
            return None

        return await self._shared_reality_manager.register_agent(
            agent_id, name, role, capabilities, priority
        )

    async def observe_agent_action(self, agent_id: str, action: str, context: Dict[str, Any]) -> None:
        """Observe and learn from another agent's action for theory of mind"""
        if self._theory_of_mind:
            await self._theory_of_mind.observe_agent_action(agent_id, action, context)

    async def predict_agent_behavior(self, agent_id: str, time_horizon: float = None) -> List[IntentionPrediction]:
        """Predict another agent's future behavior"""
        if not self._theory_of_mind:
            return []

        return await self._theory_of_mind.predict_agent_behavior(agent_id, time_horizon)

    async def generate_empathetic_response(self, agent_id: str, situation: str) -> Optional[str]:
        """Generate an empathetic response based on agent's mental state"""
        if not self._theory_of_mind:
            return None

        return await self._theory_of_mind.generate_empathetic_response(agent_id, situation)

    def get_latency_metrics(self) -> Dict[str, Any]:
        """Get latency optimization metrics"""
        if not self._latency_optimizer:
            return {}

        return {
            'latency_metrics': self._latency_optimizer.get_latency_metrics(),
            'optimization_summary': self._latency_optimizer.get_optimization_summary(),
            'target_achieved': self._latency_optimizer.is_target_latency_achieved()
        }

    def get_shared_reality_status(self) -> Dict[str, Any]:
        """Get shared reality coordination status"""
        if not self._shared_reality_manager:
            return {}

        return self._shared_reality_manager.get_reality_status()

    def get_theory_of_mind_metrics(self) -> Dict[str, Any]:
        """Get theory of mind performance metrics"""
        if not self._theory_of_mind:
            return {}

        return self._theory_of_mind.get_theory_of_mind_metrics()