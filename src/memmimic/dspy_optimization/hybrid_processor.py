"""
Hybrid DSPy Consciousness Processor

Core hybrid architecture that routes consciousness operations between fast biological
reflexes and DSPy-optimized processing based on complexity and urgency analysis.
"""

import time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from .config import DSPyConfig
from .circuit_breaker import DSPyCircuitBreaker, circuit_breaker_manager
from .signatures import (
    ConsciousnessToolSelection,
    MemorySearchOptimization,
    PatternRecognition,
    BiologicalReflexOptimization
)
from .docs_context import IntelligentDocsContextSystem, get_docs_context_system
# from ..nervous_system.core import NervousSystemCore  # Will integrate with existing system
from ..errors import MemMimicError, with_error_context, get_error_logger

logger = get_error_logger(__name__)

class ProcessingMode(Enum):
    """Processing mode for consciousness operations"""
    FAST_PATH = "fast_path"          # <5ms biological reflex
    OPTIMIZATION_PATH = "optimization_path"  # DSPy-enhanced processing
    HYBRID = "hybrid"                # Intelligent routing

@dataclass
class ProcessingRequest:
    """Request for consciousness processing"""
    operation_type: str
    context: Dict[str, Any]
    urgency_level: str  # "immediate", "normal", "complex"
    consciousness_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ProcessingResponse:
    """Response from consciousness processing"""
    result: Any
    processing_mode: ProcessingMode
    response_time_ms: float
    confidence_score: float
    optimization_applied: bool = False
    fallback_used: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConsciousnessPatternClassifier:
    """
    Classifies consciousness requests to determine optimal processing path.
    """
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        self.biological_reflex_keywords = {
            "remember", "recall", "think", "analyze", "status", "quick",
            "immediate", "urgent", "fast", "reflex", "instant"
        }
        self.complex_operation_keywords = {
            "optimize", "learn", "adapt", "complex", "analyze deeply",
            "comprehensive", "detailed", "multi-step", "workflow"
        }
    
    def classify_urgency(self, request: ProcessingRequest) -> str:
        """
        Classify request urgency level.
        
        Returns:
            "immediate" for <5ms biological reflexes
            "normal" for <50ms consciousness operations  
            "complex" for <200ms optimization operations
        """
        # Check for explicit urgency indicators
        if request.urgency_level:
            return request.urgency_level
        
        # Analyze context for urgency indicators
        context_text = str(request.context).lower()
        
        # Immediate processing (biological reflex)
        if any(keyword in context_text for keyword in self.biological_reflex_keywords):
            return "immediate"
        
        # Complex processing (optimization path)
        if any(keyword in context_text for keyword in self.complex_operation_keywords):
            return "complex"
        
        # Default to normal processing
        return "normal"
    
    def should_use_fast_path(self, request: ProcessingRequest) -> bool:
        """Determine if request should use fast biological reflex path"""
        urgency = self.classify_urgency(request)
        
        # Force fast path for immediate urgency
        if urgency == "immediate":
            return True
        
        # Check if fast path is disabled
        if not self.config.integration.enable_biological_reflex_optimization:
            return False
        
        # Simple operations that don't benefit from optimization
        simple_operations = {"status", "ping", "health_check", "quick_recall"}
        if request.operation_type in simple_operations:
            return True
        
        return False
    
    def should_use_optimization_path(self, request: ProcessingRequest) -> bool:
        """Determine if request should use DSPy optimization path"""
        urgency = self.classify_urgency(request)
        
        # Complex operations benefit from optimization
        if urgency == "complex":
            return True
        
        # Check if optimization is enabled
        if not self.config.integration.enable_dspy_optimization:
            return False
        
        # Operations that benefit from optimization
        optimization_operations = {
            "tool_selection", "memory_search", "pattern_recognition",
            "tale_generation", "consciousness_analysis"
        }
        if request.operation_type in optimization_operations:
            return True
        
        return False

class FastPathProcessor:
    """
    Fast path processor for biological reflex operations (<5ms target).
    """
    
    def __init__(self, nervous_system_core):
        self.nervous_system = nervous_system_core
        
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process request using fast biological reflex path"""
        start_time = time.time()
        
        try:
            # Direct biological reflex processing
            result = await self.nervous_system.process_biological_reflex(
                request.operation_type,
                request.context
            )
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return ProcessingResponse(
                result=result,
                processing_mode=ProcessingMode.FAST_PATH,
                response_time_ms=response_time,
                confidence_score=0.9,  # High confidence for direct reflexes
                optimization_applied=False
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Fast path processing failed: {e}")
            
            return ProcessingResponse(
                result=None,
                processing_mode=ProcessingMode.FAST_PATH,
                response_time_ms=response_time,
                confidence_score=0.0,
                error=str(e)
            )

class OptimizationPathProcessor:
    """
    Optimization path processor using DSPy for enhanced consciousness operations.
    """
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        self.circuit_breaker = circuit_breaker_manager.get_breaker(
            "dspy_optimization",
            "optimization",
            fallback_handler=self._fallback_processing
        )
        
        # Initialize DSPy modules (lazy initialization)
        self._dspy_modules = {}
    
    def _get_dspy_module(self, operation_type: str):
        """Get or create DSPy module for operation type"""
        if operation_type not in self._dspy_modules:
            try:
                import dspy
                
                # Configure DSPy with model settings
                lm = dspy.LM(
                    self.config.model.primary_model,
                    temperature=self.config.model.temperature,
                    max_tokens=self.config.model.max_tokens
                )
                dspy.configure(lm=lm)
                
                # Create appropriate DSPy module based on operation type
                if operation_type == "tool_selection":
                    self._dspy_modules[operation_type] = dspy.ChainOfThought(ConsciousnessToolSelection)
                elif operation_type == "memory_search":
                    self._dspy_modules[operation_type] = dspy.ChainOfThought(MemorySearchOptimization)
                elif operation_type == "pattern_recognition":
                    self._dspy_modules[operation_type] = dspy.Predict(PatternRecognition)
                else:
                    # Default to consciousness tool selection
                    self._dspy_modules[operation_type] = dspy.ChainOfThought(ConsciousnessToolSelection)
                    
            except ImportError:
                logger.error("DSPy not available, falling back to basic processing")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize DSPy module: {e}")
                return None
        
        return self._dspy_modules[operation_type]
    
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process request using DSPy optimization path"""
        start_time = time.time()
        
        try:
            # Get DSPy module for operation
            dspy_module = self._get_dspy_module(request.operation_type)
            if not dspy_module:
                raise Exception("DSPy module not available")
            
            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call(
                self._execute_dspy_optimization,
                dspy_module,
                request
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return ProcessingResponse(
                result=result,
                processing_mode=ProcessingMode.OPTIMIZATION_PATH,
                response_time_ms=response_time,
                confidence_score=getattr(result, 'confidence_score', 0.8),
                optimization_applied=True
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Optimization path processing failed: {e}")
            
            return ProcessingResponse(
                result=None,
                processing_mode=ProcessingMode.OPTIMIZATION_PATH,
                response_time_ms=response_time,
                confidence_score=0.0,
                error=str(e)
            )
    
    async def _execute_dspy_optimization(self, dspy_module, request: ProcessingRequest):
        """Execute DSPy optimization with proper async handling"""
        # Convert sync DSPy call to async
        loop = asyncio.get_event_loop()
        
        # Prepare DSPy inputs based on request type
        if request.operation_type == "tool_selection":
            return await loop.run_in_executor(
                None,
                dspy_module,
                request.context.get("context", ""),
                request.consciousness_patterns,
                request.context.get("available_tools", []),
                self.config.performance.biological_reflex_max_time
            )
        elif request.operation_type == "memory_search":
            return await loop.run_in_executor(
                None,
                dspy_module,
                request.context.get("query", ""),
                request.context,
                request.context.get("search_history", []),
                ["interaction", "milestone", "reflection", "technical"]
            )
        else:
            # Generic processing
            return await loop.run_in_executor(
                None,
                dspy_module,
                str(request.context)
            )
    
    async def _fallback_processing(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback processing when DSPy optimization fails"""
        logger.info("Using fallback processing for DSPy optimization failure")
        return {
            "result": "Basic processing result",
            "confidence_score": 0.5,
            "optimization_applied": False,
            "fallback_used": True
        }

class HybridConsciousnessProcessor:
    """
    Hybrid processor that intelligently routes consciousness operations between
    fast biological reflexes and DSPy-optimized processing.
    """
    
    def __init__(
        self,
        config: DSPyConfig,
        nervous_system_core
    ):
        self.config = config
        self.classifier = ConsciousnessPatternClassifier(config)
        self.fast_path = FastPathProcessor(nervous_system_core)
        self.optimization_path = OptimizationPathProcessor(config)
        
        # Initialize documentation context system
        self.docs_context = get_docs_context_system()
        if not self.docs_context:
            from .docs_context import initialize_docs_context_system
            self.docs_context = initialize_docs_context_system(config)
        
        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "fast_path_requests": 0,
            "optimization_path_requests": 0,
            "hybrid_routing_decisions": 0,
            "average_response_time": 0.0,
            "optimization_success_rate": 0.0,
            "documentation_context_requests": 0,
            "documentation_context_success_rate": 0.0
        }
    
    @with_error_context("hybrid_consciousness_processing")
    async def process_consciousness_request(
        self,
        operation_type: str,
        context: Dict[str, Any],
        urgency_level: Optional[str] = None,
        consciousness_patterns: Optional[List[str]] = None
    ) -> ProcessingResponse:
        """
        Process consciousness request with intelligent routing.
        
        Args:
            operation_type: Type of consciousness operation
            context: Context data for the operation
            urgency_level: Optional urgency override
            consciousness_patterns: Active consciousness patterns
            
        Returns:
            ProcessingResponse with results and metadata
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        # Create processing request
        request = ProcessingRequest(
            operation_type=operation_type,
            context=context,
            urgency_level=urgency_level or "normal",
            consciousness_patterns=consciousness_patterns or [],
            metadata={"processor_version": "1.0.0"}
        )
        
        # Determine processing path
        processing_mode = self._determine_processing_mode(request)
        
        try:
            # Route to appropriate processor
            if processing_mode == ProcessingMode.FAST_PATH:
                self.metrics["fast_path_requests"] += 1
                response = await self.fast_path.process(request)
                
            elif processing_mode == ProcessingMode.OPTIMIZATION_PATH:
                self.metrics["optimization_path_requests"] += 1
                response = await self.optimization_path.process(request)
                
                # Fallback to fast path if optimization fails and fallback is enabled
                if response.error and self.config.integration.fallback_strategy == "graceful":
                    logger.info("Optimization failed, falling back to fast path")
                    response = await self.fast_path.process(request)
                    response.fallback_used = True
                    
            else:  # HYBRID mode
                self.metrics["hybrid_routing_decisions"] += 1
                response = await self._hybrid_processing(request)
            
            # Update performance metrics
            self._update_metrics(response)
            
            # Add processing metadata
            response.metadata.update({
                "routing_decision": processing_mode.value,
                "total_time_ms": (time.time() - start_time) * 1000,
                "consciousness_patterns_used": len(request.consciousness_patterns)
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Consciousness processing failed: {e}")
            
            # Return error response
            return ProcessingResponse(
                result=None,
                processing_mode=processing_mode,
                response_time_ms=(time.time() - start_time) * 1000,
                confidence_score=0.0,
                error=str(e)
            )
    
    def _determine_processing_mode(self, request: ProcessingRequest) -> ProcessingMode:
        """Determine optimal processing mode for request"""
        
        # Check configuration mode
        if self.config.integration.optimization_mode == "fast_only":
            return ProcessingMode.FAST_PATH
        elif self.config.integration.optimization_mode == "optimization_only":
            return ProcessingMode.OPTIMIZATION_PATH
        
        # Intelligent routing based on request characteristics
        if self.classifier.should_use_fast_path(request):
            return ProcessingMode.FAST_PATH
        elif self.classifier.should_use_optimization_path(request):
            return ProcessingMode.OPTIMIZATION_PATH
        else:
            return ProcessingMode.HYBRID
    
    async def _hybrid_processing(self, request: ProcessingRequest) -> ProcessingResponse:
        """
        Hybrid processing that combines fast path and optimization path.
        """
        # Start with fast path for immediate response
        fast_response = await self.fast_path.process(request)
        
        # If fast path succeeds and meets confidence threshold, return it
        if (fast_response.confidence_score >= 0.8 and 
            fast_response.response_time_ms <= self.config.performance.biological_reflex_max_time):
            return fast_response
        
        # Otherwise, try optimization path for better quality
        try:
            optimization_response = await self.optimization_path.process(request)
            
            # Use optimization result if it's better
            if (optimization_response.confidence_score > fast_response.confidence_score and
                not optimization_response.error):
                optimization_response.metadata["hybrid_fallback_available"] = True
                return optimization_response
            
        except Exception as e:
            logger.warning(f"Optimization path failed in hybrid mode: {e}")
        
        # Fallback to fast path result
        fast_response.metadata["optimization_attempted"] = True
        return fast_response
    
    def _update_metrics(self, response: ProcessingResponse) -> None:
        """Update performance metrics"""
        # Update average response time
        if self.metrics["total_requests"] > 0:
            current_avg = self.metrics["average_response_time"]
            new_avg = (
                (current_avg * (self.metrics["total_requests"] - 1) + response.response_time_ms) /
                self.metrics["total_requests"]
            )
            self.metrics["average_response_time"] = new_avg
        
        # Update optimization success rate
        if response.optimization_applied:
            optimization_requests = self.metrics["optimization_path_requests"]
            if optimization_requests > 0:
                success_count = optimization_requests if not response.error else optimization_requests - 1
                self.metrics["optimization_success_rate"] = success_count / optimization_requests
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = {
            **self.metrics,
            "circuit_breaker_metrics": circuit_breaker_manager.get_all_metrics(),
            "configuration": {
                "optimization_mode": self.config.integration.optimization_mode,
                "biological_reflex_max_time": self.config.performance.biological_reflex_max_time,
                "optimization_enabled": self.config.integration.enable_dspy_optimization
            }
        }
        
        # Add documentation context metrics if available
        if self.docs_context:
            metrics["documentation_context_metrics"] = self.docs_context.get_performance_metrics()
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.metrics = {
            "total_requests": 0,
            "fast_path_requests": 0,
            "optimization_path_requests": 0,
            "hybrid_routing_decisions": 0,
            "average_response_time": 0.0,
            "optimization_success_rate": 0.0
        }
        
        # Reset circuit breaker metrics
        circuit_breaker_manager.reset_all()
    
    async def enrich_context_with_documentation(
        self,
        operation_type: str,
        context: Dict[str, Any],
        consciousness_patterns: List[str]
    ) -> Dict[str, Any]:
        """
        Enrich request context with relevant documentation.
        
        Args:
            operation_type: Type of consciousness operation
            context: Original context data
            consciousness_patterns: Active consciousness patterns
            
        Returns:
            Enhanced context with documentation
        """
        if not self.docs_context:
            return context
        
        try:
            self.metrics["documentation_context_requests"] += 1
            
            # Generate query from operation type and context
            query_parts = [operation_type]
            if "query" in context:
                query_parts.append(str(context["query"]))
            if "context" in context:
                query_parts.append(str(context["context"]))
            
            query = " ".join(query_parts)
            
            # Get documentation context
            doc_context = await self.docs_context.get_documentation_context(
                query=query,
                consciousness_patterns=consciousness_patterns,
                max_docs=3,
                relevance_threshold=0.6
            )
            
            # Enrich the context
            enhanced_context = context.copy()
            enhanced_context["documentation_context"] = {
                "relevant_docs": doc_context.relevant_docs,
                "confidence_score": doc_context.confidence_score,
                "sources_used": doc_context.sources_used,
                "fetch_time_ms": doc_context.fetch_time_ms
            }
            
            if doc_context.relevant_docs:
                self.metrics["documentation_context_success_rate"] = (
                    (self.metrics["documentation_context_success_rate"] * 
                     (self.metrics["documentation_context_requests"] - 1) + 1.0) /
                    self.metrics["documentation_context_requests"]
                )
            
            logger.debug(f"Enhanced context with {len(doc_context.relevant_docs)} docs, confidence: {doc_context.confidence_score:.3f}")
            
            return enhanced_context
            
        except Exception as e:
            logger.warning(f"Failed to enrich context with documentation: {e}")
            return context