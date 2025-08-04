"""
DSPy Consciousness Optimization Framework

Advanced DSPy integration for consciousness-aware tool selection and optimization
with biological reflex compatibility and safety-first design.

Features:
- Hybrid consciousness processing with biological reflex speeds (<5ms)
- MIPROv2 optimization pipeline for consciousness patterns
- Intelligent documentation context system with just-in-time fetching
- Performance monitoring and circuit breaker protection
- Comprehensive proof-of-concept validation framework
- Cross-pattern learning and knowledge transfer
"""

from .config import DSPyConfig, create_default_config, create_production_config
from .circuit_breaker import DSPyCircuitBreaker, circuit_breaker_manager
from .signatures import (
    ConsciousnessToolSelection,
    MemorySearchOptimization,
    PatternRecognition,
    BiologicalReflexOptimization
)
from .hybrid_processor import (
    HybridConsciousnessProcessor,
    ProcessingMode,
    ProcessingRequest,
    ProcessingResponse
)
from .performance_monitor import DSPyPerformanceMonitor, get_performance_monitor, initialize_performance_monitor
from .docs_context import (
    IntelligentDocsContextSystem,
    DocumentationContext,
    DocumentationSource,
    get_docs_context_system,
    initialize_docs_context_system
)
from .miprov2_pipeline import (
    ConsciousnessPatternOptimizer,
    OptimizationTarget,
    OptimizationResult,
    OptimizationLevel,
    MIPROv2Configuration,
    get_miprov2_optimizer,
    initialize_miprov2_optimizer
)
from .poc_implementation import DSPyPoCIntegration, DSPyPoCValidator

__all__ = [
    # Configuration
    "DSPyConfig",
    "create_default_config", 
    "create_production_config",
    
    # Circuit Breaker
    "DSPyCircuitBreaker",
    "circuit_breaker_manager",
    
    # Signatures
    "ConsciousnessToolSelection",
    "MemorySearchOptimization", 
    "PatternRecognition",
    "BiologicalReflexOptimization",
    
    # Hybrid Processor
    "HybridConsciousnessProcessor",
    "ProcessingMode",
    "ProcessingRequest", 
    "ProcessingResponse",
    
    # Performance Monitor
    "DSPyPerformanceMonitor",
    "get_performance_monitor",
    "initialize_performance_monitor",
    
    # Documentation Context System
    "IntelligentDocsContextSystem",
    "DocumentationContext",
    "DocumentationSource",
    "get_docs_context_system",
    "initialize_docs_context_system",
    
    # MIPROv2 Optimization Pipeline
    "ConsciousnessPatternOptimizer",
    "OptimizationTarget",
    "OptimizationResult",
    "OptimizationLevel",
    "MIPROv2Configuration",
    "get_miprov2_optimizer",
    "initialize_miprov2_optimizer",
    
    # Proof of Concept
    "DSPyPoCIntegration",
    "DSPyPoCValidator"
]

__version__ = "2.0.0"