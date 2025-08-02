"""
Memory Evolution Module

Comprehensive memory evolution tracking, analysis, and optimization system.
Provides intelligent insights into memory usage patterns, lifecycle management,
and system optimization opportunities.
"""

from .memory_evolution_tracker import (
    MemoryEvolutionTracker, 
    MemoryEventType, 
    MemoryEvent,
    MemoryUsageStats,
    MemoryEvolutionPattern
)

from .memory_lifecycle_manager import (
    MemoryLifecycleManager,
    LifecycleStage,
    LifecycleTransition,
    LifecyclePolicy,
    MemoryLifecycleStatus
)

from .memory_usage_analytics import (
    MemoryUsageAnalytics,
    UsagePatternType,
    UsagePattern,
    MemoryCluster,
    UsageMetrics
)

from .memory_evolution_metrics import (
    MemoryEvolutionMetrics,
    MetricCategory,
    EvolutionScore,
    SystemMetrics,
    EvolutionBenchmark
)

from .memory_evolution_reports import (
    MemoryEvolutionReporter,
    ReportType,
    ReportFormat,
    ReportConfiguration,
    VisualizationData
)

__all__ = [
    # Core tracking
    'MemoryEvolutionTracker',
    'MemoryEventType',
    'MemoryEvent',
    'MemoryUsageStats',
    'MemoryEvolutionPattern',
    
    # Lifecycle management
    'MemoryLifecycleManager',
    'LifecycleStage',
    'LifecycleTransition',
    'LifecyclePolicy',
    'MemoryLifecycleStatus',
    
    # Usage analytics
    'MemoryUsageAnalytics',
    'UsagePatternType',
    'UsagePattern',
    'MemoryCluster',
    'UsageMetrics',
    
    # Metrics and scoring
    'MemoryEvolutionMetrics',
    'MetricCategory',
    'EvolutionScore',
    'SystemMetrics',
    'EvolutionBenchmark',
    
    # Reporting
    'MemoryEvolutionReporter',
    'ReportType',
    'ReportFormat',
    'ReportConfiguration',
    'VisualizationData',
    
    # Convenience functions
    'create_evolution_system',
    'create_basic_evolution_tracker'
]


def create_evolution_system(db_path: str = "memory_evolution.db", 
                          config: dict = None) -> tuple:
    """
    Create a complete memory evolution system with all components integrated.
    
    Args:
        db_path: Path to the evolution database
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (tracker, lifecycle_manager, usage_analytics, metrics, reporter)
    """
    from ..nervous_system.core import NervousSystemCore
    
    # Create core tracker
    tracker = MemoryEvolutionTracker(db_path)
    
    # Create lifecycle manager
    lifecycle_manager = MemoryLifecycleManager(
        tracker, 
        policies_config=config.get('lifecycle_policies') if config else None
    )
    
    # Create usage analytics
    usage_analytics = MemoryUsageAnalytics(tracker, lifecycle_manager)
    
    # Create metrics system
    benchmarks = EvolutionBenchmark()
    if config and 'benchmarks' in config:
        for key, value in config['benchmarks'].items():
            if hasattr(benchmarks, key):
                setattr(benchmarks, key, value)
    
    metrics = MemoryEvolutionMetrics(
        tracker, lifecycle_manager, usage_analytics, benchmarks
    )
    
    # Create reporter
    reporter = MemoryEvolutionReporter(
        tracker, lifecycle_manager, usage_analytics, metrics,
        output_directory=config.get('reports_directory', 'reports') if config else 'reports'
    )
    
    return tracker, lifecycle_manager, usage_analytics, metrics, reporter


def create_basic_evolution_tracker(db_path: str = "memory_evolution.db") -> MemoryEvolutionTracker:
    """
    Create a basic memory evolution tracker for simple usage tracking.
    
    Args:
        db_path: Path to the evolution database
        
    Returns:
        MemoryEvolutionTracker instance
    """
    return MemoryEvolutionTracker(db_path)


# Version information
__version__ = "1.0.0"
__author__ = "MemMimic Evolution System"
__description__ = "Comprehensive memory evolution tracking and analysis system"