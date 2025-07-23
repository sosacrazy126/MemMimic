"""
Unified Enhanced Triggers Interface

Provides a single interface for all enhanced triggers with nervous system intelligence.
Maintains 100% backward compatibility while adding biological reflex processing.

This module bridges the enhanced triggers with the existing MCP system,
ensuring seamless integration and migration from external tools to internal intelligence.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union

from .remember import NervousSystemRemember
from .recall import NervousSystemRecall
from .think import NervousSystemThink
from .analyze import NervousSystemAnalyze
from ..core import NervousSystemCore
from ..db_initializer import ensure_database_ready
from ...errors import get_error_logger, with_error_context

class UnifiedEnhancedTriggers:
    """
    Unified interface for all enhanced triggers with nervous system intelligence.
    
    Provides a single point of access for all four biological reflex triggers
    while maintaining complete backward compatibility with existing MCP tools.
    """
    
    def __init__(self, db_path: str = "memmimic.db"):
        self.db_path = db_path
        self.logger = get_error_logger("unified_enhanced_triggers")
        
        # Shared nervous system core for all triggers
        self.nervous_system_core = NervousSystemCore(db_path)
        
        # Enhanced trigger instances
        self._remember_trigger = None
        self._recall_trigger = None
        self._think_trigger = None
        self._analyze_trigger = None
        
        # Performance tracking across all triggers
        self._total_operations = 0
        self._biological_reflex_achieved = 0
        self._initialization_time = 0
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all enhanced triggers with shared nervous system core"""
        if self._initialized:
            return
        
        start_time = time.perf_counter()
        
        try:
            # Ensure database is properly initialized first
            await ensure_database_ready(self.db_path)
            
            # Initialize shared nervous system core
            await self.nervous_system_core.initialize()
            
            # Initialize all enhanced triggers in parallel
            async with asyncio.TaskGroup() as tg:
                remember_task = tg.create_task(self._initialize_remember_trigger())
                recall_task = tg.create_task(self._initialize_recall_trigger())
                think_task = tg.create_task(self._initialize_think_trigger())
                analyze_task = tg.create_task(self._initialize_analyze_trigger())
            
            self._remember_trigger = remember_task.result()
            self._recall_trigger = recall_task.result()
            self._think_trigger = think_task.result()
            self._analyze_trigger = analyze_task.result()
            
            self._initialization_time = (time.perf_counter() - start_time) * 1000
            self._initialized = True
            
            self.logger.info(
                f"Unified enhanced triggers initialized successfully in {self._initialization_time:.2f}ms",
                extra={
                    "initialization_time_ms": self._initialization_time,
                    "triggers_count": 4,
                    "nervous_system_version": "2.0.0"
                }
            )
            
        except Exception as e:
            initialization_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(
                f"Failed to initialize unified enhanced triggers: {e}",
                extra={"initialization_time_ms": initialization_time}
            )
            raise
    
    async def _initialize_remember_trigger(self) -> NervousSystemRemember:
        """Initialize enhanced remember trigger"""
        remember_trigger = NervousSystemRemember(self.nervous_system_core)
        await remember_trigger.initialize()
        return remember_trigger
    
    async def _initialize_recall_trigger(self) -> NervousSystemRecall:
        """Initialize enhanced recall trigger"""
        recall_trigger = NervousSystemRecall(self.nervous_system_core)
        await recall_trigger.initialize()
        return recall_trigger
    
    async def _initialize_think_trigger(self) -> NervousSystemThink:
        """Initialize enhanced think trigger"""
        think_trigger = NervousSystemThink(self.nervous_system_core)
        await think_trigger.initialize()
        return think_trigger
    
    async def _initialize_analyze_trigger(self) -> NervousSystemAnalyze:
        """Initialize enhanced analyze trigger"""
        analyze_trigger = NervousSystemAnalyze(self.nervous_system_core)
        await analyze_trigger.initialize()
        return analyze_trigger
    
    # Enhanced Trigger Interfaces - Maintain 100% Backward Compatibility
    
    async def remember(self, content: str, memory_type: str = "interaction") -> Dict[str, Any]:
        """
        Enhanced remember with internal intelligence.
        INTERFACE: Exactly matches original remember(content: str, memory_type: str = "interaction")
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            result = await self._remember_trigger.remember(content, memory_type)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            self._track_operation_performance(processing_time)
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced remember failed: {e}")
            # Fallback to basic response format
            return {
                'status': 'error',
                'message': f"Enhanced remember failed: {str(e)}",
                'error': str(e),
                'nervous_system_version': '2.0.0'
            }
    
    async def recall_cxd(
        self, 
        query: str,
        function_filter: str = "ALL",
        limit: int = 5,
        db_name: Optional[str] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Enhanced recall with intelligent search.
        INTERFACE: Exactly matches original recall_cxd(query, function_filter="ALL", limit=5, db_name=None)
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            result = await self._recall_trigger.recall_cxd(query, function_filter, limit, db_name)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            self._track_operation_performance(processing_time)
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced recall failed: {e}")
            # Fallback to basic response format
            return {
                'error': str(e),
                'query': query,
                'nervous_system_version': '2.0.0'
            }
    
    async def think_with_memory(self, input_text: str) -> Dict[str, Any]:
        """
        Enhanced think with contextual intelligence.
        INTERFACE: Exactly matches original think_with_memory(input_text: str)
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            result = await self._think_trigger.think_with_memory(input_text)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            self._track_operation_performance(processing_time)
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced think failed: {e}")
            # Fallback to basic response format
            return {
                'status': 'error',
                'response': f"Enhanced think failed: {str(e)}",
                'error': str(e),
                'nervous_system_version': '2.0.0'
            }
    
    async def analyze_memory_patterns(self) -> Dict[str, Any]:
        """
        Enhanced analyze with pattern intelligence.
        INTERFACE: Exactly matches original analyze_memory_patterns()
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            result = await self._analyze_trigger.analyze_memory_patterns()
            processing_time = (time.perf_counter() - start_time) * 1000
            
            self._track_operation_performance(processing_time)
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced analyze failed: {e}")
            # Fallback to basic response format
            return {
                'status': 'error',
                'analysis': f"Enhanced analyze failed: {str(e)}",
                'error': str(e),
                'nervous_system_version': '2.0.0'
            }
    
    def _track_operation_performance(self, processing_time_ms: float) -> None:
        """Track operation performance for biological reflex metrics"""
        self._total_operations += 1
        
        if processing_time_ms < 5.0:
            self._biological_reflex_achieved += 1
    
    # System Health and Performance Monitoring
    
    async def unified_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all enhanced triggers"""
        if not self._initialized:
            return {'status': 'not_initialized', 'healthy': False}
        
        # Get health from all triggers
        health_checks = {}
        
        try:
            async with asyncio.TaskGroup() as tg:
                remember_health_task = tg.create_task(self._remember_trigger.health_check())
                recall_health_task = tg.create_task(self._recall_trigger.health_check())
                think_health_task = tg.create_task(self._think_trigger.health_check())
                analyze_health_task = tg.create_task(self._analyze_trigger.health_check())
            
            health_checks = {
                'remember': remember_health_task.result(),
                'recall': recall_health_task.result(),
                'think': think_health_task.result(),
                'analyze': analyze_health_task.result()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {'status': 'health_check_failed', 'healthy': False, 'error': str(e)}
        
        # Aggregate health status
        all_healthy = all(health['healthy'] for health in health_checks.values())
        
        unified_health = {
            'status': 'operational' if all_healthy else 'degraded',
            'healthy': all_healthy,
            'unified_triggers_initialized': self._initialized,
            'biological_reflex_rate': self._biological_reflex_achieved / max(1, self._total_operations),
            'total_operations': self._total_operations,
            'biological_reflex_operations': self._biological_reflex_achieved,
            'initialization_time_ms': self._initialization_time,
            'individual_trigger_health': health_checks,
            'nervous_system_version': '2.0.0'
        }
        
        return unified_health
    
    def get_unified_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics across all triggers"""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        # Collect metrics from all triggers
        individual_metrics = {
            'remember': self._remember_trigger.get_performance_metrics(),
            'recall': self._recall_trigger.get_performance_metrics(),
            'think': self._think_trigger.get_performance_metrics(),
            'analyze': self._analyze_trigger.get_performance_metrics()
        }
        
        # Calculate unified metrics
        total_operations = sum(metrics['total_operations'] if 'total_operations' in metrics else 0 
                             for metrics in individual_metrics.values())
        
        avg_processing_times = [
            metrics.get('average_processing_time_ms', 0) 
            for metrics in individual_metrics.values()
        ]
        unified_avg_time = sum(avg_processing_times) / len(avg_processing_times) if avg_processing_times else 0
        
        unified_metrics = {
            'unified_system_status': 'operational',
            'total_operations_all_triggers': total_operations,
            'biological_reflex_rate': self._biological_reflex_achieved / max(1, self._total_operations),
            'unified_average_processing_time_ms': unified_avg_time,
            'biological_reflex_target_met': unified_avg_time < 5.0,
            'initialization_time_ms': self._initialization_time,
            'individual_trigger_metrics': individual_metrics,
            'nervous_system_integration': {
                'shared_core': True,
                'parallel_processing': True,
                'intelligence_components': 3,  # Quality, Duplicate, Socratic
                'performance_optimization': True
            },
            'nervous_system_version': '2.0.0'
        }
        
        return unified_metrics
    
    async def demonstrate_biological_reflex(self) -> Dict[str, Any]:
        """
        Demonstrate biological reflex capabilities across all triggers.
        
        Tests each trigger with sample inputs to validate <5ms performance.
        """
        if not self._initialized:
            await self.initialize()
        
        demonstration_results = {
            'demonstration_timestamp': time.time(),
            'biological_reflex_target': '< 5ms per operation',
            'test_results': {},
            'overall_performance': {}
        }
        
        # Test remember reflex
        start_time = time.perf_counter()
        remember_result = await self.remember("Biological reflex test for remember trigger", "test")
        remember_time = (time.perf_counter() - start_time) * 1000
        
        demonstration_results['test_results']['remember'] = {
            'processing_time_ms': remember_time,
            'biological_reflex_achieved': remember_time < 5.0,
            'result_status': remember_result.get('status', 'unknown')
        }
        
        # Test recall reflex
        start_time = time.perf_counter()
        recall_result = await self.recall_cxd("biological reflex test", limit=3)
        recall_time = (time.perf_counter() - start_time) * 1000
        
        demonstration_results['test_results']['recall'] = {
            'processing_time_ms': recall_time,
            'biological_reflex_achieved': recall_time < 5.0,
            'results_count': len(recall_result) if isinstance(recall_result, list) else 1
        }
        
        # Test think reflex
        start_time = time.perf_counter()
        think_result = await self.think_with_memory("How do biological reflexes work in nervous systems?")
        think_time = (time.perf_counter() - start_time) * 1000
        
        demonstration_results['test_results']['think'] = {
            'processing_time_ms': think_time,
            'biological_reflex_achieved': think_time < 5.0,
            'result_status': think_result.get('status', 'unknown')
        }
        
        # Test analyze reflex
        start_time = time.perf_counter()
        analyze_result = await self.analyze_memory_patterns()
        analyze_time = (time.perf_counter() - start_time) * 1000
        
        demonstration_results['test_results']['analyze'] = {
            'processing_time_ms': analyze_time,
            'biological_reflex_achieved': analyze_time < 5.0,
            'result_status': analyze_result.get('status', 'unknown')
        }
        
        # Calculate overall performance
        all_times = [
            remember_time, recall_time, think_time, analyze_time
        ]
        
        demonstration_results['overall_performance'] = {
            'average_processing_time_ms': sum(all_times) / len(all_times),
            'all_triggers_biological_reflex': all(time < 5.0 for time in all_times),
            'biological_reflex_success_rate': len([t for t in all_times if t < 5.0]) / len(all_times),
            'fastest_operation_ms': min(all_times),
            'slowest_operation_ms': max(all_times),
            'biological_reflex_status': 'ACHIEVED' if all(time < 5.0 for time in all_times) else 'PARTIAL'
        }
        
        return demonstration_results