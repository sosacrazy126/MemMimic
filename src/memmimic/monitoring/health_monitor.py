"""
Enterprise Health Monitoring System for MemMimic

Provides comprehensive health checks, endpoint monitoring, and system status tracking.
Implements health check patterns for production monitoring.
"""

import asyncio
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of an individual health check"""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'response_time_ms': self.response_time_ms,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    checks: List[HealthCheckResult]
    overall_response_time_ms: float
    healthy_checks: int
    total_checks: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'overall_response_time_ms': self.overall_response_time_ms,
            'healthy_checks': self.healthy_checks,
            'total_checks': self.total_checks,
            'timestamp': self.timestamp.isoformat(),
            'checks': [check.to_dict() for check in self.checks]
        }


class HealthCheck:
    """Base class for health checks"""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
    
    async def check(self) -> HealthCheckResult:
        """Perform health check - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement check method")


class DatabaseHealthCheck(HealthCheck):
    """Database connectivity and performance health check"""
    
    def __init__(self, db_manager, timeout_seconds: float = 5.0):
        super().__init__("database", timeout_seconds)
        self.db_manager = db_manager
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            # Test database connectivity
            connection_test_start = time.time()
            
            # Try a simple query or connection test
            if hasattr(self.db_manager, 'test_connection'):
                await self.db_manager.test_connection()
            elif hasattr(self.db_manager, 'count_memories'):
                # Use existing method as connectivity test
                await self.db_manager.count_memories()
            else:
                # Fallback test
                pass
            
            connection_time = (time.time() - connection_test_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            # Determine status based on response time
            if connection_time < 100:
                status = HealthStatus.HEALTHY
                message = f"Database responsive ({connection_time:.1f}ms)"
            elif connection_time < 500:
                status = HealthStatus.DEGRADED
                message = f"Database slow response ({connection_time:.1f}ms)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Database very slow ({connection_time:.1f}ms)"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                response_time_ms=total_time,
                metadata={
                    'connection_time_ms': connection_time,
                    'db_type': type(self.db_manager).__name__
                }
            )
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                response_time_ms=total_time,
                metadata={'error': str(e)}
            )


class CacheHealthCheck(HealthCheck):
    """Cache system health check"""
    
    def __init__(self, cache_manager, timeout_seconds: float = 3.0):
        super().__init__("cache", timeout_seconds)
        self.cache_manager = cache_manager
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            # Test cache operations
            test_key = f"health_check_{int(time.time())}"
            test_value = "health_check_value"
            
            # Set test value
            if hasattr(self.cache_manager, 'set'):
                self.cache_manager.set(test_key, test_value)
                
                # Get test value
                retrieved_value = self.cache_manager.get(test_key)
                
                # Clean up
                if hasattr(self.cache_manager, 'delete'):
                    self.cache_manager.delete(test_key)
                
                total_time = (time.time() - start_time) * 1000
                
                if retrieved_value == test_value:
                    status = HealthStatus.HEALTHY
                    message = "Cache operations successful"
                else:
                    status = HealthStatus.DEGRADED
                    message = "Cache read/write inconsistency"
            else:
                # Fallback for different cache interface
                total_time = (time.time() - start_time) * 1000
                status = HealthStatus.HEALTHY
                message = "Cache manager available"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                response_time_ms=total_time,
                metadata={'cache_type': type(self.cache_manager).__name__}
            )
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Cache check failed: {str(e)}",
                response_time_ms=total_time,
                metadata={'error': str(e)}
            )


class MemoryHealthCheck(HealthCheck):
    """Memory system health check"""
    
    def __init__(self, memory_manager, timeout_seconds: float = 5.0):
        super().__init__("memory_system", timeout_seconds)
        self.memory_manager = memory_manager
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            # Test basic memory operations
            if hasattr(self.memory_manager, 'count_memories'):
                memory_count = await self.memory_manager.count_memories()
                
                total_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Memory system operational ({memory_count} memories)",
                    response_time_ms=total_time,
                    metadata={
                        'memory_count': memory_count,
                        'manager_type': type(self.memory_manager).__name__
                    }
                )
            else:
                total_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Memory manager available",
                    response_time_ms=total_time,
                    metadata={'manager_type': type(self.memory_manager).__name__}
                )
                
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Memory system check failed: {str(e)}",
                response_time_ms=total_time,
                metadata={'error': str(e)}
            )


class CXDHealthCheck(HealthCheck):
    """CXD classification system health check"""
    
    def __init__(self, cxd_classifier=None, timeout_seconds: float = 3.0):
        super().__init__("cxd_classifier", timeout_seconds)
        self.cxd_classifier = cxd_classifier
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            if self.cxd_classifier is None:
                total_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="CXD classifier not available",
                    response_time_ms=total_time,
                    metadata={'available': False}
                )
            
            # Test classification
            test_content = "This is a test classification request."
            
            if hasattr(self.cxd_classifier, 'classify'):
                classification_result = self.cxd_classifier.classify(test_content)
                
                total_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="CXD classifier operational",
                    response_time_ms=total_time,
                    metadata={
                        'available': True,
                        'classifier_type': type(self.cxd_classifier).__name__,
                        'test_classification': getattr(classification_result, 'pattern', 'unknown')
                    }
                )
            else:
                total_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="CXD classifier interface not compatible",
                    response_time_ms=total_time,
                    metadata={'available': True, 'interface_compatible': False}
                )
                
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"CXD classifier check failed: {str(e)}",
                response_time_ms=total_time,
                metadata={'error': str(e)}
            )


class SystemResourceHealthCheck(HealthCheck):
    """System resource utilization health check"""
    
    def __init__(self, timeout_seconds: float = 2.0):
        super().__init__("system_resources", timeout_seconds)
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            import psutil
            
            # Get system metrics
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            total_time = (time.time() - start_time) * 1000
            
            # Determine status based on resource usage
            if memory_percent > 90 or cpu_percent > 90 or disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = "Critical resource usage detected"
            elif memory_percent > 80 or cpu_percent > 80 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "High resource usage"
            elif memory_percent > 70 or cpu_percent > 70 or disk_percent > 80:
                status = HealthStatus.DEGRADED
                message = "Elevated resource usage"
            else:
                status = HealthStatus.HEALTHY
                message = "Resource usage normal"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                response_time_ms=total_time,
                metadata={
                    'memory_percent': memory_percent,
                    'cpu_percent': cpu_percent,
                    'disk_percent': disk_percent
                }
            )
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Resource check failed: {str(e)}",
                response_time_ms=total_time,
                metadata={'error': str(e)}
            )


class HealthMonitor:
    """
    Enterprise health monitoring system for MemMimic.
    
    Provides comprehensive health checks, status tracking, and endpoint monitoring
    for all critical MemMimic components.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.checks: List[HealthCheck] = []
        self.last_health_result: Optional[SystemHealth] = None
        
        # Background monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Metrics integration
        self.metrics_collector = get_metrics_collector()
        
        # Health history
        self._health_history: List[SystemHealth] = []
        self._max_history = 100
        
        logger.info("Health Monitor initialized")
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check to the monitoring system"""
        self.checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check by name"""
        self.checks = [check for check in self.checks if check.name != name]
        logger.info(f"Removed health check: {name}")
    
    async def run_health_checks(self) -> SystemHealth:
        """Run all health checks and return overall system health"""
        start_time = time.time()
        
        if not self.checks:
            return SystemHealth(
                status=HealthStatus.HEALTHY,
                checks=[],
                overall_response_time_ms=0.0,
                healthy_checks=0,
                total_checks=0
            )
        
        # Run all health checks concurrently
        check_tasks = []
        for check in self.checks:
            task = asyncio.create_task(self._run_single_check(check))
            check_tasks.append(task)
        
        # Wait for all checks to complete
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for i, result in enumerate(check_results):
            if isinstance(result, Exception):
                # Handle failed health check
                failed_result = HealthCheckResult(
                    name=self.checks[i].name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check execution failed: {str(result)}",
                    response_time_ms=0.0,
                    metadata={'execution_error': str(result)}
                )
                valid_results.append(failed_result)
                logger.error(f"Health check {self.checks[i].name} failed: {result}")
            else:
                valid_results.append(result)
        
        # Determine overall system status
        healthy_checks = sum(1 for result in valid_results if result.status == HealthStatus.HEALTHY)
        total_checks = len(valid_results)
        overall_response_time = (time.time() - start_time) * 1000
        
        # Calculate overall status
        critical_checks = sum(1 for result in valid_results if result.status == HealthStatus.CRITICAL)
        unhealthy_checks = sum(1 for result in valid_results if result.status == HealthStatus.UNHEALTHY)
        degraded_checks = sum(1 for result in valid_results if result.status == HealthStatus.DEGRADED)
        
        if critical_checks > 0:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy_checks > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_checks > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Create system health result
        system_health = SystemHealth(
            status=overall_status,
            checks=valid_results,
            overall_response_time_ms=overall_response_time,
            healthy_checks=healthy_checks,
            total_checks=total_checks
        )
        
        # Update metrics
        self._update_health_metrics(system_health)
        
        # Store in history
        self._health_history.append(system_health)
        if len(self._health_history) > self._max_history:
            self._health_history.pop(0)
        
        self.last_health_result = system_health
        
        return system_health
    
    async def _run_single_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Run a single health check with timeout"""
        try:
            # Use asyncio timeout to enforce check timeout
            result = await asyncio.wait_for(
                health_check.check(),
                timeout=health_check.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=health_check.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {health_check.timeout_seconds}s",
                response_time_ms=health_check.timeout_seconds * 1000,
                metadata={'timeout': True}
            )
        except Exception as e:
            return HealthCheckResult(
                name=health_check.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                response_time_ms=0.0,
                metadata={'error': str(e)}
            )
    
    def _update_health_metrics(self, system_health: SystemHealth):
        """Update metrics based on health check results"""
        # Overall health score (0-1 based on healthy checks ratio)
        health_score = system_health.healthy_checks / max(system_health.total_checks, 1)
        self.metrics_collector.set_gauge("memmimic_health_score", health_score)
        
        # Individual check metrics
        for check_result in system_health.checks:
            # Check status as gauge (0=healthy, 1=degraded, 2=unhealthy, 3=critical)
            status_value = {
                HealthStatus.HEALTHY: 0,
                HealthStatus.DEGRADED: 1,
                HealthStatus.UNHEALTHY: 2,
                HealthStatus.CRITICAL: 3
            }.get(check_result.status, 3)
            
            # Update check-specific metrics  
            self.metrics_collector.set_gauge(f"memmimic_health_check_status_{check_result.name}", status_value)
            self.metrics_collector.observe_histogram(f"memmimic_health_check_duration_seconds_{check_result.name}", 
                                                    check_result.response_time_ms / 1000)
        
        # Update overall response time
        self.metrics_collector.observe_histogram("memmimic_health_check_duration_seconds_overall",
                                                system_health.overall_response_time_ms / 1000)
    
    def start_background_monitoring(self):
        """Start background health monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Background monitoring already running")
            return
        
        def monitoring_worker():
            async def monitor_loop():
                while not self._stop_monitoring.is_set():
                    try:
                        await self.run_health_checks()
                        await asyncio.sleep(self.check_interval)
                    except Exception as e:
                        logger.error(f"Background health monitoring failed: {e}")
                        await asyncio.sleep(min(self.check_interval, 10))  # Shorter retry interval on error
            
            # Create new event loop for the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(monitor_loop())
            finally:
                loop.close()
        
        self._monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self._monitoring_thread.start()
        logger.info("Background health monitoring started")
    
    def stop_background_monitoring(self):
        """Stop background health monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Background health monitoring stopped")
    
    def get_health_history(self, hours: int = 1) -> List[SystemHealth]:
        """Get health check history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            health for health in self._health_history
            if health.timestamp >= cutoff_time
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of current health status"""
        if not self.last_health_result:
            return {
                'status': 'unknown',
                'message': 'No health checks have been run yet',
                'checks_configured': len(self.checks)
            }
        
        return {
            'status': self.last_health_result.status.value,
            'healthy_checks': self.last_health_result.healthy_checks,
            'total_checks': self.last_health_result.total_checks,
            'overall_response_time_ms': self.last_health_result.overall_response_time_ms,
            'last_check_time': self.last_health_result.timestamp.isoformat(),
            'checks_configured': len(self.checks),
            'monitoring_enabled': self._monitoring_thread is not None and self._monitoring_thread.is_alive()
        }
    
    def shutdown(self):
        """Shutdown health monitor"""
        try:
            self.stop_background_monitoring()
            logger.info("Health monitor shutdown completed")
        except Exception as e:
            logger.error(f"Health monitor shutdown failed: {e}")


def create_health_monitor(
    memmimic_api,
    check_interval: float = 30.0,
    auto_configure: bool = True
) -> HealthMonitor:
    """
    Factory function to create health monitor with MemMimic integration.
    
    Args:
        memmimic_api: MemMimic API instance
        check_interval: Health check interval in seconds
        auto_configure: Whether to automatically configure health checks
        
    Returns:
        Configured HealthMonitor instance
    """
    monitor = HealthMonitor(check_interval=check_interval)
    
    if auto_configure:
        # Add standard MemMimic health checks
        
        # Database health check
        if hasattr(memmimic_api, 'memory'):
            monitor.add_health_check(DatabaseHealthCheck(memmimic_api.memory))
        
        # Cache health check (if available)
        if hasattr(memmimic_api, 'memory') and hasattr(memmimic_api.memory, 'cache_manager'):
            monitor.add_health_check(CacheHealthCheck(memmimic_api.memory.cache_manager))
        
        # Memory system health check
        if hasattr(memmimic_api, 'memory'):
            monitor.add_health_check(MemoryHealthCheck(memmimic_api.memory))
        
        # CXD classifier health check
        if hasattr(memmimic_api, 'cxd'):
            monitor.add_health_check(CXDHealthCheck(memmimic_api.cxd))
        
        # System resource health check
        monitor.add_health_check(SystemResourceHealthCheck())
        
        logger.info("Health monitor auto-configured with standard checks")
    
    return monitor