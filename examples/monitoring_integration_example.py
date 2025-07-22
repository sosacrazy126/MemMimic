#!/usr/bin/env python3
"""
MemMimic Enterprise Monitoring Integration Example

This example demonstrates how to integrate the MemMimic Enterprise Monitoring System
with your application for comprehensive production observability.
"""

import asyncio
import logging
import time
from typing import Optional

from memmimic.api import create_memmimic
from memmimic.monitoring import (
    MonitoringServer,
    get_metrics_collector,
    create_health_monitor,
    create_security_monitor,
    create_alert_manager,
    create_incident_response
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemMimicApplication:
    """
    Example application demonstrating MemMimic monitoring integration.
    
    This shows how to:
    1. Initialize MemMimic with monitoring
    2. Use performance tracking decorators
    3. Implement custom health checks
    4. Handle security events
    5. Set up custom alerts
    """
    
    def __init__(self):
        # Initialize MemMimic API
        self.memmimic = create_memmimic("production.db")
        
        # Initialize monitoring system
        self.monitoring_server: Optional[MonitoringServer] = None
        
        # Get monitoring components for direct use
        self.metrics_collector = get_metrics_collector()
        
        logger.info("MemMimic application initialized")
    
    async def start_monitoring(self, dashboard_port: int = 8080):
        """Start the comprehensive monitoring system"""
        logger.info("Starting enterprise monitoring system...")
        
        # Create monitoring server with all components
        self.monitoring_server = MonitoringServer(
            memmimic_api=self.memmimic,
            dashboard_port=dashboard_port,
            dashboard_host="0.0.0.0",
            health_check_interval=30.0,
            alert_evaluation_interval=30.0
        )
        
        # Add custom health checks
        await self._add_custom_health_checks()
        
        # Configure custom alerts
        self._configure_custom_alerts()
        
        # Set up security monitoring
        self._configure_security_monitoring()
        
        # Start the monitoring server
        await self.monitoring_server.start()
        
        logger.info(f"🚀 Monitoring dashboard available at: http://localhost:{dashboard_port}/dashboard")
        logger.info(f"📊 Prometheus metrics at: http://localhost:{dashboard_port}/api/metrics/prometheus")
        
        return self.monitoring_server
    
    async def _add_custom_health_checks(self):
        """Add application-specific health checks"""
        from memmimic.monitoring.health_monitor import HealthCheck, HealthCheckResult, HealthStatus
        
        class CustomApplicationHealthCheck(HealthCheck):
            def __init__(self, app):
                super().__init__("custom_application", timeout_seconds=5.0)
                self.app = app
            
            async def check(self) -> HealthCheckResult:
                start_time = time.time()
                
                try:
                    # Perform application-specific health check
                    # Example: Check if we can store and retrieve a test memory
                    test_memory_id = await self.app.memmimic.remember("Health check test")
                    test_memories = await self.app.memmimic.recall_cxd("Health check test", limit=1)
                    
                    if test_memories and len(test_memories) > 0:
                        response_time = (time.time() - start_time) * 1000
                        
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message="Application functionality verified",
                            response_time_ms=response_time,
                            metadata={'test_memory_id': test_memory_id}
                        )
                    else:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message="Failed to verify memory operations",
                            response_time_ms=(time.time() - start_time) * 1000
                        )
                
                except Exception as e:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.CRITICAL,
                        message=f"Application health check failed: {str(e)}",
                        response_time_ms=(time.time() - start_time) * 1000,
                        metadata={'error': str(e)}
                    )
        
        # Add the custom health check
        if self.monitoring_server:
            self.monitoring_server.health_monitor.add_health_check(
                CustomApplicationHealthCheck(self)
            )
            logger.info("Custom application health check added")
    
    def _configure_custom_alerts(self):
        """Configure application-specific alerts"""
        from memmimic.monitoring.alert_manager import AlertRule, AlertSeverity
        
        if not self.monitoring_server:
            return
        
        # Custom alert rules
        custom_rules = [
            AlertRule(
                rule_id="app_memory_storage_failures",
                name="High Memory Storage Failure Rate",
                condition="memmimic_memory_storage_errors_total > 5",
                severity=AlertSeverity.ERROR,
                threshold=5,
                duration_seconds=300,
                annotations={
                    'description': 'Memory storage operations are failing frequently',
                    'runbook': 'Check database connectivity and disk space'
                },
                notify_channels=["default_log"]
            ),
            AlertRule(
                rule_id="app_search_performance_degraded",
                name="Search Performance Degraded",
                condition="memmimic_search_duration_seconds > 2.0",
                severity=AlertSeverity.WARNING,
                threshold=2.0,
                duration_seconds=600,
                annotations={
                    'description': 'Search operations are taking longer than expected',
                    'runbook': 'Check search indexes and database performance'
                },
                notify_channels=["default_log"]
            ),
            AlertRule(
                rule_id="app_cxd_classification_down",
                name="CXD Classification Unavailable",
                condition="memmimic_cxd_classifications_total < 1",
                severity=AlertSeverity.CRITICAL,
                threshold=1,
                duration_seconds=900,  # 15 minutes
                annotations={
                    'description': 'CXD classification system appears to be down',
                    'runbook': 'Check CXD classifier availability and configuration'
                },
                notify_channels=["default_log"]
            )
        ]
        
        for rule in custom_rules:
            self.monitoring_server.alert_manager.add_alert_rule(rule)
        
        logger.info(f"Added {len(custom_rules)} custom alert rules")
    
    def _configure_security_monitoring(self):
        """Configure security monitoring for the application"""
        if not self.monitoring_server:
            return
        
        # Add custom security event handlers
        def handle_security_alert(event):
            logger.warning(f"🚨 Security Alert: {event.event_type.value} - {event.message}")
            
            # Custom security response logic
            if event.severity.value in ['high', 'critical']:
                logger.critical(f"Critical security event requires immediate attention: {event.message}")
                # Here you could integrate with your incident management system
        
        def handle_security_incident(incident):
            logger.critical(f"🚨 Security Incident: {incident.incident_type} - {incident.incident_id}")
            # Custom incident handling logic
            # e.g., create ticket in JIRA, send to Slack, etc.
        
        self.monitoring_server.security_monitor.add_alert_callback(handle_security_alert)
        self.monitoring_server.security_monitor.add_incident_callback(handle_security_incident)
        
        logger.info("Security monitoring configured")
    
    async def simulate_application_load(self):
        """Simulate application load to demonstrate monitoring"""
        logger.info("Starting application load simulation...")
        
        import random
        from memmimic.monitoring.memmimic_performance_tracker import get_performance_tracker
        
        tracker = get_performance_tracker()
        
        for i in range(100):
            try:
                # Simulate various operations
                operation_type = random.choice(['store', 'search', 'classify'])
                
                if operation_type == 'store':
                    # Simulate memory storage
                    content = f"Simulated memory content {i}"
                    memory_id = await self.memmimic.remember(content)
                    logger.debug(f"Stored memory: {memory_id}")
                
                elif operation_type == 'search':
                    # Simulate search operation
                    with tracker.track_search_operation("simulation search", "hybrid") as search_tracker:
                        results = await self.memmimic.recall_cxd("simulation", limit=5)
                        search_tracker.set_results(len(results), 0.85)  # Mock relevance score
                        logger.debug(f"Search returned {len(results)} results")
                
                elif operation_type == 'classify':
                    # Simulate CXD classification
                    if self.memmimic.cxd:
                        with tracker.track_cxd_classification("Test classification content") as cxd_tracker:
                            result = self.memmimic.cxd.classify("Test classification content")
                            pattern = getattr(result, 'pattern', 'unknown')
                            cxd_tracker.set_result(pattern, 0.9)  # Mock accuracy
                            logger.debug(f"Classification: {pattern}")
                
                # Simulate some errors occasionally
                if random.random() < 0.05:  # 5% error rate
                    self.metrics_collector.increment_counter("memmimic_errors_total")
                    logger.warning(f"Simulated error in operation {i}")
                
                # Small delay between operations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in simulation step {i}: {e}")
                self.metrics_collector.increment_counter("memmimic_errors_total")
        
        logger.info("Application load simulation completed")
    
    async def demonstrate_monitoring_features(self):
        """Demonstrate various monitoring features"""
        logger.info("Demonstrating monitoring features...")
        
        if not self.monitoring_server:
            logger.error("Monitoring server not started")
            return
        
        # 1. Get system status
        system_status = await self.monitoring_server.get_system_status()
        logger.info(f"System Status: {system_status['overall_status']}")
        logger.info(f"Health Score: {system_status['health']['score']:.2%}")
        
        # 2. Run health checks
        health_result = await self.monitoring_server.run_health_check()
        logger.info(f"Health Check: {health_result.status.value} ({health_result.healthy_checks}/{health_result.total_checks} healthy)")
        
        # 3. Get performance metrics
        from memmimic.monitoring.memmimic_performance_tracker import get_performance_tracker
        tracker = get_performance_tracker()
        perf_summary = tracker.get_performance_summary()
        logger.info(f"Performance Summary: {perf_summary['search_performance']['total_searches']} searches performed")
        
        # 4. Get alert summary
        alert_summary = self.monitoring_server.get_alert_summary()
        logger.info(f"Alerts: {alert_summary['active_alerts']} active, {alert_summary['total_alerts']} total")
        
        # 5. Get security summary
        security_summary = self.monitoring_server.get_security_summary()
        logger.info(f"Security: {security_summary['recent_events_1h']} events in last hour")
        
        # 6. Get optimization recommendations
        recommendations = tracker.get_optimization_recommendations()
        if recommendations:
            logger.info("Performance Recommendations:")
            for rec in recommendations:
                logger.info(f"  - {rec['component']}: {rec['recommendation']}")
        else:
            logger.info("No performance optimization recommendations at this time")
    
    async def test_security_monitoring(self):
        """Test security monitoring capabilities"""
        logger.info("Testing security monitoring...")
        
        if not self.monitoring_server:
            return
        
        # Simulate security events
        security_monitor = self.monitoring_server.security_monitor
        
        # Simulate authentication failure
        security_monitor.record_authentication_failure(
            source_ip="192.168.1.100",
            user_agent="TestAgent/1.0",
            endpoint="/api/login",
            reason="Invalid password"
        )
        
        # Simulate successful authentication
        security_monitor.record_authentication_success(
            source_ip="192.168.1.100",
            user_id="test_user",
            endpoint="/api/login"
        )
        
        # Simulate API request for monitoring
        security_monitor.record_api_request(
            source_ip="192.168.1.100",
            endpoint="/api/memories",
            request_data={"query": "test"},
            user_id="test_user"
        )
        
        logger.info("Security monitoring test completed")
    
    async def shutdown(self):
        """Gracefully shutdown the application and monitoring"""
        logger.info("Shutting down application...")
        
        if self.monitoring_server:
            await self.monitoring_server.stop()
        
        logger.info("Application shutdown completed")


async def main():
    """Main application entry point"""
    logger.info("🚀 Starting MemMimic Enterprise Monitoring Demo")
    
    # Create application instance
    app = MemMimicApplication()
    
    try:
        # Start monitoring system
        monitoring_server = await app.start_monitoring(dashboard_port=8080)
        
        # Wait a moment for everything to initialize
        await asyncio.sleep(2)
        
        # Demonstrate monitoring features
        await app.demonstrate_monitoring_features()
        
        # Test security monitoring
        await app.test_security_monitoring()
        
        # Simulate application load
        await app.simulate_application_load()
        
        # Show final monitoring status
        logger.info("=" * 60)
        logger.info("Final System Status:")
        logger.info("=" * 60)
        
        system_status = await app.monitoring_server.get_system_status()
        logger.info(f"Overall Status: {system_status['overall_status']}")
        logger.info(f"Health Score: {system_status['health']['score']:.2%}")
        logger.info(f"Active Alerts: {system_status['alerts']['active_alerts']}")
        logger.info(f"Memory Usage: {system_status['resources']['memory_usage_percent']:.1f}%")
        logger.info(f"CPU Usage: {system_status['resources']['cpu_usage_percent']:.1f}%")
        
        logger.info("=" * 60)
        logger.info("🌐 Dashboard available at: http://localhost:8080/dashboard")
        logger.info("📊 Prometheus metrics at: http://localhost:8080/api/metrics/prometheus")
        logger.info("🔧 API documentation at: http://localhost:8080/api/status")
        logger.info("=" * 60)
        
        # Keep running for demonstration
        logger.info("Press Ctrl+C to stop the demo...")
        await monitoring_server.wait_until_stopped()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())