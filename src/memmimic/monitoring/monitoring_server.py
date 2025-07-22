"""
Enterprise Monitoring Server for MemMimic

Unified monitoring server that orchestrates all monitoring components:
metrics collection, health monitoring, security monitoring, alerting,
incident response, and dashboard services.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from .metrics_collector import get_metrics_collector
from .health_monitor import create_health_monitor
from .security_monitor import create_security_monitor
from .alert_manager import create_alert_manager
from .dashboard_server import start_dashboard_server
from .incident_response import create_incident_response

logger = logging.getLogger(__name__)


class MonitoringServer:
    """
    Unified enterprise monitoring server for MemMimic.
    
    Orchestrates all monitoring components and provides a single entry point
    for comprehensive production monitoring and observability.
    """
    
    def __init__(self, 
                 memmimic_api,
                 dashboard_port: int = 8080,
                 dashboard_host: str = "0.0.0.0",
                 health_check_interval: float = 30.0,
                 alert_evaluation_interval: float = 30.0,
                 metrics_collection_interval: float = 10.0):
        
        self.memmimic_api = memmimic_api
        self.dashboard_port = dashboard_port
        self.dashboard_host = dashboard_host
        
        # Initialize all monitoring components
        self.metrics_collector = get_metrics_collector()
        
        self.health_monitor = create_health_monitor(
            memmimic_api,
            check_interval=health_check_interval,
            auto_configure=True
        )
        
        self.security_monitor = create_security_monitor(retention_hours=24)
        
        self.alert_manager = create_alert_manager(
            self.metrics_collector,
            evaluation_interval=alert_evaluation_interval
        )
        
        self.incident_response = create_incident_response(self.metrics_collector)
        
        self.dashboard_server = start_dashboard_server(
            port=dashboard_port,
            host=dashboard_host,
            health_monitor=self.health_monitor,
            security_monitor=self.security_monitor,
            alert_manager=self.alert_manager,
            update_interval=5.0
        )
        
        # Integration hooks
        self._setup_component_integration()
        
        # Server state
        self.is_running = False
        self.dashboard_runner = None
        
        logger.info(f"Monitoring server initialized (dashboard: {dashboard_host}:{dashboard_port})")
    
    def _setup_component_integration(self):
        """Setup integration between monitoring components"""
        
        # Alert manager -> Incident response integration
        def handle_critical_alert(alert):
            """Handle critical alerts by processing them through incident response"""
            try:
                # Convert alert to format for incident detection
                self.incident_response.process_alerts([alert])
            except Exception as e:
                logger.error(f"Failed to process alert for incident response: {e}")
        
        # Register critical alert callback
        self.alert_manager.notification_manager.add_channel(
            self.alert_manager.notification_manager.NotificationChannel(
                channel_id="incident_response",
                channel_type="custom",
                config={"callback": handle_critical_alert}
            )
        )
        
        # Security monitor -> Incident response integration
        def handle_security_incident(security_incident):
            """Handle security incidents"""
            try:
                # Security incidents are already handled within security monitor
                # but we could add additional processing here
                logger.info(f"Security incident processed: {security_incident.incident_id}")
            except Exception as e:
                logger.error(f"Failed to process security incident: {e}")
        
        self.security_monitor.add_incident_callback(handle_security_incident)
        
        # Health monitor -> Alert manager integration  
        # (Health checks already update metrics which trigger alerts)
        
        logger.info("Component integration configured")
    
    async def start(self):
        """Start the monitoring server and all components"""
        if self.is_running:
            logger.warning("Monitoring server is already running")
            return
        
        try:
            logger.info("Starting MemMimic Enterprise Monitoring Server...")
            
            # Start health monitoring
            self.health_monitor.start_background_monitoring()
            
            # Start dashboard server
            self.dashboard_runner = await self.dashboard_server.start_server()
            
            self.is_running = True
            
            # Log startup status
            startup_info = {
                'dashboard_url': f"http://{self.dashboard_host}:{self.dashboard_port}/dashboard",
                'prometheus_metrics_url': f"http://{self.dashboard_host}:{self.dashboard_port}/api/metrics/prometheus",
                'api_base_url': f"http://{self.dashboard_host}:{self.dashboard_port}/api",
                'components': {
                    'metrics_collector': 'operational',
                    'health_monitor': 'operational',
                    'security_monitor': 'operational',
                    'alert_manager': 'operational',
                    'incident_response': 'operational',
                    'dashboard_server': 'operational'
                }
            }
            
            logger.info("=" * 80)
            logger.info("MemMimic Enterprise Monitoring Server Started Successfully")
            logger.info("=" * 80)
            logger.info(f"🌐 Dashboard: {startup_info['dashboard_url']}")
            logger.info(f"📊 Prometheus Metrics: {startup_info['prometheus_metrics_url']}")
            logger.info(f"🔧 API: {startup_info['api_base_url']}")
            logger.info("=" * 80)
            logger.info("Monitoring Components:")
            for component, status in startup_info['components'].items():
                logger.info(f"  ✓ {component}: {status}")
            logger.info("=" * 80)
            
            # Setup graceful shutdown
            self._setup_signal_handlers()
            
            # Start periodic status reporting
            asyncio.create_task(self._periodic_status_report())
            
        except Exception as e:
            logger.error(f"Failed to start monitoring server: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the monitoring server and all components"""
        if not self.is_running:
            return
        
        logger.info("Stopping MemMimic Enterprise Monitoring Server...")
        
        try:
            # Stop dashboard server
            if self.dashboard_runner:
                await self.dashboard_server.stop_server(self.dashboard_runner)
                self.dashboard_runner = None
            
            # Stop health monitoring
            self.health_monitor.stop_background_monitoring()
            
            # Shutdown components
            self.alert_manager.shutdown()
            self.security_monitor.shutdown()
            self.incident_response.shutdown()
            self.metrics_collector.shutdown()
            
            self.is_running = False
            
            logger.info("Monitoring server stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during monitoring server shutdown: {e}")
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _periodic_status_report(self):
        """Periodically log system status"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                # Collect status from all components
                system_status = await self.get_system_status()
                
                logger.info("=" * 60)
                logger.info("MemMimic Monitoring System Status Report")
                logger.info("=" * 60)
                logger.info(f"Overall Status: {system_status['overall_status']}")
                logger.info(f"Active Incidents: {system_status['incidents']['active']}")
                logger.info(f"Active Alerts: {system_status['alerts']['active']}")
                logger.info(f"Health Score: {system_status['health']['score']:.1%}")
                logger.info(f"Security Events (1h): {system_status['security']['recent_events']}")
                logger.info(f"Memory Usage: {system_status['resources']['memory_usage_percent']:.1f}%")
                logger.info(f"CPU Usage: {system_status['resources']['cpu_usage_percent']:.1f}%")
                logger.info("=" * 60)
                
            except Exception as e:
                logger.error(f"Failed to generate status report: {e}")
                await asyncio.sleep(60)  # Shorter retry interval on error
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get health status
            health_result = await self.health_monitor.run_health_checks()
            
            # Get current metrics
            metrics_json = self.metrics_collector.export_json_format()
            import json
            metrics_data = json.loads(metrics_json)
            
            system_status = {
                'timestamp': metrics_data['timestamp'],
                'overall_status': 'operational',
                'health': {
                    'status': health_result.status.value,
                    'score': health_result.healthy_checks / max(health_result.total_checks, 1),
                    'healthy_checks': health_result.healthy_checks,
                    'total_checks': health_result.total_checks,
                    'response_time_ms': health_result.overall_response_time_ms
                },
                'alerts': self.alert_manager.get_alert_summary(),
                'security': self.security_monitor.get_security_summary(),
                'incidents': self.incident_response.get_incident_summary(),
                'resources': {
                    'memory_usage_percent': metrics_data['metrics']['gauges'].get('memmimic_memory_usage_percent', 0),
                    'cpu_usage_percent': metrics_data['metrics']['gauges'].get('memmimic_cpu_usage_percent', 0),
                    'disk_usage_percent': metrics_data['metrics']['gauges'].get('memmimic_disk_usage_percent', 0)
                },
                'performance': {
                    'total_errors': metrics_data['metrics']['counters'].get('memmimic_errors_total', 0),
                    'total_operations': metrics_data['metrics']['counters'].get('memmimic_operations_total', 0)
                }
            }
            
            # Determine overall status
            if (system_status['incidents']['active_incidents'] > 0 or 
                system_status['alerts']['active_alerts'] > 5 or
                system_status['health']['score'] < 0.5):
                system_status['overall_status'] = 'degraded'
            
            if (system_status['health']['status'] in ['critical', 'unhealthy'] or
                system_status['resources']['memory_usage_percent'] > 95 or
                system_status['resources']['cpu_usage_percent'] > 95):
                system_status['overall_status'] = 'critical'
            
            return system_status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'timestamp': asyncio.get_event_loop().time(),
                'overall_status': 'unknown',
                'error': str(e)
            }
    
    def get_monitoring_endpoints(self) -> Dict[str, str]:
        """Get all monitoring endpoints"""
        base_url = f"http://{self.dashboard_host}:{self.dashboard_port}"
        
        return {
            'dashboard': f"{base_url}/dashboard",
            'api_status': f"{base_url}/api/status", 
            'api_health': f"{base_url}/api/health",
            'api_metrics': f"{base_url}/api/metrics",
            'api_security': f"{base_url}/api/security",
            'api_alerts': f"{base_url}/api/alerts",
            'api_performance': f"{base_url}/api/performance",
            'prometheus_metrics': f"{base_url}/api/metrics/prometheus",
            'websocket': f"ws://{self.dashboard_host}:{self.dashboard_port}/ws"
        }
    
    async def run_health_check(self):
        """Run comprehensive health check"""
        return await self.health_monitor.run_health_checks()
    
    def get_alert_summary(self):
        """Get alert summary"""
        return self.alert_manager.get_alert_summary()
    
    def get_security_summary(self):
        """Get security summary"""
        return self.security_monitor.get_security_summary()
    
    def get_incident_summary(self):
        """Get incident summary"""
        return self.incident_response.get_incident_summary()
    
    async def wait_until_stopped(self):
        """Wait until the server is stopped"""
        while self.is_running:
            await asyncio.sleep(1)


@asynccontextmanager
async def monitoring_server_context(memmimic_api, **kwargs):
    """Context manager for monitoring server lifecycle"""
    server = MonitoringServer(memmimic_api, **kwargs)
    
    try:
        await server.start()
        yield server
    finally:
        await server.stop()


async def run_monitoring_server(memmimic_api, **kwargs):
    """Run monitoring server until interrupted"""
    async with monitoring_server_context(memmimic_api, **kwargs) as server:
        try:
            await server.wait_until_stopped()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")


def main():
    """Main entry point for standalone monitoring server"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Import MemMimic API
        from ..api import create_memmimic
        
        # Create MemMimic instance
        memmimic_api = create_memmimic()
        
        # Run monitoring server
        asyncio.run(run_monitoring_server(memmimic_api))
        
    except ImportError as e:
        logger.error(f"Failed to import MemMimic API: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Monitoring server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()