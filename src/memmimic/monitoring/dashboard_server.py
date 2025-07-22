"""
Enterprise Dashboard Server for MemMimic

Provides real-time monitoring dashboards with live metrics, health status,
security events, and performance analytics via HTTP API and WebSocket streams.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import asdict
import threading

# HTTP server and WebSocket support
try:
    from aiohttp import web, WSMsgType
    from aiohttp.web_ws import WebSocketResponse
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None
    WebSocketResponse = None

from .metrics_collector import get_metrics_collector
from .health_monitor import HealthMonitor
from .security_monitor import SecurityMonitor
from .alert_manager import AlertManager

logger = logging.getLogger(__name__)


class DashboardServer:
    """
    Real-time monitoring dashboard server for MemMimic.
    
    Provides:
    - HTTP API for metrics and status data
    - WebSocket streams for real-time updates
    - Interactive dashboard interface
    - Alert and incident management
    - Performance analytics
    """
    
    def __init__(self, 
                 port: int = 8080,
                 host: str = "0.0.0.0",
                 update_interval: float = 5.0):
        
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for dashboard server. Install with: pip install aiohttp")
        
        self.port = port
        self.host = host
        self.update_interval = update_interval
        
        # Component references
        self.metrics_collector = get_metrics_collector()
        self.health_monitor: Optional[HealthMonitor] = None
        self.security_monitor: Optional[SecurityMonitor] = None
        self.alert_manager: Optional[AlertManager] = None
        
        # WebSocket connections
        self.websocket_connections: Set[WebSocketResponse] = set()
        
        # Background update task
        self._update_task = None
        
        # Web application
        self.app = web.Application()
        self._setup_routes()
        
        logger.info(f"Dashboard server initialized on {host}:{port}")
    
    def set_health_monitor(self, health_monitor: HealthMonitor):
        """Set health monitor reference"""
        self.health_monitor = health_monitor
    
    def set_security_monitor(self, security_monitor: SecurityMonitor):
        """Set security monitor reference"""
        self.security_monitor = security_monitor
    
    def set_alert_manager(self, alert_manager: AlertManager):
        """Set alert manager reference"""
        self.alert_manager = alert_manager
    
    def _setup_routes(self):
        """Setup HTTP routes"""
        # API routes
        self.app.router.add_get('/api/metrics', self._api_metrics)
        self.app.router.add_get('/api/metrics/prometheus', self._api_prometheus_metrics)
        self.app.router.add_get('/api/health', self._api_health)
        self.app.router.add_get('/api/security', self._api_security)
        self.app.router.add_get('/api/alerts', self._api_alerts)
        self.app.router.add_get('/api/status', self._api_status)
        self.app.router.add_get('/api/performance', self._api_performance)
        
        # WebSocket endpoint
        self.app.router.add_get('/ws', self._websocket_handler)
        
        # Static dashboard (basic HTML interface)
        self.app.router.add_get('/', self._dashboard_index)
        self.app.router.add_get('/dashboard', self._dashboard_index)
        
        # Alert management endpoints
        self.app.router.add_post('/api/alerts/{alert_id}/acknowledge', self._api_acknowledge_alert)
        self.app.router.add_post('/api/alerts/{alert_id}/resolve', self._api_resolve_alert)
        
        # Security incident management
        self.app.router.add_post('/api/security/incidents/{incident_id}/resolve', self._api_resolve_incident)
        
        logger.info("Dashboard routes configured")
    
    async def _api_metrics(self, request) -> web.Response:
        """Get current metrics in JSON format"""
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': json.loads(self.metrics_collector.export_json_format())
            }
            
            return web.json_response(metrics_data)
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_prometheus_metrics(self, request) -> web.Response:
        """Get metrics in Prometheus format"""
        try:
            prometheus_data = self.metrics_collector.export_prometheus_format()
            
            return web.Response(
                text=prometheus_data,
                content_type='text/plain; version=0.0.4; charset=utf-8'
            )
            
        except Exception as e:
            logger.error(f"Failed to get Prometheus metrics: {e}")
            return web.Response(text=f"# Error: {e}", status=500)
    
    async def _api_health(self, request) -> web.Response:
        """Get system health status"""
        try:
            if self.health_monitor:
                health_status = await self.health_monitor.run_health_checks()
                return web.json_response(health_status.to_dict())
            else:
                return web.json_response({
                    'status': 'unknown',
                    'message': 'Health monitoring not configured'
                })
                
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_security(self, request) -> web.Response:
        """Get security monitoring data"""
        try:
            if self.security_monitor:
                security_data = {
                    'summary': self.security_monitor.get_security_summary(),
                    'recent_events': self.security_monitor.get_recent_events(hours=1, limit=50),
                    'incidents': self.security_monitor.get_incidents()
                }
                return web.json_response(security_data)
            else:
                return web.json_response({
                    'message': 'Security monitoring not configured'
                })
                
        except Exception as e:
            logger.error(f"Failed to get security data: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_alerts(self, request) -> web.Response:
        """Get alert management data"""
        try:
            if self.alert_manager:
                alert_data = {
                    'summary': self.alert_manager.get_alert_summary(),
                    'active_alerts': self.alert_manager.get_alerts(),
                    'alert_rules': self.alert_manager.get_alert_rules()
                }
                return web.json_response(alert_data)
            else:
                return web.json_response({
                    'message': 'Alert management not configured'
                })
                
        except Exception as e:
            logger.error(f"Failed to get alert data: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_status(self, request) -> web.Response:
        """Get overall system status"""
        try:
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'operational',
                'components': {
                    'metrics_collector': 'operational',
                    'health_monitor': 'operational' if self.health_monitor else 'not_configured',
                    'security_monitor': 'operational' if self.security_monitor else 'not_configured',
                    'alert_manager': 'operational' if self.alert_manager else 'not_configured'
                },
                'metrics_summary': self.metrics_collector.get_metric_summary()
            }
            
            # Add health status if available
            if self.health_monitor and self.health_monitor.last_health_result:
                status_data['health'] = self.health_monitor.get_health_summary()
            
            # Add security status if available
            if self.security_monitor:
                status_data['security'] = self.security_monitor.get_security_summary()
            
            # Add alert status if available
            if self.alert_manager:
                status_data['alerts'] = self.alert_manager.get_alert_summary()
            
            return web.json_response(status_data)
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_performance(self, request) -> web.Response:
        """Get performance analytics"""
        try:
            # Get query parameters
            hours = int(request.query.get('hours', 1))
            
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'historical_metrics': self.metrics_collector.get_historical_metrics(hours=hours),
                'current_metrics': json.loads(self.metrics_collector.export_json_format())['metrics']
            }
            
            # Add health trends if available
            if self.health_monitor:
                performance_data['health_history'] = [
                    health.to_dict() for health in self.health_monitor.get_health_history(hours=hours)
                ]
            
            return web.json_response(performance_data)
            
        except Exception as e:
            logger.error(f"Failed to get performance data: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_acknowledge_alert(self, request) -> web.Response:
        """Acknowledge an alert"""
        try:
            alert_id = request.match_info['alert_id']
            
            if self.alert_manager:
                success = self.alert_manager.acknowledge_alert(alert_id, "dashboard_user")
                if success:
                    return web.json_response({'status': 'acknowledged'})
                else:
                    return web.json_response({'error': 'Alert not found'}, status=404)
            else:
                return web.json_response({'error': 'Alert manager not configured'}, status=500)
                
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_resolve_alert(self, request) -> web.Response:
        """Resolve an alert"""
        try:
            alert_id = request.match_info['alert_id']
            
            if self.alert_manager:
                success = self.alert_manager.resolve_alert(alert_id)
                if success:
                    return web.json_response({'status': 'resolved'})
                else:
                    return web.json_response({'error': 'Alert not found'}, status=404)
            else:
                return web.json_response({'error': 'Alert manager not configured'}, status=500)
                
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_resolve_incident(self, request) -> web.Response:
        """Resolve a security incident"""
        try:
            incident_id = request.match_info['incident_id']
            
            if self.security_monitor:
                success = self.security_monitor.resolve_incident(incident_id)
                if success:
                    return web.json_response({'status': 'resolved'})
                else:
                    return web.json_response({'error': 'Incident not found'}, status=404)
            else:
                return web.json_response({'error': 'Security monitor not configured'}, status=500)
                
        except Exception as e:
            logger.error(f"Failed to resolve incident: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _websocket_handler(self, request) -> WebSocketResponse:
        """Handle WebSocket connections for real-time updates"""
        ws = WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        logger.info(f"WebSocket connection established. Total connections: {len(self.websocket_connections)}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Handle client messages (ping, subscribe to specific data, etc.)
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({'error': 'Invalid JSON'}))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        
        finally:
            self.websocket_connections.discard(ws)
            logger.info(f"WebSocket connection closed. Total connections: {len(self.websocket_connections)}")
        
        return ws
    
    async def _handle_websocket_message(self, ws: WebSocketResponse, data: Dict[str, Any]):
        """Handle WebSocket message from client"""
        message_type = data.get('type', 'unknown')
        
        if message_type == 'ping':
            await ws.send_str(json.dumps({'type': 'pong', 'timestamp': time.time()}))
        elif message_type == 'subscribe':
            # Client wants to subscribe to specific data streams
            # This could be extended to support selective subscriptions
            await ws.send_str(json.dumps({'type': 'subscribed', 'streams': ['all']}))
        else:
            await ws.send_str(json.dumps({'type': 'error', 'message': f'Unknown message type: {message_type}'}))
    
    async def _dashboard_index(self, request) -> web.Response:
        """Serve basic dashboard HTML"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MemMimic Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background-color: #f5f5f5;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
        .status-critical { color: #c0392b; }
        .btn { 
            background: #3498db; color: white; border: none; padding: 8px 16px; 
            border-radius: 4px; cursor: pointer; margin: 2px;
        }
        .btn:hover { background: #2980b9; }
        .btn-warning { background: #f39c12; }
        .btn-danger { background: #e74c3c; }
        .alert-item { 
            padding: 10px; margin: 5px 0; border-radius: 4px; 
            border-left: 4px solid #3498db;
        }
        .alert-warning { border-left-color: #f39c12; background-color: #fef9e7; }
        .alert-error { border-left-color: #e74c3c; background-color: #fdedec; }
        .alert-critical { border-left-color: #c0392b; background-color: #fadbd8; }
        #status { margin-top: 10px; padding: 10px; background: #ecf0f1; border-radius: 4px; }
        .loading { text-align: center; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MemMimic Monitoring Dashboard</h1>
            <p>Enterprise-grade production monitoring and observability</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>System Status</h3>
                <div id="system-status" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Health Checks</h3>
                <div id="health-status" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Performance Metrics</h3>
                <div id="performance-metrics" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Active Alerts</h3>
                <div id="active-alerts" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Security Events</h3>
                <div id="security-events" class="loading">Loading...</div>
            </div>
        </div>
        
        <div id="status">
            Connection Status: <span id="connection-status">Connecting...</span>
        </div>
    </div>
    
    <script>
        class MemMimicDashboard {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.init();
            }
            
            init() {
                this.connectWebSocket();
                this.loadInitialData();
                
                // Refresh data every 30 seconds as fallback
                setInterval(() => this.loadInitialData(), 30000);
            }
            
            connectWebSocket() {
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.reconnectAttempts = 0;
                    document.getElementById('connection-status').textContent = 'Connected';
                    document.getElementById('connection-status').style.color = '#27ae60';
                };
                
                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleWebSocketMessage(data);
                    } catch (e) {
                        console.error('WebSocket message parse error:', e);
                    }
                };
                
                this.ws.onclose = () => {
                    document.getElementById('connection-status').textContent = 'Disconnected';
                    document.getElementById('connection-status').style.color = '#e74c3c';
                    this.scheduleReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            
            scheduleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
                    setTimeout(() => this.connectWebSocket(), delay);
                }
            }
            
            handleWebSocketMessage(data) {
                if (data.type === 'update') {
                    // Handle real-time updates
                    this.updateDashboard(data.payload);
                }
            }
            
            async loadInitialData() {
                try {
                    await Promise.all([
                        this.loadSystemStatus(),
                        this.loadHealthStatus(),
                        this.loadPerformanceMetrics(),
                        this.loadActiveAlerts(),
                        this.loadSecurityEvents()
                    ]);
                } catch (e) {
                    console.error('Failed to load initial data:', e);
                }
            }
            
            async loadSystemStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    this.renderSystemStatus(data);
                } catch (e) {
                    document.getElementById('system-status').innerHTML = '<div class="status-error">Error loading system status</div>';
                }
            }
            
            async loadHealthStatus() {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    this.renderHealthStatus(data);
                } catch (e) {
                    document.getElementById('health-status').innerHTML = '<div class="status-error">Error loading health status</div>';
                }
            }
            
            async loadPerformanceMetrics() {
                try {
                    const response = await fetch('/api/metrics');
                    const data = await response.json();
                    this.renderPerformanceMetrics(data);
                } catch (e) {
                    document.getElementById('performance-metrics').innerHTML = '<div class="status-error">Error loading metrics</div>';
                }
            }
            
            async loadActiveAlerts() {
                try {
                    const response = await fetch('/api/alerts');
                    const data = await response.json();
                    this.renderActiveAlerts(data);
                } catch (e) {
                    document.getElementById('active-alerts').innerHTML = '<div class="status-error">Error loading alerts</div>';
                }
            }
            
            async loadSecurityEvents() {
                try {
                    const response = await fetch('/api/security');
                    const data = await response.json();
                    this.renderSecurityEvents(data);
                } catch (e) {
                    document.getElementById('security-events').innerHTML = '<div class="status-error">Error loading security data</div>';
                }
            }
            
            renderSystemStatus(data) {
                const statusClass = data.system_status === 'operational' ? 'status-healthy' : 'status-error';
                const html = `
                    <div class="metric">
                        <span>System Status:</span>
                        <span class="${statusClass}">${data.system_status}</span>
                    </div>
                    <div class="metric">
                        <span>Metrics Collector:</span>
                        <span class="status-healthy">${data.components.metrics_collector}</span>
                    </div>
                    <div class="metric">
                        <span>Health Monitor:</span>
                        <span class="${data.components.health_monitor === 'operational' ? 'status-healthy' : 'status-warning'}">${data.components.health_monitor}</span>
                    </div>
                    <div class="metric">
                        <span>Security Monitor:</span>
                        <span class="${data.components.security_monitor === 'operational' ? 'status-healthy' : 'status-warning'}">${data.components.security_monitor}</span>
                    </div>
                    <div class="metric">
                        <span>Alert Manager:</span>
                        <span class="${data.components.alert_manager === 'operational' ? 'status-healthy' : 'status-warning'}">${data.components.alert_manager}</span>
                    </div>
                `;
                document.getElementById('system-status').innerHTML = html;
            }
            
            renderHealthStatus(data) {
                if (data.status === 'unknown') {
                    document.getElementById('health-status').innerHTML = '<div>Health monitoring not configured</div>';
                    return;
                }
                
                const statusClass = this.getStatusClass(data.status);
                const html = `
                    <div class="metric">
                        <span>Overall Status:</span>
                        <span class="${statusClass}">${data.status}</span>
                    </div>
                    <div class="metric">
                        <span>Healthy Checks:</span>
                        <span>${data.healthy_checks}/${data.total_checks}</span>
                    </div>
                    <div class="metric">
                        <span>Response Time:</span>
                        <span>${data.overall_response_time_ms?.toFixed(1)}ms</span>
                    </div>
                `;
                document.getElementById('health-status').innerHTML = html;
            }
            
            renderPerformanceMetrics(data) {
                const metrics = data.metrics;
                const gauges = metrics.gauges || {};
                const counters = metrics.counters || {};
                
                const html = `
                    <div class="metric">
                        <span>Memory Usage:</span>
                        <span>${this.formatBytes(gauges.memmimic_memory_usage_bytes || 0)}</span>
                    </div>
                    <div class="metric">
                        <span>CPU Usage:</span>
                        <span>${(gauges.memmimic_cpu_usage_percent || 0).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Total Errors:</span>
                        <span>${counters.memmimic_errors_total || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Health Score:</span>
                        <span>${((gauges.memmimic_health_score || 0) * 100).toFixed(1)}%</span>
                    </div>
                `;
                document.getElementById('performance-metrics').innerHTML = html;
            }
            
            renderActiveAlerts(data) {
                if (!data.active_alerts || data.active_alerts.length === 0) {
                    document.getElementById('active-alerts').innerHTML = '<div class="status-healthy">No active alerts</div>';
                    return;
                }
                
                const html = data.active_alerts.slice(0, 5).map(alert => `
                    <div class="alert-item alert-${alert.severity}">
                        <div><strong>${alert.name}</strong></div>
                        <div>${alert.message}</div>
                        <div>
                            <button class="btn btn-warning" onclick="dashboard.acknowledgeAlert('${alert.alert_id}')">Acknowledge</button>
                            <button class="btn btn-danger" onclick="dashboard.resolveAlert('${alert.alert_id}')">Resolve</button>
                        </div>
                    </div>
                `).join('');
                
                document.getElementById('active-alerts').innerHTML = html;
            }
            
            renderSecurityEvents(data) {
                if (!data.recent_events || data.recent_events.length === 0) {
                    document.getElementById('security-events').innerHTML = '<div class="status-healthy">No recent security events</div>';
                    return;
                }
                
                const html = `
                    <div class="metric">
                        <span>Events (1h):</span>
                        <span>${data.summary?.recent_events_1h || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Open Incidents:</span>
                        <span>${data.summary?.open_incidents || 0}</span>
                    </div>
                    <div style="max-height: 200px; overflow-y: auto;">
                        ${data.recent_events.slice(0, 5).map(event => `
                            <div class="alert-item alert-${event.severity}">
                                <div><strong>${event.event_type}</strong></div>
                                <div>${event.message}</div>
                                <div><small>${new Date(event.timestamp).toLocaleString()}</small></div>
                            </div>
                        `).join('')}
                    </div>
                `;
                
                document.getElementById('security-events').innerHTML = html;
            }
            
            getStatusClass(status) {
                switch (status) {
                    case 'healthy': return 'status-healthy';
                    case 'degraded': return 'status-warning';
                    case 'unhealthy': return 'status-error';
                    case 'critical': return 'status-critical';
                    default: return '';
                }
            }
            
            formatBytes(bytes) {
                if (bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            async acknowledgeAlert(alertId) {
                try {
                    const response = await fetch(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' });
                    if (response.ok) {
                        this.loadActiveAlerts();
                    }
                } catch (e) {
                    console.error('Failed to acknowledge alert:', e);
                }
            }
            
            async resolveAlert(alertId) {
                try {
                    const response = await fetch(`/api/alerts/${alertId}/resolve`, { method: 'POST' });
                    if (response.ok) {
                        this.loadActiveAlerts();
                    }
                } catch (e) {
                    console.error('Failed to resolve alert:', e);
                }
            }
        }
        
        // Initialize dashboard when page loads
        const dashboard = new MemMimicDashboard();
    </script>
</body>
</html>
        """
        
        return web.Response(text=html_content, content_type='text/html')
    
    async def _start_update_task(self):
        """Start background task for pushing updates to WebSocket clients"""
        while True:
            try:
                if self.websocket_connections:
                    # Prepare update data
                    update_data = {
                        'type': 'update',
                        'timestamp': time.time(),
                        'payload': {
                            'metrics': json.loads(self.metrics_collector.export_json_format()),
                            'system_status': 'operational'
                        }
                    }
                    
                    # Add health data if available
                    if self.health_monitor and self.health_monitor.last_health_result:
                        update_data['payload']['health'] = self.health_monitor.get_health_summary()
                    
                    # Add security data if available
                    if self.security_monitor:
                        update_data['payload']['security'] = self.security_monitor.get_security_summary()
                    
                    # Add alert data if available
                    if self.alert_manager:
                        update_data['payload']['alerts'] = self.alert_manager.get_alert_summary()
                    
                    # Send to all connected clients
                    disconnected = set()
                    for ws in self.websocket_connections:
                        try:
                            await ws.send_str(json.dumps(update_data))
                        except Exception as e:
                            logger.debug(f"Failed to send WebSocket update: {e}")
                            disconnected.add(ws)
                    
                    # Remove disconnected clients
                    self.websocket_connections -= disconnected
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Update task error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def start_server(self):
        """Start the dashboard server"""
        # Start update task
        self._update_task = asyncio.create_task(self._start_update_task())
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
        logger.info(f"Dashboard available at http://{self.host}:{self.port}/dashboard")
        logger.info(f"Prometheus metrics at http://{self.host}:{self.port}/api/metrics/prometheus")
        
        return runner
    
    async def stop_server(self, runner):
        """Stop the dashboard server"""
        if self._update_task:
            self._update_task.cancel()
        
        await runner.cleanup()
        logger.info("Dashboard server stopped")


def start_dashboard_server(
    port: int = 8080,
    host: str = "0.0.0.0",
    health_monitor: Optional[HealthMonitor] = None,
    security_monitor: Optional[SecurityMonitor] = None,
    alert_manager: Optional[AlertManager] = None,
    update_interval: float = 5.0
) -> DashboardServer:
    """
    Factory function to create and configure dashboard server.
    
    Args:
        port: HTTP server port
        host: Host to bind to
        health_monitor: HealthMonitor instance
        security_monitor: SecurityMonitor instance
        alert_manager: AlertManager instance
        update_interval: WebSocket update interval
        
    Returns:
        Configured DashboardServer instance
    """
    server = DashboardServer(port=port, host=host, update_interval=update_interval)
    
    if health_monitor:
        server.set_health_monitor(health_monitor)
    
    if security_monitor:
        server.set_security_monitor(security_monitor)
    
    if alert_manager:
        server.set_alert_manager(alert_manager)
    
    return server