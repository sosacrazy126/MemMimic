"""
Visual Monitoring Dashboard for MemMimic v2.0

Real-time visual monitoring interface with comprehensive metrics display,
performance visualization, and interactive monitoring capabilities.
"""

import json
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

from ..errors import get_error_logger
from .collector import TelemetryCollector, get_telemetry_collector
from .aggregator import MetricsAggregator, get_metrics_aggregator
from .monitor import PerformanceMonitor, get_performance_monitor
from .alerts import AlertingSystem, get_alerting_system, AlertSeverity

logger = get_error_logger("telemetry.dashboard")


class DashboardUpdateMode(Enum):
    """Dashboard update modes"""
    REAL_TIME = "real_time"     # Live updates every few seconds
    PERIODIC = "periodic"       # Scheduled updates  
    ON_DEMAND = "on_demand"     # Manual refresh only


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    title: str
    widget_type: str  # chart, table, metric, alert, status
    data_source: str  # Which component provides data
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "width": 4, "height": 3})
    refresh_interval: int = 30  # seconds
    enabled: bool = True


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    name: str
    description: str
    widgets: List[DashboardWidget]
    auto_refresh: bool = True
    refresh_interval: int = 30
    theme: str = "light"


class TelemetryDashboard:
    """
    Visual Monitoring Dashboard with real-time metrics display.
    
    Features:
    - Real-time performance visualization
    - Customizable dashboard layouts
    - Interactive metric exploration
    - Alert status display
    - System health overview
    - Performance trend analysis
    - Export capabilities (JSON, HTML)
    - Web interface compatible data
    """
    
    def __init__(
        self,
        collector: Optional[TelemetryCollector] = None,
        aggregator: Optional[MetricsAggregator] = None,
        monitor: Optional[PerformanceMonitor] = None,
        alerting: Optional[AlertingSystem] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.collector = collector or get_telemetry_collector()
        self.aggregator = aggregator or get_metrics_aggregator()
        self.monitor = monitor or get_performance_monitor()
        self.alerting = alerting or get_alerting_system()
        self.config = config or {}
        
        # Configuration
        self.update_mode = DashboardUpdateMode(self.config.get('update_mode', 'real_time'))
        self.default_refresh_interval = self.config.get('refresh_interval', 30)
        self.max_data_points = self.config.get('max_data_points', 100)
        self.enable_web_interface = self.config.get('enable_web_interface', False)
        
        # Dashboard layouts
        self._layouts: Dict[str, DashboardLayout] = {}
        self._current_layout = None
        
        # Data cache for performance
        self._data_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 30  # seconds
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background data collection
        self._data_thread = None
        self._stop_data_collection = threading.Event()
        
        # Initialize default layouts
        self._initialize_default_layouts()
        
        # Set default layout
        if self._layouts:
            self._current_layout = list(self._layouts.keys())[0]
        
        # Start background data collection if real-time mode
        if self.update_mode == DashboardUpdateMode.REAL_TIME:
            self._start_background_data_collection()
        
        logger.info(f"TelemetryDashboard initialized in {self.update_mode.value} mode")
    
    def _initialize_default_layouts(self):
        """Initialize default dashboard layouts"""
        # Performance Overview Layout
        performance_layout = DashboardLayout(
            name="performance_overview",
            description="Core performance metrics and trends",
            auto_refresh=True,
            refresh_interval=30,
            widgets=[
                DashboardWidget(
                    id="system_health",
                    title="System Health Score",
                    widget_type="metric",
                    data_source="monitor",
                    config={"metric": "health_score", "format": "percentage", "threshold": 85},
                    position={"x": 0, "y": 0, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    id="active_alerts",
                    title="Active Alerts",
                    widget_type="alert",
                    data_source="alerting",
                    config={"show_severity": True, "max_items": 5},
                    position={"x": 3, "y": 0, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    id="response_times",
                    title="Response Time Trends",
                    widget_type="chart",
                    data_source="aggregator",
                    config={"chart_type": "line", "metrics": ["p50", "p95", "p99"], "time_window": 900},
                    position={"x": 0, "y": 2, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="operation_stats",
                    title="Operation Statistics",
                    widget_type="table",
                    data_source="collector",
                    config={"columns": ["operation", "count", "p95_ms", "success_rate"], "sort_by": "p95_ms"},
                    position={"x": 0, "y": 6, "width": 6, "height": 3}
                ),
                DashboardWidget(
                    id="system_resources",
                    title="System Resources",
                    widget_type="chart",
                    data_source="monitor",
                    config={"chart_type": "gauge", "metrics": ["cpu_percent", "memory_percent", "disk_usage_percent"]},
                    position={"x": 6, "y": 0, "width": 3, "height": 4}
                ),
                DashboardWidget(
                    id="telemetry_overhead",
                    title="Telemetry Overhead",
                    widget_type="metric",
                    data_source="collector",
                    config={"metric": "overhead_p95", "format": "milliseconds", "threshold": 1.0},
                    position={"x": 6, "y": 4, "width": 3, "height": 2}
                )
            ]
        )
        
        # Alert Management Layout
        alert_layout = DashboardLayout(
            name="alert_management",
            description="Alert status, escalations, and incident management",
            auto_refresh=True,
            refresh_interval=15,
            widgets=[
                DashboardWidget(
                    id="alert_summary",
                    title="Alert Summary",
                    widget_type="status",
                    data_source="alerting",
                    config={"show_counts": True, "group_by_severity": True},
                    position={"x": 0, "y": 0, "width": 4, "height": 2}
                ),
                DashboardWidget(
                    id="critical_alerts",
                    title="Critical Alerts",
                    widget_type="alert",
                    data_source="alerting",
                    config={"severity_filter": ["critical", "emergency"], "show_details": True},
                    position={"x": 4, "y": 0, "width": 5, "height": 4}
                ),
                DashboardWidget(
                    id="alert_timeline",
                    title="Alert Timeline",
                    widget_type="chart",
                    data_source="alerting",
                    config={"chart_type": "timeline", "time_window": 3600},
                    position={"x": 0, "y": 2, "width": 4, "height": 4}
                ),
                DashboardWidget(
                    id="escalation_status",
                    title="Escalation Status",
                    widget_type="table",
                    data_source="alerting",
                    config={"columns": ["alert_id", "severity", "escalation_level", "status"]},
                    position={"x": 0, "y": 6, "width": 9, "height": 3}
                )
            ]
        )
        
        # System Metrics Layout  
        system_layout = DashboardLayout(
            name="system_metrics",
            description="Detailed system performance and resource metrics",
            auto_refresh=True,
            refresh_interval=60,
            widgets=[
                DashboardWidget(
                    id="cpu_usage_chart",
                    title="CPU Usage Over Time",
                    widget_type="chart",
                    data_source="monitor",
                    config={"chart_type": "area", "metric": "cpu_percent", "time_window": 1800},
                    position={"x": 0, "y": 0, "width": 4, "height": 3}
                ),
                DashboardWidget(
                    id="memory_usage_chart",
                    title="Memory Usage Over Time",
                    widget_type="chart",
                    data_source="monitor",
                    config={"chart_type": "area", "metric": "memory_percent", "time_window": 1800},
                    position={"x": 4, "y": 0, "width": 4, "height": 3}
                ),
                DashboardWidget(
                    id="performance_metrics_table",
                    title="Detailed Performance Metrics",
                    widget_type="table",
                    data_source="aggregator",
                    config={"show_all_operations": True, "include_percentiles": True},
                    position={"x": 0, "y": 3, "width": 8, "height": 4}
                ),
                DashboardWidget(
                    id="telemetry_stats",
                    title="Telemetry System Statistics",
                    widget_type="status",
                    data_source="collector",
                    config={"show_buffer_utilization": True, "show_overhead_details": True},
                    position={"x": 8, "y": 0, "width": 3, "height": 7}
                )
            ]
        )
        
        # Store layouts
        self.add_layout(performance_layout)
        self.add_layout(alert_layout)
        self.add_layout(system_layout)
        
        logger.info(f"Initialized {len(self._layouts)} default dashboard layouts")
    
    def add_layout(self, layout: DashboardLayout) -> None:
        """Add dashboard layout"""
        with self._lock:
            self._layouts[layout.name] = layout
        logger.info(f"Added dashboard layout: {layout.name}")
    
    def get_layout(self, layout_name: str) -> Optional[DashboardLayout]:
        """Get dashboard layout by name"""
        with self._lock:
            return self._layouts.get(layout_name)
    
    def set_current_layout(self, layout_name: str) -> bool:
        """Set current active layout"""
        with self._lock:
            if layout_name in self._layouts:
                self._current_layout = layout_name
                logger.info(f"Set current layout to: {layout_name}")
                return True
        return False
    
    def get_widget_data(self, widget: DashboardWidget, use_cache: bool = True) -> Dict[str, Any]:
        """Get data for specific widget"""
        cache_key = f"{widget.data_source}_{widget.id}"
        current_time = time.time()
        
        # Check cache first
        if (use_cache and 
            cache_key in self._data_cache and 
            cache_key in self._cache_timestamps and
            current_time - self._cache_timestamps[cache_key] < self._cache_ttl):
            return self._data_cache[cache_key]
        
        # Generate fresh data
        try:
            if widget.data_source == "collector":
                data = self._get_collector_widget_data(widget)
            elif widget.data_source == "aggregator":
                data = self._get_aggregator_widget_data(widget)
            elif widget.data_source == "monitor":
                data = self._get_monitor_widget_data(widget)
            elif widget.data_source == "alerting":
                data = self._get_alerting_widget_data(widget)
            else:
                data = {"error": f"Unknown data source: {widget.data_source}"}
            
            # Add metadata
            data.update({
                "widget_id": widget.id,
                "widget_type": widget.widget_type,
                "timestamp": datetime.now().isoformat(),
                "refresh_interval": widget.refresh_interval
            })
            
            # Cache data
            if use_cache:
                with self._lock:
                    self._data_cache[cache_key] = data
                    self._cache_timestamps[cache_key] = current_time
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to get data for widget {widget.id}: {e}")
            return {
                "widget_id": widget.id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_collector_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get telemetry collector data for widget"""
        if widget.widget_type == "metric":
            # Single metric display
            system_stats = self.collector.get_system_stats()
            metric_name = widget.config.get('metric', 'total_events')
            
            if metric_name == "overhead_p95":
                value = system_stats.get('overhead_stats', {}).get('p95_ms', 0.0)
                threshold = widget.config.get('threshold', 1.0)
                status = "good" if value <= threshold else "warning" if value <= threshold * 1.5 else "critical"
            else:
                value = system_stats.get(metric_name, 0)
                status = "good"
            
            return {
                "type": "metric",
                "value": value,
                "status": status,
                "format": widget.config.get('format', 'number'),
                "threshold": widget.config.get('threshold')
            }
        
        elif widget.widget_type == "table":
            # Operations statistics table
            operations_data = []
            for operation in self.collector._operation_buffers.keys():
                op_stats = self.collector.get_operation_stats(operation)
                if op_stats:
                    operations_data.append({
                        "operation": operation,
                        "count": op_stats.get('total_count', 0),
                        "p95_ms": round(op_stats.get('p95_duration_ms', 0), 2),
                        "success_rate": round(op_stats.get('success_rate', 1.0) * 100, 1)
                    })
            
            # Sort by configured column
            sort_by = widget.config.get('sort_by', 'p95_ms')
            operations_data.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
            
            return {
                "type": "table",
                "columns": widget.config.get('columns', ['operation', 'count', 'p95_ms', 'success_rate']),
                "data": operations_data
            }
        
        elif widget.widget_type == "status":
            # System status overview
            system_stats = self.collector.get_system_stats()
            
            return {
                "type": "status",
                "telemetry_enabled": system_stats.get('enabled', False),
                "total_events": system_stats.get('total_events', 0),
                "total_errors": system_stats.get('total_errors', 0),
                "success_rate": system_stats.get('success_rate_percent', 100.0),
                "operation_types": system_stats.get('operation_types', 0),
                "overhead_stats": system_stats.get('overhead_stats', {}),
                "buffer_utilization": system_stats.get('buffer_utilization', {}),
                "memory_usage": system_stats.get('memory_usage', {})
            }
        
        return {"type": "unknown", "data": {}}
    
    def _get_aggregator_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get metrics aggregator data for widget"""
        if widget.widget_type == "chart":
            # Time series chart data
            chart_type = widget.config.get('chart_type', 'line')
            time_window = widget.config.get('time_window', 900)  # 15 minutes default
            metrics = widget.config.get('metrics', ['p95'])
            
            # Get aggregation summary
            summary = self.aggregator.get_aggregation_summary()
            operations = summary.get('operations', {})
            
            chart_data = {
                "type": "chart",
                "chart_type": chart_type,
                "series": []
            }
            
            # Generate series data for each operation and metric
            for operation, op_data in operations.items():
                for metric in metrics:
                    series_name = f"{operation}_{metric}"
                    
                    # Get time series data
                    time_series = self.aggregator.get_time_series_data(operation, interval_seconds=60)
                    
                    series_data = []
                    for point in time_series[-50:]:  # Last 50 points
                        if metric == 'p95':
                            value = point.value
                        elif metric == 'p50':
                            value = point.metadata.get('p50', point.value)
                        elif metric == 'mean':
                            value = point.metadata.get('mean', point.value)
                        else:
                            value = point.value
                        
                        series_data.append({
                            "x": point.timestamp,
                            "y": value
                        })
                    
                    chart_data["series"].append({
                        "name": series_name,
                        "data": series_data
                    })
            
            return chart_data
        
        elif widget.widget_type == "table":
            # Aggregated performance table
            summary = self.aggregator.get_aggregation_summary()
            operations = summary.get('operations', {})
            
            table_data = []
            for operation, op_data in operations.items():
                recent_stats = op_data.get('recent_stats', {})
                trend = op_data.get('trend_analysis', {})
                
                table_data.append({
                    "operation": operation,
                    "mean_ms": round(recent_stats.get('mean_duration_ms', 0), 2),
                    "p50_ms": round(recent_stats.get('p50_duration_ms', 0), 2),
                    "p95_ms": round(recent_stats.get('p95_duration_ms', 0), 2),
                    "p99_ms": round(recent_stats.get('p99_duration_ms', 0), 2),
                    "success_rate": round(recent_stats.get('success_rate', 1.0) * 100, 1),
                    "trend": trend.get('trend_direction', 'stable'),
                    "anomalies": len(op_data.get('anomalies', []))
                })
            
            return {
                "type": "table",
                "columns": ["operation", "mean_ms", "p95_ms", "success_rate", "trend", "anomalies"],
                "data": table_data
            }
        
        return {"type": "unknown", "data": {}}
    
    def _get_monitor_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get performance monitor data for widget"""
        if widget.widget_type == "metric":
            # Health score or other single metrics
            snapshot = self.monitor.get_performance_snapshot()
            metric_name = widget.config.get('metric', 'health_score')
            
            if metric_name == "health_score":
                value = snapshot.health_score
                threshold = widget.config.get('threshold', 85)
                status = "good" if value >= threshold else "warning" if value >= threshold - 15 else "critical"
            else:
                value = getattr(snapshot.system_resources, metric_name, 0)
                status = "good"
            
            return {
                "type": "metric",
                "value": round(value, 1),
                "status": status,
                "format": widget.config.get('format', 'number'),
                "threshold": widget.config.get('threshold')
            }
        
        elif widget.widget_type == "chart":
            # System resource charts
            chart_type = widget.config.get('chart_type', 'line')
            metric = widget.config.get('metric', 'cpu_percent')
            time_window = widget.config.get('time_window', 1800)  # 30 minutes
            
            # Get historical performance data
            history = self.monitor.get_performance_history(minutes=time_window // 60)
            
            chart_data = {
                "type": "chart",
                "chart_type": chart_type,
                "series": [{
                    "name": metric,
                    "data": [
                        {
                            "x": snapshot.timestamp.timestamp(),
                            "y": getattr(snapshot.system_resources, metric, 0)
                        }
                        for snapshot in history
                    ]
                }]
            }
            
            return chart_data
        
        elif widget.widget_type == "status":
            # Monitoring summary
            summary = self.monitor.get_monitoring_summary()
            
            return {
                "type": "status",
                "health_score": summary.get('health_score', 0),
                "health_status": summary.get('health_status', 'unknown'),
                "active_violations": len(summary.get('active_violations', [])),
                "system_resources": summary.get('system_resources', {}),
                "threshold_summary": summary.get('threshold_summary', {}),
                "performance_targets": summary.get('performance_targets', {})
            }
        
        return {"type": "unknown", "data": {}}
    
    def _get_alerting_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get alerting system data for widget"""
        if widget.widget_type == "alert":
            # Active alerts display
            severity_filter = widget.config.get('severity_filter')
            if severity_filter:
                severity_set = {AlertSeverity(s) for s in severity_filter}
                alerts = self.alerting.get_active_alerts(severity_filter=severity_set)
            else:
                alerts = self.alerting.get_active_alerts()
            
            max_items = widget.config.get('max_items', 10)
            alerts = alerts[:max_items]
            
            alert_data = []
            for alert in alerts:
                alert_data.append({
                    "id": alert.id,
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "created_at": alert.created_at.isoformat(),
                    "escalation_level": alert.escalation_level,
                    "notification_count": alert.notification_count
                })
            
            return {
                "type": "alert",
                "alerts": alert_data,
                "total_count": len(alerts)
            }
        
        elif widget.widget_type == "status":
            # Alert system status
            summary = self.alerting.get_alert_summary()
            
            return {
                "type": "status",
                "active_alerts": summary.get('active_alerts', 0),
                "alerts_by_severity": summary.get('alerts_by_severity', {}),
                "alerts_by_status": summary.get('alerts_by_status', {}),
                "notification_channels": summary.get('notification_channels', 0),
                "escalation_policies": summary.get('escalation_policies', 0),
                "metrics": summary.get('metrics', {})
            }
        
        elif widget.widget_type == "table":
            # Alert details table
            alerts = self.alerting.get_active_alerts()
            
            table_data = []
            for alert in alerts:
                table_data.append({
                    "alert_id": alert.id[:8],  # Shortened ID
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "escalation_level": alert.escalation_level,
                    "created_at": alert.created_at.strftime("%H:%M:%S"),
                    "notifications": alert.notification_count
                })
            
            return {
                "type": "table",
                "columns": ["alert_id", "severity", "status", "escalation_level", "created_at", "notifications"],
                "data": table_data
            }
        
        return {"type": "unknown", "data": {}}
    
    def get_dashboard_data(self, layout_name: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """Get complete dashboard data for specified layout"""
        if layout_name is None:
            layout_name = self._current_layout
        
        if layout_name not in self._layouts:
            return {"error": f"Layout '{layout_name}' not found"}
        
        layout = self._layouts[layout_name]
        dashboard_data = {
            "layout": {
                "name": layout.name,
                "description": layout.description,
                "auto_refresh": layout.auto_refresh,
                "refresh_interval": layout.refresh_interval,
                "theme": layout.theme
            },
            "widgets": {},
            "timestamp": datetime.now().isoformat(),
            "update_mode": self.update_mode.value
        }
        
        # Get data for each widget
        for widget in layout.widgets:
            if widget.enabled:
                try:
                    widget_data = self.get_widget_data(widget, use_cache=use_cache)
                    widget_data["position"] = widget.position
                    widget_data["title"] = widget.title
                    dashboard_data["widgets"][widget.id] = widget_data
                
                except Exception as e:
                    logger.error(f"Failed to get data for widget {widget.id}: {e}")
                    dashboard_data["widgets"][widget.id] = {
                        "widget_id": widget.id,
                        "title": widget.title,
                        "error": str(e),
                        "position": widget.position
                    }
        
        return dashboard_data
    
    def get_dashboard_json(self, layout_name: Optional[str] = None, pretty: bool = True) -> str:
        """Get dashboard data as JSON string"""
        dashboard_data = self.get_dashboard_data(layout_name)
        
        if pretty:
            return json.dumps(dashboard_data, indent=2, default=str)
        else:
            return json.dumps(dashboard_data, default=str)
    
    def export_dashboard_html(self, layout_name: Optional[str] = None) -> str:
        """Export dashboard as HTML (basic visualization)"""
        dashboard_data = self.get_dashboard_data(layout_name)
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MemMimic v2.0 Telemetry Dashboard - {dashboard_data['layout']['name']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; }}
        .dashboard-header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .widget {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .widget-title {{ font-weight: bold; font-size: 16px; margin-bottom: 10px; color: #333; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-good {{ color: #4CAF50; }}
        .metric-warning {{ color: #FF9800; }}
        .metric-critical {{ color: #F44336; }}
        .table {{ width: 100%; border-collapse: collapse; }}
        .table th, .table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .alert-item {{ padding: 10px; margin: 5px 0; border-left: 4px solid; border-radius: 4px; }}
        .alert-critical {{ border-color: #F44336; background-color: #ffebee; }}
        .alert-warning {{ border-color: #FF9800; background-color: #fff3e0; }}
        .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .status-item {{ text-align: center; }}
        .status-label {{ font-size: 14px; color: #666; margin-bottom: 5px; }}
        .status-value {{ font-size: 20px; font-weight: bold; color: #333; }}
        .footer {{ margin-top: 30px; text-align: center; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="dashboard-header">
            <h1>MemMimic v2.0 Telemetry Dashboard</h1>
            <h2>{dashboard_data['layout']['name']}</h2>
            <p>{dashboard_data['layout']['description']}</p>
            <p><strong>Last Updated:</strong> {dashboard_data['timestamp']}</p>
        </div>
        """
        
        # Add widgets
        for widget_id, widget_data in dashboard_data.get('widgets', {}).items():
            html_template += f'<div class="widget">'
            html_template += f'<div class="widget-title">{widget_data.get("title", widget_id)}</div>'
            
            if "error" in widget_data:
                html_template += f'<p style="color: red;">Error: {widget_data["error"]}</p>'
            elif widget_data.get("type") == "metric":
                value = widget_data.get("value", 0)
                status = widget_data.get("status", "good")
                format_type = widget_data.get("format", "number")
                
                if format_type == "percentage":
                    display_value = f"{value:.1f}%"
                elif format_type == "milliseconds":
                    display_value = f"{value:.2f} ms"
                else:
                    display_value = f"{value}"
                
                html_template += f'<div class="metric-value metric-{status}">{display_value}</div>'
                
            elif widget_data.get("type") == "table":
                columns = widget_data.get("columns", [])
                data = widget_data.get("data", [])
                
                html_template += '<table class="table">'
                html_template += '<tr>' + ''.join(f'<th>{col}</th>' for col in columns) + '</tr>'
                
                for row in data[:10]:  # Limit to 10 rows for HTML
                    html_template += '<tr>'
                    for col in columns:
                        html_template += f'<td>{row.get(col, "")}</td>'
                    html_template += '</tr>'
                
                html_template += '</table>'
                
            elif widget_data.get("type") == "alert":
                alerts = widget_data.get("alerts", [])
                
                for alert in alerts:
                    severity = alert.get("severity", "info")
                    css_class = "alert-critical" if severity in ["critical", "emergency"] else "alert-warning"
                    html_template += f'<div class="alert-item {css_class}">'
                    html_template += f'<strong>[{severity.upper()}]</strong> {alert.get("title", "")}'
                    html_template += f'<br><small>{alert.get("message", "")}</small>'
                    html_template += '</div>'
                
            elif widget_data.get("type") == "status":
                # Extract key status metrics
                status_items = []
                for key, value in widget_data.items():
                    if key not in ["type", "widget_id", "title", "timestamp", "position"] and not isinstance(value, dict):
                        status_items.append((key.replace("_", " ").title(), value))
                
                html_template += '<div class="status-grid">'
                for label, value in status_items[:8]:  # Limit display items
                    html_template += '<div class="status-item">'
                    html_template += f'<div class="status-label">{label}</div>'
                    html_template += f'<div class="status-value">{value}</div>'
                    html_template += '</div>'
                html_template += '</div>'
            
            html_template += '</div>'
        
        html_template += f"""
        <div class="footer">
            <p>Generated by MemMimic v2.0 Telemetry System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_template
    
    def get_available_layouts(self) -> List[Dict[str, str]]:
        """Get list of available dashboard layouts"""
        return [
            {
                "name": layout.name,
                "description": layout.description,
                "widget_count": len(layout.widgets)
            }
            for layout in self._layouts.values()
        ]
    
    def clear_cache(self):
        """Clear dashboard data cache"""
        with self._lock:
            self._data_cache.clear()
            self._cache_timestamps.clear()
        logger.info("Dashboard cache cleared")
    
    def _start_background_data_collection(self):
        """Start background data collection for real-time updates"""
        def data_collection_worker():
            while not self._stop_data_collection.wait(self.default_refresh_interval):
                try:
                    # Pre-populate cache with fresh data
                    if self._current_layout in self._layouts:
                        layout = self._layouts[self._current_layout]
                        for widget in layout.widgets:
                            if widget.enabled:
                                self.get_widget_data(widget, use_cache=False)
                
                except Exception as e:
                    logger.error(f"Background data collection failed: {e}")
        
        self._data_thread = threading.Thread(target=data_collection_worker, daemon=True)
        self._data_thread.start()
        logger.info("Background dashboard data collection started")
    
    def shutdown(self):
        """Shutdown telemetry dashboard"""
        try:
            self._stop_data_collection.set()
            if self._data_thread and self._data_thread.is_alive():
                self._data_thread.join(timeout=5.0)
            
            self.clear_cache()
            logger.info("TelemetryDashboard shutdown completed")
            
        except Exception as e:
            logger.error(f"TelemetryDashboard shutdown failed: {e}")


# Global dashboard instance
_global_dashboard: Optional[TelemetryDashboard] = None
_dashboard_lock = threading.Lock()


def get_telemetry_dashboard() -> TelemetryDashboard:
    """Get global telemetry dashboard instance"""
    global _global_dashboard
    
    if _global_dashboard is None:
        with _dashboard_lock:
            if _global_dashboard is None:
                _global_dashboard = TelemetryDashboard()
    
    return _global_dashboard