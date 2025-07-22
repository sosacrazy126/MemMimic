"""
MemMimic v2.0 Production Monitoring and Alerting System
Real-time performance monitoring with automated alerting for production deployment.

Features:
- Continuous performance monitoring 
- Real-time alert generation with severity levels
- Production health scoring and recommendations
- Integration with external monitoring systems (Prometheus, Grafana)
- Automated remediation suggestions
- SLA monitoring and reporting
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum

from memmimic.memory.enhanced_memory import EnhancedMemory
from memmimic.memory.enhanced_amms_storage import EnhancedAMMSStorage
from performance_benchmarking import PerformanceBenchmarkSuite, PerformanceBenchmark


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OUTAGE = "outage"


@dataclass
class Alert:
    """System alert with detailed information"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    description: str
    operation: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    recommendation: str = ""
    tags: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLATarget:
    """Service Level Agreement target definition"""
    name: str
    operation: str
    metric: str  # p95_ms, p99_ms, throughput_ops_sec, success_rate
    target_value: float
    measurement_window_minutes: int = 5
    description: str = ""


@dataclass
class SLAViolation:
    """SLA violation record"""
    sla_name: str
    timestamp: datetime
    target_value: float
    actual_value: float
    violation_duration_minutes: float
    severity: AlertSeverity


@dataclass
class HealthScore:
    """Comprehensive system health score"""
    overall_score: float  # 0.0 - 1.0
    component_scores: Dict[str, float]
    active_alerts: int
    sla_violations: int
    performance_degradation: float
    recommendation: str
    status: HealthStatus


class ProductionMonitor:
    """
    Real-time production monitoring system
    Continuous monitoring with automated alerting and health scoring
    """
    
    # Production SLA Targets based on v2.0 specifications
    DEFAULT_SLA_TARGETS = [
        SLATarget(
            name="summary_retrieval_p95",
            operation="summary_retrieval", 
            metric="p95_ms",
            target_value=5.0,
            description="Summary retrieval 95th percentile latency"
        ),
        SLATarget(
            name="full_context_retrieval_p95",
            operation="full_context_retrieval",
            metric="p95_ms", 
            target_value=50.0,
            description="Full context retrieval 95th percentile latency"
        ),
        SLATarget(
            name="enhanced_store_p95", 
            operation="enhanced_store",
            metric="p95_ms",
            target_value=15.0,
            description="Enhanced memory storage 95th percentile latency"
        ),
        SLATarget(
            name="concurrent_throughput",
            operation="concurrent_mixed_workload",
            metric="throughput_ops_sec",
            target_value=100.0,
            description="Concurrent operations throughput"
        ),
        SLATarget(
            name="cache_hit_rate",
            operation="cache_effectiveness",
            metric="hit_rate", 
            target_value=0.8,
            description="Summary cache hit rate"
        )
    ]
    
    def __init__(self, storage: EnhancedAMMSStorage, environment: str = "production"):
        self.storage = storage
        self.environment = environment
        self.benchmark_suite = PerformanceBenchmarkSuite(storage, environment)
        
        # Monitoring configuration
        self.monitoring_interval_seconds = 60  # Monitor every minute
        self.alert_retention_hours = 24
        self.health_check_interval_seconds = 30
        
        # Alert and SLA tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.sla_targets = self.DEFAULT_SLA_TARGETS.copy()
        self.sla_violations: deque = deque(maxlen=100)
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=288)  # 24 hours at 5-min intervals
        self.baseline_performance: Dict[str, PerformanceBenchmark] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.last_health_check = datetime.now()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler (e.g., for Slack, email, PagerDuty integration)"""
        self.alert_handlers.append(handler)
    
    def add_sla_target(self, target: SLATarget):
        """Add custom SLA target"""
        self.sla_targets.append(target)
    
    async def start_monitoring(self):
        """Start continuous production monitoring"""
        if self.monitoring_active:
            print("⚠️  Monitoring is already active")
            return
        
        print(f"🚀 Starting production monitoring for {self.environment} environment")
        
        # Establish baseline performance if not exists
        if not self.baseline_performance:
            print("📊 Establishing baseline performance...")
            self.baseline_performance = await self.benchmark_suite.establish_performance_baseline()
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._alert_cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except asyncio.CancelledError:
            print("🛑 Monitoring stopped")
        finally:
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        print("🛑 Production monitoring stopped")
    
    async def _performance_monitoring_loop(self):
        """Main performance monitoring loop"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Run performance measurements
                current_performance = await self._measure_current_performance()
                
                # Store performance history
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'performance': current_performance,
                    'measurement_duration_ms': (time.time() - start_time) * 1000
                })
                
                # Check for SLA violations
                await self._check_sla_violations(current_performance)
                
                # Check for performance regressions
                await self._check_performance_regressions(current_performance)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                await self._generate_alert(
                    Alert(
                        id=f"monitoring_error_{int(time.time())}",
                        timestamp=datetime.now(),
                        severity=AlertSeverity.HIGH,
                        title="Performance Monitoring Error",
                        description=f"Error in performance monitoring loop: {e}",
                        recommendation="Check monitoring system logs and restart if necessary",
                        tags=["monitoring", "error"]
                    )
                )
                await asyncio.sleep(30)  # Wait before retry
    
    async def _health_check_loop(self):
        """Health check monitoring loop"""
        while self.monitoring_active:
            try:
                # Calculate current health score
                health_score = await self._calculate_health_score()
                
                # Generate health-based alerts
                if health_score.status in [HealthStatus.CRITICAL, HealthStatus.OUTAGE]:
                    await self._generate_alert(
                        Alert(
                            id=f"health_critical_{int(time.time())}",
                            timestamp=datetime.now(),
                            severity=AlertSeverity.CRITICAL,
                            title="System Health Critical",
                            description=f"System health score: {health_score.overall_score:.1%}",
                            recommendation=health_score.recommendation,
                            tags=["health", "critical"]
                        )
                    )
                elif health_score.status == HealthStatus.DEGRADED:
                    await self._generate_alert(
                        Alert(
                            id=f"health_degraded_{int(time.time())}",
                            timestamp=datetime.now(),
                            severity=AlertSeverity.HIGH,
                            title="System Health Degraded", 
                            description=f"System health score: {health_score.overall_score:.1%}",
                            recommendation=health_score.recommendation,
                            tags=["health", "degraded"]
                        )
                    )
                
                self.last_health_check = datetime.now()
                await asyncio.sleep(self.health_check_interval_seconds)
                
            except Exception as e:
                print(f"❌ Health check error: {e}")
                await asyncio.sleep(30)
    
    async def _alert_cleanup_loop(self):
        """Clean up old alerts and maintain alert history"""
        while self.monitoring_active:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
                
                # Clean up resolved alerts older than retention period
                alerts_to_remove = []
                for alert_id, alert in self.active_alerts.items():
                    if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time:
                        alerts_to_remove.append(alert_id)
                
                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]
                
                # Clean up old SLA violations
                cutoff_time_sla = datetime.now() - timedelta(days=7)
                while self.sla_violations and self.sla_violations[0].timestamp < cutoff_time_sla:
                    self.sla_violations.popleft()
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except Exception as e:
                print(f"❌ Alert cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _measure_current_performance(self) -> Dict[str, Any]:
        """Measure current system performance"""
        # Run lightweight performance measurements
        performance_data = {}
        
        # Test summary retrieval (sample of recent memories)
        summary_times = []
        for i in range(10):  # Quick sample
            start_time = time.perf_counter()
            # Test retrieval of a memory that might exist
            await self.storage.retrieve_summary_optimized(str(i + 1))
            elapsed = (time.perf_counter() - start_time) * 1000
            summary_times.append(elapsed)
        
        if summary_times:
            performance_data['summary_retrieval'] = {
                'mean_ms': sum(summary_times) / len(summary_times),
                'p95_ms': sorted(summary_times)[int(len(summary_times) * 0.95)],
                'samples': len(summary_times)
            }
        
        # Test storage operation
        test_memory = EnhancedMemory(
            content=f"Monitor test {datetime.now().isoformat()}",
            tags=["monitoring", "test"]
        )
        
        start_time = time.perf_counter()
        try:
            memory_id = await self.storage.store_enhanced_memory_optimized(test_memory)
            store_time = (time.perf_counter() - start_time) * 1000
            success = memory_id is not None
        except Exception:
            store_time = (time.perf_counter() - start_time) * 1000
            success = False
        
        performance_data['enhanced_store'] = {
            'mean_ms': store_time,
            'p95_ms': store_time,
            'success': success
        }
        
        # Get cache statistics if available
        if hasattr(self.storage, 'get_enhanced_stats'):
            storage_stats = self.storage.get_enhanced_stats()
            cache_stats = storage_stats.get('cache_stats', {})
            if cache_stats.get('enabled'):
                performance_data['cache_effectiveness'] = {
                    'hit_rate': cache_stats.get('hit_rate', 0.0),
                    'size': cache_stats.get('size', 0),
                    'max_size': cache_stats.get('max_size', 1000)
                }
        
        return performance_data
    
    async def _check_sla_violations(self, performance_data: Dict[str, Any]):
        """Check current performance against SLA targets"""
        for sla in self.sla_targets:
            operation_data = performance_data.get(sla.operation)
            if not operation_data:
                continue
            
            # Extract metric value
            metric_value = None
            if sla.metric == "p95_ms":
                metric_value = operation_data.get('p95_ms')
            elif sla.metric == "throughput_ops_sec":
                metric_value = operation_data.get('throughput_ops_sec')
            elif sla.metric == "hit_rate":
                metric_value = operation_data.get('hit_rate')
            elif sla.metric == "success_rate":
                metric_value = operation_data.get('success_rate', 1.0)
            
            if metric_value is None:
                continue
            
            # Check for SLA violation
            violation = False
            if sla.metric in ["p95_ms", "p99_ms"]:  # Lower is better
                violation = metric_value > sla.target_value
            else:  # Higher is better
                violation = metric_value < sla.target_value
            
            if violation:
                # Record SLA violation
                sla_violation = SLAViolation(
                    sla_name=sla.name,
                    timestamp=datetime.now(),
                    target_value=sla.target_value,
                    actual_value=metric_value,
                    violation_duration_minutes=1.0,  # Simplified - would track actual duration
                    severity=self._calculate_violation_severity(sla, metric_value)
                )
                self.sla_violations.append(sla_violation)
                
                # Generate alert
                await self._generate_alert(
                    Alert(
                        id=f"sla_violation_{sla.name}_{int(time.time())}",
                        timestamp=datetime.now(),
                        severity=sla_violation.severity,
                        title=f"SLA Violation: {sla.name}",
                        description=f"{sla.description}: {metric_value:.2f} violates target {sla.target_value}",
                        operation=sla.operation,
                        metric_value=metric_value,
                        threshold_value=sla.target_value,
                        recommendation=self._get_sla_violation_recommendation(sla, metric_value),
                        tags=["sla", "violation", sla.operation]
                    )
                )
    
    def _calculate_violation_severity(self, sla: SLATarget, actual_value: float) -> AlertSeverity:
        """Calculate violation severity based on how far from target"""
        if sla.metric in ["p95_ms", "p99_ms"]:  # Higher is worse
            violation_ratio = actual_value / sla.target_value
            if violation_ratio > 5.0:
                return AlertSeverity.CRITICAL
            elif violation_ratio > 3.0:
                return AlertSeverity.HIGH
            elif violation_ratio > 1.5:
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW
        else:  # Lower is worse
            violation_ratio = sla.target_value / actual_value if actual_value > 0 else float('inf')
            if violation_ratio > 3.0:
                return AlertSeverity.CRITICAL
            elif violation_ratio > 2.0:
                return AlertSeverity.HIGH
            elif violation_ratio > 1.3:
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW
    
    def _get_sla_violation_recommendation(self, sla: SLATarget, actual_value: float) -> str:
        """Get specific recommendation for SLA violation"""
        if sla.operation == "summary_retrieval":
            if actual_value > sla.target_value * 2:
                return "Check cache performance and database query optimization. Consider increasing cache size."
            else:
                return "Monitor cache hit rate and consider database index optimization."
        
        elif sla.operation == "full_context_retrieval":
            if actual_value > sla.target_value * 2:
                return "Investigate database performance and connection pool settings. Check for slow queries."
            else:
                return "Monitor database performance and consider query optimization."
        
        elif sla.operation == "enhanced_store":
            if actual_value > sla.target_value * 2:
                return "Check database write performance and transaction settings. Monitor disk I/O."
            else:
                return "Monitor storage performance and consider batch optimization."
        
        elif sla.operation == "concurrent_mixed_workload":
            return "Increase connection pool size or optimize database connection handling."
        
        elif sla.operation == "cache_effectiveness":
            return "Increase cache size or review cache eviction policy. Check memory usage."
        
        return "Review system performance and resource utilization."
    
    async def _check_performance_regressions(self, performance_data: Dict[str, Any]):
        """Check for performance regressions against baseline"""
        if not self.baseline_performance:
            return
        
        for operation, baseline in self.baseline_performance.items():
            current_data = performance_data.get(operation)
            if not current_data:
                continue
            
            current_p95 = current_data.get('p95_ms', 0)
            if current_p95 <= 0 or baseline.p95_ms <= 0:
                continue
            
            regression_percent = ((current_p95 - baseline.p95_ms) / baseline.p95_ms) * 100
            
            # Generate regression alert if significant
            if regression_percent > 25:  # >25% regression
                severity = AlertSeverity.CRITICAL if regression_percent > 100 else AlertSeverity.HIGH
                await self._generate_alert(
                    Alert(
                        id=f"regression_{operation}_{int(time.time())}",
                        timestamp=datetime.now(),
                        severity=severity,
                        title=f"Performance Regression: {operation}",
                        description=f"P95 latency increased by {regression_percent:.1f}%: {current_p95:.2f}ms vs baseline {baseline.p95_ms:.2f}ms",
                        operation=operation,
                        metric_value=current_p95,
                        threshold_value=baseline.p95_ms,
                        recommendation=f"Investigate performance regression in {operation}. Consider rollback if severe.",
                        tags=["regression", "performance", operation]
                    )
                )
    
    async def _calculate_health_score(self) -> HealthScore:
        """Calculate comprehensive system health score"""
        # Count active alerts by severity
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL and not a.resolved])
        high_alerts = len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.HIGH and not a.resolved])
        medium_alerts = len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.MEDIUM and not a.resolved])
        
        # Count recent SLA violations
        recent_violations = len([
            v for v in self.sla_violations 
            if v.timestamp > datetime.now() - timedelta(hours=1)
        ])
        
        # Calculate component scores
        component_scores = {}
        
        # Performance score (based on recent performance vs baseline)
        performance_score = 1.0
        if self.performance_history:
            recent_performance = self.performance_history[-1]['performance']
            for operation, baseline in self.baseline_performance.items():
                current_data = recent_performance.get(operation)
                if current_data and baseline.p95_ms > 0:
                    current_p95 = current_data.get('p95_ms', 0)
                    if current_p95 > 0:
                        ratio = current_p95 / baseline.p95_ms
                        # Score decreases as performance degrades
                        op_score = max(0.0, min(1.0, 2.0 - ratio))
                        performance_score = min(performance_score, op_score)
        
        component_scores['performance'] = performance_score
        
        # Alert score (based on active alerts)
        alert_penalty = critical_alerts * 0.4 + high_alerts * 0.2 + medium_alerts * 0.1
        alert_score = max(0.0, 1.0 - alert_penalty)
        component_scores['alerts'] = alert_score
        
        # SLA score (based on recent violations)
        sla_penalty = recent_violations * 0.1
        sla_score = max(0.0, 1.0 - sla_penalty)
        component_scores['sla_compliance'] = sla_score
        
        # Availability score (simplified - would integrate with actual uptime monitoring)
        availability_score = 1.0 if critical_alerts == 0 else 0.7
        component_scores['availability'] = availability_score
        
        # Calculate overall score (weighted average)
        overall_score = (
            performance_score * 0.4 +
            alert_score * 0.3 + 
            sla_score * 0.2 +
            availability_score * 0.1
        )
        
        # Determine status
        if overall_score >= 0.9:
            status = HealthStatus.HEALTHY
        elif overall_score >= 0.7:
            status = HealthStatus.WARNING
        elif overall_score >= 0.5:
            status = HealthStatus.DEGRADED
        elif overall_score >= 0.2:
            status = HealthStatus.CRITICAL
        else:
            status = HealthStatus.OUTAGE
        
        # Generate recommendation
        recommendation = self._generate_health_recommendation(component_scores, critical_alerts, high_alerts, recent_violations)
        
        return HealthScore(
            overall_score=overall_score,
            component_scores=component_scores,
            active_alerts=len(self.active_alerts),
            sla_violations=recent_violations,
            performance_degradation=1.0 - performance_score,
            recommendation=recommendation,
            status=status
        )
    
    def _generate_health_recommendation(self, component_scores: Dict[str, float], critical_alerts: int, high_alerts: int, recent_violations: int) -> str:
        """Generate health-based recommendation"""
        recommendations = []
        
        if critical_alerts > 0:
            recommendations.append(f"Address {critical_alerts} critical alerts immediately")
        
        if high_alerts > 0:
            recommendations.append(f"Review {high_alerts} high-priority alerts")
        
        if component_scores.get('performance', 1.0) < 0.7:
            recommendations.append("Performance degradation detected - investigate slow operations")
        
        if recent_violations > 0:
            recommendations.append(f"{recent_violations} SLA violations in last hour - review service targets")
        
        if component_scores.get('availability', 1.0) < 0.9:
            recommendations.append("Availability concerns - ensure system redundancy")
        
        if not recommendations:
            recommendations.append("System operating normally - continue monitoring")
        
        return "; ".join(recommendations)
    
    async def _generate_alert(self, alert: Alert):
        """Generate and process new alert"""
        # Check for duplicate alerts
        existing_alert_key = f"{alert.title}_{alert.operation}"
        existing_alert = None
        for existing in self.active_alerts.values():
            if existing.title == alert.title and existing.operation == alert.operation and not existing.resolved:
                existing_alert = existing
                break
        
        if existing_alert:
            # Update existing alert if severity increased
            if alert.severity.value in ["critical", "high"] and existing_alert.severity.value not in ["critical", "high"]:
                existing_alert.severity = alert.severity
                existing_alert.description = alert.description
                existing_alert.timestamp = alert.timestamp
                print(f"🔄 Updated alert: {alert.title}")
            return
        
        # Store new alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Print alert (in production, this would go to logging system)
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.LOW: "🟡",
            AlertSeverity.MEDIUM: "🟠", 
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        print(f"{severity_emoji[alert.severity]} {alert.severity.value.upper()}: {alert.title}")
        print(f"   {alert.description}")
        if alert.recommendation:
            print(f"   💡 {alert.recommendation}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"❌ Alert handler error: {e}")
    
    async def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Manually resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            if resolution_note:
                alert.metadata['resolution_note'] = resolution_note
            print(f"✅ Resolved alert: {alert.title}")
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        health_score = asyncio.run(self._calculate_health_score())
        
        # Recent performance data
        recent_performance = {}
        if self.performance_history:
            recent_performance = self.performance_history[-1]['performance']
        
        # Active alerts summary
        alerts_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            if not alert.resolved:
                alerts_by_severity[alert.severity.value] += 1
        
        # SLA compliance summary
        sla_status = {}
        for sla in self.sla_targets:
            recent_violations = len([
                v for v in self.sla_violations
                if v.sla_name == sla.name and v.timestamp > datetime.now() - timedelta(hours=1)
            ])
            sla_status[sla.name] = {
                'target': sla.target_value,
                'violations_last_hour': recent_violations,
                'compliant': recent_violations == 0
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'health_score': {
                'overall': health_score.overall_score,
                'components': health_score.component_scores,
                'status': health_score.status.value,
                'recommendation': health_score.recommendation
            },
            'performance': recent_performance,
            'alerts': {
                'active': len([a for a in self.active_alerts.values() if not a.resolved]),
                'by_severity': dict(alerts_by_severity),
                'recent': [
                    {
                        'id': alert.id,
                        'title': alert.title,
                        'severity': alert.severity.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            },
            'sla_compliance': sla_status,
            'monitoring_stats': {
                'uptime_minutes': (datetime.now() - self.last_health_check).total_seconds() / 60,
                'performance_samples': len(self.performance_history),
                'alert_handlers': len(self.alert_handlers)
            }
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format for external monitoring"""
        metrics = []
        timestamp = int(time.time() * 1000)
        
        # Health score metrics
        health_score = asyncio.run(self._calculate_health_score())
        metrics.append(f'memmimic_health_score{{environment="{self.environment}"}} {health_score.overall_score} {timestamp}')
        
        for component, score in health_score.component_scores.items():
            metrics.append(f'memmimic_component_health{{environment="{self.environment}",component="{component}"}} {score} {timestamp}')
        
        # Alert metrics
        alerts_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            if not alert.resolved:
                alerts_by_severity[alert.severity.value] += 1
        
        for severity, count in alerts_by_severity.items():
            metrics.append(f'memmimic_active_alerts{{environment="{self.environment}",severity="{severity}"}} {count} {timestamp}')
        
        # Performance metrics
        if self.performance_history:
            recent_performance = self.performance_history[-1]['performance']
            for operation, data in recent_performance.items():
                for metric, value in data.items():
                    if isinstance(value, (int, float)):
                        metrics.append(f'memmimic_performance_{metric}{{environment="{self.environment}",operation="{operation}"}} {value} {timestamp}')
        
        # SLA metrics
        for sla in self.sla_targets:
            recent_violations = len([
                v for v in self.sla_violations
                if v.sla_name == sla.name and v.timestamp > datetime.now() - timedelta(hours=1)
            ])
            metrics.append(f'memmimic_sla_violations_1h{{environment="{self.environment}",sla="{sla.name}"}} {recent_violations} {timestamp}')
        
        return '\n'.join(metrics)


# Example alert handlers
def console_alert_handler(alert: Alert):
    """Simple console alert handler for development"""
    print(f"🔔 ALERT: {alert.title} [{alert.severity.value}]")
    print(f"   {alert.description}")

def webhook_alert_handler(webhook_url: str):
    """Create webhook alert handler for integration with external systems"""
    def handler(alert: Alert):
        import requests
        payload = {
            'alert_id': alert.id,
            'timestamp': alert.timestamp.isoformat(),
            'severity': alert.severity.value,
            'title': alert.title,
            'description': alert.description,
            'recommendation': alert.recommendation
        }
        try:
            requests.post(webhook_url, json=payload, timeout=5)
        except requests.RequestException as e:
            print(f"❌ Webhook alert failed: {e}")
    return handler

def slack_alert_handler(webhook_url: str):
    """Create Slack alert handler"""
    def handler(alert: Alert):
        import requests
        
        severity_colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.LOW: "#ffeb3b",
            AlertSeverity.MEDIUM: "#ff9800",
            AlertSeverity.HIGH: "#f44336", 
            AlertSeverity.CRITICAL: "#9c27b0"
        }
        
        payload = {
            "attachments": [
                {
                    "color": severity_colors.get(alert.severity, "#36a64f"),
                    "title": f"{alert.severity.value.upper()}: {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Operation",
                            "value": alert.operation or "N/A",
                            "short": True
                        },
                        {
                            "title": "Recommendation", 
                            "value": alert.recommendation or "No recommendation",
                            "short": False
                        }
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }
            ]
        }
        
        try:
            requests.post(webhook_url, json=payload, timeout=5)
        except requests.RequestException as e:
            print(f"❌ Slack alert failed: {e}")
    
    return handler


# CLI interface for monitoring
async def main():
    """Run production monitoring"""
    import tempfile
    from pathlib import Path
    
    print("🚀 MemMimic v2.0 Production Monitoring System")
    print("=" * 50)
    
    # Create temporary storage for monitoring demo
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        config = {
            'enable_summary_cache': True,
            'summary_cache_size': 1000
        }
        
        storage = EnhancedAMMSStorage(db_path, config=config)
        monitor = ProductionMonitor(storage, environment="demo")
        
        # Add alert handlers
        monitor.add_alert_handler(console_alert_handler)
        
        print("🎯 Setting up monitoring system...")
        
        # Run monitoring for demo (limited time)
        print("🔍 Starting monitoring (will run for 2 minutes for demo)...")
        
        try:
            # Start monitoring with timeout
            await asyncio.wait_for(monitor.start_monitoring(), timeout=120)
        except asyncio.TimeoutError:
            print("⏱️  Demo monitoring period completed")
        
        await monitor.stop_monitoring()
        
        # Show final dashboard
        print("\n📊 Final Dashboard Data:")
        dashboard_data = monitor.get_monitoring_dashboard_data()
        print(json.dumps(dashboard_data, indent=2, default=str))
        
        # Show Prometheus metrics
        print("\n📈 Prometheus Metrics:")
        prometheus_metrics = monitor.export_prometheus_metrics()
        print(prometheus_metrics[:1000] + "..." if len(prometheus_metrics) > 1000 else prometheus_metrics)
        
        await storage.close()
        
    finally:
        # Cleanup
        try:
            Path(db_path).unlink()
        except:
            pass
    
    print(f"\n🏁 Production monitoring demo completed!")


if __name__ == "__main__":
    asyncio.run(main())