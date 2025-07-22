"""
MemMimic v2.0 Immutable Audit Logging - Usage Examples

Comprehensive examples demonstrating the audit logging system integration
with MemMimic storage, governance, and telemetry components.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Import audit logging components
from .immutable_logger import ImmutableAuditLogger, AuditEntry
from .cryptographic_verifier import CryptographicVerifier, HashChainVerifier
from .tamper_detector import TamperDetector, TamperAlert, TamperSeverity
from .audit_trail_manager import AuditTrailManager, AuditQuery, AuditQueryFilter, QueryOperation
from .security_metrics import AuditMetrics


class MemMimicAuditSystem:
    """
    Complete MemMimic v2.0 Audit System Integration.
    
    Demonstrates enterprise-grade audit logging with cryptographic verification,
    tamper detection, and comprehensive reporting for all MemMimic operations.
    """
    
    def __init__(self, db_path: str, config: Dict[str, Any] = None):
        """
        Initialize complete audit system.
        
        Args:
            db_path: Path to audit database
            config: System configuration
        """
        self.config = config or {}
        
        # Initialize core audit logger
        self.audit_logger = ImmutableAuditLogger(
            db_path=db_path,
            retention_days=self.config.get('retention_days', 90),
            enable_memory_buffer=True,
            buffer_size=self.config.get('buffer_size', 1000),
            config=self.config
        )
        
        # Initialize cryptographic verification
        verification_key = self.config.get('verification_key', 'default_memmimic_audit_key')
        self.crypto_verifier = CryptographicVerifier(verification_key, self.config)
        self.hash_chain_verifier = HashChainVerifier(self.crypto_verifier)
        
        # Initialize tamper detection
        self.tamper_detector = TamperDetector(self.config.get('tamper_detection', {}))
        
        # Initialize audit trail management
        self.audit_manager = AuditTrailManager(
            audit_logger=self.audit_logger,
            crypto_verifier=self.crypto_verifier,
            hash_chain_verifier=self.hash_chain_verifier,
            tamper_detector=self.tamper_detector,
            config=self.config.get('audit_manager', {})
        )
        
        # Initialize security metrics
        self.metrics = AuditMetrics(self.config.get('metrics', {}))
        
        # Setup alert callbacks
        self._setup_alert_callbacks()
        
        print("MemMimic v2.0 Audit System initialized with enterprise security features")
    
    def _setup_alert_callbacks(self):
        """Setup automated alert handling."""
        def handle_tamper_alert(alert: TamperAlert):
            """Handle tamper detection alerts."""
            print(f"🚨 SECURITY ALERT: {alert.tamper_type.value}")
            print(f"   Severity: {alert.severity.value}")
            print(f"   Description: {alert.description}")
            print(f"   Affected entries: {len(alert.affected_entries)}")
            print(f"   Confidence: {alert.confidence_score:.2%}")
            
            # Log security event
            self.metrics.increment_counter(
                'security_alerts',
                labels={
                    'type': alert.tamper_type.value,
                    'severity': alert.severity.value
                }
            )
            
            # Auto-acknowledge low severity alerts
            if alert.severity in [TamperSeverity.LOW, TamperSeverity.INFO]:
                self.tamper_detector.acknowledge_alert(
                    alert.alert_id, 
                    "Auto-acknowledged low severity alert"
                )
        
        def handle_metric_alert(alert_data: Dict[str, Any]):
            """Handle metric threshold alerts."""
            print(f"📊 METRIC ALERT: {alert_data['metric_name']}")
            print(f"   Threshold: {alert_data['threshold_type']}")
            print(f"   Current: {alert_data['current_value']}")
            print(f"   Limit: {alert_data['threshold_value']}")
        
        # Register callbacks
        self.tamper_detector.register_alert_callback(handle_tamper_alert)
        self.metrics.register_alert_callback(handle_metric_alert)
    
    def audit_memory_operation(
        self,
        operation: str,
        memory_id: str,
        user_context: Dict[str, Any],
        operation_result: Dict[str, Any],
        governance_result: Dict[str, Any] = None,
        performance_data: Dict[str, float] = None
    ) -> str:
        """
        Audit a MemMimic memory operation with comprehensive logging.
        
        Args:
            operation: Operation name (remember, recall, forget, etc.)
            memory_id: Associated memory ID
            user_context: User/session context
            operation_result: Operation result data
            governance_result: Governance validation result
            performance_data: Performance metrics
            
        Returns:
            Audit entry ID
        """
        start_time = time.perf_counter()
        
        try:
            # Prepare governance status
            governance_status = "unknown"
            if governance_result:
                governance_status = governance_result.get('status', 'unknown')
            
            # Prepare performance metrics
            perf_metrics = performance_data or {}
            
            # Determine security level based on operation
            security_level = self._determine_security_level(operation, user_context)
            
            # Generate compliance flags
            compliance_flags = self._generate_compliance_flags(
                operation, governance_result, user_context
            )
            
            # Log audit entry
            entry_id = self.audit_logger.log_audit_entry(
                operation=operation,
                component="memmimic_storage",
                memory_id=memory_id,
                user_context=user_context,
                operation_result=operation_result,
                governance_status=governance_status,
                performance_metrics=perf_metrics,
                security_level=security_level,
                compliance_flags=compliance_flags
            )
            
            # Record metrics
            audit_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_audit_operation(
                operation=operation,
                component="memmimic_storage",
                duration_ms=audit_time,
                success=operation_result.get('success', True),
                memory_id=memory_id
            )
            
            # Record governance metrics if applicable
            if governance_result:
                self.metrics.record_governance_validation(
                    operation=operation,
                    status=governance_status,
                    violations_count=len(governance_result.get('violations', [])),
                    validation_time_ms=governance_result.get('processing_time', 0.0)
                )
            
            return entry_id
            
        except Exception as e:
            print(f"❌ Audit logging failed for operation {operation}: {e}")
            # Record failure metric
            self.metrics.increment_counter(
                'audit_failures',
                labels={'operation': operation, 'error': str(e)[:100]}
            )
            raise
    
    def _determine_security_level(self, operation: str, user_context: Dict[str, Any]) -> str:
        """Determine security classification level for operation."""
        # Check for sensitive operations
        sensitive_operations = {'forget', 'purge', 'admin_operation'}
        if operation in sensitive_operations:
            return "confidential"
        
        # Check user context for elevated permissions
        if user_context.get('admin_user', False):
            return "confidential"
        
        # Check for sensitive data indicators
        if any(key in user_context for key in ['pii', 'financial', 'medical']):
            return "confidential"
        
        return "internal"  # Default level
    
    def _generate_compliance_flags(
        self,
        operation: str,
        governance_result: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> list:
        """Generate compliance flags based on operation context."""
        flags = ['audit_required']
        
        # Add retention requirements
        if operation in ['remember', 'update']:
            flags.append('retention_required')
        
        # Add governance flags
        if governance_result:
            if governance_result.get('violations'):
                flags.append('governance_violations')
            if governance_result.get('warnings'):
                flags.append('governance_warnings')
        
        # Add user context flags
        if user_context.get('gdpr_applicable', False):
            flags.append('gdpr_compliance')
        
        if user_context.get('hipaa_applicable', False):
            flags.append('hipaa_compliance')
        
        return flags
    
    def verify_system_integrity(self) -> Dict[str, Any]:
        """
        Perform comprehensive system integrity verification.
        
        Returns:
            Integrity verification results
        """
        print("🔍 Performing system integrity verification...")
        
        start_time = time.perf_counter()
        
        # Verify hash chain integrity
        chain_result = self.audit_logger.verify_hash_chain_integrity()
        
        # Get tamper detection summary
        recent_alerts = self.tamper_detector.get_recent_alerts(hours=24)
        critical_alerts = [a for a in recent_alerts if a.severity == TamperSeverity.CRITICAL]
        
        # Calculate integrity score
        integrity_score = 100.0
        if not chain_result['valid']:
            integrity_score -= 30.0
        if len(critical_alerts) > 0:
            integrity_score -= len(critical_alerts) * 10.0
        
        integrity_score = max(0, integrity_score)
        
        # Update system health metric
        self.metrics.update_system_health(integrity_score)
        
        verification_time = (time.perf_counter() - start_time) * 1000
        
        result = {
            'integrity_score': integrity_score,
            'hash_chain_valid': chain_result['valid'],
            'entries_verified': chain_result['entries_verified'],
            'broken_links': len(chain_result['broken_links']),
            'critical_alerts': len(critical_alerts),
            'total_alerts': len(recent_alerts),
            'verification_time_ms': verification_time
        }
        
        print(f"✅ System integrity: {integrity_score:.1f}/100")
        print(f"   Hash chain: {'✅ Valid' if chain_result['valid'] else '❌ Compromised'}")
        print(f"   Critical alerts: {len(critical_alerts)}")
        print(f"   Verification time: {verification_time:.2f}ms")
        
        return result
    
    def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        report_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate compliance report for audit trail.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            report_type: Type of report (comprehensive, security, performance)
            
        Returns:
            Compliance report data
        """
        print(f"📋 Generating {report_type} compliance report...")
        
        # Generate audit report
        report = self.audit_manager.generate_audit_report(
            start_time=start_time,
            end_time=end_time,
            title=f"MemMimic v2.0 {report_type.title()} Compliance Report",
            include_performance=(report_type in ['comprehensive', 'performance']),
            include_security=(report_type in ['comprehensive', 'security']),
            include_compliance=True
        )
        
        # Add audit metrics summary
        metrics_summary = self.metrics.get_audit_summary(
            hours=int((end_time - start_time).total_seconds() / 3600)
        )
        
        # Create comprehensive report
        compliance_report = {
            'report_metadata': {
                'report_id': report.report_id,
                'title': report.title,
                'generated_at': report.generated_at.isoformat(),
                'time_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': (end_time - start_time).total_seconds() / 3600
                }
            },
            'audit_summary': {
                'total_entries': report.total_entries,
                'operations_breakdown': report.operations_summary,
                'components_breakdown': report.components_summary
            },
            'security_analysis': {
                'tamper_alerts': len(report.tamper_alerts),
                'integrity_status': report.integrity_status,
                'critical_issues': [
                    alert.to_dict() for alert in report.tamper_alerts
                    if alert.severity == TamperSeverity.CRITICAL
                ]
            },
            'performance_metrics': metrics_summary,
            'compliance_status': report.compliance_status,
            'recommendations': self._generate_recommendations(report, metrics_summary)
        }
        
        print(f"✅ Report generated: {report.total_entries} entries analyzed")
        print(f"   Time period: {(end_time - start_time).days} days")
        print(f"   Security alerts: {len(report.tamper_alerts)}")
        print(f"   System health: {metrics_summary.get('current_health_score', 0):.1f}/100")
        
        return compliance_report
    
    def _generate_recommendations(self, report, metrics_summary) -> list:
        """Generate recommendations based on report analysis."""
        recommendations = []
        
        # Performance recommendations
        avg_audit_time = metrics_summary['system_performance']['avg_audit_time_ms']
        if avg_audit_time > 10.0:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'description': f"Average audit time ({avg_audit_time:.2f}ms) exceeds optimal threshold",
                'suggested_action': 'Consider increasing buffer size or optimizing database indexes'
            })
        
        # Security recommendations
        if report.tamper_alerts:
            critical_alerts = [a for a in report.tamper_alerts if a.severity == TamperSeverity.CRITICAL]
            if critical_alerts:
                recommendations.append({
                    'type': 'security',
                    'priority': 'high',
                    'description': f'{len(critical_alerts)} critical security alerts detected',
                    'suggested_action': 'Investigate tamper attempts and enhance security monitoring'
                })
        
        # Integrity recommendations
        if not report.integrity_status.get('hash_chain_valid', True):
            recommendations.append({
                'type': 'integrity',
                'priority': 'critical',
                'description': 'Hash chain integrity compromised',
                'suggested_action': 'Immediate security investigation required'
            })
        
        # Compliance recommendations
        violation_rate = metrics_summary.get('governance_validations', {}).get('count', 0)
        if violation_rate > 100:  # High volume of governance activity
            recommendations.append({
                'type': 'compliance',
                'priority': 'low',
                'description': f'High governance validation activity ({violation_rate} validations)',
                'suggested_action': 'Review governance policies for optimization opportunities'
            })
        
        return recommendations
    
    def query_audit_history(
        self,
        operation: str = None,
        component: str = None,
        memory_id: str = None,
        hours: int = 24,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Query audit history with common filtering patterns.
        
        Args:
            operation: Filter by operation name
            component: Filter by component
            memory_id: Filter by memory ID
            hours: Hours of history to query
            limit: Maximum entries to return
            
        Returns:
            Query results with entries and metadata
        """
        print(f"🔍 Querying audit history (last {hours}h)...")
        
        # Build query
        start_time = datetime.now(timezone.utc).replace(
            hour=datetime.now().hour - hours,
            minute=0, second=0, microsecond=0
        )
        
        query = AuditQuery(
            start_time=start_time,
            operations=[operation] if operation else None,
            components=[component] if component else None,
            memory_ids=[memory_id] if memory_id else None,
            limit=limit,
            order_by="timestamp",
            order_direction="DESC",
            include_verification_status=True
        )
        
        # Execute query
        result = self.audit_manager.query_audit_trail(query)
        
        # Analyze results
        analysis = {
            'total_found': result.total_count,
            'entries_returned': len(result.entries),
            'query_time_ms': result.query_time_ms,
            'has_more': result.has_more,
            'verification_summary': result.verification_results,
            'entries': [entry.to_dict() for entry in result.entries]
        }
        
        print(f"✅ Found {result.total_count} entries ({len(result.entries)} returned)")
        print(f"   Query time: {result.query_time_ms:.2f}ms")
        if result.verification_results:
            verified = result.verification_results.get('verified_entries', 0)
            total = result.verification_results.get('total_entries', 0)
            print(f"   Integrity: {verified}/{total} entries verified")
        
        return analysis
    
    def export_audit_data(self, file_path: str, format: str = "json") -> str:
        """
        Export audit data for external analysis or compliance.
        
        Args:
            file_path: Export file path
            format: Export format (json, csv)
            
        Returns:
            Export file path
        """
        print(f"📤 Exporting audit data to {file_path} ({format} format)...")
        
        # Query all recent entries
        query = AuditQuery(
            limit=10000,  # Large limit for export
            order_by="timestamp",
            order_direction="ASC"  # Chronological order for export
        )
        
        result = self.audit_manager.query_audit_trail(query)
        
        # Export based on format
        export_path = Path(file_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            import json
            with open(export_path, 'w') as f:
                json.dump({
                    'export_metadata': {
                        'exported_at': datetime.now(timezone.utc).isoformat(),
                        'total_entries': len(result.entries),
                        'format_version': '2.0'
                    },
                    'entries': [entry.to_dict() for entry in result.entries]
                }, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            import csv
            with open(export_path, 'w', newline='') as f:
                if result.entries:
                    # Use first entry to determine fieldnames
                    fieldnames = result.entries[0].to_dict().keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for entry in result.entries:
                        # Flatten complex fields for CSV
                        row = entry.to_dict()
                        for key, value in row.items():
                            if isinstance(value, (dict, list)):
                                row[key] = json.dumps(value)
                        writer.writerow(row)
        
        print(f"✅ Exported {len(result.entries)} entries to {export_path}")
        return str(export_path)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # Get component metrics
        logger_metrics = self.audit_logger.get_metrics()
        crypto_metrics = self.crypto_verifier.get_metrics()
        detector_metrics = self.tamper_detector.get_metrics()
        manager_metrics = self.audit_manager.get_metrics()
        audit_metrics = self.metrics.get_system_metrics()
        
        # Calculate overall health
        health_factors = [
            logger_metrics.get('avg_log_time_ms', 0) < 5.0,  # Fast logging
            crypto_metrics.get('avg_verification_time_ms', 0) < 2.0,  # Fast verification
            detector_metrics.get('critical_alerts', 0) == 0,  # No critical alerts
            logger_metrics.get('hash_chain_length', 0) > 0  # Active logging
        ]
        
        health_score = sum(health_factors) / len(health_factors) * 100
        
        return {
            'overall_health': health_score,
            'audit_logger': logger_metrics,
            'crypto_verifier': crypto_metrics,
            'tamper_detector': detector_metrics,
            'audit_manager': manager_metrics,
            'security_metrics': audit_metrics,
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 60 else 'critical'
        }


def main():
    """Demonstrate complete audit system usage."""
    print("🚀 MemMimic v2.0 Immutable Audit Logging System Demo")
    print("=" * 60)
    
    # Create temporary database for demo
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Initialize audit system
        audit_config = {
            'retention_days': 30,
            'buffer_size': 500,
            'verification_key': 'demo_verification_key_secure_12345',
            'tamper_detection': {
                'max_alerts': 1000,
                'data_retention_hours': 72
            }
        }
        
        audit_system = MemMimicAuditSystem(db_path, audit_config)
        
        print("\n1. 📝 Simulating MemMimic operations with audit logging...")
        
        # Simulate various memory operations
        operations = [
            ("remember", "Creating new memory", {"content": "Important user data"}),
            ("recall", "Retrieving memory", {"query": "user preferences"}),
            ("update", "Updating memory", {"changes": "updated preferences"}),
            ("search", "Searching memories", {"query": "recent activities"}),
            ("forget", "Removing memory", {"reason": "user requested deletion"})
        ]
        
        entry_ids = []
        for i, (op, description, data) in enumerate(operations):
            memory_id = f"mem_{op}_{i}"
            
            # Simulate governance validation
            governance_result = {
                'status': 'approved' if i % 4 != 0 else 'approved_with_warnings',
                'violations': [],
                'warnings': ['size_warning'] if i % 4 == 0 else [],
                'processing_time': 2.5 + (i * 0.5)
            }
            
            # Simulate performance data
            performance_data = {
                'operation_duration_ms': 15.0 + (i * 5),
                'database_time_ms': 8.0 + (i * 2),
                'network_time_ms': 3.0
            }
            
            # Audit the operation
            entry_id = audit_system.audit_memory_operation(
                operation=op,
                memory_id=memory_id,
                user_context={
                    'user_id': f'user_{i % 3}',
                    'session_id': f'session_{i // 2}',
                    'ip_address': f'192.168.1.{100 + i}',
                    'operation_description': description
                },
                operation_result={
                    'success': True,
                    'data': data,
                    'processing_time_ms': performance_data['operation_duration_ms']
                },
                governance_result=governance_result,
                performance_data=performance_data
            )
            
            entry_ids.append(entry_id)
            print(f"   ✅ {op.capitalize()} operation audited: {entry_id[:8]}...")
        
        print(f"\n2. 🔐 Verifying system integrity...")
        integrity_result = audit_system.verify_system_integrity()
        
        print(f"\n3. 🔍 Querying audit history...")
        history = audit_system.query_audit_history(hours=1, limit=10)
        
        print(f"\n4. 📋 Generating compliance report...")
        start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        end_time = datetime.now(timezone.utc)
        
        report = audit_system.generate_compliance_report(
            start_time=start_time,
            end_time=end_time,
            report_type="comprehensive"
        )
        
        print(f"\n5. 📤 Exporting audit data...")
        export_path = audit_system.export_audit_data(
            file_path="/tmp/memmimic_audit_export.json",
            format="json"
        )
        
        print(f"\n6. 📊 System status summary...")
        status = audit_system.get_system_status()
        print(f"   Overall health: {status['overall_health']:.1f}%")
        print(f"   System status: {status['status']}")
        print(f"   Entries logged: {status['audit_logger']['entries_logged']}")
        print(f"   Hash chain length: {status['audit_logger']['hash_chain_length']}")
        print(f"   Verifications performed: {status['crypto_verifier']['verifications_performed']}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   Database: {db_path}")
        print(f"   Export: {export_path}")
        print(f"   System integrity: {'✅ Valid' if integrity_result['hash_chain_valid'] else '❌ Compromised'}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
    
    finally:
        # Cleanup
        try:
            Path(db_path).unlink()
            Path("/tmp/memmimic_audit_export.json").unlink(missing_ok=True)
        except:
            pass


if __name__ == "__main__":
    main()