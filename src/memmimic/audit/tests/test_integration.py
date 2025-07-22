"""
Integration tests for MemMimic v2.0 Audit Logging System.

Comprehensive integration testing covering all audit logging components
working together with realistic usage patterns.
"""

import asyncio
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pytest

from ..immutable_logger import ImmutableAuditLogger, AuditEntry
from ..cryptographic_verifier import CryptographicVerifier, HashChainVerifier
from ..tamper_detector import TamperDetector, TamperAlert
from ..audit_trail_manager import AuditTrailManager, AuditQuery, AuditQueryFilter, QueryOperation
from ..security_metrics import AuditMetrics


class TestAuditLoggingIntegration:
    """Integration tests for complete audit logging system."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()
        yield temp_file.name
        Path(temp_file.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def audit_system(self, temp_db_path):
        """Create complete integrated audit system."""
        # Initialize components
        audit_logger = ImmutableAuditLogger(
            db_path=temp_db_path,
            retention_days=7,
            enable_memory_buffer=True,
            buffer_size=1000
        )
        
        crypto_verifier = CryptographicVerifier(
            verification_key="test_verification_key_12345"
        )
        
        hash_chain_verifier = HashChainVerifier(crypto_verifier)
        
        tamper_detector = TamperDetector({
            'max_alerts': 1000,
            'event_history_size': 5000
        })
        
        audit_trail_manager = AuditTrailManager(
            audit_logger=audit_logger,
            crypto_verifier=crypto_verifier,
            hash_chain_verifier=hash_chain_verifier,
            tamper_detector=tamper_detector
        )
        
        audit_metrics = AuditMetrics()
        
        return {
            'logger': audit_logger,
            'crypto_verifier': crypto_verifier,
            'chain_verifier': hash_chain_verifier,
            'tamper_detector': tamper_detector,
            'trail_manager': audit_trail_manager,
            'metrics': audit_metrics
        }
    
    def test_complete_audit_workflow(self, audit_system):
        """Test complete audit logging workflow from entry to verification."""
        logger = audit_system['logger']
        trail_manager = audit_system['trail_manager']
        metrics = audit_system['metrics']
        
        # Log multiple audit entries
        entry_ids = []
        for i in range(10):
            entry_id = logger.log_audit_entry(
                operation=f"test_operation_{i}",
                component="test_component",
                memory_id=str(uuid.uuid4()),
                user_context={'user_id': f'test_user_{i}'},
                operation_result={'success': True, 'duration_ms': 10 + i},
                governance_status="approved",
                performance_metrics={'latency_ms': float(i * 5)},
                security_level="standard",
                compliance_flags=["audit_required", "retention_7d"]
            )
            entry_ids.append(entry_id)
            
            # Record metrics
            metrics.record_audit_operation(
                operation=f"test_operation_{i}",
                component="test_component",
                duration_ms=10 + i,
                success=True
            )
        
        # Verify all entries
        for entry_id in entry_ids:
            is_valid = logger.verify_entry_integrity(entry_id)
            assert is_valid, f"Entry {entry_id} failed integrity check"
            
            # Record verification metrics
            metrics.record_verification_result(
                entry_id=entry_id,
                verification_time_ms=1.5,
                success=True,
                tamper_detected=False
            )
        
        # Verify hash chain integrity
        chain_result = logger.verify_hash_chain_integrity()
        assert chain_result['valid'], f"Hash chain integrity failed: {chain_result}"
        
        # Query audit trail
        query = AuditQuery(
            components=["test_component"],
            limit=5,
            order_by="timestamp",
            order_direction="DESC"
        )
        
        query_result = trail_manager.query_audit_trail(query)
        assert len(query_result.entries) == 5
        assert query_result.total_count == 10
        assert query_result.has_more is True
        
        # Check metrics
        summary = metrics.get_audit_summary(hours=1)
        assert summary['entries_logged']['count'] == 10
        assert summary['verifications_performed']['count'] == 10
        assert summary['tamper_attempts']['count'] == 0
    
    def test_tamper_detection_workflow(self, audit_system):
        """Test tamper detection and alerting workflow."""
        logger = audit_system['logger']
        tamper_detector = audit_system['tamper_detector']
        crypto_verifier = audit_system['crypto_verifier']
        
        # Set up alert callback
        alerts_received = []
        
        def alert_callback(alert: TamperAlert):
            alerts_received.append(alert)
        
        tamper_detector.register_alert_callback(alert_callback)
        
        # Log a legitimate entry
        entry_id = logger.log_audit_entry(
            operation="legitimate_operation",
            component="test_component"
        )
        
        # Verify entry normally
        result = crypto_verifier.verify_entry_hash(
            entry_data={'operation': 'legitimate_operation', 'component': 'test_component'},
            expected_hash="valid_hash",
            salt="test_salt",
            entry_id=entry_id
        )
        
        # Simulate tampered verification
        tampered_result = crypto_verifier.verify_entry_hash(
            entry_data={'operation': 'tampered_operation'},  # Different data
            expected_hash="original_hash",
            salt="test_salt",
            entry_id="tampered_entry"
        )
        
        # Process verification results
        alert1 = tamper_detector.detect_from_verification_result(result)
        alert2 = tamper_detector.detect_from_verification_result(tampered_result)
        
        # Should detect tampering
        assert alert2 is not None
        assert alert2.tamper_type.value == "hash_mismatch"
        
        # Check alerts were received
        assert len(alerts_received) >= 1
        
        # Get recent alerts
        recent_alerts = tamper_detector.get_recent_alerts(hours=1)
        assert len(recent_alerts) >= 1
    
    def test_advanced_querying_workflow(self, audit_system):
        """Test advanced audit trail querying capabilities."""
        logger = audit_system['logger']
        trail_manager = audit_system['trail_manager']
        
        # Create diverse audit entries
        operations = ["create", "read", "update", "delete"]
        components = ["storage", "governance", "api"]
        
        for i in range(20):
            operation = operations[i % len(operations)]
            component = components[i % len(components)]
            
            logger.log_audit_entry(
                operation=operation,
                component=component,
                memory_id=f"memory_{i}",
                user_context={'user_id': f'user_{i % 5}'},
                operation_result={'success': i % 7 != 0},  # Some failures
                governance_status="approved" if i % 3 == 0 else "rejected",
                performance_metrics={'duration_ms': float(i * 10)},
                security_level="confidential" if i % 4 == 0 else "standard"
            )
        
        # Test complex queries
        
        # 1. Filter by operation and success status
        query1 = AuditQuery(
            operations=["create", "update"],
            filters=[
                AuditQueryFilter(
                    field="operation_result",
                    operation=QueryOperation.CONTAINS,
                    value="success",
                    case_sensitive=False
                )
            ],
            limit=10
        )
        result1 = trail_manager.query_audit_trail(query1)
        assert len(result1.entries) <= 10
        
        # 2. Filter by component and governance status
        query2 = AuditQuery(
            components=["storage"],
            filters=[
                AuditQueryFilter(
                    field="governance_status",
                    operation=QueryOperation.EQUALS,
                    value="approved"
                )
            ]
        )
        result2 = trail_manager.query_audit_trail(query2)
        
        # All results should be from storage component with approved status
        for entry in result2.entries:
            assert entry.component == "storage"
            assert entry.governance_status == "approved"
        
        # 3. Time-based query with performance filtering
        start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        query3 = AuditQuery(
            start_time=start_time,
            filters=[
                AuditQueryFilter(
                    field="security_level",
                    operation=QueryOperation.EQUALS,
                    value="confidential"
                )
            ]
        )
        result3 = trail_manager.query_audit_trail(query3)
        
        # All results should be confidential
        for entry in result3.entries:
            assert entry.security_level == "confidential"
        
        # Test pagination
        paginated_query = AuditQuery(
            limit=5,
            offset=0,
            order_by="timestamp",
            order_direction="DESC"
        )
        
        page1 = trail_manager.query_audit_trail(paginated_query)
        assert len(page1.entries) == 5
        assert page1.has_more is True
        assert page1.next_offset == 5
        
        # Get next page
        paginated_query.offset = page1.next_offset
        page2 = trail_manager.query_audit_trail(paginated_query)
        assert len(page2.entries) == 5
        
        # Ensure no overlap
        page1_ids = {entry.entry_id for entry in page1.entries}
        page2_ids = {entry.entry_id for entry in page2.entries}
        assert page1_ids.isdisjoint(page2_ids)
    
    def test_performance_benchmarks(self, audit_system):
        """Test performance benchmarks for audit logging operations."""
        logger = audit_system['logger']
        crypto_verifier = audit_system['crypto_verifier']
        metrics = audit_system['metrics']
        
        # Benchmark audit entry logging
        num_entries = 100
        start_time = time.perf_counter()
        
        entry_ids = []
        for i in range(num_entries):
            entry_id = logger.log_audit_entry(
                operation=f"benchmark_operation_{i}",
                component="benchmark_component",
                memory_id=f"memory_{i}",
                operation_result={'index': i, 'success': True}
            )
            entry_ids.append(entry_id)
        
        logging_time = (time.perf_counter() - start_time) * 1000  # ms
        avg_logging_time = logging_time / num_entries
        
        print(f"Average logging time: {avg_logging_time:.2f}ms per entry")
        assert avg_logging_time < 5.0, f"Logging too slow: {avg_logging_time:.2f}ms > 5ms"
        
        # Benchmark verification
        start_time = time.perf_counter()
        
        for entry_id in entry_ids:
            is_valid = logger.verify_entry_integrity(entry_id)
            assert is_valid
        
        verification_time = (time.perf_counter() - start_time) * 1000  # ms
        avg_verification_time = verification_time / num_entries
        
        print(f"Average verification time: {avg_verification_time:.2f}ms per entry")
        assert avg_verification_time < 2.0, f"Verification too slow: {avg_verification_time:.2f}ms > 2ms"
        
        # Benchmark hash chain verification
        start_time = time.perf_counter()
        
        chain_result = logger.verify_hash_chain_integrity()
        
        chain_verification_time = (time.perf_counter() - start_time) * 1000  # ms
        
        print(f"Hash chain verification time: {chain_verification_time:.2f}ms for {num_entries} entries")
        assert chain_verification_time < 100.0, f"Chain verification too slow: {chain_verification_time:.2f}ms > 100ms"
        
        # Test metrics performance
        metrics_start = time.perf_counter()
        
        for i in range(100):
            metrics.record_audit_operation(
                operation="benchmark_metric",
                component="metrics_test",
                duration_ms=1.0,
                success=True
            )
        
        metrics_time = (time.perf_counter() - metrics_start) * 1000  # ms
        avg_metrics_time = metrics_time / 100
        
        print(f"Average metrics recording time: {avg_metrics_time:.3f}ms per metric")
        assert avg_metrics_time < 0.5, f"Metrics recording too slow: {avg_metrics_time:.3f}ms > 0.5ms"
    
    def test_report_generation_workflow(self, audit_system):
        """Test comprehensive audit report generation."""
        logger = audit_system['logger']
        trail_manager = audit_system['trail_manager']
        tamper_detector = audit_system['tamper_detector']
        
        # Create audit data over time period
        base_time = datetime.now(timezone.utc)
        
        for i in range(50):
            # Vary entry characteristics
            operation = ["create", "read", "update", "delete"][i % 4]
            component = ["storage", "api", "governance"][i % 3]
            success = i % 10 != 0  # 10% failures
            
            logger.log_audit_entry(
                operation=operation,
                component=component,
                memory_id=f"memory_{i}",
                operation_result={'success': success, 'error_code': 500 if not success else 0},
                governance_status="approved" if success else "rejected",
                performance_metrics={'duration_ms': float(i * 2)},
                security_level="confidential" if i % 5 == 0 else "standard"
            )
        
        # Generate comprehensive report
        start_time = base_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = base_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        report = trail_manager.generate_audit_report(
            start_time=start_time,
            end_time=end_time,
            title="Integration Test Audit Report",
            include_performance=True,
            include_security=True,
            include_compliance=True
        )
        
        # Validate report contents
        assert report.total_entries == 50
        assert len(report.operations_summary) == 4  # create, read, update, delete
        assert len(report.components_summary) == 3  # storage, api, governance
        
        # Check operations breakdown
        expected_operations = {"create", "read", "update", "delete"}
        actual_operations = set(report.operations_summary.keys())
        assert actual_operations == expected_operations
        
        # Check components breakdown
        expected_components = {"storage", "api", "governance"}
        actual_components = set(report.components_summary.keys())
        assert actual_components == expected_components
        
        # Validate integrity status
        assert 'hash_chain_valid' in report.integrity_status
        assert 'entries_verified' in report.integrity_status
        
        # Validate performance summary
        assert 'audit_logger' in report.performance_summary
        assert 'crypto_verifier' in report.performance_summary
        assert 'tamper_detector' in report.performance_summary
        
        # Validate compliance status
        assert 'total_entries' in report.compliance_status
        assert 'security_levels' in report.compliance_status
        
        # Convert to dict for serialization test
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert 'report_id' in report_dict
        assert 'generated_at' in report_dict
    
    def test_memory_vs_database_consistency(self, audit_system):
        """Test consistency between memory buffer and database storage."""
        logger = audit_system['logger']
        
        # Log entries that will be in memory buffer
        memory_entry_ids = []
        for i in range(10):
            entry_id = logger.log_audit_entry(
                operation=f"memory_test_{i}",
                component="memory_component",
                memory_id=f"mem_{i}"
            )
            memory_entry_ids.append(entry_id)
        
        # Force persistence to database
        if hasattr(logger, '_persist_buffer'):
            logger._persist_buffer()
        
        # Wait a moment for persistence thread
        time.sleep(0.1)
        
        # Verify all entries can be retrieved and verified
        for entry_id in memory_entry_ids:
            is_valid = logger.verify_entry_integrity(entry_id)
            assert is_valid, f"Entry {entry_id} failed integrity check"
        
        # Test hash chain integrity across memory and database
        chain_result = logger.verify_hash_chain_integrity()
        assert chain_result['valid'], f"Hash chain integrity failed: {chain_result}"
        assert chain_result['entries_verified'] >= len(memory_entry_ids)
    
    def test_concurrent_operations(self, audit_system):
        """Test thread safety with concurrent operations."""
        import threading
        
        logger = audit_system['logger']
        tamper_detector = audit_system['tamper_detector']
        metrics = audit_system['metrics']
        
        # Concurrent logging test
        entry_ids = []
        entry_ids_lock = threading.Lock()
        
        def log_entries(thread_id: int, count: int):
            thread_entries = []
            for i in range(count):
                entry_id = logger.log_audit_entry(
                    operation=f"concurrent_op_{thread_id}_{i}",
                    component=f"thread_{thread_id}",
                    memory_id=f"concurrent_mem_{thread_id}_{i}",
                    performance_metrics={'thread_id': thread_id, 'index': i}
                )
                thread_entries.append(entry_id)
                
                # Record metrics concurrently
                metrics.record_audit_operation(
                    operation=f"concurrent_op_{thread_id}_{i}",
                    component=f"thread_{thread_id}",
                    duration_ms=1.0,
                    success=True
                )
            
            with entry_ids_lock:
                entry_ids.extend(thread_entries)
        
        # Create multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=log_entries, args=(thread_id, 20))
            threads.append(thread)
        
        # Start all threads
        start_time = time.perf_counter()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        concurrent_time = (time.perf_counter() - start_time) * 1000
        print(f"Concurrent logging time: {concurrent_time:.2f}ms for {len(entry_ids)} entries")
        
        # Verify all entries
        assert len(entry_ids) == 100  # 5 threads * 20 entries each
        
        # Verify integrity of all entries
        for entry_id in entry_ids:
            is_valid = logger.verify_entry_integrity(entry_id)
            assert is_valid, f"Entry {entry_id} failed integrity check after concurrent operations"
        
        # Verify hash chain integrity
        chain_result = logger.verify_hash_chain_integrity()
        assert chain_result['valid'], "Hash chain integrity failed after concurrent operations"
    
    def test_system_resilience(self, audit_system):
        """Test system resilience under various conditions."""
        logger = audit_system['logger']
        metrics = audit_system['metrics']
        
        # Test with large payloads
        large_context = {'large_data': 'x' * 10000}  # 10KB payload
        large_result = {'output': 'y' * 5000}  # 5KB result
        
        large_entry_id = logger.log_audit_entry(
            operation="large_payload_test",
            component="resilience_test",
            user_context=large_context,
            operation_result=large_result,
            compliance_flags=['large_payload', 'stress_test']
        )
        
        # Verify large entry
        is_valid = logger.verify_entry_integrity(large_entry_id)
        assert is_valid, "Large payload entry failed integrity check"
        
        # Test with special characters and unicode
        unicode_entry_id = logger.log_audit_entry(
            operation="unicode_test_操作",
            component="unicode_测试",
            user_context={'message': 'Test with émojis 🚀 and símböls'},
            operation_result={'status': 'succès ✅'},
            compliance_flags=['unicode', '多语言支持']
        )
        
        # Verify unicode entry
        is_valid = logger.verify_entry_integrity(unicode_entry_id)
        assert is_valid, "Unicode entry failed integrity check"
        
        # Test rapid sequential operations
        rapid_entry_ids = []
        start_time = time.perf_counter()
        
        for i in range(100):
            entry_id = logger.log_audit_entry(
                operation=f"rapid_op_{i}",
                component="rapid_test",
                memory_id=f"rapid_{i}",
                operation_result={'index': i, 'timestamp': time.time()}
            )
            rapid_entry_ids.append(entry_id)
            
            # Add some metrics
            metrics.increment_counter('rapid_operations')
        
        rapid_time = (time.perf_counter() - start_time) * 1000
        avg_rapid_time = rapid_time / 100
        
        print(f"Rapid sequential operations: {avg_rapid_time:.2f}ms per operation")
        assert avg_rapid_time < 10.0, f"Rapid operations too slow: {avg_rapid_time:.2f}ms"
        
        # Verify all rapid entries
        for entry_id in rapid_entry_ids:
            is_valid = logger.verify_entry_integrity(entry_id)
            assert is_valid, f"Rapid entry {entry_id} failed integrity check"
        
        # Final system health check
        system_metrics = metrics.get_system_metrics()
        assert system_metrics['metrics_collected'] > 0
        assert system_metrics['avg_collection_time_ms'] < 1.0  # <1ms average
        
        # Check audit logger metrics
        logger_metrics = logger.get_metrics()
        assert logger_metrics['entries_logged'] > 200  # All our test entries
        assert logger_metrics['avg_log_time_ms'] < 5.0  # Performance target


if __name__ == "__main__":
    # Run tests manually for development
    import tempfile
    
    # Create test instance
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db_path = f.name
    
    try:
        test = TestAuditLoggingIntegration()
        
        # Create audit system
        audit_system = test.audit_system(test_db_path).__next__()
        
        print("Running integration tests...")
        
        # Run key tests
        print("1. Testing complete audit workflow...")
        test.test_complete_audit_workflow(audit_system)
        print("✓ Complete workflow test passed")
        
        print("2. Testing performance benchmarks...")
        test.test_performance_benchmarks(audit_system)
        print("✓ Performance benchmarks passed")
        
        print("3. Testing tamper detection...")
        test.test_tamper_detection_workflow(audit_system)
        print("✓ Tamper detection test passed")
        
        print("4. Testing advanced querying...")
        test.test_advanced_querying_workflow(audit_system)
        print("✓ Advanced querying test passed")
        
        print("5. Testing system resilience...")
        test.test_system_resilience(audit_system)
        print("✓ System resilience test passed")
        
        print("\n🎉 All integration tests passed!")
        
    finally:
        Path(test_db_path).unlink(missing_ok=True)