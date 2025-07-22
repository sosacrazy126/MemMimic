"""
Tests for MemMimic v2.0 Simple Governance Framework
Comprehensive test suite validating governance functionality, performance, and integration.
"""

import asyncio
import json
import tempfile
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pytest

from memmimic.memory.enhanced_memory import EnhancedMemory
from memmimic.memory.governance import (
    SimpleGovernance, GovernanceConfig, GovernanceValidator, 
    ThresholdManager, GovernanceMetrics, GovernanceViolation,
    GovernanceResult, GovernanceConfigError
)
from memmimic.memory.governance_integrated_storage import (
    GovernanceIntegratedStorage, GovernanceAwareResult
)


class TestGovernanceConfig:
    """Test governance configuration management"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = GovernanceConfig()
        
        assert config.content_size == 1_000_000
        assert config.summary_length == 1000
        assert config.metadata_size == 10_000
        assert config.tag_count == 100
        assert config.tag_length == 50
        assert config.relationship_depth == 3
        assert config.governance_timeout == 10
        assert config.enabled is True
        assert config.enforcement_mode == "strict"
        assert config.environment == "development"
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary"""
        config_data = {
            'thresholds': {
                'content_size': 500_000,
                'summary_length': 500,
                'tag_count': 50
            },
            'enabled': True,
            'enforcement_mode': 'permissive',
            'environment': 'production'
        }
        
        config = GovernanceConfig.from_dict(config_data)
        
        assert config.content_size == 500_000
        assert config.summary_length == 500
        assert config.tag_count == 50
        assert config.enforcement_mode == 'permissive'
        assert config.environment == 'production'
        # Verify defaults for missing values
        assert config.tag_length == 50
        assert config.governance_timeout == 10
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config = GovernanceConfig()
        config.validate()  # Should not raise
    
    def test_config_validation_failure(self):
        """Test configuration validation with invalid values"""
        config = GovernanceConfig()
        config.content_size = -1
        
        with pytest.raises(GovernanceConfigError, match="Invalid threshold content_size"):
            config.validate()
        
        config.content_size = 1000
        config.enforcement_mode = "invalid_mode"
        
        with pytest.raises(GovernanceConfigError, match="Invalid enforcement_mode"):
            config.validate()


class TestThresholdManager:
    """Test dynamic threshold management"""
    
    def test_base_thresholds(self):
        """Test base threshold retrieval"""
        base_config = GovernanceConfig(content_size=1000, tag_count=50)
        manager = ThresholdManager(base_config)
        
        effective = manager.get_effective_thresholds()
        
        assert effective.content_size == 1000
        assert effective.tag_count == 50
    
    def test_context_adjustments(self):
        """Test context-specific threshold adjustments"""
        base_config = GovernanceConfig(content_size=1000, tag_count=50)
        manager = ThresholdManager(base_config)
        
        # Apply adjustments for test context
        manager.adjust_thresholds('testing', {'content_size': 500, 'tag_count': 25})
        
        # Get effective thresholds for testing context
        effective = manager.get_effective_thresholds('testing')
        assert effective.content_size == 500
        assert effective.tag_count == 25
        
        # Base thresholds unchanged for other contexts
        base_effective = manager.get_effective_thresholds('production')
        assert base_effective.content_size == 1000
        assert base_effective.tag_count == 50
    
    def test_adjustment_reset(self):
        """Test resetting threshold adjustments"""
        base_config = GovernanceConfig(content_size=1000)
        manager = ThresholdManager(base_config)
        
        manager.adjust_thresholds('test1', {'content_size': 500})
        manager.adjust_thresholds('test2', {'content_size': 200})
        
        # Reset specific context
        manager.reset_adjustments('test1')
        effective = manager.get_effective_thresholds('test1')
        assert effective.content_size == 1000  # Back to base
        
        # Other context still has adjustment
        effective = manager.get_effective_thresholds('test2')
        assert effective.content_size == 200
        
        # Reset all
        manager.reset_adjustments()
        effective = manager.get_effective_thresholds('test2')
        assert effective.content_size == 1000  # Back to base


class TestGovernanceMetrics:
    """Test governance metrics tracking"""
    
    def test_metrics_initialization(self):
        """Test metrics start with zero values"""
        metrics = GovernanceMetrics()
        
        assert metrics.metrics['total_validations'] == 0
        assert metrics.metrics['approvals'] == 0
        assert metrics.metrics['rejections'] == 0
        assert metrics.metrics['warnings'] == 0
        assert metrics.metrics['errors'] == 0
        assert metrics.processing_times == []
    
    def test_record_approval(self):
        """Test recording approval metrics"""
        metrics = GovernanceMetrics()
        
        result = GovernanceResult(
            approved=True,
            status="approved",
            violations=[],
            warnings=[],
            processing_time=5.0
        )
        
        metrics.record_validation(result)
        
        assert metrics.metrics['total_validations'] == 1
        assert metrics.metrics['approvals'] == 1
        assert metrics.metrics['rejections'] == 0
        assert metrics.metrics['avg_processing_time_ms'] == 5.0
    
    def test_record_rejection(self):
        """Test recording rejection metrics"""
        metrics = GovernanceMetrics()
        
        violation = GovernanceViolation(
            type="content_size_exceeded",
            message="Too large",
            severity="critical"
        )
        
        result = GovernanceResult(
            approved=False,
            status="rejected",
            violations=[violation],
            processing_time=8.0
        )
        
        metrics.record_validation(result)
        
        assert metrics.metrics['total_validations'] == 1
        assert metrics.metrics['approvals'] == 0
        assert metrics.metrics['rejections'] == 1
        assert metrics.metrics['violation_types']['content_size_exceeded'] == 1
        assert metrics.metrics['avg_processing_time_ms'] == 8.0
    
    def test_performance_percentiles(self):
        """Test performance percentile calculations"""
        metrics = GovernanceMetrics()
        
        # Record multiple processing times
        processing_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        for time_ms in processing_times:
            result = GovernanceResult(
                approved=True,
                status="approved",
                violations=[],
                processing_time=time_ms
            )
            metrics.record_validation(result)
        
        # Check percentile calculations
        assert metrics.metrics['performance_percentiles']['p50'] == 5.0  # Median
        assert metrics.metrics['performance_percentiles']['p95'] == 10.0  # 95th percentile
    
    def test_performance_summary(self):
        """Test comprehensive performance summary"""
        metrics = GovernanceMetrics()
        
        # Record some results
        for i in range(10):
            result = GovernanceResult(
                approved=i % 2 == 0,  # Alternate approval/rejection
                status="approved" if i % 2 == 0 else "rejected",
                violations=[],
                warnings=["warning"] if i % 3 == 0 else [],
                processing_time=float(i + 1)
            )
            metrics.record_validation(result)
        
        summary = metrics.get_performance_summary()
        
        assert summary['total_validations'] == 10
        assert summary['approval_rate'] == 50.0  # 50% approval rate
        assert summary['rejection_rate'] == 50.0  # 50% rejection rate
        assert summary['warning_rate'] == 40.0    # 40% warning rate (4 out of 10)
        assert 'performance' in summary
        assert 'avg_processing_time_ms' in summary['performance']


@pytest.fixture
def sample_memory():
    """Create sample EnhancedMemory for testing"""
    return EnhancedMemory(
        content="Test content for governance validation",
        summary="Test summary",
        full_context="This is a test content for governance validation. It contains enough text to test various governance rules.",
        tags=["test", "governance", "validation"],
        metadata={"test": True, "priority": "high"},
        importance_score=0.8
    )


@pytest.fixture
def governance_config():
    """Create test governance configuration"""
    return GovernanceConfig(
        content_size=500,  # Small limit for testing
        summary_length=100,
        tag_count=5,
        tag_length=20,
        governance_timeout=50,  # Generous timeout for tests
        enabled=True,
        enforcement_mode="strict"
    )


class TestGovernanceValidator:
    """Test governance validation logic"""
    
    @pytest.mark.asyncio
    async def test_validation_approved(self, sample_memory, governance_config):
        """Test memory that passes all governance rules"""
        # Adjust memory to pass validation
        sample_memory.full_context = "Short content"  # Within 500 char limit
        sample_memory.summary = "Short summary"       # Within 100 char limit
        sample_memory.tags = ["test", "short"]        # Within 5 tag limit
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        assert result.approved is True
        assert result.status == "approved"
        assert len(result.violations) == 0
        assert result.processing_time < governance_config.governance_timeout
    
    @pytest.mark.asyncio
    async def test_validation_content_size_violation(self, sample_memory, governance_config):
        """Test content size violation"""
        # Make content too large
        sample_memory.full_context = "x" * 1000  # Exceeds 500 char limit
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        assert result.approved is False
        assert result.status == "rejected"
        assert len(result.violations) == 1
        assert result.violations[0].type == "content_size_exceeded"
        assert result.violations[0].severity == "critical"
    
    @pytest.mark.asyncio
    async def test_validation_tag_count_violation(self, sample_memory, governance_config):
        """Test tag count violation"""
        # Too many tags
        sample_memory.tags = [f"tag{i}" for i in range(10)]  # Exceeds 5 tag limit
        sample_memory.full_context = "Short content"  # Keep content valid
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        assert result.approved is False
        assert result.status == "rejected"
        assert len(result.violations) == 1
        assert result.violations[0].type == "tag_count_exceeded"
        assert result.violations[0].severity == "high"
    
    @pytest.mark.asyncio
    async def test_validation_tag_length_violation(self, sample_memory, governance_config):
        """Test individual tag length violation"""
        # Tag too long
        sample_memory.tags = ["x" * 30]  # Exceeds 20 char limit
        sample_memory.full_context = "Short content"
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        assert result.approved is False
        assert result.status == "rejected"
        assert len(result.violations) == 1
        assert result.violations[0].type == "tag_length_exceeded"
        assert result.violations[0].severity == "medium"
    
    @pytest.mark.asyncio
    async def test_validation_summary_length_violation(self, sample_memory, governance_config):
        """Test summary length violation"""
        # Summary too long
        sample_memory.summary = "x" * 200  # Exceeds 100 char limit
        sample_memory.full_context = "Short content"
        sample_memory.tags = ["short"]
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        assert result.approved is False
        assert result.status == "rejected"
        assert len(result.violations) == 1
        assert result.violations[0].type == "summary_length_exceeded"
        assert result.violations[0].severity == "medium"
    
    @pytest.mark.asyncio
    async def test_validation_multiple_violations(self, sample_memory, governance_config):
        """Test multiple simultaneous violations"""
        # Multiple violations
        sample_memory.full_context = "x" * 1000      # Content too large
        sample_memory.summary = "x" * 200            # Summary too long
        sample_memory.tags = [f"verylongtagname{i}" for i in range(10)]  # Too many long tags
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        assert result.approved is False
        assert result.status == "rejected"
        assert len(result.violations) >= 3  # At least content, summary, and tag count violations
        
        violation_types = [v.type for v in result.violations]
        assert "content_size_exceeded" in violation_types
        assert "summary_length_exceeded" in violation_types
        assert "tag_count_exceeded" in violation_types
    
    @pytest.mark.asyncio
    async def test_validation_warnings(self, sample_memory, governance_config):
        """Test validation with warnings but no violations"""
        # Content at 80% of limit (should trigger warning)
        sample_memory.full_context = "x" * 400  # 80% of 500 char limit
        sample_memory.summary = "Short summary"
        sample_memory.tags = ["test"]
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        assert result.approved is True
        assert result.status == "approved_with_warnings"
        assert len(result.violations) == 0
        assert len(result.warnings) > 0
        assert "approaching limit" in result.warnings[0].lower()
    
    @pytest.mark.asyncio
    async def test_enforcement_mode_permissive(self, sample_memory, governance_config):
        """Test permissive enforcement mode"""
        governance_config.enforcement_mode = "permissive"
        
        # Create violations
        sample_memory.full_context = "x" * 1000  # Exceeds limit
        sample_memory.summary = "x" * 200        # Exceeds limit
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        # Should be approved in permissive mode, but with warnings
        assert result.approved is True
        assert result.status == "approved_with_warnings"
        assert len(result.violations) >= 2
        assert len(result.warnings) >= 2
        assert any("PERMISSIVE" in warning for warning in result.warnings)
    
    @pytest.mark.asyncio
    async def test_enforcement_mode_audit_only(self, sample_memory, governance_config):
        """Test audit-only enforcement mode"""
        governance_config.enforcement_mode = "audit_only"
        
        # Create critical violations
        sample_memory.full_context = "x" * 1000  # Exceeds limit
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        # Should always be approved in audit-only mode
        assert result.approved is True
        assert result.status == "approved"
        assert len(result.violations) >= 1
        assert len(result.warnings) >= 1
        assert any("AUDIT" in warning for warning in result.warnings)
    
    @pytest.mark.asyncio
    async def test_validation_performance(self, sample_memory, governance_config):
        """Test governance validation performance target (<10ms)"""
        validator = GovernanceValidator(governance_config)
        
        # Run multiple validations and measure performance
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            result = await validator.validate_memory_governance(sample_memory, "store")
            elapsed = (time.perf_counter() - start_time) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        p95_time = sorted(times)[95]
        
        # Performance targets
        assert avg_time < 10.0, f"Average validation time {avg_time:.2f}ms exceeds 10ms target"
        assert p95_time < 20.0, f"P95 validation time {p95_time:.2f}ms exceeds 20ms target"
        
        # Check that reported processing time is accurate
        result = await validator.validate_memory_governance(sample_memory, "store")
        assert 0.1 <= result.processing_time <= 50.0, f"Reported time {result.processing_time}ms seems incorrect"
    
    @pytest.mark.asyncio
    async def test_validation_disabled(self, sample_memory, governance_config):
        """Test validation when governance is disabled"""
        governance_config.enabled = False
        
        # Create violations (should be ignored)
        sample_memory.full_context = "x" * 1000
        
        validator = GovernanceValidator(governance_config)
        result = await validator.validate_memory_governance(sample_memory, "store")
        
        assert result.approved is True
        assert result.status == "approved"
        assert len(result.violations) == 0
        assert "disabled" in result.warnings[0].lower()


class TestSimpleGovernance:
    """Test main governance framework"""
    
    def test_initialization_default(self):
        """Test default governance initialization"""
        governance = SimpleGovernance()
        
        assert governance.config.enabled is True
        assert governance.config.enforcement_mode == "strict"
        assert governance.config.content_size == 1_000_000
    
    def test_initialization_from_dict(self):
        """Test governance initialization from dictionary"""
        config_data = {
            'enabled': True,
            'enforcement_mode': 'permissive',
            'thresholds': {
                'content_size': 500_000,
                'tag_count': 50
            }
        }
        
        governance = SimpleGovernance(config_data)
        
        assert governance.config.enabled is True
        assert governance.config.enforcement_mode == 'permissive'
        assert governance.config.content_size == 500_000
        assert governance.config.tag_count == 50
    
    def test_initialization_from_config_object(self, governance_config):
        """Test governance initialization from config object"""
        governance = SimpleGovernance(governance_config)
        
        assert governance.config == governance_config
    
    @pytest.mark.asyncio
    async def test_validate_memory(self, sample_memory, governance_config):
        """Test main memory validation interface"""
        governance = SimpleGovernance(governance_config)
        
        # Adjust memory to pass validation
        sample_memory.full_context = "Short content"
        sample_memory.summary = "Short"
        sample_memory.tags = ["test"]
        
        result = await governance.validate_memory(sample_memory, "store")
        
        assert isinstance(result, GovernanceResult)
        assert result.approved is True
        assert result.operation_context == "store"
    
    def test_get_thresholds(self, governance_config):
        """Test threshold retrieval"""
        governance = SimpleGovernance(governance_config)
        
        thresholds = governance.get_thresholds()
        
        assert thresholds == governance_config
        assert thresholds.content_size == 500
    
    def test_adjust_thresholds(self, governance_config):
        """Test dynamic threshold adjustment"""
        governance = SimpleGovernance(governance_config)
        
        # Adjust thresholds
        governance.adjust_thresholds('testing', {'content_size': 200})
        
        # Get adjusted thresholds
        thresholds = governance.get_thresholds('testing')
        assert thresholds.content_size == 200
        
        # Base thresholds unchanged
        base_thresholds = governance.get_thresholds('production')
        assert base_thresholds.content_size == 500
    
    def test_get_performance_metrics(self, governance_config):
        """Test performance metrics retrieval"""
        governance = SimpleGovernance(governance_config)
        
        metrics = governance.get_performance_metrics()
        
        assert 'governance_framework' in metrics
        assert metrics['governance_framework'] == 'simple'
        assert 'version' in metrics
        assert 'configuration' in metrics
        assert 'thresholds' in metrics
        assert 'performance' in metrics
    
    def test_reload_config(self, governance_config):
        """Test configuration hot-reload"""
        governance = SimpleGovernance(governance_config)
        
        # Create new config
        new_config_data = {
            'enabled': True,
            'enforcement_mode': 'audit_only',
            'thresholds': {
                'content_size': 1000,
                'tag_count': 20
            }
        }
        
        # Reload configuration
        governance.reload_config(new_config_data)
        
        assert governance.config.enforcement_mode == 'audit_only'
        assert governance.config.content_size == 1000
        assert governance.config.tag_count == 20
    
    def test_from_yaml_file_success(self, governance_config):
        """Test loading configuration from YAML file"""
        config_data = {
            'governance': {
                'enabled': True,
                'enforcement_mode': 'strict',
                'thresholds': {
                    'content_size': 750_000,
                    'tag_count': 75
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config_data, f)
            yaml_path = f.name
        
        try:
            governance = SimpleGovernance.from_yaml_file(yaml_path)
            
            assert governance.config.enabled is True
            assert governance.config.enforcement_mode == 'strict'
            assert governance.config.content_size == 750_000
            assert governance.config.tag_count == 75
        finally:
            Path(yaml_path).unlink()  # Clean up
    
    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent YAML file"""
        with pytest.raises(GovernanceConfigError, match="not found"):
            SimpleGovernance.from_yaml_file("nonexistent.yaml")


@pytest.fixture
async def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


class TestGovernanceIntegratedStorage:
    """Test governance integration with storage operations"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, temp_db):
        """Test storage initialization with governance"""
        config = {
            'governance': {
                'enabled': True,
                'enforcement_mode': 'strict',
                'thresholds': {
                    'content_size': 500,
                    'tag_count': 5
                }
            }
        }
        
        storage = GovernanceIntegratedStorage(temp_db, config=config)
        
        assert storage.governance is not None
        assert storage.governance.config.enabled is True
        assert storage.governance.config.enforcement_mode == 'strict'
        assert storage.governance.config.content_size == 500
        
        await storage.close()
    
    @pytest.mark.asyncio
    async def test_store_with_governance_approved(self, temp_db, sample_memory):
        """Test storing memory with governance approval"""
        config = {
            'governance': {
                'enabled': True,
                'enforcement_mode': 'strict',
                'thresholds': {
                    'content_size': 1000,
                    'tag_count': 10,
                    'summary_length': 200
                }
            }
        }
        
        storage = GovernanceIntegratedStorage(temp_db, config=config)
        
        try:
            # Ensure memory will pass governance
            sample_memory.full_context = "Valid content"
            sample_memory.summary = "Valid summary"
            sample_memory.tags = ["valid", "test"]
            
            result = await storage.store_with_governance(sample_memory)
            
            assert isinstance(result, GovernanceAwareResult)
            assert result.success is True
            assert result.memory_id is not None
            assert result.governance_result.approved is True
            assert result.governance_result.status == "approved"
            assert result.governance_time > 0
            assert result.storage_time > 0
            assert result.total_processing_time > 0
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_store_with_governance_rejected(self, temp_db, sample_memory):
        """Test storing memory with governance rejection"""
        config = {
            'governance': {
                'enabled': True,
                'enforcement_mode': 'strict',
                'thresholds': {
                    'content_size': 50,  # Very small limit
                    'tag_count': 1,
                    'summary_length': 10
                }
            }
        }
        
        storage = GovernanceIntegratedStorage(temp_db, config=config)
        
        try:
            # Ensure memory will fail governance
            sample_memory.full_context = "This content is too long for the strict limit"
            sample_memory.summary = "This summary is also too long"
            sample_memory.tags = ["tag1", "tag2", "tag3"]  # Too many tags
            
            result = await storage.store_with_governance(sample_memory)
            
            assert isinstance(result, GovernanceAwareResult)
            assert result.success is False
            assert result.memory_id is None
            assert result.governance_result.approved is False
            assert result.governance_result.status == "rejected"
            assert len(result.governance_result.violations) >= 2  # Content and tags violations
            assert result.governance_time > 0
            assert result.storage_time == 0  # No storage attempted
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_store_with_governance_audit_only(self, temp_db, sample_memory):
        """Test storing memory with audit-only enforcement"""
        config = {
            'governance': {
                'enabled': True,
                'enforcement_mode': 'audit_only',
                'thresholds': {
                    'content_size': 50,  # Very small limit
                    'tag_count': 1
                }
            }
        }
        
        storage = GovernanceIntegratedStorage(temp_db, config=config)
        
        try:
            # Memory that would fail in strict mode
            sample_memory.full_context = "This content exceeds the limit but should be stored in audit mode"
            sample_memory.tags = ["tag1", "tag2", "tag3"]
            
            result = await storage.store_with_governance(sample_memory)
            
            assert result.success is True  # Stored despite violations
            assert result.memory_id is not None
            assert result.governance_result.approved is True
            assert len(result.governance_result.violations) >= 2
            assert len(result.governance_result.warnings) >= 2
            assert any("AUDIT" in warning for warning in result.governance_result.warnings)
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_retrieve_with_governance_check(self, temp_db, sample_memory):
        """Test retrieving memory with governance compliance check"""
        storage = GovernanceIntegratedStorage(temp_db)
        
        try:
            # First store a memory
            sample_memory.full_context = "Valid content"
            sample_memory.summary = "Valid summary"
            sample_memory.tags = ["test"]
            
            store_result = await storage.store_with_governance(sample_memory)
            assert store_result.success is True
            memory_id = store_result.memory_id
            
            # Retrieve with governance check
            retrieve_result = await storage.retrieve_with_governance_check(memory_id, "summary")
            
            assert isinstance(retrieve_result, GovernanceAwareResult)
            assert retrieve_result.success is True
            assert retrieve_result.memory_id == memory_id
            assert retrieve_result.governance_result.approved is True
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_batch_store_with_governance(self, temp_db):
        """Test batch storing multiple memories with governance"""
        storage = GovernanceIntegratedStorage(temp_db)
        
        try:
            # Create multiple memories
            memories = []
            for i in range(5):
                memory = EnhancedMemory(
                    content=f"Test content {i}",
                    summary=f"Summary {i}",
                    full_context=f"This is test content number {i} for batch storage testing",
                    tags=[f"test{i}", "batch"],
                    metadata={"batch_index": i},
                    importance_score=0.5 + (i * 0.1)
                )
                memories.append(memory)
            
            # Batch store
            results = await storage.batch_store_with_governance(memories)
            
            assert len(results) == 5
            for i, result in enumerate(results):
                assert isinstance(result, GovernanceAwareResult)
                assert result.success is True
                assert result.memory_id is not None
                assert result.governance_result.approved is True
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_governance_configuration_management(self, temp_db):
        """Test governance configuration retrieval and updates"""
        storage = GovernanceIntegratedStorage(temp_db)
        
        try:
            # Get current configuration
            config = storage.get_governance_configuration()
            
            assert 'enabled' in config
            assert 'enforcement_mode' in config
            assert 'thresholds' in config
            assert config['enabled'] is True
            
            # Update configuration
            new_config = {
                'enabled': True,
                'enforcement_mode': 'permissive',
                'thresholds': {
                    'content_size': 2000,
                    'tag_count': 200
                }
            }
            
            storage.update_governance_configuration(new_config)
            
            # Verify configuration updated
            updated_config = storage.get_governance_configuration()
            assert updated_config['enforcement_mode'] == 'permissive'
            assert updated_config['thresholds']['content_size'] == 2000
            assert updated_config['thresholds']['tag_count'] == 200
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_governance_threshold_adjustment(self, temp_db):
        """Test dynamic governance threshold adjustments"""
        storage = GovernanceIntegratedStorage(temp_db)
        
        try:
            # Adjust thresholds for testing context
            storage.adjust_governance_thresholds('testing', {
                'content_size': 100,
                'tag_count': 2
            })
            
            # The adjustment is applied to the governance framework
            # We can verify by checking the governance object directly
            testing_thresholds = storage.governance.get_thresholds('testing')
            assert testing_thresholds.content_size == 100
            assert testing_thresholds.tag_count == 2
            
            # Base thresholds unchanged
            base_thresholds = storage.governance.get_thresholds('production')
            assert base_thresholds.content_size == 1_000_000  # Default value
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_comprehensive_stats(self, temp_db, sample_memory):
        """Test comprehensive statistics including governance metrics"""
        storage = GovernanceIntegratedStorage(temp_db)
        
        try:
            # Store a few memories to generate stats
            for i in range(3):
                test_memory = EnhancedMemory(
                    content=f"Test content {i}",
                    summary=f"Summary {i}",
                    full_context=f"Full context for memory {i}",
                    tags=[f"test{i}"],
                    importance_score=0.5
                )
                await storage.store_with_governance(test_memory)
            
            # Get comprehensive stats
            stats = storage.get_comprehensive_stats()
            
            assert 'governance_integration' in stats
            governance_stats = stats['governance_integration']
            
            assert governance_stats['enabled'] is True
            assert governance_stats['framework_version'] == '2.0'
            assert governance_stats['total_validations'] >= 3
            assert governance_stats['approval_rate'] > 0
            assert 'performance' in governance_stats
            assert 'avg_governance_time_ms' in governance_stats['performance']
            assert 'avg_storage_time_ms' in governance_stats['performance']
            assert 'governance_framework_metrics' in stats
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_governance_performance_targets(self, temp_db, sample_memory):
        """Test governance performance meets <10ms target"""
        storage = GovernanceIntegratedStorage(temp_db)
        
        try:
            # Prepare memory for validation
            sample_memory.full_context = "Test content for performance validation"
            sample_memory.summary = "Performance test"
            sample_memory.tags = ["performance", "test"]
            
            # Measure governance performance over multiple operations
            governance_times = []
            total_times = []
            
            for _ in range(50):
                result = await storage.store_with_governance(sample_memory)
                governance_times.append(result.governance_time)
                total_times.append(result.total_processing_time)
            
            # Calculate performance metrics
            avg_governance_time = sum(governance_times) / len(governance_times)
            p95_governance_time = sorted(governance_times)[int(len(governance_times) * 0.95)]
            
            # Performance assertions
            assert avg_governance_time < 10.0, f"Average governance time {avg_governance_time:.2f}ms exceeds 10ms target"
            assert p95_governance_time < 15.0, f"P95 governance time {p95_governance_time:.2f}ms exceeds 15ms target"
            
            # Total end-to-end time should be reasonable
            avg_total_time = sum(total_times) / len(total_times)
            assert avg_total_time < 50.0, f"Average total time {avg_total_time:.2f}ms exceeds 50ms target"
            
        finally:
            await storage.close()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])