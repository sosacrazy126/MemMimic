#!/usr/bin/env python3
"""
MemMimic v2.0 Governance Framework Demo
Demonstrates governance capabilities with various scenarios and enforcement modes.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

from memmimic.memory import (
    EnhancedMemory, GovernanceIntegratedStorage, 
    GovernanceConfig, SimpleGovernance, GovernanceAwareResult
)


async def demo_governance_scenarios():
    """Demonstrate various governance scenarios"""
    print("🏛️  MemMimic v2.0 Governance Framework Demo")
    print("=" * 60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Demo configurations for different enforcement modes
    demo_configs = {
        "strict": {
            'governance': {
                'enabled': True,
                'enforcement_mode': 'strict',
                'environment': 'production',
                'thresholds': {
                    'content_size': 200,      # Small for demo
                    'summary_length': 50,
                    'tag_count': 3,
                    'tag_length': 15,
                    'governance_timeout': 10
                }
            }
        },
        "permissive": {
            'governance': {
                'enabled': True,
                'enforcement_mode': 'permissive',
                'environment': 'development',
                'thresholds': {
                    'content_size': 200,
                    'summary_length': 50,
                    'tag_count': 3,
                    'tag_length': 15,
                    'governance_timeout': 10
                }
            }
        },
        "audit_only": {
            'governance': {
                'enabled': True,
                'enforcement_mode': 'audit_only',
                'environment': 'testing',
                'thresholds': {
                    'content_size': 200,
                    'summary_length': 50,
                    'tag_count': 3,
                    'tag_length': 15,
                    'governance_timeout': 10
                }
            }
        }
    }
    
    # Test memories with various compliance levels
    test_memories = [
        {
            'name': 'Compliant Memory',
            'memory': EnhancedMemory(
                content="Valid content",
                summary="Valid summary",
                full_context="This is a valid memory that complies with all governance rules",
                tags=["valid", "demo"],
                metadata={"test": True},
                importance_score=0.7
            )
        },
        {
            'name': 'Content Size Violation',
            'memory': EnhancedMemory(
                content="Long content",
                summary="Summary for long content",
                full_context="This is a very long content that exceeds the governance limit for content size. " * 10,
                tags=["long", "violation"],
                metadata={"test": True, "violation_type": "content_size"},
                importance_score=0.8
            )
        },
        {
            'name': 'Tag Violations',
            'memory': EnhancedMemory(
                content="Tag violation content",
                summary="Tag violation summary",
                full_context="Content with tag violations",
                tags=["tag1", "tag2", "tag3", "tag4", "this_tag_is_way_too_long_for_governance"],
                metadata={"test": True, "violation_type": "tags"},
                importance_score=0.6
            )
        },
        {
            'name': 'Multiple Violations',
            'memory': EnhancedMemory(
                content="Multiple violations",
                summary="This summary is way too long for the governance rules and exceeds the limit significantly",
                full_context="This content has multiple governance violations including being too long. " * 8,
                tags=["tag1", "tag2", "tag3", "tag4", "way_too_long_tag_name"],
                metadata={"test": True, "violation_type": "multiple"},
                importance_score=0.9
            )
        }
    ]
    
    # Test each enforcement mode
    for mode_name, config in demo_configs.items():
        print(f"\n📋 Testing {mode_name.upper()} Enforcement Mode")
        print("-" * 50)
        
        # Initialize storage with governance
        storage = GovernanceIntegratedStorage(db_path, config=config)
        
        try:
            print(f"Configuration: {storage.get_governance_configuration()}")
            print()
            
            # Test each memory type
            for test_case in test_memories:
                print(f"  Testing: {test_case['name']}")
                
                # Store with governance
                result = await storage.store_with_governance(test_case['memory'])
                
                # Display results
                print(f"    ✅ Success: {result.success}")
                print(f"    📊 Status: {result.governance_result.status}")
                print(f"    ⏱️  Governance Time: {result.governance_time:.2f}ms")
                print(f"    💾 Storage Time: {result.storage_time:.2f}ms")
                print(f"    📝 Message: {result.message}")
                
                if result.governance_result.violations:
                    print(f"    ⚠️  Violations ({len(result.governance_result.violations)}):")
                    for violation in result.governance_result.violations:
                        print(f"      • {violation.type} ({violation.severity}): {violation.message}")
                
                if result.governance_result.warnings:
                    print(f"    ⚡ Warnings ({len(result.governance_result.warnings)}):")
                    for warning in result.governance_result.warnings[:2]:  # Show first 2
                        print(f"      • {warning}")
                
                print()
            
            # Display governance statistics
            stats = storage.get_comprehensive_stats()
            gov_stats = stats['governance_integration']
            
            print(f"  📈 {mode_name.upper()} Mode Statistics:")
            print(f"    Total Validations: {gov_stats['total_validations']}")
            print(f"    Approval Rate: {gov_stats['approval_rate']:.1f}%")
            print(f"    Rejection Rate: {gov_stats['rejection_rate']:.1f}%")
            print(f"    Avg Governance Time: {gov_stats['performance']['avg_governance_time_ms']:.2f}ms")
            print(f"    Governance Overhead: {gov_stats['performance']['governance_overhead_percent']:.1f}%")
            
        finally:
            await storage.close()
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


async def demo_performance_validation():
    """Demo governance performance validation"""
    print("\n⚡ Performance Validation Demo")
    print("-" * 40)
    
    # Create high-performance governance config
    perf_config = GovernanceConfig(
        content_size=1_000_000,
        summary_length=1000,
        tag_count=100,
        tag_length=50,
        governance_timeout=10,  # Target: <10ms
        enforcement_mode="strict"
    )
    
    governance = SimpleGovernance(perf_config)
    
    # Create test memory
    test_memory = EnhancedMemory(
        content="Performance test content",
        summary="Performance test summary",
        full_context="This memory is designed for performance testing of the governance framework",
        tags=["performance", "test", "benchmark"],
        metadata={"benchmark": True, "iteration": 0},
        importance_score=0.5
    )
    
    # Run performance test
    print("Running governance performance test (100 iterations)...")
    
    times = []
    results = []
    
    for i in range(100):
        test_memory.metadata['iteration'] = i
        result = await governance.validate_memory(test_memory, "performance_test")
        times.append(result.processing_time)
        results.append(result.approved)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    p95_time = sorted(times)[95]
    approval_rate = sum(results) / len(results) * 100
    
    print(f"\n📊 Performance Results:")
    print(f"  Average Time: {avg_time:.3f}ms")
    print(f"  Min Time: {min_time:.3f}ms")
    print(f"  Max Time: {max_time:.3f}ms")
    print(f"  P95 Time: {p95_time:.3f}ms")
    print(f"  Target Met: {'✅ YES' if avg_time < 10.0 else '❌ NO'} (target: <10ms)")
    print(f"  Approval Rate: {approval_rate:.1f}%")
    
    # Get governance metrics
    metrics = governance.get_performance_metrics()
    print(f"\n📈 Framework Metrics:")
    print(f"  Total Validations: {metrics['performance']['total_validations']}")
    print(f"  Framework Version: {metrics['version']}")
    print(f"  Configuration Mode: {metrics['configuration']['enforcement_mode']}")


async def demo_dynamic_thresholds():
    """Demo dynamic threshold adjustment"""
    print("\n🔧 Dynamic Threshold Adjustment Demo")  
    print("-" * 45)
    
    governance = SimpleGovernance()
    
    # Base memory that passes default thresholds
    base_memory = EnhancedMemory(
        content="Base content",
        summary="Base summary",
        full_context="This is base content for threshold testing",
        tags=["base", "test"],
        importance_score=0.5
    )
    
    print("1. Testing with default thresholds:")
    result = await governance.validate_memory(base_memory, "default")
    print(f"   Result: {result.status} (approved: {result.approved})")
    print(f"   Content limit: {governance.get_thresholds().content_size}")
    
    print("\n2. Adjusting thresholds for 'strict' context:")
    governance.adjust_thresholds('strict', {
        'content_size': 30,  # Very strict
        'tag_count': 1
    })
    
    result = await governance.validate_memory(base_memory, "strict")
    print(f"   Result: {result.status} (approved: {result.approved})")
    print(f"   Violations: {len(result.violations)}")
    strict_thresholds = governance.get_thresholds('strict')
    print(f"   Strict content limit: {strict_thresholds.content_size}")
    
    print("\n3. Testing with 'lenient' context (no adjustments):")
    result = await governance.validate_memory(base_memory, "lenient") 
    print(f"   Result: {result.status} (approved: {result.approved})")
    lenient_thresholds = governance.get_thresholds('lenient')
    print(f"   Lenient content limit: {lenient_thresholds.content_size}")
    
    print("\n4. Hot-reload configuration:")
    new_config = {
        'enabled': True,
        'enforcement_mode': 'permissive',
        'thresholds': {
            'content_size': 2000,
            'tag_count': 20
        }
    }
    governance.reload_config(new_config)
    
    result = await governance.validate_memory(base_memory, "default")
    print(f"   After reload - Result: {result.status} (approved: {result.approved})")
    print(f"   New enforcement mode: {governance.config.enforcement_mode}")
    print(f"   New content limit: {governance.config.content_size}")


async def demo_yaml_configuration():
    """Demo YAML configuration loading"""
    print("\n📄 YAML Configuration Demo")
    print("-" * 35)
    
    # Create a sample YAML configuration
    yaml_config = """
governance:
  enabled: true
  enforcement_mode: "permissive"
  environment: "demo"
  
  thresholds:
    content_size: 1500
    summary_length: 200
    tag_count: 10
    tag_length: 25
    governance_timeout: 15
    
  custom_settings:
    demo_mode: true
    logging_level: "debug"
"""
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_config)
        yaml_path = f.name
    
    try:
        # Load governance from YAML
        print(f"Loading configuration from: {yaml_path}")
        governance = SimpleGovernance.from_yaml_file(yaml_path)
        
        print(f"Loaded configuration:")
        print(f"  Enforcement Mode: {governance.config.enforcement_mode}")
        print(f"  Environment: {governance.config.environment}")
        print(f"  Content Size Limit: {governance.config.content_size}")
        print(f"  Tag Count Limit: {governance.config.tag_count}")
        print(f"  Governance Timeout: {governance.config.governance_timeout}ms")
        
        # Test with the loaded configuration
        test_memory = EnhancedMemory(
            content="YAML config test",
            summary="Testing YAML configuration loading",
            full_context="This memory tests the YAML configuration loading feature",
            tags=["yaml", "config", "demo"],
            importance_score=0.6
        )
        
        result = await governance.validate_memory(test_memory, "yaml_test")
        print(f"\nValidation with YAML config:")
        print(f"  Status: {result.status}")
        print(f"  Processing Time: {result.processing_time:.2f}ms")
        print(f"  Approved: {result.approved}")
        
    finally:
        # Cleanup
        Path(yaml_path).unlink(missing_ok=True)


async def main():
    """Main demo function"""
    print("🚀 Starting MemMimic v2.0 Governance Framework Demo")
    print("=" * 60)
    
    try:
        await demo_governance_scenarios()
        await demo_performance_validation() 
        await demo_dynamic_thresholds()
        await demo_yaml_configuration()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Three enforcement modes: strict, permissive, audit_only")
        print("  • Real-time validation with <10ms performance target")
        print("  • Dynamic threshold adjustment")
        print("  • YAML configuration loading")
        print("  • Comprehensive violation reporting")
        print("  • Performance metrics and monitoring")
        print("  • Integration with enhanced storage")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())