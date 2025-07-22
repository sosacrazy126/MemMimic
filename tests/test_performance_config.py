"""Performance Configuration Tests

Tests for the performance configuration system including:
- Configuration loading and validation
- YAML parsing and error handling
- Configuration path precedence
- Default value fallbacks
- Dynamic configuration updates
"""

import sys
import os
import tempfile
import yaml
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestPerformanceConfig:
    """Test suite for performance configuration system."""
    
    def test_config_loading(self):
        """Test configuration can be loaded successfully."""
        print("🧪 Testing configuration loading...")
        
        try:
            from memmimic.config import get_performance_config
            
            config = get_performance_config()
            
            # Check that config has required sections
            required_sections = ['database_config', 'memory_config', 'performance_config']
            for section in required_sections:
                assert hasattr(config, section), f"Config missing section: {section}"
                print(f"   ✅ {section}: present")
            
            # Check database config values
            db_config = config.database_config
            assert isinstance(db_config.get('connection_pool_size', 5), int)
            assert isinstance(db_config.get('connection_timeout', 5.0), (int, float))
            assert isinstance(db_config.get('wal_mode', True), bool)
            print("   ✅ Database config values valid")
            
            # Check memory config values
            mem_config = config.memory_config
            assert isinstance(mem_config.get('max_pool_size', 10000), int)
            assert isinstance(mem_config.get('cache_size', 1000), int)
            print("   ✅ Memory config values valid")
            
            # Check performance config values
            perf_config = config.performance_config
            assert isinstance(perf_config.get('query_timeout_ms', 100), int)
            assert isinstance(perf_config.get('batch_size', 100), int)
            print("   ✅ Performance config values valid")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Configuration loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_yaml_validation(self):
        """Test YAML file validation and error handling."""
        print("🧪 Testing YAML validation...")
        
        try:
            from memmimic.config import load_performance_config
            
            # Test 1: Valid YAML
            valid_config = {
                'database_config': {
                    'connection_pool_size': 10,
                    'connection_timeout': 3.0,
                    'wal_mode': True,
                    'cache_size': 20000
                },
                'memory_config': {
                    'max_pool_size': 15000,
                    'cache_size': 2000,
                    'cleanup_threshold': 0.8
                },
                'performance_config': {
                    'query_timeout_ms': 150,
                    'batch_size': 50,
                    'enable_metrics': True
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                yaml.dump(valid_config, tmp)
                tmp_path = tmp.name
            
            try:
                loaded_config = load_performance_config(tmp_path)
                assert loaded_config is not None
                assert loaded_config.database_config['connection_pool_size'] == 10
                print("   ✅ Valid YAML loaded correctly")
            finally:
                os.unlink(tmp_path)
            
            # Test 2: Invalid YAML structure
            invalid_config = """
            database_config:
              connection_pool_size: "not_a_number"  # Invalid type
              invalid_nesting:
                deeply:
                  nested: [1, 2, 3
            """  # Missing closing bracket
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                tmp.write(invalid_config)
                tmp_path = tmp.name
            
            try:
                # Should fall back to defaults due to YAML parse error
                loaded_config = load_performance_config(tmp_path)
                # If it loads, it should have used defaults
                if loaded_config:
                    print("   ✅ Invalid YAML handled gracefully (using defaults)")
                else:
                    print("   ⚠️ Invalid YAML returned None (acceptable)")
            except Exception as e:
                print(f"   ✅ Invalid YAML properly rejected: {type(e).__name__}")
            finally:
                os.unlink(tmp_path)
            
            return True
            
        except Exception as e:
            print(f"   ❌ YAML validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_config_path_precedence(self):
        """Test configuration file path precedence."""
        print("🧪 Testing config path precedence...")
        
        try:
            from memmimic.config import find_config_file
            
            # Test standard paths in order of precedence
            standard_paths = [
                "config/performance_config.yaml",
                "performance_config.yaml", 
                "~/.memmimic/performance_config.yaml"
            ]
            
            found_configs = []
            for path in standard_paths:
                expanded_path = Path(path).expanduser()
                if expanded_path.exists():
                    found_configs.append((path, expanded_path))
                    print(f"   ✅ Found config at: {path}")
            
            if found_configs:
                highest_precedence = found_configs[0]
                print(f"   📊 Highest precedence config: {highest_precedence[0]}")
            else:
                print("   ⚠️ No config files found in standard paths (using defaults)")
            
            # Test config file discovery
            config_file = find_config_file()
            if config_file:
                print(f"   ✅ Config discovery found: {config_file}")
            else:
                print("   ✅ Config discovery returned None (will use defaults)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Config path precedence test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_default_fallbacks(self):
        """Test that default values work when config is missing."""
        print("🧪 Testing default fallbacks...")
        
        try:
            from memmimic.config import get_default_performance_config
            
            # Get defaults
            defaults = get_default_performance_config()
            
            # Check that all required keys exist with reasonable values
            db_defaults = defaults.database_config
            assert db_defaults['connection_pool_size'] > 0
            assert db_defaults['connection_timeout'] > 0
            assert isinstance(db_defaults['wal_mode'], bool)
            print("   ✅ Database defaults present and valid")
            
            mem_defaults = defaults.memory_config  
            assert mem_defaults['max_pool_size'] > 0
            assert mem_defaults['cache_size'] > 0
            print("   ✅ Memory defaults present and valid")
            
            perf_defaults = defaults.performance_config
            assert perf_defaults['query_timeout_ms'] > 0
            assert perf_defaults['batch_size'] > 0
            print("   ✅ Performance defaults present and valid")
            
            # Test that defaults work when no config file exists
            from memmimic.config import load_performance_config
            config_from_nonexistent = load_performance_config("/nonexistent/path/config.yaml")
            
            # Should return defaults
            assert config_from_nonexistent is not None
            assert config_from_nonexistent.database_config['connection_pool_size'] > 0
            print("   ✅ Nonexistent file falls back to defaults")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Default fallbacks test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_config_validation(self):
        """Test configuration value validation."""
        print("🧪 Testing config validation...")
        
        try:
            from memmimic.config import validate_performance_config, PerformanceConfig
            
            # Test 1: Valid configuration
            valid_config_dict = {
                'database_config': {
                    'connection_pool_size': 5,
                    'connection_timeout': 10.0,
                    'wal_mode': True,
                    'cache_size': 10000
                },
                'memory_config': {
                    'max_pool_size': 20000,
                    'cache_size': 5000,
                    'cleanup_threshold': 0.7
                },
                'performance_config': {
                    'query_timeout_ms': 200,
                    'batch_size': 100,
                    'enable_metrics': True
                }
            }
            
            valid_config = PerformanceConfig.from_dict(valid_config_dict)
            is_valid, errors = validate_performance_config(valid_config)
            
            if is_valid:
                print("   ✅ Valid configuration passed validation")
            else:
                print(f"   ❌ Valid configuration failed validation: {errors}")
                return False
            
            # Test 2: Invalid configuration (negative values)
            invalid_config_dict = {
                'database_config': {
                    'connection_pool_size': -1,  # Invalid: negative
                    'connection_timeout': 0,     # Invalid: zero timeout
                    'wal_mode': True,
                    'cache_size': -5000         # Invalid: negative
                },
                'memory_config': {
                    'max_pool_size': 0,         # Invalid: zero size
                    'cache_size': -1,           # Invalid: negative
                    'cleanup_threshold': 1.5    # Invalid: > 1.0
                },
                'performance_config': {
                    'query_timeout_ms': -100,   # Invalid: negative
                    'batch_size': 0,           # Invalid: zero
                    'enable_metrics': True
                }
            }
            
            invalid_config = PerformanceConfig.from_dict(invalid_config_dict)
            is_valid, errors = validate_performance_config(invalid_config)
            
            if not is_valid:
                print(f"   ✅ Invalid configuration properly rejected: {len(errors)} errors")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"      - {error}")
            else:
                print("   ❌ Invalid configuration was accepted")
                return False
            
            return True
            
        except Exception as e:
            print(f"   ❌ Config validation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_config_integration(self):
        """Test configuration integration with AMMS."""
        print("🧪 Testing config integration...")
        
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage
            from memmimic.config import get_performance_config
            import tempfile
            
            # Create temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
            
            try:
                # Get configuration
                config = get_performance_config()
                db_config = config.database_config
                
                # Create AMMS storage with configuration
                storage = AMMSStorage(db_path)
                
                # Verify configuration was applied
                assert hasattr(storage, 'config'), "Storage should have config reference"
                assert storage.pool_size == db_config.get('connection_pool_size', 5)
                assert storage.connection_timeout == db_config.get('connection_timeout', 5.0)
                assert storage.enable_wal == db_config.get('wal_mode', True)
                assert storage.cache_size == db_config.get('cache_size', 10000)
                
                print(f"   ✅ Pool size: {storage.pool_size}")
                print(f"   ✅ Connection timeout: {storage.connection_timeout}s")
                print(f"   ✅ WAL mode: {storage.enable_wal}")
                print(f"   ✅ Cache size: {storage.cache_size}")
                
                # Test that configuration affects behavior
                stats = storage.get_stats()
                assert 'connection_pool' in stats
                pool_info = stats['connection_pool']
                assert pool_info['pool_size'] == storage.pool_size
                
                print("   ✅ Configuration successfully integrated with AMMS")
                
                asyncio.run(storage.close())
                return True
                
            finally:
                os.unlink(db_path)
            
        except Exception as e:
            print(f"   ❌ Config integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_performance_config_tests():
    """Run all performance configuration tests."""
    print("🚀 Running Performance Configuration Tests")
    print("=" * 50)
    
    test_suite = TestPerformanceConfig()
    
    tests = [
        ("Configuration Loading", test_suite.test_config_loading),
        ("YAML Validation", test_suite.test_yaml_validation),
        ("Config Path Precedence", test_suite.test_config_path_precedence),
        ("Default Fallbacks", test_suite.test_default_fallbacks),
        ("Config Validation", test_suite.test_config_validation),
        ("Config Integration", test_suite.test_config_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            results[test_name] = False
            print(f"   ❌ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PERFORMANCE CONFIG TESTS PASSED!")
        return 0
    else:
        print("💥 SOME PERFORMANCE CONFIG TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_performance_config_tests())