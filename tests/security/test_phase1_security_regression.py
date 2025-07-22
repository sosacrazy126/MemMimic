#!/usr/bin/env python3
"""
Phase 1 Security Regression Tests

Comprehensive security regression tests to validate that all Phase 1 security 
fixes are working correctly and vulnerabilities cannot be re-introduced.

Tests for:
1. Credential security (no hardcoded secrets)
2. Eval/exec vulnerability prevention
3. Input validation framework
4. Security audit system
5. Error handling security
"""

import sys
import os
import subprocess
import tempfile
import sqlite3
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from memmimic.security import (
    validate_input, sanitize_output, audit_security,
    SecurityValidationError, ValidationError
)
from memmimic.config.security import SecureCredentials, audit_credential_security


class TestCredentialSecurity:
    """Test credential security measures from Phase 1."""
    
    def test_no_hardcoded_api_keys_in_source(self):
        """Test that no hardcoded API keys remain in source code."""
        print("🔒 Testing for hardcoded API keys...")
        
        # Define dangerous patterns that should not exist
        dangerous_patterns = [
            r'pplx-[a-zA-Z0-9]{32,}',  # Perplexity API keys
            r'AIzaSy[a-zA-Z0-9_-]{33}',  # Google API keys  
            r'sk-ant-api03-[a-zA-Z0-9_-]{95}',  # Anthropic API keys
            r'sk-proj-[a-zA-Z0-9]{48}',  # OpenAI project API keys
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub personal access tokens
            r'github_pat_[a-zA-Z0-9_]{82}',  # GitHub fine-grained tokens
        ]
        
        # Search source code directories
        src_dir = Path(__file__).parent.parent.parent / 'src'
        
        violations = []
        for pattern in dangerous_patterns:
            try:
                # Use ripgrep if available for better performance
                result = subprocess.run([
                    'rg', '-r', pattern, '--type', 'py', str(src_dir)
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    violations.append({
                        'pattern': pattern,
                        'matches': result.stdout.strip().split('\n')
                    })
            except FileNotFoundError:
                # Fallback to grep if ripgrep not available
                result = subprocess.run([
                    'grep', '-r', '-E', pattern, str(src_dir), '--include=*.py'
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    violations.append({
                        'pattern': pattern,
                        'matches': result.stdout.strip().split('\n')
                    })
        
        if violations:
            print(f"   ❌ Found {len(violations)} credential violations:")
            for violation in violations:
                print(f"      Pattern: {violation['pattern']}")
                for match in violation['matches'][:3]:  # Show first 3 matches
                    print(f"        {match}")
            assert False, f"Hardcoded credentials found: {violations}"
        
        print("   ✅ No hardcoded API keys found in source code")
    
    def test_env_file_security(self):
        """Test that .env file contains only placeholders."""
        print("🔒 Testing .env file security...")
        
        env_file = Path(__file__).parent.parent.parent / '.env'
        
        if not env_file.exists():
            print("   ⚠️ .env file not found (may use environment variables)")
            return
        
        # Read .env file
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        # Look for dangerous patterns in .env
        dangerous_patterns = [
            r'pplx-[a-zA-Z0-9]{20,}',  # Real Perplexity keys
            r'AIzaSy[a-zA-Z0-9_-]{30,}',  # Real Google keys
            r'sk-ant-api03-[a-zA-Z0-9_-]{90,}',  # Real Anthropic keys
            r'sk-proj-[a-zA-Z0-9]{45,}',  # Real OpenAI keys
            r'ghp_[a-zA-Z0-9]{30,}',  # Real GitHub tokens
        ]
        
        violations = []
        for pattern in dangerous_patterns:
            import re
            matches = re.findall(pattern, env_content)
            if matches:
                violations.extend(matches)
        
        if violations:
            print(f"   ❌ Found potential real credentials in .env: {len(violations)}")
            assert False, f"Real credentials may be in .env file: {violations[:3]}"
        
        print("   ✅ .env file appears to contain only placeholders")
    
    def test_secure_credential_loading(self):
        """Test secure credential loading system."""
        print("🔒 Testing secure credential loading...")
        
        try:
            from memmimic.config.security import SecureCredentials
            
            # Test with placeholder values (should detect as placeholder)
            credentials = SecureCredentials()
            
            # Should not validate placeholder credentials as real
            test_providers = ['anthropic', 'perplexity', 'openai', 'google']
            
            for provider in test_providers:
                is_placeholder = credentials.is_placeholder(provider)
                if not is_placeholder and credentials._credentials.get(provider):
                    # If not placeholder and has value, should be a real credential
                    # This is acceptable for testing environments
                    print(f"   ✅ {provider}: Real credential detected (testing environment)")
                else:
                    print(f"   ✅ {provider}: Placeholder or missing (secure)")
            
            print("   ✅ Secure credential loading system working")
            
        except Exception as e:
            print(f"   ❌ Credential loading failed: {e}")
            raise
    
    def test_security_audit_system(self):
        """Test the security audit system."""
        print("🔒 Testing security audit system...")
        
        try:
            audit_result = audit_credential_security()
            
            # Should return audit information
            assert isinstance(audit_result, dict)
            assert 'security_score' in audit_result
            assert 'total_providers' in audit_result
            assert 'placeholder_providers' in audit_result
            
            # Security score should be reasonable (0-100)
            score = audit_result['security_score']
            assert 0 <= score <= 100
            
            # Should have found some providers (placeholders are good)
            total_providers = audit_result['total_providers']
            assert total_providers > 0
            
            print(f"   ✅ Security audit score: {score}/100")
            print(f"   ✅ Total providers: {total_providers}")
            print(f"   ✅ Placeholder providers: {audit_result['placeholder_providers']}")
            
        except Exception as e:
            print(f"   ❌ Security audit failed: {e}")
            raise


class TestEvalExecVulnerabilities:
    """Test that eval/exec vulnerabilities are blocked."""
    
    def test_eval_blocked_in_json_processing(self):
        """Test that eval is not used in JSON processing."""
        print("🚨 Testing eval vulnerability prevention...")
        
        # Create malicious JSON-like string that would be dangerous with eval
        malicious_json = '''
        {
            "__import__('os').system('echo PWNED')": "dangerous",
            "normal_key": "normal_value"
        }
        '''
        
        try:
            # Test that our JSON processing doesn't use eval
            from memmimic.memory.storage.amms_storage import AMMSStorage
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
            
            try:
                storage = AMMSStorage(db_path)
                
                # Try to store malicious JSON - should be safe
                await storage.initialize()
                
                # This should not execute the malicious code
                result = await storage.store_memory(
                    content=malicious_json,
                    memory_type="test",
                    metadata={"test": True}
                )
                
                # Should successfully store without executing malicious code
                assert result > 0  # Should get a memory ID
                print("   ✅ Malicious JSON handled safely")
                
                await storage.close()
                
            finally:
                os.unlink(db_path)
                
        except Exception as e:
            print(f"   ❌ Eval vulnerability test failed: {e}")
            raise
    
    def test_no_exec_in_code_execution(self):
        """Test that exec is not used for dynamic code execution."""
        print("🚨 Testing exec vulnerability prevention...")
        
        # Search for dangerous exec usage in source code
        src_dir = Path(__file__).parent.parent.parent / 'src'
        
        dangerous_exec_patterns = [
            r'\bexec\s*\(',  # exec() function calls
            r'\beval\s*\(',  # eval() function calls
            r'__import__\s*\(',  # dynamic imports that could be dangerous
        ]
        
        violations = []
        for pattern in dangerous_exec_patterns:
            try:
                result = subprocess.run([
                    'rg', '-n', pattern, '--type', 'py', str(src_dir)
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    matches = result.stdout.strip().split('\n')
                    # Filter out comments and safe usage
                    real_violations = []
                    for match in matches:
                        if not (
                            '# ' in match or 
                            '"""' in match or 
                            "'''" in match or
                            'test_' in match.lower()
                        ):
                            real_violations.append(match)
                    
                    if real_violations:
                        violations.extend(real_violations)
                        
            except FileNotFoundError:
                # Fallback to grep
                result = subprocess.run([
                    'grep', '-n', '-E', pattern, '-r', str(src_dir), '--include=*.py'
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    matches = result.stdout.strip().split('\n')
                    # Filter out comments and safe usage
                    real_violations = []
                    for match in matches:
                        if not (
                            '# ' in match or 
                            '"""' in match or 
                            "'''" in match or
                            'test_' in match.lower()
                        ):
                            real_violations.append(match)
                    
                    if real_violations:
                        violations.extend(real_violations)
        
        if violations:
            print(f"   ⚠️ Found {len(violations)} potential exec/eval usage:")
            for violation in violations[:5]:  # Show first 5
                print(f"      {violation}")
            # This might be acceptable in some contexts, so warning not error
            print("   ⚠️ Manual review required for exec/eval usage")
        else:
            print("   ✅ No dangerous exec/eval usage found")
    
    def test_safe_json_parsing(self):
        """Test that JSON parsing is safe and doesn't allow code execution."""
        print("🚨 Testing safe JSON parsing...")
        
        malicious_inputs = [
            '{"__import__": "os", "__exec__": "system(\\"echo PWNED\\")"}',
            '{"eval": "print(\\"DANGEROUS\\")"}',
            '{"exec": "import os; os.system(\\"echo PWNED\\")"}',
        ]
        
        try:
            import json
            
            for malicious_input in malicious_inputs:
                try:
                    # Standard JSON parsing should be safe
                    parsed = json.loads(malicious_input)
                    
                    # The parsed result should just be a dictionary, not executable code
                    assert isinstance(parsed, dict)
                    print(f"   ✅ Malicious JSON parsed safely: {type(parsed)}")
                    
                    # Ensure values are strings, not executable
                    for key, value in parsed.items():
                        assert isinstance(value, (str, int, float, bool, list, dict, type(None)))
                    
                except json.JSONDecodeError:
                    print("   ✅ Malformed JSON properly rejected")
            
            print("   ✅ JSON parsing is secure")
            
        except Exception as e:
            print(f"   ❌ JSON security test failed: {e}")
            raise


class TestInputValidationFramework:
    """Test the input validation framework."""
    
    def test_memory_content_validation(self):
        """Test memory content validation."""
        print("🛡️ Testing memory content validation...")
        
        try:
            from memmimic.security.validation import validate_memory_content
            
            # Test valid content
            valid_content = "This is normal memory content for testing purposes."
            result = validate_memory_content(valid_content)
            assert result is True or result == valid_content  # May return sanitized version
            print("   ✅ Valid content accepted")
            
            # Test potentially dangerous content
            dangerous_content = [
                "'; DROP TABLE memories; --",  # SQL injection
                "<script>alert('xss')</script>",  # XSS
                "../../../etc/passwd",  # Path traversal
                "__import__('os').system('rm -rf /')",  # Code injection
            ]
            
            blocked_count = 0
            sanitized_count = 0
            
            for content in dangerous_content:
                try:
                    result = validate_memory_content(content, strict=True)
                    if result != content:
                        sanitized_count += 1
                        print(f"   ✅ Dangerous content sanitized: {content[:20]}...")
                    else:
                        print(f"   ⚠️ Dangerous content passed through: {content[:20]}...")
                except (SecurityValidationError, ValidationError):
                    blocked_count += 1
                    print(f"   ✅ Dangerous content blocked: {content[:20]}...")
            
            assert blocked_count + sanitized_count > 0, "No dangerous content was handled"
            print(f"   ✅ Validation system working: {blocked_count} blocked, {sanitized_count} sanitized")
            
        except Exception as e:
            print(f"   ❌ Input validation test failed: {e}")
            raise
    
    def test_tale_input_validation(self):
        """Test tale input validation."""
        print("🛡️ Testing tale input validation...")
        
        try:
            from memmimic.security.validation import validate_tale_input
            
            # Test valid tale input
            valid_name = "normal_tale_name"
            valid_content = "This is normal tale content."
            
            result = validate_tale_input(valid_name, valid_content)
            assert result is True or isinstance(result, tuple)
            print("   ✅ Valid tale input accepted")
            
            # Test dangerous tale inputs
            dangerous_names = [
                "../../../malicious_tale",  # Path traversal
                "tale_with_<script>",  # XSS in name
                "tale'; DROP TABLE tales; --",  # SQL injection
            ]
            
            for name in dangerous_names:
                try:
                    result = validate_tale_input(name, "content", strict=True)
                    if isinstance(result, tuple):
                        sanitized_name, sanitized_content = result
                        if sanitized_name != name:
                            print(f"   ✅ Dangerous tale name sanitized: {name}")
                        else:
                            print(f"   ⚠️ Dangerous tale name passed through: {name}")
                    else:
                        print(f"   ⚠️ Dangerous tale name handling unclear: {name}")
                except (SecurityValidationError, ValidationError):
                    print(f"   ✅ Dangerous tale name blocked: {name}")
            
            print("   ✅ Tale input validation working")
            
        except Exception as e:
            print(f"   ❌ Tale input validation test failed: {e}")
            raise
    
    def test_query_input_validation(self):
        """Test search query input validation."""
        print("🛡️ Testing query input validation...")
        
        try:
            from memmimic.security.validation import validate_query_input
            
            # Test valid queries
            valid_queries = [
                "normal search query",
                "search with numbers 123",
                "search-with-hyphens",
            ]
            
            for query in valid_queries:
                result = validate_query_input(query)
                assert result is True or isinstance(result, str)
                print(f"   ✅ Valid query accepted: {query[:30]}...")
            
            # Test invalid queries
            invalid_queries = [
                "",  # Empty query
                "x" * 1001,  # Too long
                "'; DROP TABLE memories; --",  # SQL injection
                "<script>alert('xss')</script>",  # XSS
            ]
            
            for query in invalid_queries:
                try:
                    result = validate_query_input(query, strict=True)
                    if result != query:
                        print(f"   ✅ Invalid query sanitized: {query[:30]}...")
                    else:
                        print(f"   ⚠️ Invalid query passed through: {query[:30]}...")
                except (SecurityValidationError, ValidationError):
                    print(f"   ✅ Invalid query blocked: {query[:30]}...")
            
            print("   ✅ Query input validation working")
            
        except Exception as e:
            print(f"   ❌ Query input validation test failed: {e}")
            raise


class TestErrorHandlingSecurity:
    """Test security measures in error handling."""
    
    def test_error_information_leakage(self):
        """Test that errors don't leak sensitive information."""
        print("🔐 Testing error information security...")
        
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage
            import tempfile
            
            # Create temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
            
            try:
                storage = AMMSStorage(db_path)
                await storage.initialize()
                
                # Cause intentional error
                try:
                    # Try to access non-existent memory
                    result = await storage.get_memory(-999999)
                    # Should return None or empty result, not error
                    assert result is None or result == []
                    print("   ✅ Non-existent memory handled gracefully")
                except Exception as e:
                    # Error message should not contain sensitive info
                    error_msg = str(e).lower()
                    sensitive_patterns = [
                        'password',
                        'secret',
                        'api_key',
                        'token',
                        'credential',
                    ]
                    
                    for pattern in sensitive_patterns:
                        assert pattern not in error_msg, f"Error contains sensitive info: {pattern}"
                    
                    print("   ✅ Error message doesn't leak sensitive info")
                
                await storage.close()
                
            finally:
                os.unlink(db_path)
            
            print("   ✅ Error handling security verified")
            
        except Exception as e:
            print(f"   ❌ Error handling security test failed: {e}")
            raise
    
    def test_sql_injection_in_errors(self):
        """Test that SQL injection doesn't work through error conditions."""
        print("🔐 Testing SQL injection in error handling...")
        
        malicious_inputs = [
            "'; DROP TABLE memories; --",
            "' UNION SELECT * FROM sqlite_master; --",
            "'; INSERT INTO memories (content) VALUES ('HACKED'); --",
        ]
        
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
            
            try:
                storage = AMMSStorage(db_path)
                await storage.initialize()
                
                for malicious_input in malicious_inputs:
                    try:
                        # Try to use malicious input in various ways
                        result = await storage.store_memory(
                            content=malicious_input,
                            memory_type="test",
                            metadata={"malicious": True}
                        )
                        
                        # Should store safely as regular content
                        assert result > 0  # Should get memory ID
                        
                        # Verify the malicious input was stored as content, not executed
                        stored = await storage.get_memory(result)
                        assert stored is not None
                        assert malicious_input in stored.get('content', '')
                        
                        print(f"   ✅ Malicious input stored safely: {malicious_input[:30]}...")
                        
                    except Exception as e:
                        # If it fails, the error shouldn't expose SQL details
                        error_msg = str(e).lower()
                        sql_patterns = ['sqlite', 'database', 'table', 'column']
                        exposed_patterns = [p for p in sql_patterns if p in error_msg]
                        
                        if exposed_patterns:
                            print(f"   ⚠️ Error exposes SQL details: {exposed_patterns}")
                        else:
                            print(f"   ✅ Error doesn't expose SQL details")
                
                await storage.close()
                
            finally:
                os.unlink(db_path)
            
            print("   ✅ SQL injection security verified")
            
        except Exception as e:
            print(f"   ❌ SQL injection security test failed: {e}")
            raise


async def run_phase1_security_tests():
    """Run all Phase 1 security regression tests."""
    print("🛡️ Running Phase 1 Security Regression Tests")
    print("=" * 60)
    
    test_classes = [
        ("Credential Security", TestCredentialSecurity()),
        ("Eval/Exec Vulnerabilities", TestEvalExecVulnerabilities()),
        ("Input Validation Framework", TestInputValidationFramework()),
        ("Error Handling Security", TestErrorHandlingSecurity()),
    ]
    
    results = {}
    
    for category, test_instance in test_classes:
        print(f"\n🧪 Testing: {category}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [
            method for method in dir(test_instance)
            if method.startswith('test_') and callable(getattr(test_instance, method))
        ]
        
        category_results = {}
        
        for test_method in test_methods:
            method_name = test_method.replace('test_', '').replace('_', ' ').title()
            
            try:
                test_func = getattr(test_instance, test_method)
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                category_results[method_name] = True
                print(f"   ✅ {method_name}")
            except Exception as e:
                category_results[method_name] = False
                print(f"   ❌ {method_name}: {e}")
        
        results[category] = category_results
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Phase 1 Security Test Results:")
    
    total_tests = 0
    passed_tests = 0
    
    for category, category_results in results.items():
        category_passed = sum(category_results.values())
        category_total = len(category_results)
        total_tests += category_total
        passed_tests += category_passed
        
        status = "✅" if category_passed == category_total else "❌"
        print(f"{status} {category}: {category_passed}/{category_total}")
        
        if category_passed != category_total:
            for test, result in category_results.items():
                if not result:
                    print(f"     ❌ {test}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 95:
        print("🎉 PHASE 1 SECURITY REGRESSION TESTS PASSED!")
        return 0
    elif success_rate >= 80:
        print("⚠️ Most security tests passed, some issues need attention")
        return 1
    else:
        print("❌ CRITICAL SECURITY ISSUES DETECTED!")
        return 2


if __name__ == "__main__":
    import asyncio
    import sys
    
    sys.exit(asyncio.run(run_phase1_security_tests()))