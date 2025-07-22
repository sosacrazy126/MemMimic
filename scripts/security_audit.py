#!/usr/bin/env python3
"""
Security Audit Script for MemMimic

This script performs a comprehensive security audit of MemMimic's credential
management and configuration security.
"""

import sys
import json
from pathlib import Path

# Add src to path to import MemMimic modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from memmimic.config.security import audit_credential_security, get_credentials
except ImportError as e:
    print(f"Error importing MemMimic security module: {e}")
    print("Make sure MemMimic is properly installed.")
    sys.exit(1)


def main():
    """Run comprehensive security audit."""
    print("🔒 MemMimic Security Audit")
    print("=" * 50)
    
    try:
        # Perform security audit
        audit_result = audit_credential_security()
        
        # Display results
        print(f"Security Score: {audit_result['security_score']}/100")
        
        if audit_result['security_score'] >= 90:
            status_emoji = "✅"
            status_text = "EXCELLENT"
        elif audit_result['security_score'] >= 75:
            status_emoji = "⚠️"
            status_text = "GOOD"
        elif audit_result['security_score'] >= 50:
            status_emoji = "⚠️"
            status_text = "NEEDS IMPROVEMENT"
        else:
            status_emoji = "❌"
            status_text = "CRITICAL ISSUES"
        
        print(f"Status: {status_emoji} {status_text}")
        print()
        
        # Configuration details
        print("📋 Configuration Summary:")
        print(f"  Configured Providers: {len(audit_result['configured_providers'])}")
        if audit_result['configured_providers']:
            print(f"    - {', '.join(audit_result['configured_providers'])}")
        
        print(f"  Placeholder Providers: {len(audit_result['placeholder_providers'])}")
        if audit_result['placeholder_providers']:
            print(f"    - {', '.join(audit_result['placeholder_providers'])}")
        
        print(f"  Validation Errors: {audit_result['total_validation_errors']}")
        if audit_result['validation_errors']:
            for error in audit_result['validation_errors']:
                print(f"    ⚠️ {error}")
        
        print()
        
        # File security check
        print("📁 File Security:")
        print(f"  .env file exists: {'✅' if audit_result['env_file_exists'] else '❌'}")
        print(f"  .env.template exists: {'✅' if audit_result['env_template_exists'] else '❌'}")
        print(f"  .gitignore excludes .env: {'✅' if audit_result['gitignore_excludes_env'] else '❌'}")
        print()
        
        # Security settings
        print("⚙️ Security Settings:")
        print(f"  Validation Enabled: {'✅' if audit_result['validation_enabled'] else '❌'}")
        print(f"  Secure Defaults: {'✅' if audit_result['secure_defaults'] else '❌'}")
        print()
        
        # Recommendations
        if audit_result['recommendations']:
            print("💡 Security Recommendations:")
            for i, recommendation in enumerate(audit_result['recommendations'], 1):
                print(f"  {i}. {recommendation}")
            print()
        
        # JSON output option
        if len(sys.argv) > 1 and sys.argv[1] == '--json':
            print("\n🔍 Full Audit Data (JSON):")
            print(json.dumps(audit_result, indent=2))
        
        # Exit with appropriate code
        if audit_result['security_score'] < 50:
            print("❌ SECURITY AUDIT FAILED - Critical issues found!")
            sys.exit(1)
        elif audit_result['security_score'] < 75:
            print("⚠️ SECURITY AUDIT WARNING - Improvements recommended")
            sys.exit(0)
        else:
            print("✅ SECURITY AUDIT PASSED")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Security audit failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()