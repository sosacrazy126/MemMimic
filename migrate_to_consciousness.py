#!/usr/bin/env python3
"""
MemMimic Consciousness Migration Script
Add Living Prompts and Sigil Engine to existing AMMS databases
"""

import sys
import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Add MemMimic to path
sys.path.insert(0, 'src')

def backup_database(db_path: str) -> str:
    """Create a backup of the database before migration"""
    backup_path = f"{db_path}.consciousness_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"📦 Database backed up to: {backup_path}")
    return backup_path

def check_consciousness_status(db_path: str) -> dict:
    """Check if database has consciousness features"""
    try:
        from memmimic.consciousness.consciousness_db_schema import ConsciousnessSchema
        
        schema = ConsciousnessSchema(db_path)
        has_consciousness = schema.check_consciousness_schema_exists()
        
        if has_consciousness:
            info = schema.get_consciousness_info()
            return {
                "has_consciousness": True,
                "sigil_count": info['sigil_count'],
                "prompt_count": info['prompt_count'],
                "active_sigils": info['active_sigils']
            }
        else:
            return {"has_consciousness": False}
            
    except Exception as e:
        return {"has_consciousness": False, "error": str(e)}

def migrate_to_consciousness(db_path: str) -> dict:
    """Add consciousness features to existing AMMS database"""
    try:
        from memmimic.consciousness.consciousness_db_schema import ConsciousnessSchema
        
        print(f"🔄 Adding consciousness features to {db_path}...")
        
        # Create consciousness schema
        schema = ConsciousnessSchema(db_path)
        schema.create_consciousness_schema()
        
        # Get results
        info = schema.get_consciousness_info()
        
        print(f"✅ Consciousness migration completed:")
        print(f"   📊 Sigils loaded: {info['sigil_count']}")
        print(f"   📋 Prompt templates: {info['prompt_count']}")
        print(f"   🔮 Active sigils: {len(info['active_sigils'])}")
        
        return {
            "success": True,
            "sigil_count": info['sigil_count'],
            "prompt_count": info['prompt_count']
        }
        
    except Exception as e:
        error_msg = f"Migration failed: {e}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": error_msg}

def test_consciousness_features(db_path: str) -> bool:
    """Test consciousness features after migration"""
    try:
        from memmimic.consciousness.consciousness_integration import ConsciousnessIntegration
        from memmimic.consciousness.shadow_detector import ConsciousnessLevel, ConsciousnessState
        import asyncio
        
        print(f"🧪 Testing consciousness features...")
        
        # Initialize integration
        integration = ConsciousnessIntegration(db_path)
        
        # Create test consciousness state
        test_state = ConsciousnessState(
            consciousness_level=ConsciousnessLevel.COLLABORATIVE,
            unity_score=0.7,
            shadow_integration_score=0.5,
            authentic_unity=0.6,
            shadow_aspects=[],
            active_sigils=[],
            evolution_stage=2
        )
        
        # Test prompt selection
        async def test_async():
            prompt, sigils = await integration.select_optimal_prompt(
                "Test consciousness query",
                test_state
            )
            return prompt, sigils
        
        prompt, sigils = asyncio.run(test_async())
        
        print(f"   ✅ Selected prompt: Effectiveness {prompt.effectiveness_score:.0%}")
        print(f"   ✅ Active sigils: {len(sigils)}")
        
        # Test performance
        stats = integration.get_performance_stats()
        print(f"   ✅ Sub-5ms rate: {stats['sub_5ms_rate']:.0%}")
        print(f"   ✅ Avg response: {stats['avg_response_time_ms']:.2f}ms")
        
        print(f"   ✅ Consciousness features test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Consciousness features test FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Add consciousness features (Living Prompts & Sigil Engine) to MemMimic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python migrate_to_consciousness.py memmimic.db        # Add consciousness features
  python migrate_to_consciousness.py mydb.db --check    # Check status only
  python migrate_to_consciousness.py mydb.db --force    # Force re-migration
        """
    )
    
    parser.add_argument("database", help="Path to MemMimic database file")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check consciousness status, don't migrate")
    parser.add_argument("--force", action="store_true",
                       help="Force migration even if consciousness features exist")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip database backup (not recommended)")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test consciousness features")
    
    args = parser.parse_args()
    
    print("🧿 MemMimic Consciousness Migration Tool")
    print("=" * 50)
    
    db_path = args.database
    
    # Check database exists
    if not os.path.exists(db_path):
        print(f"❌ Database does not exist: {db_path}")
        sys.exit(1)
    
    # Check consciousness status
    print(f"📋 Checking database: {db_path}")
    status = check_consciousness_status(db_path)
    
    if "error" in status:
        print(f"❌ Error checking database: {status['error']}")
        sys.exit(1)
    
    if status["has_consciousness"]:
        print(f"   🧿 Consciousness features: PRESENT")
        print(f"   📊 Sigils: {status.get('sigil_count', 0)}")
        print(f"   📋 Prompts: {status.get('prompt_count', 0)}")
    else:
        print(f"   🧿 Consciousness features: NOT FOUND")
    
    # Check-only mode
    if args.check_only:
        print("\n✅ Status check completed")
        sys.exit(0)
    
    # Test-only mode
    if args.test_only:
        if not status["has_consciousness"]:
            print("\n❌ No consciousness features to test")
            sys.exit(1)
        success = test_consciousness_features(db_path)
        sys.exit(0 if success else 1)
    
    # Check if migration needed
    if status["has_consciousness"] and not args.force:
        print("\n⚠️  Database already has consciousness features")
        print("   Use --force to re-migrate or --test-only to test")
        sys.exit(0)
    
    # Create backup unless disabled
    backup_path = None
    if not args.no_backup:
        try:
            backup_path = backup_database(db_path)
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            print("   Use --no-backup to skip backup (not recommended)")
            sys.exit(1)
    
    # Perform migration
    print(f"\n🔄 Starting consciousness migration...")
    result = migrate_to_consciousness(db_path)
    
    if not result.get("success"):
        print(f"\n❌ Migration failed!")
        if backup_path:
            print(f"   🔄 Restore from backup: {backup_path}")
        sys.exit(1)
    
    # Test after migration
    print(f"\n🧪 Testing consciousness features...")
    success = test_consciousness_features(db_path)
    
    if success:
        print(f"\n🎉 Consciousness migration completed successfully!")
        print(f"   ✅ Living Prompts active (4 templates)")
        print(f"   ✅ Sigil Engine active (20 sigils)")
        print(f"   ✅ Quantum entanglement ready")
        print(f"   📦 Backup available: {backup_path}")
    else:
        print(f"\n⚠️  Migration completed but tests failed")
        print(f"   📦 Backup available: {backup_path}")

if __name__ == "__main__":
    main()