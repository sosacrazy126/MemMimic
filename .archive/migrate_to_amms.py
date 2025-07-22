#!/usr/bin/env python3
"""
MemMimic AMMS Migration Script
Migrate from legacy MemoryStore to Active Memory Management System
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
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"📦 Database backed up to: {backup_path}")
    return backup_path

def check_database_compatibility(db_path: str) -> dict:
    """Check if database needs migration and return status"""
    try:
        from memmimic.memory import MemoryStore
        from memmimic.memory.active_schema import ActiveMemorySchema
        
        # Check if legacy database exists
        if not os.path.exists(db_path):
            return {"exists": False, "legacy": False, "enhanced": False}
        
        # Check if it has legacy schema
        legacy_store = MemoryStore(db_path)
        legacy_memories = legacy_store.get_all()
        
        # Check if it has enhanced schema
        schema = ActiveMemorySchema(db_path)
        has_enhanced = schema.check_enhanced_schema_exists()
        
        return {
            "exists": True,
            "legacy": len(legacy_memories) > 0,
            "enhanced": has_enhanced,
            "legacy_count": len(legacy_memories)
        }
        
    except Exception as e:
        return {"exists": True, "error": str(e)}

def migrate_database(db_path: str, config_path: str = None) -> dict:
    """Perform the migration from legacy to AMMS"""
    try:
        from memmimic.memory import UnifiedMemoryStore
        
        print(f"🔄 Starting migration of {db_path}...")
        
        # Initialize UnifiedMemoryStore (this will set up enhanced schema)
        unified_store = UnifiedMemoryStore(db_path, config_path)
        
        # Perform migration
        result = unified_store.migrate_from_legacy()
        
        print(f"✅ Migration completed:")
        print(f"   📊 Legacy memories: {result.get('total_legacy_memories', 0)}")
        print(f"   ✅ Successfully migrated: {result.get('successfully_migrated', 0)}")
        print(f"   ❌ Errors: {result.get('errors', 0)}")
        
        if result.get('errors', 0) > 0:
            print(f"   📋 Error details:")
            for error in result.get('error_details', []):
                print(f"      • {error}")
        
        return result
        
    except Exception as e:
        error_msg = f"Migration failed: {e}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}

def test_amms_functionality(db_path: str, config_path: str = None) -> bool:
    """Test that AMMS is working correctly after migration"""
    try:
        from memmimic import MemMimicAPI
        
        print(f"🧪 Testing AMMS functionality...")
        
        # Initialize API with AMMS
        api = MemMimicAPI(db_path, config_path)
        
        # Test basic operations
        status = api.status()
        print(f"   📊 Memories: {status.get('memories', 0)}")
        print(f"   🔄 AMMS Active: {status.get('amms_active', False)}")
        
        # Test memory addition
        memory_id = api.remember("AMMS migration test memory", "test")
        print(f"   ✅ Added test memory: ID {memory_id}")
        
        # Test search
        results = api.recall_cxd("migration test", limit=1)
        print(f"   🔍 Search test: {len(results)} results")
        
        # Test enhanced features
        if hasattr(api.memory, 'get_active_pool_status'):
            pool_status = api.memory.get_active_pool_status()
            print(f"   📊 Active pool status: {len(pool_status)} metrics")
        
        print(f"   ✅ AMMS functionality test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ AMMS functionality test FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Migrate MemMimic database to Active Memory Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python migrate_to_amms.py memmimic.db                    # Migrate with defaults
  python migrate_to_amms.py mydb.db --config myconfig.yaml # Migrate with custom config
  python migrate_to_amms.py memmimic.db --check-only       # Check status only
  python migrate_to_amms.py memmimic.db --force            # Force migration
        """
    )
    
    parser.add_argument("database", help="Path to MemMimic database file")
    parser.add_argument("--config", help="Path to AMMS configuration file")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check database status, don't migrate")
    parser.add_argument("--force", action="store_true",
                       help="Force migration even if enhanced schema exists")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip database backup (not recommended)")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test AMMS functionality, don't migrate")
    
    args = parser.parse_args()
    
    print("🚀 MemMimic AMMS Migration Tool")
    print("=" * 50)
    
    db_path = args.database
    
    # Check database status
    print(f"📋 Checking database: {db_path}")
    status = check_database_compatibility(db_path)
    
    if "error" in status:
        print(f"❌ Error checking database: {status['error']}")
        sys.exit(1)
    
    if not status["exists"]:
        print(f"❌ Database does not exist: {db_path}")
        sys.exit(1)
    
    print(f"   📊 Legacy memories: {status.get('legacy_count', 0)}")
    print(f"   🔄 Enhanced schema: {'Yes' if status.get('enhanced', False) else 'No'}")
    print(f"   📋 Migration needed: {'No' if status.get('enhanced', False) else 'Yes'}")
    
    # Check-only mode
    if args.check_only:
        print("\n✅ Database status check completed")
        sys.exit(0)
    
    # Test-only mode
    if args.test_only:
        print("\n🧪 Testing AMMS functionality...")
        success = test_amms_functionality(db_path, args.config)
        sys.exit(0 if success else 1)
    
    # Check if migration is needed
    if status.get("enhanced", False) and not args.force:
        print("\n⚠️  Database already has enhanced schema")
        print("   Use --force to migrate anyway or --test-only to test functionality")
        sys.exit(0)
    
    if not status.get("legacy", False):
        print("\n⚠️  No legacy memories found to migrate")
        print("   Database appears to be empty or already migrated")
        
        # Still test AMMS functionality
        success = test_amms_functionality(db_path, args.config)
        sys.exit(0 if success else 1)
    
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
    print(f"\n🔄 Starting migration...")
    result = migrate_database(db_path, args.config)
    
    if "error" in result:
        print(f"\n❌ Migration failed!")
        if backup_path:
            print(f"   🔄 Restore from backup: {backup_path}")
        sys.exit(1)
    
    # Test functionality after migration
    print(f"\n🧪 Testing AMMS after migration...")
    success = test_amms_functionality(db_path, args.config)
    
    if success:
        print(f"\n🎉 Migration completed successfully!")
        print(f"   ✅ AMMS is now active")
        print(f"   📦 Backup available: {backup_path}")
        print(f"   🚀 You can now use enhanced MemMimic features")
    else:
        print(f"\n⚠️  Migration completed but functionality test failed")
        print(f"   📦 Backup available: {backup_path}")
        print(f"   🔧 Check configuration and try again")

if __name__ == "__main__":
    main()