#!/usr/bin/env python3
"""
Execute SQLite to Markdown Migration
User-friendly migration tool with validation and rollback
"""

import os
import sys
import sqlite3
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import Dict, List, Tuple

# Import our migration components
from storage_adapter import SQLiteAdapter, MarkdownAdapter, HybridAdapter, Memory
from migrate_to_markdown import migrate_database_to_markdown
from test_migration import TestMigrationIntegrity


class MigrationExecutor:
    """Main migration executor with validation and rollback"""
    
    def __init__(self, db_path: str, output_dir: str, verbose: bool = False):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.backup_dir = self.output_dir / '.migration_backup'
        self.log_file = self.output_dir / 'migration.log'
        self.verbose = verbose
        
        # Setup logging
        self._setup_logging()
        
        # Migration state
        self.state = {
            'status': 'not_started',
            'total_memories': 0,
            'migrated': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
            'validation_passed': False
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup logger
        self.logger = logging.getLogger('migration')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def pre_migration_checks(self) -> bool:
        """Run pre-migration validation checks"""
        print("\n🔍 Running pre-migration checks...")
        
        # Check database exists
        if not self.db_path.exists():
            self.logger.error(f"Database not found: {self.db_path}")
            return False
        
        # Check database is valid SQLite
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            conn.close()
            
            self.state['total_memories'] = count
            print(f"✅ Found {count} memories in database")
            
        except Exception as e:
            self.logger.error(f"Invalid database: {e}")
            return False
        
        # Check output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory ready: {self.output_dir}")
        
        # Check disk space
        db_size = self.db_path.stat().st_size
        required_space = db_size * 2  # Conservative estimate
        
        import shutil
        free_space = shutil.disk_usage(str(self.output_dir)).free
        
        if free_space < required_space:
            self.logger.error(f"Insufficient disk space. Need {required_space/1024/1024:.1f}MB, have {free_space/1024/1024:.1f}MB")
            return False
        
        print(f"✅ Sufficient disk space available")
        
        return True
    
    def create_backup(self) -> bool:
        """Create backup of database before migration"""
        print("\n💾 Creating backup...")
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup database
            backup_db = self.backup_dir / f"memmimic_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(self.db_path, backup_db)
            
            # Verify backup
            conn = sqlite3.connect(str(backup_db))
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            backup_count = cursor.fetchone()[0]
            conn.close()
            
            if backup_count != self.state['total_memories']:
                raise ValueError("Backup verification failed")
            
            print(f"✅ Backup created: {backup_db}")
            self.logger.info(f"Backup created at {backup_db}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def execute_migration(self) -> bool:
        """Execute the actual migration"""
        print("\n🔄 Starting migration...")
        
        self.state['status'] = 'in_progress'
        self.state['start_time'] = datetime.now()
        
        try:
            # Use our migration function
            count = migrate_database_to_markdown(str(self.db_path), str(self.output_dir))
            
            self.state['migrated'] = count
            self.state['status'] = 'completed'
            
            print(f"✅ Migrated {count} memories successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            self.state['status'] = 'failed'
            return False
        
        finally:
            self.state['end_time'] = datetime.now()
    
    def validate_migration(self) -> bool:
        """Validate the migration was successful"""
        print("\n🔍 Validating migration...")
        
        validation_results = {
            'count_match': False,
            'content_match': False,
            'metadata_match': False,
            'index_valid': False
        }
        
        try:
            # Initialize adapters
            sqlite_adapter = SQLiteAdapter(str(self.db_path))
            md_adapter = MarkdownAdapter(str(self.output_dir))
            
            # 1. Count validation
            sql_count = sqlite_adapter.count()
            md_count = md_adapter.count()
            
            if sql_count == md_count:
                validation_results['count_match'] = True
                print(f"✅ Count validation passed: {sql_count} memories")
            else:
                print(f"❌ Count mismatch: SQLite={sql_count}, Markdown={md_count}")
            
            # 2. Content validation (sample)
            sample_size = min(100, sql_count)
            content_matches = 0
            
            all_memories = sqlite_adapter.get_all(limit=sample_size)
            for memory in all_memories:
                md_memory = md_adapter.retrieve(memory.id)
                if md_memory and memory.content == md_memory.content:
                    content_matches += 1
            
            if content_matches == sample_size:
                validation_results['content_match'] = True
                print(f"✅ Content validation passed ({sample_size} samples)")
            else:
                print(f"❌ Content mismatch: {sample_size - content_matches} differences")
            
            # 3. Metadata validation
            metadata_matches = 0
            for memory in all_memories[:20]:  # Check first 20
                md_memory = md_adapter.retrieve(memory.id)
                if md_memory and memory.metadata == md_memory.metadata:
                    metadata_matches += 1
            
            if metadata_matches == min(20, len(all_memories)):
                validation_results['metadata_match'] = True
                print(f"✅ Metadata validation passed")
            else:
                print(f"❌ Metadata validation failed")
            
            # 4. Index validation
            index_path = self.output_dir / 'memories' / 'index.json'
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index = json.load(f)
                    if len(index) == md_count:
                        validation_results['index_valid'] = True
                        print(f"✅ Index validation passed")
                    else:
                        print(f"❌ Index count mismatch")
            else:
                print(f"❌ Index file not found")
            
            # Overall validation
            all_passed = all(validation_results.values())
            self.state['validation_passed'] = all_passed
            
            if all_passed:
                print("\n✅ All validation checks passed!")
            else:
                print("\n⚠️ Some validation checks failed")
                
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False
    
    def generate_report(self):
        """Generate migration report"""
        report_path = self.output_dir / 'migration_report.md'
        
        duration = None
        if self.state['start_time'] and self.state['end_time']:
            duration = self.state['end_time'] - self.state['start_time']
        
        report = f"""# Migration Report

## Summary
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Source Database**: {self.db_path}
- **Output Directory**: {self.output_dir}
- **Status**: {self.state['status']}

## Statistics
- **Total Memories**: {self.state['total_memories']}
- **Successfully Migrated**: {self.state['migrated']}
- **Failed**: {self.state['failed']}
- **Success Rate**: {(self.state['migrated'] / max(1, self.state['total_memories']) * 100):.1f}%

## Performance
- **Duration**: {duration if duration else 'N/A'}
- **Memories per second**: {self.state['migrated'] / max(1, duration.total_seconds() if duration else 1):.1f}

## Validation
- **Validation Passed**: {'Yes' if self.state['validation_passed'] else 'No'}

## Files Created
- Markdown files: `{self.output_dir}/memories/YYYY/MM/DD/*.md`
- Search index: `{self.output_dir}/memories/index.json`
- Relationships: `{self.output_dir}/memories/relationships.json`

## Next Steps
1. Test the migrated memories with MCP tools
2. Set environment variable: `export MEMMIMIC_STORAGE=markdown`
3. Update MCP configuration to use markdown directory
4. Consider removing the SQLite database after verification
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: {report_path}")
    
    def rollback(self):
        """Rollback migration if needed"""
        print("\n⏪ Rolling back migration...")
        
        try:
            # Remove migrated files
            memories_dir = self.output_dir / 'memories'
            if memories_dir.exists():
                shutil.rmtree(memories_dir)
                print(f"✅ Removed migrated files")
            
            # Restore from backup if exists
            backup_files = list(self.backup_dir.glob('*.db'))
            if backup_files:
                latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
                print(f"ℹ️ Backup available at: {latest_backup}")
            
            self.logger.info("Rollback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def run(self, skip_backup: bool = False, skip_validation: bool = False) -> bool:
        """Run the complete migration process"""
        print("="*50)
        print("MemMimic SQLite to Markdown Migration")
        print("="*50)
        
        # Pre-checks
        if not self.pre_migration_checks():
            print("\n❌ Pre-migration checks failed")
            return False
        
        # Backup
        if not skip_backup:
            if not self.create_backup():
                print("\n❌ Backup failed")
                return False
        
        # Migration
        if not self.execute_migration():
            print("\n❌ Migration failed")
            if input("\nRollback migration? (y/n): ").lower() == 'y':
                self.rollback()
            return False
        
        # Validation
        if not skip_validation:
            if not self.validate_migration():
                print("\n⚠️ Validation failed")
                if input("\nRollback migration? (y/n): ").lower() == 'y':
                    self.rollback()
                    return False
        
        # Report
        self.generate_report()
        
        print("\n🎉 Migration completed successfully!")
        return True


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Migrate MemMimic from SQLite to Markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --auto                            # Automatic migration
  %(prog)s --db custom.db --output ./md      # Custom paths
  %(prog)s --skip-backup --skip-validation   # Fast mode (dangerous!)
  %(prog)s --test                            # Run tests only
        """
    )
    
    parser.add_argument('--db', default='data/databases/memmimic.db',
                       help='Path to SQLite database (default: data/databases/memmimic.db)')
    parser.add_argument('--output', default='.',
                       help='Output directory for markdown files (default: current directory)')
    parser.add_argument('--auto', action='store_true',
                       help='Run migration automatically without prompts')
    parser.add_argument('--skip-backup', action='store_true',
                       help='Skip backup creation (not recommended)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip post-migration validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--test', action='store_true',
                       help='Run migration tests only')
    
    args = parser.parse_args()
    
    # Run tests if requested
    if args.test:
        print("Running migration tests...")
        from test_migration import run_all_tests
        success = run_all_tests()
        return 0 if success else 1
    
    # Interactive mode
    if not args.auto:
        print("="*50)
        print("MemMimic Migration Tool")
        print("="*50)
        print(f"\nDatabase: {args.db}")
        print(f"Output: {args.output}")
        print(f"Backup: {'Disabled' if args.skip_backup else 'Enabled'}")
        print(f"Validation: {'Disabled' if args.skip_validation else 'Enabled'}")
        
        if input("\nProceed with migration? (y/n): ").lower() != 'y':
            print("Migration cancelled")
            return 0
    
    # Execute migration
    executor = MigrationExecutor(args.db, args.output, args.verbose)
    success = executor.run(args.skip_backup, args.skip_validation)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())