#!/usr/bin/env python3
"""
Comprehensive testing suite for SQLite to Markdown migration
Ensures data integrity, performance, and functionality
"""

import unittest
import tempfile
import shutil
import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import time
import random
import string

from storage_adapter import (
    Memory, SQLiteAdapter, MarkdownAdapter, HybridAdapter,
    create_storage_adapter
)
from updated_mcp_tools import MemMimicMCP


class TestMigrationIntegrity(unittest.TestCase):
    """Test data integrity during migration"""
    
    def setUp(self):
        """Create temporary directories and databases"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.md_dir = Path(self.temp_dir) / 'markdown'
        self.md_dir.mkdir()
        
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
    
    def test_single_memory_migration(self):
        """Test migrating a single memory"""
        # Create memory in SQLite
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        
        memory = Memory(
            content="Test memory content",
            metadata={'cxd': 'CONTEXT', 'tags': ['test']},
            importance=0.75
        )
        
        memory_id = sqlite_adapter.store(memory)
        
        # Migrate to markdown
        md_adapter = MarkdownAdapter(str(self.md_dir))
        retrieved = sqlite_adapter.retrieve(memory_id)
        md_adapter.store(retrieved)
        
        # Verify content matches
        md_retrieved = md_adapter.retrieve(memory_id)
        
        self.assertEqual(retrieved.content, md_retrieved.content)
        self.assertEqual(retrieved.metadata, md_retrieved.metadata)
        self.assertAlmostEqual(retrieved.importance, md_retrieved.importance, places=2)
    
    def test_bulk_migration(self):
        """Test migrating multiple memories"""
        # Create 100 memories in SQLite
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        memory_ids = []
        
        for i in range(100):
            memory = Memory(
                content=f"Memory {i}: " + ''.join(random.choices(string.ascii_letters, k=50)),
                metadata={
                    'cxd': random.choice(['CONTROL', 'CONTEXT', 'DATA']),
                    'index': i
                },
                importance=random.random()
            )
            memory_ids.append(sqlite_adapter.store(memory))
        
        # Migrate all to markdown
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        for memory_id in memory_ids:
            memory = sqlite_adapter.retrieve(memory_id)
            md_adapter.store(memory)
        
        # Verify counts match
        self.assertEqual(sqlite_adapter.count(), md_adapter.count())
        
        # Verify each memory
        for memory_id in memory_ids:
            sql_mem = sqlite_adapter.retrieve(memory_id)
            md_mem = md_adapter.retrieve(memory_id)
            
            self.assertEqual(sql_mem.content, md_mem.content)
            self.assertEqual(sql_mem.metadata['index'], md_mem.metadata['index'])
    
    def test_special_characters(self):
        """Test memories with special characters"""
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        # Test various special characters
        test_contents = [
            "Memory with 'quotes' and \"double quotes\"",
            "Memory with\nnewlines\nand\ttabs",
            "Memory with emojis 😀 🎉 🚀",
            "Memory with markdown: **bold** *italic* `code`",
            "Memory with ---\nfrontmatter\n---\ndelimiters",
            "Memory with unicode: café, naïve, 日本語"
        ]
        
        for content in test_contents:
            memory = Memory(content=content)
            memory_id = sqlite_adapter.store(memory)
            
            # Migrate
            retrieved = sqlite_adapter.retrieve(memory_id)
            md_adapter.store(retrieved)
            
            # Verify
            md_retrieved = md_adapter.retrieve(memory_id)
            self.assertEqual(content, md_retrieved.content)
    
    def test_metadata_preservation(self):
        """Test that all metadata is preserved"""
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        complex_metadata = {
            'cxd': 'CONTEXT',
            'tags': ['test', 'migration', 'complex'],
            'nested': {
                'level1': {
                    'level2': 'value'
                }
            },
            'numbers': [1, 2, 3, 4.5],
            'boolean': True,
            'null_value': None
        }
        
        memory = Memory(
            content="Complex metadata test",
            metadata=complex_metadata,
            importance=0.99
        )
        
        memory_id = sqlite_adapter.store(memory)
        
        # Migrate
        retrieved = sqlite_adapter.retrieve(memory_id)
        md_adapter.store(retrieved)
        
        # Verify
        md_retrieved = md_adapter.retrieve(memory_id)
        self.assertEqual(complex_metadata, md_retrieved.metadata)
    
    def test_date_preservation(self):
        """Test that dates are preserved correctly"""
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        # Create memory with specific dates
        memory = Memory(content="Date test")
        memory.created_at = datetime(2025, 1, 1, 12, 0, 0)
        memory.updated_at = datetime(2025, 6, 15, 18, 30, 45)
        
        memory_id = sqlite_adapter.store(memory)
        
        # Migrate
        retrieved = sqlite_adapter.retrieve(memory_id)
        md_adapter.store(retrieved)
        
        # Verify dates
        md_retrieved = md_adapter.retrieve(memory_id)
        self.assertEqual(retrieved.created_at, md_retrieved.created_at)
        self.assertEqual(retrieved.updated_at, md_retrieved.updated_at)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.md_dir = Path(self.temp_dir) / 'markdown'
        self.md_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_search_performance(self):
        """Compare search performance between SQLite and Markdown"""
        # Create 1000 memories
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        for i in range(1000):
            memory = Memory(
                content=f"Memory {i}: " + ' '.join(random.choices(['apple', 'banana', 'cherry', 'date', 'elderberry'], k=10)),
                metadata={'index': i}
            )
            sqlite_adapter.store(memory)
            md_adapter.store(memory)
        
        # Test SQLite search performance
        start = time.perf_counter()
        sql_results = sqlite_adapter.search("apple", limit=10)
        sql_time = time.perf_counter() - start
        
        # Test Markdown search performance
        start = time.perf_counter()
        md_results = md_adapter.search("apple", limit=10)
        md_time = time.perf_counter() - start
        
        print(f"\nSearch Performance (1000 memories):")
        print(f"SQLite: {sql_time*1000:.2f}ms")
        print(f"Markdown: {md_time*1000:.2f}ms")
        print(f"Ratio: {md_time/sql_time:.2f}x")
        
        # Markdown should be within 10x of SQLite performance
        self.assertLess(md_time, sql_time * 10)
    
    def test_write_performance(self):
        """Compare write performance"""
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        # Test SQLite write performance
        start = time.perf_counter()
        for i in range(100):
            memory = Memory(content=f"Write test {i}")
            sqlite_adapter.store(memory)
        sql_time = time.perf_counter() - start
        
        # Test Markdown write performance
        start = time.perf_counter()
        for i in range(100):
            memory = Memory(content=f"Write test {i}")
            md_adapter.store(memory)
        md_time = time.perf_counter() - start
        
        print(f"\nWrite Performance (100 memories):")
        print(f"SQLite: {sql_time*1000:.2f}ms")
        print(f"Markdown: {md_time*1000:.2f}ms")
        print(f"Ratio: {md_time/sql_time:.2f}x")


class TestHybridAdapter(unittest.TestCase):
    """Test hybrid adapter functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.md_dir = Path(self.temp_dir) / 'markdown'
        self.md_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_hybrid_write_both(self):
        """Test writing to both backends"""
        hybrid = HybridAdapter(str(self.db_path), str(self.md_dir), write_to='both')
        
        memory = Memory(content="Hybrid test")
        memory_id = hybrid.store(memory)
        
        # Verify in both backends
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        self.assertIsNotNone(sqlite_adapter.retrieve(memory_id))
        self.assertIsNotNone(md_adapter.retrieve(memory_id))
    
    def test_hybrid_read_fallback(self):
        """Test read fallback from SQLite when not in Markdown"""
        hybrid = HybridAdapter(str(self.db_path), str(self.md_dir))
        
        # Store only in SQLite
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        memory = Memory(content="SQLite only")
        memory_id = sqlite_adapter.store(memory)
        
        # Hybrid should find it
        retrieved = hybrid.retrieve(memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "SQLite only")
    
    def test_hybrid_search_deduplication(self):
        """Test that hybrid search deduplicates results"""
        hybrid = HybridAdapter(str(self.db_path), str(self.md_dir), write_to='both')
        
        # Store same memory in both
        memory = Memory(content="Duplicate test memory")
        memory_id = hybrid.store(memory)
        
        # Search should return only one result
        results = hybrid.search("Duplicate", limit=10)
        
        # Count how many times this memory appears
        count = sum(1 for r in results if r.id == memory_id)
        self.assertEqual(count, 1, "Memory should appear only once in search results")


class TestMCPCompatibility(unittest.TestCase):
    """Test MCP tools compatibility"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.md_dir = Path(self.temp_dir) / 'markdown'
        self.md_dir.mkdir()
        
        # Set environment for MCP tools
        import os
        os.environ['MEMMIMIC_STORAGE'] = 'hybrid'
        os.environ['MEMMIMIC_DB_PATH'] = str(self.db_path)
        os.environ['MEMMIMIC_MD_DIR'] = str(self.md_dir)
        os.environ['MEMMIMIC_WRITE_TO'] = 'both'
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_mcp_remember(self):
        """Test MCP remember tool"""
        mcp = MemMimicMCP()
        
        result = mcp.remember(
            content="Test memory from MCP",
            memory_type="interaction",
            metadata={'test': True}
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('memory_id', result)
        
        # Verify in both backends
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        memory_id = result['memory_id']
        self.assertIsNotNone(sqlite_adapter.retrieve(memory_id))
        self.assertIsNotNone(md_adapter.retrieve(memory_id))
    
    def test_mcp_recall(self):
        """Test MCP recall tool"""
        mcp = MemMimicMCP()
        
        # Store some memories
        mcp.remember("Control action test", metadata={'cxd': 'CONTROL'})
        mcp.remember("Context explanation test", metadata={'cxd': 'CONTEXT'})
        mcp.remember("Data information test", metadata={'cxd': 'DATA'})
        
        # Test recall with CXD filter
        results = mcp.recall_cxd("test", function_filter="CONTEXT")
        
        self.assertTrue(len(results) > 0)
        for result in results:
            if 'cxd' in result:
                self.assertEqual(result['cxd'], 'CONTEXT')
    
    def test_mcp_think(self):
        """Test MCP think tool"""
        mcp = MemMimicMCP()
        
        # Store context memories
        mcp.remember("Architecture involves designing systems")
        mcp.remember("Refactoring improves code quality")
        mcp.remember("Testing ensures reliability")
        
        # Think with context
        result = mcp.think_with_memory("How should I approach architecture?")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('relevant_memories', result)
        self.assertIn('analysis', result)
    
    def test_mcp_status(self):
        """Test MCP status tool"""
        mcp = MemMimicMCP()
        
        # Store some memories
        for i in range(5):
            mcp.remember(f"Memory {i}")
        
        # Get status
        status = mcp.status()
        
        self.assertEqual(status['status'], 'success')
        self.assertIn('stats', status)
        self.assertEqual(status['stats']['total_memories'], 5)


class TestDataValidation(unittest.TestCase):
    """Test data validation and integrity checks"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.md_dir = Path(self.temp_dir) / 'markdown'
        self.md_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_content_hash_validation(self):
        """Validate content integrity using hashes"""
        sqlite_adapter = SQLiteAdapter(str(self.db_path))
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        # Create and migrate memories
        memory_hashes = {}
        
        for i in range(50):
            memory = Memory(
                content=f"Memory {i}: " + ''.join(random.choices(string.ascii_letters, k=100))
            )
            memory_id = sqlite_adapter.store(memory)
            
            # Calculate hash
            content_hash = hashlib.sha256(memory.content.encode()).hexdigest()
            memory_hashes[memory_id] = content_hash
            
            # Migrate
            retrieved = sqlite_adapter.retrieve(memory_id)
            md_adapter.store(retrieved)
        
        # Validate hashes match
        for memory_id, expected_hash in memory_hashes.items():
            md_memory = md_adapter.retrieve(memory_id)
            actual_hash = hashlib.sha256(md_memory.content.encode()).hexdigest()
            self.assertEqual(expected_hash, actual_hash, f"Hash mismatch for {memory_id}")
    
    def test_index_integrity(self):
        """Test that the index stays consistent"""
        md_adapter = MarkdownAdapter(str(self.md_dir))
        
        # Store memories
        memory_ids = []
        for i in range(20):
            memory = Memory(content=f"Index test {i}")
            memory_ids.append(md_adapter.store(memory))
        
        # Verify index has all memories
        self.assertEqual(len(md_adapter.index), 20)
        
        # Delete some memories
        for memory_id in memory_ids[:5]:
            md_adapter.delete(memory_id)
        
        # Verify index updated
        self.assertEqual(len(md_adapter.index), 15)
        
        # Verify remaining memories still accessible
        for memory_id in memory_ids[5:]:
            self.assertIsNotNone(md_adapter.retrieve(memory_id))


def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMigrationIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)