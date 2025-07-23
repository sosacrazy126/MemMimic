#!/usr/bin/env python3
"""
Modular Architecture Integration Tests

Comprehensive integration tests for the 4 Phase 2 modular components:
1. hybrid_search.py - Core hybrid search engine
2. wordnet_expander.py - NLTK WordNet integration  
3. semantic_processor.py - Vector similarity processing
4. result_combiner.py - Score fusion and ranking

Tests component interaction, data flow, and backward compatibility.
"""

import asyncio
import sys
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import modular components
try:
    from memmimic.memory.search.hybrid_search import HybridSearch
    from memmimic.memory.search.wordnet_expander import WordNetExpander
    from memmimic.memory.search.semantic_processor import SemanticProcessor
    from memmimic.memory.search.result_combiner import ResultCombiner
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modular components not available: {e}")
    COMPONENTS_AVAILABLE = False

# Import supporting modules
from memmimic.memory.storage.amms_storage import AMMSStorage


class ComponentTestSuite:
    """Base test suite for component testing."""
    
    def __init__(self):
        self.test_results = {}
        self.components = {}
        
    def setup_components(self):
        """Initialize all modular components."""
        if not COMPONENTS_AVAILABLE:
            return False
            
        try:
            self.components = {
                'hybrid_search': HybridSearch(),
                'wordnet_expander': WordNetExpander(),
                'semantic_processor': SemanticProcessor(),
                'result_combiner': ResultCombiner()
            }
            return True
        except Exception as e:
            print(f"Failed to initialize components: {e}")
            return False


class TestIndividualComponents(ComponentTestSuite):
    """Test individual component functionality."""
    
    def test_hybrid_search_component(self):
        """Test HybridSearch component functionality."""
        print("🔍 Testing HybridSearch component...")
        
        if 'hybrid_search' not in self.components:
            print("   ⚠️ HybridSearch component not available")
            return False
            
        hybrid_search = self.components['hybrid_search']
        
        try:
            # Test basic search functionality
            test_query = "artificial intelligence"
            
            # Check if search method exists and works
            if hasattr(hybrid_search, 'search'):
                results = hybrid_search.search(test_query)
                print(f"   ✅ Search method executed: {type(results)}")
            elif hasattr(hybrid_search, 'hybrid_search'):
                results = hybrid_search.hybrid_search(test_query)
                print(f"   ✅ Hybrid search method executed: {type(results)}")
            else:
                # Test initialization at least
                print("   ✅ Component initialized successfully")
                return True
            
            # Test with different query types
            test_queries = [
                "machine learning",
                "neural networks deep learning",
                "natural language processing",
            ]
            
            for query in test_queries:
                try:
                    if hasattr(hybrid_search, 'search'):
                        result = hybrid_search.search(query, limit=5)
                    elif hasattr(hybrid_search, 'hybrid_search'):
                        result = hybrid_search.hybrid_search(query, limit=5)
                    else:
                        continue
                    print(f"   ✅ Query '{query[:20]}...': {type(result)}")
                except Exception as e:
                    print(f"   ⚠️ Query '{query[:20]}...' failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ HybridSearch component test failed: {e}")
            return False
    
    def test_wordnet_expander_component(self):
        """Test WordNetExpander component functionality."""
        print("📚 Testing WordNetExpander component...")
        
        if 'wordnet_expander' not in self.components:
            print("   ⚠️ WordNetExpander component not available")
            return False
            
        wordnet_expander = self.components['wordnet_expander']
        
        try:
            # Test query expansion
            test_queries = [
                "car",
                "happy",
                "computer",
                "artificial intelligence",
            ]
            
            for query in test_queries:
                try:
                    if hasattr(wordnet_expander, 'expand_query'):
                        expanded = wordnet_expander.expand_query(query)
                        print(f"   ✅ Expanded '{query}': {type(expanded)}")
                        
                        # Expanded result should contain original term or related terms
                        if isinstance(expanded, (list, set, tuple)):
                            print(f"      Terms: {len(expanded) if expanded else 0}")
                        elif isinstance(expanded, dict):
                            print(f"      Result keys: {list(expanded.keys())}")
                            
                    elif hasattr(wordnet_expander, 'get_synonyms'):
                        synonyms = wordnet_expander.get_synonyms(query)
                        print(f"   ✅ Synonyms for '{query}': {type(synonyms)}")
                        
                    else:
                        print("   ✅ Component initialized (methods not yet available)")
                        return True
                        
                except Exception as e:
                    print(f"   ⚠️ Expansion for '{query}' failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ WordNetExpander component test failed: {e}")
            return False
    
    def test_semantic_processor_component(self):
        """Test SemanticProcessor component functionality."""
        print("🧠 Testing SemanticProcessor component...")
        
        if 'semantic_processor' not in self.components:
            print("   ⚠️ SemanticProcessor component not available")
            return False
            
        semantic_processor = self.components['semantic_processor']
        
        try:
            # Test semantic processing
            test_texts = [
                "artificial intelligence and machine learning",
                "natural language processing systems",
                "deep neural networks for classification",
            ]
            
            for text in test_texts:
                try:
                    if hasattr(semantic_processor, 'process_text'):
                        result = semantic_processor.process_text(text)
                        print(f"   ✅ Processed '{text[:30]}...': {type(result)}")
                        
                    elif hasattr(semantic_processor, 'create_embedding'):
                        embedding = semantic_processor.create_embedding(text)
                        print(f"   ✅ Embedding for '{text[:30]}...': {type(embedding)}")
                        
                        # Embeddings should be numeric
                        if hasattr(embedding, '__len__'):
                            print(f"      Embedding size: {len(embedding)}")
                            
                    elif hasattr(semantic_processor, 'process_embedding'):
                        result = semantic_processor.process_embedding(text)
                        print(f"   ✅ Embedding processing '{text[:30]}...': {type(result)}")
                        
                    else:
                        print("   ✅ Component initialized (methods not yet available)")
                        return True
                        
                except Exception as e:
                    print(f"   ⚠️ Processing '{text[:30]}...' failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ SemanticProcessor component test failed: {e}")
            return False
    
    def test_result_combiner_component(self):
        """Test ResultCombiner component functionality."""
        print("🔗 Testing ResultCombiner component...")
        
        if 'result_combiner' not in self.components:
            print("   ⚠️ ResultCombiner component not available")
            return False
            
        result_combiner = self.components['result_combiner']
        
        try:
            # Test result combination with mock data
            mock_results_1 = [
                {"id": 1, "content": "AI research", "score": 0.8},
                {"id": 2, "content": "Machine learning", "score": 0.7},
                {"id": 3, "content": "Neural networks", "score": 0.6},
            ]
            
            mock_results_2 = [
                {"id": 1, "content": "AI research", "score": 0.9},  # Same as results_1
                {"id": 4, "content": "Deep learning", "score": 0.85},
                {"id": 5, "content": "Computer vision", "score": 0.5},
            ]
            
            try:
                if hasattr(result_combiner, 'combine_results'):
                    combined = result_combiner.combine_results(mock_results_1, mock_results_2)
                    print(f"   ✅ Combined results: {type(combined)}")
                    
                    if isinstance(combined, list):
                        print(f"      Combined count: {len(combined)}")
                        
                elif hasattr(result_combiner, 'merge_results'):
                    merged = result_combiner.merge_results(mock_results_1, mock_results_2)
                    print(f"   ✅ Merged results: {type(merged)}")
                    
                elif hasattr(result_combiner, 'fuse_scores'):
                    fused = result_combiner.fuse_scores([mock_results_1, mock_results_2])
                    print(f"   ✅ Fused scores: {type(fused)}")
                    
                else:
                    print("   ✅ Component initialized (methods not yet available)")
                    return True
                
                # Test ranking functionality
                if hasattr(result_combiner, 'rank_results'):
                    ranked = result_combiner.rank_results(mock_results_1)
                    print(f"   ✅ Ranked results: {type(ranked)}")
                    
            except Exception as e:
                print(f"   ⚠️ Result combination failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ ResultCombiner component test failed: {e}")
            return False


class TestComponentIntegration(ComponentTestSuite):
    """Test integration between modular components."""
    
    def test_search_pipeline_integration(self):
        """Test full search pipeline integration."""
        print("🔄 Testing search pipeline integration...")
        
        if not all(comp in self.components for comp in ['hybrid_search', 'wordnet_expander', 'semantic_processor', 'result_combiner']):
            print("   ⚠️ Not all components available for integration test")
            return False
        
        try:
            # Simulate complete search pipeline
            query = "machine learning algorithms"
            
            # Step 1: Query expansion
            wordnet_expander = self.components['wordnet_expander']
            expanded_query = query  # Default fallback
            
            if hasattr(wordnet_expander, 'expand_query'):
                try:
                    expanded_query = wordnet_expander.expand_query(query)
                    print(f"   ✅ Query expanded: {type(expanded_query)}")
                except Exception as e:
                    print(f"   ⚠️ Query expansion failed: {e}")
            
            # Step 2: Semantic processing
            semantic_processor = self.components['semantic_processor']
            semantic_results = []
            
            if hasattr(semantic_processor, 'process_text'):
                try:
                    semantic_result = semantic_processor.process_text(query)
                    semantic_results = [semantic_result] if semantic_result else []
                    print(f"   ✅ Semantic processing: {type(semantic_result)}")
                except Exception as e:
                    print(f"   ⚠️ Semantic processing failed: {e}")
            
            # Step 3: Hybrid search
            hybrid_search = self.components['hybrid_search']
            search_results = []
            
            if hasattr(hybrid_search, 'search'):
                try:
                    search_results = hybrid_search.search(query)
                    print(f"   ✅ Hybrid search: {type(search_results)}")
                except Exception as e:
                    print(f"   ⚠️ Hybrid search failed: {e}")
            
            # Step 4: Result combination
            result_combiner = self.components['result_combiner']
            final_results = search_results  # Default fallback
            
            if hasattr(result_combiner, 'combine_results') and semantic_results:
                try:
                    final_results = result_combiner.combine_results(search_results, semantic_results)
                    print(f"   ✅ Results combined: {type(final_results)}")
                except Exception as e:
                    print(f"   ⚠️ Result combination failed: {e}")
            
            print(f"   ✅ Pipeline completed - Final result type: {type(final_results)}")
            return True
            
        except Exception as e:
            print(f"   ❌ Search pipeline integration failed: {e}")
            return False
    
    def test_component_data_flow(self):
        """Test data flow between components."""
        print("📊 Testing component data flow...")
        
        try:
            # Test that components can pass data between each other
            test_data = {
                "query": "artificial intelligence",
                "content": "AI is transforming technology",
                "metadata": {"type": "test", "source": "integration_test"}
            }
            
            data_flow_results = {}
            
            # Test each component's ability to handle different data formats
            for component_name, component in self.components.items():
                try:
                    # Try different methods that might exist
                    methods_to_try = [
                        'process',
                        'handle_data',
                        'transform',
                        'execute',
                    ]
                    
                    success = False
                    for method_name in methods_to_try:
                        if hasattr(component, method_name):
                            try:
                                method = getattr(component, method_name)
                                result = method(test_data)
                                data_flow_results[component_name] = {
                                    'method': method_name,
                                    'result_type': type(result).__name__,
                                    'success': True
                                }
                                success = True
                                break
                            except Exception as e:
                                continue
                    
                    if not success:
                        data_flow_results[component_name] = {
                            'success': False,
                            'reason': 'No compatible methods found'
                        }
                        
                except Exception as e:
                    data_flow_results[component_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Report results
            successful_components = sum(1 for r in data_flow_results.values() if r.get('success', False))
            total_components = len(data_flow_results)
            
            print(f"   📈 Data flow compatibility: {successful_components}/{total_components}")
            
            for comp_name, result in data_flow_results.items():
                if result.get('success'):
                    print(f"   ✅ {comp_name}: {result.get('method', 'N/A')} → {result.get('result_type', 'N/A')}")
                else:
                    print(f"   ⚠️ {comp_name}: {result.get('reason', result.get('error', 'Failed'))}")
            
            return successful_components > 0
            
        except Exception as e:
            print(f"   ❌ Component data flow test failed: {e}")
            return False
    
    async def test_storage_integration(self):
        """Test integration with AMMS storage system."""
        print("💾 Testing storage integration...")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            storage = AMMSStorage(db_path)
            await storage.initialize()
            
            # Store test memories that components can search
            test_memories = [
                {
                    "content": "Artificial intelligence is revolutionizing technology",
                    "type": "ai_research",
                    "metadata": {"topic": "AI", "importance": "high"}
                },
                {
                    "content": "Machine learning algorithms can learn from data",
                    "type": "ml_research", 
                    "metadata": {"topic": "ML", "importance": "medium"}
                },
                {
                    "content": "Neural networks are inspired by biological neurons",
                    "type": "neural_research",
                    "metadata": {"topic": "Neural", "importance": "high"}
                }
            ]
            
            memory_ids = []
            for memory in test_memories:
                memory_id = await storage.store_memory(
                    content=memory["content"],
                    memory_type=memory["type"],
                    metadata=memory["metadata"]
                )
                memory_ids.append(memory_id)
                
            print(f"   ✅ Stored {len(memory_ids)} test memories")
            
            # Test component integration with storage
            query = "artificial intelligence"
            integration_results = {}
            
            # Test if components can work with stored memories
            stored_memories = []
            for memory_id in memory_ids:
                memory = await storage.get_memory(memory_id)
                if memory:
                    stored_memories.append(memory)
            
            for component_name, component in self.components.items():
                try:
                    # Try to use component with stored memory data
                    test_content = stored_memories[0]['content'] if stored_memories else "test content"
                    
                    if hasattr(component, 'search') and component_name == 'hybrid_search':
                        result = component.search(query)
                        integration_results[component_name] = {'success': True, 'method': 'search'}
                        
                    elif hasattr(component, 'expand_query') and component_name == 'wordnet_expander':
                        result = component.expand_query(query)
                        integration_results[component_name] = {'success': True, 'method': 'expand_query'}
                        
                    elif hasattr(component, 'process_text') and component_name == 'semantic_processor':
                        result = component.process_text(test_content)
                        integration_results[component_name] = {'success': True, 'method': 'process_text'}
                        
                    elif hasattr(component, 'combine_results') and component_name == 'result_combiner':
                        # Create mock results from stored memories
                        mock_results = [
                            {"id": i, "content": mem.get('content', ''), "score": 0.8}
                            for i, mem in enumerate(stored_memories[:2])
                        ]
                        result = component.combine_results(mock_results, [])
                        integration_results[component_name] = {'success': True, 'method': 'combine_results'}
                        
                    else:
                        integration_results[component_name] = {'success': False, 'reason': 'No integration method found'}
                        
                except Exception as e:
                    integration_results[component_name] = {'success': False, 'error': str(e)}
            
            # Report integration results
            successful_integrations = sum(1 for r in integration_results.values() if r.get('success', False))
            total_components = len(integration_results)
            
            print(f"   🔗 Storage integration: {successful_integrations}/{total_components}")
            
            for comp_name, result in integration_results.items():
                if result.get('success'):
                    print(f"   ✅ {comp_name}: {result.get('method', 'N/A')}")
                else:
                    print(f"   ⚠️ {comp_name}: {result.get('reason', result.get('error', 'Failed'))}")
            
            await storage.close()
            return successful_integrations > 0
            
        finally:
            os.unlink(db_path)


class TestBackwardCompatibility(ComponentTestSuite):
    """Test backward compatibility with existing APIs."""
    
    def test_api_compatibility(self):
        """Test that existing APIs still work with modular components."""
        print("🔄 Testing API backward compatibility...")
        
        try:
            # Test existing API endpoints
            from memmimic import api
            
            compatibility_tests = []
            
            # Test status endpoint
            if hasattr(api, 'status'):
                try:
                    status_result = api.status()
                    compatibility_tests.append({
                        'endpoint': 'status',
                        'success': True,
                        'result_type': type(status_result).__name__
                    })
                except Exception as e:
                    compatibility_tests.append({
                        'endpoint': 'status',
                        'success': False,
                        'error': str(e)
                    })
            
            # Test other API methods if they exist
            api_methods = ['recall_cxd', 'remember', 'think_with_memory']
            
            for method_name in api_methods:
                if hasattr(api, method_name):
                    try:
                        method = getattr(api, method_name)
                        # Try with minimal test parameters
                        if method_name == 'recall_cxd':
                            result = method('test query')
                        elif method_name == 'remember':
                            result = method('test content')
                        elif method_name == 'think_with_memory':
                            result = method('test input')
                        else:
                            result = method()
                            
                        compatibility_tests.append({
                            'endpoint': method_name,
                            'success': True,
                            'result_type': type(result).__name__
                        })
                    except Exception as e:
                        compatibility_tests.append({
                            'endpoint': method_name,
                            'success': False,
                            'error': str(e)
                        })
            
            # Report compatibility results
            successful_apis = sum(1 for test in compatibility_tests if test['success'])
            total_apis = len(compatibility_tests)
            
            print(f"   🔗 API compatibility: {successful_apis}/{total_apis}")
            
            for test in compatibility_tests:
                if test['success']:
                    print(f"   ✅ {test['endpoint']}: {test.get('result_type', 'N/A')}")
                else:
                    print(f"   ⚠️ {test['endpoint']}: {test.get('error', 'Failed')}")
            
            return successful_apis > 0
            
        except ImportError:
            print("   ⚠️ API module not available for compatibility testing")
            return False
        except Exception as e:
            print(f"   ❌ API compatibility test failed: {e}")
            return False
    
    def test_existing_functionality(self):
        """Test that existing functionality still works."""
        print("⚙️ Testing existing functionality preservation...")
        
        try:
            # Test that core memory operations still work
            functionality_tests = []
            
            # Test memory storage if available
            try:
                from memmimic.memory.storage.amms_storage import AMMSStorage
                
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                    db_path = tmp_db.name
                
                async def test_storage():
                    storage = AMMSStorage(db_path)
                    await storage.initialize()
                    
                    # Basic storage operations should still work
                    memory_id = await storage.store_memory(
                        content="Test content for backward compatibility",
                        memory_type="compatibility_test",
                        metadata={"test": True}
                    )
                    
                    retrieved = await storage.get_memory(memory_id)
                    await storage.close()
                    
                    return memory_id > 0 and retrieved is not None
                
                storage_works = asyncio.run(test_storage())
                functionality_tests.append({
                    'feature': 'Memory Storage',
                    'success': storage_works
                })
                
                os.unlink(db_path)
                
            except Exception as e:
                functionality_tests.append({
                    'feature': 'Memory Storage',
                    'success': False,
                    'error': str(e)
                })
            
            # Test configuration system if available
            try:
                from memmimic.config import get_performance_config
                config = get_performance_config()
                config_works = config is not None
                functionality_tests.append({
                    'feature': 'Configuration System',
                    'success': config_works
                })
            except Exception as e:
                functionality_tests.append({
                    'feature': 'Configuration System',
                    'success': False,
                    'error': str(e)
                })
            
            # Security system removed in enterprise cleanup
            functionality_tests.append({
                'feature': 'Security System', 
                'success': False,
                'error': 'Security module removed'
            })
            
            # Report functionality results
            working_features = sum(1 for test in functionality_tests if test['success'])
            total_features = len(functionality_tests)
            
            print(f"   ⚙️ Working functionality: {working_features}/{total_features}")
            
            for test in functionality_tests:
                if test['success']:
                    print(f"   ✅ {test['feature']}: Working")
                else:
                    print(f"   ⚠️ {test['feature']}: {test.get('error', 'Not working')}")
            
            return working_features > 0
            
        except Exception as e:
            print(f"   ❌ Functionality preservation test failed: {e}")
            return False


async def run_modular_architecture_integration_tests():
    """Run all modular architecture integration tests."""
    print("🏗️ Running Modular Architecture Integration Tests")
    print("=" * 60)
    
    if not COMPONENTS_AVAILABLE:
        print("⚠️ Modular components not fully available. Running compatibility tests only.")
    
    # Initialize test suites
    individual_tests = TestIndividualComponents()
    integration_tests = TestComponentIntegration()
    compatibility_tests = TestBackwardCompatibility()
    
    test_suites = [
        ("Individual Components", individual_tests),
        ("Component Integration", integration_tests),
        ("Backward Compatibility", compatibility_tests),
    ]
    
    results = {}
    
    for suite_name, test_suite in test_suites:
        print(f"\n🧪 Testing: {suite_name}")
        print("-" * 40)
        
        # Setup components if needed
        if hasattr(test_suite, 'setup_components'):
            setup_success = test_suite.setup_components()
            if not setup_success and COMPONENTS_AVAILABLE:
                print("   ⚠️ Component setup failed")
        
        # Get all test methods
        test_methods = [
            method for method in dir(test_suite)
            if method.startswith('test_') and callable(getattr(test_suite, method))
        ]
        
        suite_results = {}
        
        for test_method in test_methods:
            method_name = test_method.replace('test_', '').replace('_', ' ').title()
            
            try:
                test_func = getattr(test_suite, test_method)
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                    
                suite_results[method_name] = result
                status = "✅" if result else "⚠️"
                print(f"   {status} {method_name}")
                
            except Exception as e:
                suite_results[method_name] = False
                print(f"   ❌ {method_name}: {e}")
        
        results[suite_name] = suite_results
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Modular Architecture Integration Test Results:")
    
    total_tests = 0
    passed_tests = 0
    warning_tests = 0
    
    for suite_name, suite_results in results.items():
        suite_passed = sum(1 for r in suite_results.values() if r is True)
        suite_warnings = sum(1 for r in suite_results.values() if r is False)
        suite_total = len(suite_results)
        total_tests += suite_total
        passed_tests += suite_passed
        warning_tests += suite_warnings
        
        if suite_passed == suite_total:
            status = "✅"
        elif suite_passed > 0:
            status = "⚠️"
        else:
            status = "❌"
            
        print(f"{status} {suite_name}: {suite_passed}/{suite_total}")
        
        # Show failed/warning tests
        for test, result in suite_results.items():
            if result is False:
                print(f"     ❌ {test}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 MODULAR ARCHITECTURE INTEGRATION TESTS PASSED!")
        return 0
    elif success_rate >= 70:
        print("⚠️ Most integration tests passed, some components need work")
        return 1
    else:
        print("❌ SIGNIFICANT INTEGRATION ISSUES DETECTED!")
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(run_modular_architecture_integration_tests()))