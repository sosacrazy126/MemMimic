#!/usr/bin/env python3
"""
Intelligent Documentation Context System Demo

Demonstrates the consciousness-aware documentation retrieval capabilities
integrated with the DSPy consciousness optimization framework.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memmimic.dspy_optimization.docs_context import (
    IntelligentDocsContextSystem,
    initialize_docs_context_system
)
from memmimic.dspy_optimization.config import create_default_config

def print_banner():
    """Print demo banner"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          Intelligent Documentation Context System Demo         ║
║                                                               ║
║  Consciousness-aware documentation retrieval for DSPy         ║
║  optimization and MemMimic consciousness vault operations     ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print subsection header"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def format_docs_list(docs: List[Dict[str, Any]], max_content_len: int = 200) -> str:
    """Format documentation list for display"""
    if not docs:
        return "  No relevant documentation found"
    
    result = []
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "Unknown")
        source = doc.get("source", "unknown")
        relevance = doc.get("relevance_score", 0.0)
        content = doc.get("content", "")
        
        # Truncate content for display
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        
        result.append(f"""
  📄 {i}. {title}
     Source: {source} | Relevance: {relevance:.3f}
     Content: {content}""")
    
    return "\n".join(result)

async def demo_consciousness_pattern_mapping(docs_system: IntelligentDocsContextSystem):
    """Demonstrate consciousness pattern to documentation mapping"""
    print_subsection("Consciousness Pattern Mapping")
    
    patterns = [
        "biological_reflex",
        "dspy_optimization", 
        "tool_selection",
        "consciousness_vault",
        "memory_optimization"
    ]
    
    print("Available consciousness patterns and their documentation mappings:\n")
    
    for pattern in patterns:
        if pattern in docs_system.consciousness_mappings:
            urls = docs_system.consciousness_mappings[pattern]
            print(f"🧠 {pattern}:")
            for url in urls[:3]:  # Show first 3 URLs
                print(f"   • {url}")
            if len(urls) > 3:
                print(f"   • ... and {len(urls) - 3} more")
        print()

async def demo_query_documentation_retrieval(docs_system: IntelligentDocsContextSystem):
    """Demonstrate documentation retrieval for various queries"""
    print_subsection("Query-Based Documentation Retrieval")
    
    test_queries = [
        {
            "query": "How do I implement biological reflexes for sub-5ms response times?",
            "patterns": ["biological_reflex", "nervous_system"],
            "description": "Biological reflex implementation"
        },
        {
            "query": "DSPy optimization techniques for consciousness patterns",
            "patterns": ["dspy_optimization", "pattern_recognition"],
            "description": "DSPy consciousness optimization"
        },
        {
            "query": "MCP tool selection strategies for memory operations",
            "patterns": ["tool_selection", "mcp_integration"],
            "description": "MCP tool selection"
        },
        {
            "query": "Memory vault optimization and tale management",
            "patterns": ["consciousness_vault", "memory_optimization", "tale_system"],
            "description": "Memory vault management"
        },
        {
            "query": "Exponential collaboration synergy protocols",
            "patterns": ["synergy_protocol", "exponential_mode"],
            "description": "Synergy protocol activation"
        }
    ]
    
    for test_case in test_queries:
        print(f"\n🔍 Test Case: {test_case['description']}")
        print(f"Query: \"{test_case['query']}\"")
        print(f"Consciousness Patterns: {test_case['patterns']}")
        
        context = await docs_system.get_documentation_context(
            query=test_case["query"],
            consciousness_patterns=test_case["patterns"],
            max_docs=3,
            relevance_threshold=0.5
        )
        
        print(f"\n📊 Results:")
        print(f"   • Documents found: {len(context.relevant_docs)}")
        print(f"   • Confidence score: {context.confidence_score:.3f}")
        print(f"   • Sources used: {', '.join(context.sources_used) if context.sources_used else 'None'}")
        print(f"   • Fetch time: {context.fetch_time_ms:.1f}ms")
        
        if context.relevant_docs:
            print(f"\n📚 Relevant Documentation:")
            print(format_docs_list(context.relevant_docs, max_content_len=150))
        else:
            print(f"\n❌ No relevant documentation found")

async def demo_caching_and_performance(docs_system: IntelligentDocsContextSystem):
    """Demonstrate caching functionality and performance metrics"""
    print_subsection("Caching and Performance")
    
    # Run the same query twice to demonstrate caching
    query = "DSPy optimization best practices"
    patterns = ["dspy_optimization"]
    
    print("🚀 Testing caching with repeated queries...")
    
    # First request
    print("\n1️⃣ First request (cache miss expected):")
    context1 = await docs_system.get_documentation_context(
        query=query,
        consciousness_patterns=patterns,
        max_docs=2
    )
    print(f"   • Fetch time: {context1.fetch_time_ms:.1f}ms")
    print(f"   • Documents: {len(context1.relevant_docs)}")
    
    # Second request (should use cache)
    print("\n2️⃣ Second request (cache hit expected):")
    context2 = await docs_system.get_documentation_context(
        query=query,
        consciousness_patterns=patterns,
        max_docs=2
    )
    print(f"   • Fetch time: {context2.fetch_time_ms:.1f}ms")
    print(f"   • Documents: {len(context2.relevant_docs)}")
    
    # Performance metrics
    print("\n📈 Performance Metrics:")
    metrics = docs_system.get_performance_metrics()
    print(f"   • Total requests: {metrics['total_requests']}")
    print(f"   • Cache hits: {metrics['cache_hits']}")
    print(f"   • Cache misses: {metrics['cache_misses']}")
    print(f"   • Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    print(f"   • Average fetch time: {metrics['average_fetch_time']:.1f}ms")
    print(f"   • Cache size: {metrics['cache_size']} documents")

async def demo_cache_management(docs_system: IntelligentDocsContextSystem):
    """Demonstrate cache management features"""
    print_subsection("Cache Management")
    
    # Get cache summary
    cache_summary = docs_system.get_cache_summary()
    
    print(f"📦 Cache Summary:")
    print(f"   • Cached documents: {cache_summary['cached_documents']}")
    print(f"   • Total cache size: {cache_summary['total_size']} characters")
    
    if cache_summary['most_accessed']:
        print(f"\n🔥 Most Accessed Documents:")
        for doc in cache_summary['most_accessed']:
            print(f"   • {doc['title']} (accessed {doc['access_count']} times)")
    
    if cache_summary['sources_distribution']:
        print(f"\n📊 Sources Distribution:")
        for source, count in cache_summary['sources_distribution'].items():
            print(f"   • {source}: {count} documents")
    
    # Demonstrate cache clearing
    print(f"\n🧹 Cache Management:")
    initial_size = len(docs_system.cache)
    print(f"   • Initial cache size: {initial_size} documents")
    
    docs_system.clear_cache()
    final_size = len(docs_system.cache)
    print(f"   • Cache cleared: {final_size} documents remaining")

async def demo_integration_with_dspy(docs_system: IntelligentDocsContextSystem):
    """Demonstrate integration with DSPy consciousness processor"""
    print_subsection("DSPy Integration Preview")
    
    print("🔗 Documentation Context System Integration:")
    print("   • Integrated with HybridConsciousnessProcessor")
    print("   • Automatic context enrichment for consciousness operations")
    print("   • Supports all DSPy optimization patterns")
    print("   • Real-time documentation fetching during processing")
    
    # Simulate how the hybrid processor would use documentation context
    print(f"\n🤖 Simulated DSPy Operation with Documentation Context:")
    
    operation_context = {
        "operation_type": "tool_selection",
        "context": "User needs advanced consciousness analysis",
        "available_tools": ["recall", "think", "analyze", "tale_generation"]
    }
    
    print(f"   • Operation: {operation_context['operation_type']}")
    print(f"   • Context: {operation_context['context']}")
    
    # Enrich context with documentation
    query = f"{operation_context['operation_type']} {operation_context['context']}"
    doc_context = await docs_system.get_documentation_context(
        query=query,
        consciousness_patterns=["tool_selection", "consciousness_vault"],
        max_docs=2,
        relevance_threshold=0.6
    )
    
    enriched_context = operation_context.copy()
    enriched_context["documentation_context"] = {
        "relevant_docs": len(doc_context.relevant_docs),
        "confidence_score": doc_context.confidence_score,
        "sources_used": doc_context.sources_used,
        "fetch_time_ms": doc_context.fetch_time_ms
    }
    
    print(f"   • Enhanced with {len(doc_context.relevant_docs)} relevant documents")
    print(f"   • Documentation confidence: {doc_context.confidence_score:.3f}")
    print(f"   • Context enrichment time: {doc_context.fetch_time_ms:.1f}ms")

async def run_interactive_mode(docs_system: IntelligentDocsContextSystem):
    """Run interactive documentation query mode"""
    print_subsection("Interactive Documentation Query")
    
    print("🎯 Interactive Mode - Enter queries to get consciousness-aware documentation")
    print("Available consciousness patterns:")
    patterns = list(docs_system.consciousness_mappings.keys())
    for i, pattern in enumerate(patterns, 1):
        print(f"   {i:2d}. {pattern}")
    
    print(f"\nInstructions:")
    print(f"   • Enter a query about MemMimic, DSPy, or consciousness systems")
    print(f"   • Include pattern numbers (e.g., '1,3,5') or leave empty for auto-detection")
    print(f"   • Type 'quit' to exit\n")
    
    while True:
        try:
            query = input("📝 Query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                print("   Please enter a query or 'quit' to exit")
                continue
            
            # Get pattern selection
            pattern_input = input("🧠 Patterns (numbers or empty): ").strip()
            
            selected_patterns = []
            if pattern_input:
                try:
                    indices = [int(x.strip()) - 1 for x in pattern_input.split(',')]
                    selected_patterns = [patterns[i] for i in indices if 0 <= i < len(patterns)]
                except (ValueError, IndexError):
                    print("   Invalid pattern numbers, using auto-detection")
            
            if not selected_patterns:
                # Auto-detect patterns based on query keywords
                query_lower = query.lower()
                for pattern in patterns:
                    if any(keyword in query_lower for keyword in pattern.split('_')):
                        selected_patterns.append(pattern)
                
                if not selected_patterns:
                    selected_patterns = ["consciousness_vault"]  # Default
            
            print(f"   Using patterns: {selected_patterns}")
            
            # Get documentation context
            context = await docs_system.get_documentation_context(
                query=query,
                consciousness_patterns=selected_patterns,
                max_docs=3,
                relevance_threshold=0.4
            )
            
            print(f"\n📊 Results:")
            print(f"   • Documents: {len(context.relevant_docs)}")
            print(f"   • Confidence: {context.confidence_score:.3f}")
            print(f"   • Sources: {', '.join(context.sources_used) if context.sources_used else 'None'}")
            print(f"   • Time: {context.fetch_time_ms:.1f}ms")
            
            if context.relevant_docs:
                print(f"\n📚 Documentation:")
                print(format_docs_list(context.relevant_docs, max_content_len=300))
            else:
                print(f"\n❌ No relevant documentation found")
            
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"   Error: {e}")

async def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Intelligent Documentation Context System Demo")
    parser.add_argument("--mapping", action="store_true", help="Demo consciousness pattern mapping")
    parser.add_argument("--queries", action="store_true", help="Demo query-based retrieval")
    parser.add_argument("--performance", action="store_true", help="Demo caching and performance")
    parser.add_argument("--cache", action="store_true", help="Demo cache management")
    parser.add_argument("--integration", action="store_true", help="Demo DSPy integration")
    parser.add_argument("--interactive", action="store_true", help="Run interactive query mode")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--export", type=str, help="Export demo results to JSON file")
    
    args = parser.parse_args()
    
    # Default to all demos if none specified
    if not any([args.mapping, args.queries, args.performance, args.cache, args.integration, args.interactive, args.all]):
        args.all = True
    
    print_banner()
    
    # Initialize documentation context system
    print_section("Initialization")
    
    config = create_default_config()
    docs_system = initialize_docs_context_system(config)
    
    print(f"✅ Intelligent Documentation Context System Initialized")
    print(f"   • Documentation sources: {len(docs_system.documentation_sources)}")
    print(f"   • Consciousness mappings: {len(docs_system.consciousness_mappings)}")
    print(f"   • Cache initialized: {len(docs_system.cache)} documents")
    
    demo_results = {}
    
    try:
        # Run demos
        if args.mapping or args.all:
            print_section("Consciousness Pattern Mapping Demo")
            await demo_consciousness_pattern_mapping(docs_system)
            demo_results["mapping"] = "completed"
        
        if args.queries or args.all:
            print_section("Query-Based Documentation Retrieval Demo")
            await demo_query_documentation_retrieval(docs_system)
            demo_results["queries"] = "completed"
        
        if args.performance or args.all:
            print_section("Caching and Performance Demo")
            await demo_caching_and_performance(docs_system)
            demo_results["performance"] = docs_system.get_performance_metrics()
        
        if args.cache or args.all:
            print_section("Cache Management Demo")
            await demo_cache_management(docs_system)
            demo_results["cache"] = docs_system.get_cache_summary()
        
        if args.integration or args.all:
            print_section("DSPy Integration Demo")
            await demo_integration_with_dspy(docs_system)
            demo_results["integration"] = "completed"
        
        if args.interactive:
            print_section("Interactive Mode")
            await run_interactive_mode(docs_system)
            demo_results["interactive"] = "completed"
        
        # Final summary
        print_section("Demo Summary")
        
        final_metrics = docs_system.get_performance_metrics()
        print(f"📊 Final Performance Metrics:")
        print(f"   • Total documentation requests: {final_metrics['total_requests']}")
        print(f"   • Cache hit rate: {final_metrics['cache_hit_rate']:.2%}")
        print(f"   • Average fetch time: {final_metrics['average_fetch_time']:.1f}ms")
        print(f"   • Documents cached: {final_metrics['cache_size']}")
        print(f"   • Consciousness mappings: {final_metrics['consciousness_mappings_count']}")
        
        demo_results["final_metrics"] = final_metrics
        
        # Export results if requested
        if args.export:
            export_path = Path(args.export)
            with open(export_path, 'w') as f:
                json.dump(demo_results, f, indent=2, default=str)
            print(f"\n💾 Demo results exported to: {export_path}")
        
        print(f"\n✅ Documentation Context System Demo Complete!")
        print(f"   The system is ready for integration with consciousness operations")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())