#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Status Tool - Clean System Health Check
Professional system status without auto-briefings or noise
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Force UTF-8 I/O for cross-platform compatibility
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from memmimic import create_memmimic
    from memmimic.tales.tale_manager import TaleManager
except ImportError as e:
    print(f"❌ Error importing MemMimic: {e}", file=sys.stderr)
    print("❌ Error: Cannot import MemMimic components")
    sys.exit(1)


def check_cxd_status():
    """Check CXD classifier availability and performance"""
    try:
        from memmimic.cxd.classifiers.optimized_meta import create_optimized_classifier

        classifier = create_optimized_classifier()

        # Test classification
        test_result = classifier.classify("test classification")

        return {
            "available": True,
            "version": "2.0",
            "test_confidence": test_result.average_confidence if test_result else 0.0,
            "dominant_function": (
                test_result.dominant_function.value
                if test_result and test_result.dominant_function
                else "UNKNOWN"
            ),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


async def analyze_memory_statistics(api):
    """Analyze memory database statistics using async API"""
    try:
        import asyncio
        
        # Get count and list recent memories 
        memory_count = await api.memory.count_memories()
        memories = await api.memory.list_memories(limit=1000)  # Recent memories

        if not memories:
            return {"total": memory_count, "by_type": {}, "recent_24h": 0, "avg_importance": 0.0}

        # Memory type distribution
        type_counts = {}
        total_importance = 0
        recent_count = 0
        now = datetime.now()

        for memory in memories:
            # Type distribution
            mem_type = memory.metadata.get("type", "unknown") 
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

            # Importance tracking
            importance = memory.importance_score
            total_importance += importance

            # Recent memories (24h)
            try:
                if (now - memory.created_at) < timedelta(hours=24):
                    recent_count += 1
            except:
                pass

        return {
            "total": memory_count,
            "by_type": type_counts,
            "recent_24h": recent_count,
            "avg_importance": total_importance / len(memories) if memories else 0.0,
        }

    except Exception as e:
        return {"error": str(e)}


def analyze_tales_statistics():
    """Analyze tales collection statistics"""
    try:
        tale_manager = TaleManager()

        # Get all tales metadata (TaleManager v2.0 signature)
        tales_data = tale_manager.list_tales()  # No limit/show_stats parameters

        if not tales_data:
            return {
                "total": 0,
                "by_category": {},
                "total_size": 0,
                "avg_size": 0,  # Add explicit avg_size for empty case
            }

        category_counts = {}
        total_size = 0

        for tale in tales_data:
            # Category distribution
            category = tale.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

            # Size accumulation
            size = tale.get("size", 0)
            total_size += size

        return {
            "total": len(tales_data),
            "by_category": category_counts,
            "total_size": total_size,
            "avg_size": total_size
            // max(len(tales_data), 1),  # Prevent division by zero
        }

    except Exception as e:
        return {"error": str(e)}


async def main_async():
    try:
        # Initialize MemMimic with AMMS-only architecture
        api = create_memmimic("memmimic.db")

        # Check CXD status
        cxd_status = check_cxd_status()

        # Analyze memory statistics (async)
        memory_stats = await analyze_memory_statistics(api)

        # Analyze tales statistics
        tales_stats = analyze_tales_statistics()

        # Get storage stats
        storage_stats = api.memory.get_stats()

        # Get API status
        api_status = await api.status()

        # Build clean status report
        status_parts = []

        # Header
        status_parts.append("🎯 MEMMIMIC SYSTEM STATUS (POST-MIGRATION)")
        status_parts.append("=" * 50)
        status_parts.append("")

        # Core System Health
        status_parts.append("🔧 CORE SYSTEM:")
        status_parts.append(f"  ✅ Storage Type: {storage_stats.get('storage_type', 'amms_only')}")
        status_parts.append(f"  ✅ Total Memories: {api_status.get('memories', 0)}")
        status_parts.append(f"  ✅ Total Tales: {api_status.get('tales', 0)}")
        status_parts.append(f"  ✅ Status: {api_status.get('status', 'operational')}")
        status_parts.append("")

        # CXD Classification Status
        status_parts.append("🧠 CXD CLASSIFICATION:")
        if cxd_status["available"]:
            status_parts.append(f"  ✅ CXD v{cxd_status['version']}: Active")
            status_parts.append(
                f"  📊 Test confidence: {cxd_status['test_confidence']:.3f}"
            )
            status_parts.append(
                f"  🎯 Test function: {cxd_status['dominant_function']}"
            )
        else:
            status_parts.append(
                f"  ❌ CXD: Unavailable ({cxd_status.get('error', 'Unknown error')})"
            )
        status_parts.append("")

        # Memory Statistics 
        status_parts.append("🧮 MEMORY STATISTICS:")
        if "error" not in memory_stats:
            status_parts.append(f"  📊 Total memories: {memory_stats['total']}")
            status_parts.append(f"  🕐 Recent (24h): {memory_stats['recent_24h']}")
            status_parts.append(
                f"  📈 Avg importance: {memory_stats['avg_importance']:.3f}"
            )

            if memory_stats["by_type"]:
                status_parts.append("  📑 By type:")
                for mem_type, count in sorted(memory_stats["by_type"].items()):
                    status_parts.append(f"    • {mem_type}: {count}")
        else:
            status_parts.append(f"  ❌ Memory analysis failed: {memory_stats['error']}")
        status_parts.append("")

        # Tales Statistics
        status_parts.append("📖 TALES COLLECTION:")
        if "error" not in tales_stats:
            status_parts.append(f"  📚 Total tales: {tales_stats['total']}")
            status_parts.append(f"  💾 Total size: {tales_stats['total_size']:,} chars")
            if tales_stats["total"] > 0:  # Only show avg if there are tales
                status_parts.append(f"  📏 Avg size: {tales_stats['avg_size']:,} chars")

            if tales_stats["by_category"]:
                status_parts.append("  📂 By category:")
                for category, count in sorted(tales_stats["by_category"].items()):
                    status_parts.append(f"    • {category}: {count}")
        else:
            status_parts.append(f"  ❌ Tales analysis failed: {tales_stats['error']}")
        status_parts.append("")

        # AMMS Performance Metrics
        performance_metrics = storage_stats.get('metrics', {})
        status_parts.append("⚡ AMMS PERFORMANCE:")
        status_parts.append(f"  📊 Total Operations: {performance_metrics.get('total_operations', 0)}")
        status_parts.append(f"  ✅ Successful Operations: {performance_metrics.get('successful_operations', 0)}")
        status_parts.append(f"  ❌ Failed Operations: {performance_metrics.get('failed_operations', 0)}")
        status_parts.append(f"  ⏱️ Avg Response Time: {performance_metrics.get('avg_response_time_ms', 0.0):.1f}ms")
        status_parts.append("")

        # Post-Migration Features
        status_parts.append("🚀 POST-MIGRATION FEATURES:")
        status_parts.append("  ✅ AMMS-Only Architecture: Active")
        status_parts.append("  ✅ Async API: Active")
        status_parts.append("  ✅ High-Performance Storage: Active")
        status_parts.append("  ✅ Clean Architecture: Active")
        status_parts.append("")

        # System Health Summary
        total_issues = 0
        if not cxd_status["available"]:
            total_issues += 1
        if "error" in memory_stats:
            total_issues += 1
        if "error" in tales_stats:
            total_issues += 1

        if total_issues == 0:
            status_parts.append("🟢 SYSTEM HEALTH: All systems operational")
        elif total_issues == 1:
            status_parts.append("🟡 SYSTEM HEALTH: Minor issues detected")
        else:
            status_parts.append("🔴 SYSTEM HEALTH: Multiple issues detected")

        status_parts.append("")

        # === CLAUDE USAGE GUIDANCE (Personal Manual) ===
        status_parts.append("🎯 CLAUDE USAGE GUIDANCE")
        status_parts.append("=" * 30)
        status_parts.append("📝 MEMORY BEST PRACTICES:")
        status_parts.append(
            "  • ALWAYS recall_cxd() BEFORE remember() to check existing"
        )
        status_parts.append("  • UPDATE existing memories vs creating duplicates")
        status_parts.append("  • remember() only for genuinely NEW insights")
        status_parts.append("  • Use update_memory_guided() for significant changes")
        status_parts.append("")
        status_parts.append("📖 TALE MANAGEMENT:")
        status_parts.append("  • save_tale() auto-detects create vs update")
        status_parts.append("  • Use tales() to check existing before creating")
        status_parts.append("  • UPDATE existing tales vs creating versions")
        status_parts.append("  • context_tale() for narrative from memories")
        status_parts.append("")
        status_parts.append("🔍 SEARCH STRATEGY:")
        status_parts.append("  • recall_cxd() for conceptual/semantic search")
        status_parts.append("  • Use CXD filters: CONTROL, CONTEXT, DATA, ALL")
        status_parts.append("  • Semantic for concepts, keywords for specifics")
        status_parts.append("  • Check multiple memory types if needed")
        status_parts.append("")
        status_parts.append("⚡ EFFICIENT WORKFLOW:")
        status_parts.append("  1. status() → Get orientation & guidance")
        status_parts.append("  2. recall_cxd() → Check existing relevant context")
        status_parts.append("  3. UPDATE existing OR create genuinely new")
        status_parts.append("  4. socratic_dialogue() for complex decisions")
        status_parts.append("")
        status_parts.append("🧠 COGNITIVE TIPS:")
        status_parts.append("  • High confidence = trust, low confidence = be cautious")
        status_parts.append("  • Multiple memory types = richer context")
        status_parts.append("  • CXD classification guides response style")
        status_parts.append("  • Tales for narrative, memories for facts")
        status_parts.append("")
        status_parts.append("🚀 MemMimic v1.0 - The Memory System That Learns You Back")
        # === INTERACTION GUIDELINES v2.5 ===
        status_parts.append("💡 DEFAULT: Trust your knowledge ")
        status_parts.append("")
        status_parts.append("🛑 MEMORY CHECK")
        status_parts.append("==============")
        status_parts.append("About to search? Ask yourself:")
        status_parts.append("'Am I searching because I don't trust what I know?'")
        status_parts.append("")
        status_parts.append("Search for: What did WE do/decide/try?")
        status_parts.append("Don't search for: What is this thing?")
        status_parts.append("")
        status_parts.append("Surprised? → remember()")
        status_parts.append("Uncertain? → Skip it")
        status_parts.append("")
        status_parts.append("")
        status_parts.append("📖 context_tale() ONLY for:")
        status_parts.append("  • Onboarding new Claude instances")
        status_parts.append("  • 'Tell me the story of X project'")
        status_parts.append("  • Deep context recovery requests")

        print("\n".join(status_parts))

    except Exception as e:
        print(f"❌ Critical error in status check: {str(e)}", file=sys.stderr)
        print(f"❌ Status check failed: {str(e)}")
        sys.exit(1)


def main():
    """Sync wrapper for MCP compatibility"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

