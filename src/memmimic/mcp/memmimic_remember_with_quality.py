#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Enhanced Remember Tool - With Quality Gate
Professional memory storage with intelligent quality control and duplicate detection
"""

import os
import sys
import argparse

# Force UTF-8 I/O for cross-platform compatibility
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import asyncio
    from memmimic.assistant import ContextualAssistant
    from memmimic.memory.quality_gate import MemoryQualityGate
    from memmimic.memory.storage.amms_storage import Memory
except ImportError as e:
    print(f"❌ Error importing MemMimic: {e}", file=sys.stderr)
    print("❌ Error: Cannot import MemMimic core")
    sys.exit(1)


def init_cxd_classifier():
    """Initialize CXD classifier for automatic classification"""
    try:
        from memmimic.cxd.classifiers.optimized_meta import create_optimized_classifier
        classifier = create_optimized_classifier()
        return classifier
    except ImportError as e:
        print(f"⚠️ CXD not available: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"⚠️ CXD initialization failed: {e}", file=sys.stderr)
        return None


def classify_content_with_cxd(classifier, content):
    """Classify content and return CXD metadata"""
    if not classifier:
        return {}

    try:
        result = classifier.classify(content)
        cxd_metadata = {
            "cxd": result.function.name,
            "cxd_function": result.function.name,
            "cxd_confidence": result.confidence,
            "cxd_execution_pattern": result.execution_pattern,
            "cxd_version": "2.0",
        }
        return cxd_metadata
    except Exception as e:
        print(f"⚠️ CXD classification failed: {e}", file=sys.stderr)
        return {"cxd_error": str(e)}


async def main_async():
    parser = argparse.ArgumentParser(description="MemMimic Enhanced Remember with Quality Control")
    parser.add_argument("content", help="Content to remember")
    parser.add_argument("memory_type", nargs="?", default="interaction", help="Type of memory")
    parser.add_argument("--force", action="store_true", help="Force save without quality check")
    parser.add_argument("--review", action="store_true", help="Show pending reviews instead of saving")
    parser.add_argument("--approve", help="Approve pending memory by ID")
    parser.add_argument("--reject", help="Reject pending memory by ID")
    parser.add_argument("--note", help="Note for approval")
    parser.add_argument("--reason", help="Reason for rejection")
    
    args = parser.parse_args()

    try:
        # Initialize assistant and quality gate
        assistant = ContextualAssistant("memmimic")
        quality_gate = MemoryQualityGate(assistant)
        
        # Handle review management commands
        if args.review:
            await handle_review_list(quality_gate)
            return
        
        if args.approve:
            await handle_approval(quality_gate, args.approve, args.note or "")
            return
            
        if args.reject:
            await handle_rejection(quality_gate, args.reject, args.reason or "Rejected by reviewer")
            return
        
        # Initialize CXD classifier
        cxd_classifier = init_cxd_classifier()
        cxd_status = "CXD v2.0 active" if cxd_classifier else "CXD unavailable"
        
        content = args.content
        memory_type = args.memory_type
        
        if args.force:
            # Force save without quality check
            await save_memory_directly(assistant, content, memory_type, cxd_classifier)
        else:
            # Use quality gate
            await save_memory_with_quality_check(quality_gate, content, memory_type, cxd_classifier)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


async def save_memory_with_quality_check(quality_gate, content, memory_type, cxd_classifier):
    """Save memory with quality control"""
    
    # Evaluate memory quality
    quality_result = await quality_gate.evaluate_memory(content, memory_type)
    
    # Classify with CXD
    cxd_metadata = classify_content_with_cxd(cxd_classifier, content)
    
    if quality_result.approved and quality_result.auto_decision:
        # Auto-approved - save directly
        memory = Memory(
            content=content,
            metadata={"type": memory_type, "quality_approved": True}
        )
        
        # Add CXD metadata
        if cxd_metadata and "cxd_function" in cxd_metadata:
            memory.metadata.update(cxd_metadata)
        
        memory_id = await quality_gate.memory_store.store_memory(memory)
        
        print("✅ MEMORY APPROVED AND SAVED")
        print("=" * 40)
        print(f"📝 Memory ID: {memory_id}")
        print(f"🎯 Type: {memory_type}")
        print(f"💡 Quality: AUTO-APPROVED (confidence: {quality_result.confidence:.2f})")
        print(f"📄 Content: {content}")
        
        if cxd_metadata and "cxd_function" in cxd_metadata:
            print(f"🧠 CXD: {cxd_metadata['cxd_function']} (confidence: {cxd_metadata['cxd_confidence']:.2f})")
    
    elif not quality_result.approved and quality_result.auto_decision:
        # Auto-rejected
        print("❌ MEMORY REJECTED")
        print("=" * 40)
        print(f"📝 Content: {content}")
        print(f"🚫 Reason: {quality_result.reason}")
        print(f"💡 Confidence: {quality_result.confidence:.2f}")
        
        if quality_result.suggested_content:
            print(f"💬 Suggestion: {quality_result.suggested_content}")
        
        if quality_result.duplicates:
            print(f"🔍 Found {len(quality_result.duplicates)} similar memories:")
            for i, dup in enumerate(quality_result.duplicates[:3]):
                print(f"   {i+1}. {dup.content[:80]}...")
        
        print("\n🔧 To force save anyway: --force")
    
    else:
        # Requires human review
        queue_id = await quality_gate.queue_for_review(content, memory_type, quality_result)
        
        print("⏳ MEMORY QUEUED FOR REVIEW")
        print("=" * 40)
        print(f"📝 Content: {content}")
        print(f"🎯 Type: {memory_type}")
        print(f"💡 Quality: BORDERLINE (confidence: {quality_result.confidence:.2f})")
        print(f"🔍 Queue ID: {queue_id}")
        print(f"📋 Reason: {quality_result.reason}")
        
        if quality_result.suggested_content:
            print(f"💬 Suggestion: {quality_result.suggested_content}")
        
        print("\n🔧 Review Commands:")
        print(f"   --review                    # Show all pending reviews")
        print(f"   --approve {queue_id}        # Approve this memory")
        print(f"   --reject {queue_id}         # Reject this memory")


async def save_memory_directly(assistant, content, memory_type, cxd_classifier):
    """Save memory directly without quality check (force mode)"""
    
    # Classify with CXD
    cxd_metadata = classify_content_with_cxd(cxd_classifier, content)
    
    memory = Memory(
        content=content,
        metadata={"type": memory_type, "forced_save": True}
    )
    
    # Add CXD metadata
    if cxd_metadata and "cxd_function" in cxd_metadata:
        memory.metadata.update(cxd_metadata)
    
    memory_id = await assistant.memory_store.store_memory(memory)
    
    print("✅ MEMORY FORCE SAVED")
    print("=" * 40)
    print(f"📝 Memory ID: {memory_id}")
    print(f"🎯 Type: {memory_type}")
    print(f"⚠️ Quality: BYPASSED (forced save)")
    print(f"📄 Content: {content}")
    
    if cxd_metadata and "cxd_function" in cxd_metadata:
        print(f"🧠 CXD: {cxd_metadata['cxd_function']} (confidence: {cxd_metadata['cxd_confidence']:.2f})")


async def handle_review_list(quality_gate):
    """Handle --review command"""
    pending_reviews = quality_gate.get_pending_reviews()
    
    if not pending_reviews:
        print("✅ NO MEMORIES PENDING REVIEW")
        return
    
    print("📋 MEMORIES PENDING REVIEW")
    print("=" * 50)
    
    for review in pending_reviews:
        print(f"\n🔍 Queue ID: {review['id']}")
        print(f"📝 Content: {review['content'][:100]}{'...' if len(review['content']) > 100 else ''}")
        print(f"🎯 Type: {review['memory_type']}")
        print(f"💡 Confidence: {review['quality_result'].confidence:.2f}")
        print(f"📋 Reason: {review['quality_result'].reason}")
        print(f"⏰ Queued: {review['queued_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n🔧 Found {len(pending_reviews)} memories awaiting review")
    print("Use --approve <id> or --reject <id> to process them")


async def handle_approval(quality_gate, queue_id, note):
    """Handle --approve command"""
    success = await quality_gate.approve_pending(queue_id, note or "Human approved via CLI")
    
    if success:
        print(f"✅ MEMORY APPROVED: {queue_id}")
        print("Memory has been saved to the database")
        if note:
            print(f"Reviewer note: {note}")
    else:
        print(f"❌ APPROVAL FAILED: {queue_id}")
        print("Queue ID not found or approval failed")


async def handle_rejection(quality_gate, queue_id, reason):
    """Handle --reject command"""
    success = await quality_gate.reject_pending(queue_id, reason or "Human rejected via CLI")
    
    if success:
        print(f"❌ MEMORY REJECTED: {queue_id}")
        print("Memory has been removed from review queue")
        if reason:
            print(f"Rejection reason: {reason}")
    else:
        print(f"❌ REJECTION FAILED: {queue_id}")
        print("Queue ID not found")


def main():
    """Entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()