#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Memory Tool - Remember with CXD Classification
Professional-grade memory storage with automatic cognitive classification
"""

import os
import sys

# Force UTF-8 I/O for cross-platform compatibility
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import asyncio
    from memmimic import create_memmimic
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
            "cxd_function": (
                result.dominant_function.value
                if result.dominant_function
                else "UNKNOWN"
            ),
            "cxd_confidence": result.average_confidence,
            "cxd_pattern": result.pattern,
            "cxd_execution_pattern": result.execution_pattern,
            "cxd_version": "2.0",
        }

        return cxd_metadata

    except Exception as e:
        print(f"⚠️ CXD classification failed: {e}", file=sys.stderr)
        return {"cxd_error": str(e)}


async def main_async():
    if len(sys.argv) < 2:
        print("❌ Error: Missing content argument")
        sys.exit(1)

    try:
        content = sys.argv[1]
        memory_type = sys.argv[2] if len(sys.argv) > 2 else "interaction"

        # Initialize CXD classifier
        cxd_classifier = init_cxd_classifier()
        cxd_status = "CXD v2.0 active" if cxd_classifier else "CXD unavailable"

        # Initialize MemMimic with AMMS-only architecture
        api = create_memmimic("memmimic.db")

        # Classify content with CXD
        cxd_metadata = classify_content_with_cxd(cxd_classifier, content)

        # Create memory with CXD metadata
        memory = Memory(
            content=content,
            metadata={"type": memory_type}
        )

        # Add CXD metadata if classification succeeded
        if cxd_metadata and "cxd_function" in cxd_metadata:
            memory.metadata.update(cxd_metadata)

        # Store memory using async API
        memory_id = await api.remember(content, memory_type)

        # Create professional success message
        result_parts = [f"✅ Memory stored (ID: {memory_id}, Type: {memory_type})"]

        if cxd_metadata and "cxd_function" in cxd_metadata:
            cxd_func = cxd_metadata["cxd_function"]
            cxd_conf = cxd_metadata["cxd_confidence"]
            result_parts.append(f"🎯 CXD: {cxd_func} (confidence: {cxd_conf:.2f})")

            # Add cognitive function interpretation
            interpretations = {
                "CONTROL": "Control - search, filtering, management",
                "CONTEXT": "Context - references, relationships, memory",
                "DATA": "Data - processing, analysis, transformation",
            }
            if cxd_func in interpretations:
                result_parts.append(f"📝 {interpretations[cxd_func]}")

        result_parts.append(f"🔧 Status: {cxd_status}")

        print("\n".join(result_parts))

    except Exception as e:
        print(f"❌ Error storing memory: {str(e)}", file=sys.stderr)
        print(f"❌ Failed to store memory: {str(e)}")
        sys.exit(1)


def main():
    """Sync wrapper for MCP compatibility"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
