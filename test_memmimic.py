#!/usr/bin/env python3
"""
Test MemMimic functionality with correct paths
"""

import sys
import os
from pathlib import Path

# Set up paths for MemMimic
MEMMIMIC_ROOT = Path("/home/sigilzo/tools-raw/MemMimic")
sys.path.insert(0, str(MEMMIMIC_ROOT))

# Configure storage
os.environ['MEMMIMIC_STORAGE'] = 'markdown'
os.environ['MEMMIMIC_MD_DIR'] = str(MEMMIMIC_ROOT / "memories")

# Import MemMimic
try:
    from updated_mcp_tools import MemMimicMCP
    print("✅ MemMimicMCP imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MemMimicMCP: {e}")
    sys.exit(1)

def test_status():
    """Test system status"""
    print("\n" + "="*60)
    print("TEST 1: System Status")
    print("="*60)

    try:
        mcp = MemMimicMCP()
        status = mcp.status()

        print(f"✅ Status: {status['status']}")
        print(f"📊 Total memories: {status['stats']['total_memories']}")
        print(f"🗄️  Storage type: {status['stats']['storage_type']}")
        print(f"📂 Directory: {status['stats']['markdown_dir']}")

        return True
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return False

def test_remember():
    """Test storing a memory"""
    print("\n" + "="*60)
    print("TEST 2: Store Memory")
    print("="*60)

    try:
        mcp = MemMimicMCP()
        result = mcp.remember(
            content="Testing MemMimic integration with Amplifier",
            memory_type="test",
            metadata={"cxd": "CONTEXT"}
        )

        print(f"✅ Memory stored: {result['memory_id']}")
        print(f"🏷️  Type: {result.get('type', 'N/A')}")
        print(f"🧠 CXD: {result.get('cxd', 'N/A')}")
        print(f"⭐ Importance: {result.get('importance', 0):.2f}")

        return True
    except Exception as e:
        print(f"❌ Remember test failed: {e}")
        return False

def test_recall():
    """Test retrieving memories"""
    print("\n" + "="*60)
    print("TEST 3: Recall Memories")
    print("="*60)

    try:
        mcp = MemMimicMCP()
        results = mcp.recall_cxd(
            query="integration test",
            function_filter="CONTEXT",
            limit=5
        )

        memories = results if isinstance(results, list) else results.get('memories', [])
        print(f"✅ Found {len(memories)} memories")

        for i, mem in enumerate(memories[:3], 1):
            print(f"\n  Memory {i}:")
            print(f"    ID: {mem.get('id', 'N/A')}")
            print(f"    Content: {mem.get('content', '')[:50]}...")
            print(f"    CXD: {mem.get('cxd', 'N/A')}")

        return True
    except Exception as e:
        print(f"❌ Recall test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 MEMMIMIC TESTING SUITE")
    print("Location:", MEMMIMIC_ROOT)
    print("Storage:", os.environ.get('MEMMIMIC_MD_DIR'))

    tests = [
        test_status,
        test_remember,
        test_recall
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! MemMimic is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
