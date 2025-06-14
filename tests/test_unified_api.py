"""
Test MemMimic Unified API

Quick test to verify the 11-tool API works.
"""

def test_memmimic_api_creation():
    """Test API can be created."""
    try:
        import memmimic
        api = memmimic.create_memmimic()
        print("✅ MemMimic API created successfully")
        return True
    except Exception as e:
        print(f"❌ API creation failed: {e}")
        return False

def test_memmimic_11_tools():
    """Test all 11 core tools exist."""
    try:
        import memmimic
        api = memmimic.create_memmimic()
        
        # Test each tool exists
        tools = [
            'remember', 'recall_cxd', 'think_with_memory', 'status',  # Memory core (4)
            'tales', 'save_tale', 'load_tale', 'delete_tale', 'context_tale',  # Tales (5)
            'update_memory_guided', 'delete_memory_guided', 'analyze_memory_patterns',  # Management (3)
            'socratic_dialogue'  # Cognitive (1)
        ]
        
        for tool in tools:
            if not hasattr(api, tool):
                print(f"❌ Missing tool: {tool}")
                return False
        
        print(f"✅ All {len(tools)} tools present")
        return True
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality works."""
    try:
        import memmimic
        api = memmimic.create_memmimic()
        
        # Test status
        status = api.status()
        print(f"✅ Status: {status}")
        
        # Test CXD integration
        memory = api.remember("Test memory for classification")
        print(f"✅ Memory created: {memory}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing MemMimic Unified API...")
    
    # Run tests
    creation_ok = test_memmimic_api_creation()
    if creation_ok:
        tools_ok = test_memmimic_11_tools()
        if tools_ok:
            basic_ok = test_basic_functionality()
            if basic_ok:
                print("🎉 MemMimic API fully functional!")
            else:
                print("⚠️ API created but basic functionality needs work")
        else:
            print("❌ Missing tools - API incomplete")
    else:
        print("❌ Cannot create API - imports need fixing")
