"""
Test CXD Framework Integration

Quick test to verify CXD imports work correctly.
"""

def test_cxd_imports():
    """Test that CXD classes can be imported."""
    try:
        from memmimic.cxd import (
            CXDFunction,
            ExecutionState,
            CXDTag,
            CXDSequence,
            OptimizedMetaCXDClassifier,
            create_optimized_classifier,
        )
        print("✅ CXD imports successful")
        return True
    except ImportError as e:
        print(f"❌ CXD import failed: {e}")
        return False

def test_cxd_basic_functionality():
    """Test basic CXD functionality."""
    try:
        from memmimic.cxd import create_optimized_classifier
        
        # Create classifier
        classifier = create_optimized_classifier()
        
        # Test classification
        result = classifier.classify("Search for files related to the project")
        print(f"✅ CXD classification successful: {result.pattern}")
        return True
    except Exception as e:
        print(f"❌ CXD functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing CXD Framework Integration...")
    
    # Test imports
    imports_ok = test_cxd_imports()
    
    if imports_ok:
        # Test functionality
        functionality_ok = test_cxd_basic_functionality()
        
        if functionality_ok:
            print("🎉 CXD Framework integration successful!")
        else:
            print("⚠️ CXD imports work but functionality needs fixing")
    else:
        print("❌ CXD imports need fixing first")
