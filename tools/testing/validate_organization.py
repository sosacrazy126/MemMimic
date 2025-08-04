#!/usr/bin/env python3
"""
Organization Validation Script

Validates that the MemMimic project follows the clean, specification-driven
architecture with proper organization and explicit intent for every file.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set

def validate_root_structure():
    """Validate root directory structure follows specification-driven principles"""
    print("🏗️ Validating Root Directory Structure")
    
    required_files = {
        'README.md': 'Project overview and quick start guide',
        'ARCHITECTURE.md': 'System architecture and design principles',
        'pyproject.toml': 'Python project configuration',
        'requirements.txt': 'Python dependencies',
        'LICENSE': 'Legal license information',
        '.gitignore': 'Git ignore patterns to prevent noise'
    }
    
    required_dirs = {
        'specs/': 'Source of truth specifications',
        'src/': 'Implementation code',
        'tools/': 'Development and operational tools',
        'data/': 'Persistent data and caches',
        'docs/': 'Generated documentation',
        'tests/': 'Test suites'
    }
    
    # Check required files
    missing_files = []
    for file_name, purpose in required_files.items():
        if not Path(file_name).exists():
            missing_files.append(f"❌ Missing: {file_name} ({purpose})")
        else:
            print(f"✅ Found: {file_name} - {purpose}")
    
    # Check required directories
    missing_dirs = []
    for dir_name, purpose in required_dirs.items():
        if not Path(dir_name).exists():
            missing_dirs.append(f"❌ Missing: {dir_name} ({purpose})")
        else:
            print(f"✅ Found: {dir_name} - {purpose}")
    
    if missing_files or missing_dirs:
        print("\n⚠️ Missing Required Items:")
        for item in missing_files + missing_dirs:
            print(f"  {item}")
        return False
    
    print("✅ Root structure validation passed!")
    return True

def validate_specs_structure():
    """Validate specs directory follows specification-driven organization"""
    print("\n📋 Validating Specifications Structure")
    
    required_spec_dirs = {
        'specs/nervous-system/': 'Core nervous system specifications',
        'specs/memory-management/': 'Memory system specifications',
        'specs/multi-agent/': 'Multi-agent coordination specifications',
        'specs/narrative-fusion/': 'Narrative-memory fusion specifications',
        'specs/evolution-phases/': 'Development phase tracking specifications',
        'specs/patterns/': 'Reusable patterns and templates'
    }
    
    missing_spec_dirs = []
    for dir_path, purpose in required_spec_dirs.items():
        if not Path(dir_path).exists():
            missing_spec_dirs.append(f"❌ Missing: {dir_path} ({purpose})")
        else:
            print(f"✅ Found: {dir_path} - {purpose}")
    
    # Check for specs README
    if Path('specs/README.md').exists():
        print("✅ Found: specs/README.md - Specification documentation")
    else:
        missing_spec_dirs.append("❌ Missing: specs/README.md (Specification documentation)")
    
    if missing_spec_dirs:
        print("\n⚠️ Missing Specification Items:")
        for item in missing_spec_dirs:
            print(f"  {item}")
        return False
    
    print("✅ Specifications structure validation passed!")
    return True

def validate_data_organization():
    """Validate data directory organization"""
    print("\n💾 Validating Data Organization")
    
    required_data_dirs = {
        'data/databases/': 'SQLite databases',
        'data/caches/': 'Embedding and processing caches',
        'data/logs/': 'Application logs'
    }
    
    missing_data_dirs = []
    for dir_path, purpose in required_data_dirs.items():
        if not Path(dir_path).exists():
            missing_data_dirs.append(f"❌ Missing: {dir_path} ({purpose})")
        else:
            print(f"✅ Found: {dir_path} - {purpose}")
            
            # Check for .gitkeep files
            gitkeep_path = Path(dir_path) / '.gitkeep'
            if gitkeep_path.exists():
                print(f"  ✅ Has .gitkeep to preserve empty directory")
            else:
                print(f"  ⚠️ Missing .gitkeep file")
    
    if missing_data_dirs:
        print("\n⚠️ Missing Data Organization Items:")
        for item in missing_data_dirs:
            print(f"  {item}")
        return False
    
    print("✅ Data organization validation passed!")
    return True

def validate_tools_organization():
    """Validate tools directory organization"""
    print("\n🛠️ Validating Tools Organization")
    
    required_tool_dirs = {
        'tools/scripts/': 'Automation scripts',
        'tools/config/': 'Configuration files',
        'tools/testing/': 'Test utilities and runners',
        'tools/monitoring/': 'Performance and health monitoring'
    }
    
    missing_tool_dirs = []
    for dir_path, purpose in required_tool_dirs.items():
        if not Path(dir_path).exists():
            missing_tool_dirs.append(f"❌ Missing: {dir_path} ({purpose})")
        else:
            print(f"✅ Found: {dir_path} - {purpose}")
    
    # Check for this validation script
    if Path('tools/testing/validate_organization.py').exists():
        print("✅ Found: tools/testing/validate_organization.py - This validation script")
    
    if missing_tool_dirs:
        print("\n⚠️ Missing Tools Organization Items:")
        for item in missing_tool_dirs:
            print(f"  {item}")
        return False
    
    print("✅ Tools organization validation passed!")
    return True

def check_for_noise_files():
    """Check for noise files that should have been removed"""
    print("\n🗑️ Checking for Noise Files")
    
    noise_patterns = [
        'AGENT.md', 'CLAUDE.md', 'think.md',
        'CHANGELOG.md', 'PROJECT_STATUS_COMPLETE.md', 'STRUCTURE.md',
        '*.tmp', '*.cache', 'results_*.json',
        'workspace/', 'scratch/', '*.draft.md'
    ]
    
    found_noise = []
    for pattern in noise_patterns:
        if '*' in pattern:
            # Handle glob patterns
            import glob
            matches = glob.glob(pattern)
            for match in matches:
                if Path(match).exists():
                    found_noise.append(f"⚠️ Noise file: {match}")
        else:
            if Path(pattern).exists():
                found_noise.append(f"⚠️ Noise file: {pattern}")
    
    if found_noise:
        print("\n⚠️ Found Noise Files (should be cleaned up):")
        for item in found_noise:
            print(f"  {item}")
        return False
    
    print("✅ No noise files detected!")
    return True

def validate_implementation_structure():
    """Validate that implementation follows specifications"""
    print("\n🔧 Validating Implementation Structure")
    
    # Check that src/ contains the enhanced nervous system
    nervous_system_path = Path('src/memmimic/nervous_system/')
    if nervous_system_path.exists():
        print("✅ Found: Enhanced nervous system implementation")
        
        # Check for key components
        key_components = [
            'core.py',
            'archive_intelligence.py',
            'phase_evolution_tracker.py',
            'tale_memory_binder.py',
            'reflex_latency_optimizer.py',
            'shared_reality_manager.py',
            'theory_of_mind.py'
        ]
        
        missing_components = []
        for component in key_components:
            component_path = nervous_system_path / component
            if component_path.exists():
                print(f"  ✅ Found: {component}")
            else:
                missing_components.append(f"  ❌ Missing: {component}")
        
        if missing_components:
            print("\n⚠️ Missing Implementation Components:")
            for item in missing_components:
                print(item)
            return False
    else:
        print("❌ Missing: Enhanced nervous system implementation")
        return False
    
    print("✅ Implementation structure validation passed!")
    return True

def validate_gitignore():
    """Validate .gitignore prevents noise accumulation"""
    print("\n🚫 Validating .gitignore Configuration")
    
    if not Path('.gitignore').exists():
        print("❌ Missing .gitignore file")
        return False
    
    with open('.gitignore', 'r') as f:
        gitignore_content = f.read()
    
    required_patterns = [
        'data/databases/*.db',
        'data/caches/',
        'data/logs/*.log',
        'AGENT.md',
        'CLAUDE.md',
        'think.md',
        '*.draft.md',
        'workspace/',
        'scratch/'
    ]
    
    missing_patterns = []
    for pattern in required_patterns:
        if pattern not in gitignore_content:
            missing_patterns.append(f"❌ Missing pattern: {pattern}")
        else:
            print(f"✅ Found pattern: {pattern}")
    
    if missing_patterns:
        print("\n⚠️ Missing .gitignore Patterns:")
        for item in missing_patterns:
            print(f"  {item}")
        return False
    
    print("✅ .gitignore validation passed!")
    return True

def main():
    """Run all validation checks"""
    print("🔍 MemMimic Organization Validation")
    print("=" * 50)
    
    validations = [
        validate_root_structure,
        validate_specs_structure,
        validate_data_organization,
        validate_tools_organization,
        validate_implementation_structure,
        validate_gitignore,
        check_for_noise_files
    ]
    
    results = []
    for validation in validations:
        try:
            result = validation()
            results.append(result)
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} validations passed!")
        print("✅ MemMimic organization follows specification-driven architecture")
        print("🚀 Project is ready for production use")
        return True
    else:
        print(f"⚠️ {passed}/{total} validations passed")
        print("❌ Some organizational issues need to be addressed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
