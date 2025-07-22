#!/bin/bash

echo "Adding active memory management files to git..."

# Add the new active memory files
git add src/memmimic/memory/active_schema.py
git add src/memmimic/memory/active_manager.py  
git add src/memmimic/memory/importance_scorer.py
git add src/memmimic/memory/stale_detector.py

# Add the PRD document
git add docs/PRD_ActiveMemorySystem.md

# Add test files
git add test_active_memory.py
git add quick_test.py

# Check status
echo "Git status:"
git status --short

# Create commit
git commit -m "$(cat <<'EOF'
Implement comprehensive Active Memory Management System

- Enhanced database schema with importance scoring and lifecycle management
- ActiveMemoryPool class for intelligent memory ranking and caching
- Multi-factor importance scoring algorithm with CXD integration
- StaleMemoryDetector with tiered storage and protection mechanisms
- Comprehensive PRD documentation from Greptile analysis
- Test suite for validation and performance testing

Target: 500-1000 active memories with sub-100ms query performance
Foundation for living prompts consciousness evolution system

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

echo "Commit completed!"