#!/usr/bin/env python3
import subprocess
import os

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/home/evilbastardxd/Desktop/tools/memmimic')
    return result.returncode, result.stdout, result.stderr

print("Adding files to git...")

# Add all generated files
files = [
    "src/memmimic/memory/active_schema.py",
    "src/memmimic/memory/active_manager.py", 
    "src/memmimic/memory/importance_scorer.py",
    "src/memmimic/memory/stale_detector.py",
    "src/memmimic/mcp/memmimic_load_tale.py",
    "docs/PRD_ActiveMemorySystem.md",
    "Memory/README.md",
    "test_active_memory.py",
    "quick_test.py",
    "commit_summary.md"
]

for file in files:
    ret, out, err = run_cmd(f"git add {file}")
    print(f"Added {file}: {'✅' if ret == 0 else '❌'}")

# Commit
ret, out, err = run_cmd('git commit -m "Implement Active Memory Management System with importance scoring and lifecycle management"')
print(f"Commit: {'✅' if ret == 0 else '❌'}")
if ret != 0:
    print(f"Error: {err}")

# Push to remote
ret, out, err = run_cmd("git push origin main")
print(f"Push: {'✅' if ret == 0 else '❌'}")
if ret != 0:
    print(f"Error: {err}")
else:
    print("✅ Successfully pushed to remote!")