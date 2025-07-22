#!/usr/bin/env python3
import subprocess
import os

def run_git_command(cmd):
    """Run a git command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/home/evilbastardxd/Desktop/tools/memmimic')
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def main():
    print("Starting git operations...")
    
    # Add files
    files_to_add = [
        "src/memmimic/memory/active_schema.py",
        "src/memmimic/memory/active_manager.py", 
        "src/memmimic/memory/importance_scorer.py",
        "src/memmimic/memory/stale_detector.py",
        "docs/PRD_ActiveMemorySystem.md",
        "test_active_memory.py",
        "quick_test.py"
    ]
    
    for file in files_to_add:
        cmd = f"git add {file}"
        ret, out, err = run_git_command(cmd)
        print(f"Adding {file}: {'✅' if ret == 0 else '❌'}")
        if ret != 0:
            print(f"  Error: {err}")
    
    # Check status
    ret, out, err = run_git_command("git status --short")
    print(f"\nGit status:\n{out}")
    
    # Create commit
    commit_msg = """Implement comprehensive Active Memory Management System

- Enhanced database schema with importance scoring and lifecycle management
- ActiveMemoryPool class for intelligent memory ranking and caching
- Multi-factor importance scoring algorithm with CXD integration
- StaleMemoryDetector with tiered storage and protection mechanisms
- Comprehensive PRD documentation from Greptile analysis
- Test suite for validation and performance testing

Target: 500-1000 active memories with sub-100ms query performance
Foundation for living prompts consciousness evolution system

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""
    
    cmd = f'git commit -m "{commit_msg}"'
    ret, out, err = run_git_command(cmd)
    print(f"\nCommit result: {'✅' if ret == 0 else '❌'}")
    if ret == 0:
        print(f"Commit output: {out}")
    else:
        print(f"Commit error: {err}")
    
    # Check if we need to push
    ret, out, err = run_git_command("git status")
    print(f"\nFinal status:\n{out}")

if __name__ == "__main__":
    main()