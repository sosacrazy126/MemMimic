#!/usr/bin/env python3
import subprocess
import os

print("Testing execution...")
print(f"Current directory: {os.getcwd()}")

try:
    result = subprocess.run(['pwd'], capture_output=True, text=True, timeout=5)
    print(f"pwd result: {result.stdout.strip()}")
    print(f"pwd stderr: {result.stderr}")
    print(f"pwd returncode: {result.returncode}")
except Exception as e:
    print(f"Error: {e}")

try:
    result = subprocess.run(['echo', 'hello'], capture_output=True, text=True, timeout=5)
    print(f"echo result: {result.stdout.strip()}")
except Exception as e:
    print(f"Echo error: {e}")

try:
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, timeout=10)
    print(f"git status result: {result.stdout}")
    print(f"git status stderr: {result.stderr}")
except Exception as e:
    print(f"Git error: {e}")