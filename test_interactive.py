#!/usr/bin/env python3
import subprocess
import sys
import time

# Test the interactive mode with automated input
process = subprocess.Popen(
    [sys.executable, 'hf_generate.py', 'interactive'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send commands to interactive mode
commands = [
    "y",  # Load default model
    "Hello world",  # First prompt
    "n",  # Don't customize parameters
    "Tell me a joke",  # Second prompt  
    "n",  # Don't customize parameters
    "quit"  # Exit
]

input_text = "\n".join(commands) + "\n"

try:
    stdout, stderr = process.communicate(input=input_text, timeout=180)
    print("STDOUT:")
    print(stdout)
    if stderr:
        print("STDERR:")
        print(stderr)
    print(f"Return code: {process.returncode}")
except subprocess.TimeoutExpired:
    process.kill()
    print("Process timed out")
except Exception as e:
    print(f"Error: {e}")
