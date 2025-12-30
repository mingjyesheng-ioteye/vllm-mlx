#!/usr/bin/env python3
"""
Entry point for PyInstaller build.
This wrapper ensures proper package imports.
"""

import sys
import os

# Force unbuffered output for PyInstaller binary
# This is critical for Tauri sidecar to receive stdout/stderr
os.environ['PYTHONUNBUFFERED'] = '1'

# Ensure the package is importable
if getattr(sys, 'frozen', False):
    # Running as compiled
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, bundle_dir)

    # Force unbuffered stdout/stderr
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

from vllm_mlx.cli import main

if __name__ == "__main__":
    main()
