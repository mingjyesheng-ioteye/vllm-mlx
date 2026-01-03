#!/usr/bin/env python3
"""
Build script for creating a vllm-mlx binary using cx_Freeze.

cx_Freeze is a mature Python freezing tool that:
- Creates standalone executables with fast startup
- Has good support for macOS including Apple Silicon
- Produces smaller binaries than PyInstaller in many cases

Usage:
    python build_cxfreeze.py

Output:
    dist-cxfreeze/vllm-mlx-server - Standalone executable

BUILD STATUS: FAILED
=====================
Attempted: 2026-01-02

Failure reason:
    cx_Freeze fails during the dependency collection phase when processing
    the complex package dependencies (transformers, mlx, mlx_lm, mlx_vlm).
    The recursion limit was increased to 5000 but the build still fails
    due to:

    1. Deep import chains in the transformers library
    2. Native extension modules in mlx that cx_Freeze cannot properly handle
    3. Complex conditional imports that confuse the dependency scanner

    cx_Freeze is designed for simpler Python applications and struggles
    with the ML ecosystem's complex dependency graphs.

Possible workarounds (not yet attempted):
    1. Manually specify all modules instead of using package discovery
    2. Create a minimal wrapper that lazy-loads heavy dependencies
    3. Use cx_Freeze's hooks system to handle problematic imports

Recommendation:
    Use PyInstaller (build_binary.py) instead - it works successfully.
"""

import subprocess
import sys
import os
import shutil

# Increase recursion limit for complex packages like transformers
sys.setrecursionlimit(5000)


def check_cxfreeze():
    """Check if cx_Freeze is installed, install if not."""
    try:
        import cx_Freeze
        print(f"cx_Freeze {cx_Freeze.__version__} found")
        return True
    except ImportError:
        print("Installing cx_Freeze...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "cx_Freeze"
        ])
        return True


def create_setup_script():
    """Create the cx_Freeze setup script."""
    setup_content = '''
import sys
sys.setrecursionlimit(5000)
from cx_Freeze import setup, Executable

# Dependencies to include
build_exe_options = {
    "packages": [
        # Core vllm_mlx - let cx_Freeze discover submodules
        "vllm_mlx",
        # MLX ecosystem
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_lm",
        "mlx_vlm",
        # Server
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.protocols",
        "uvicorn.lifespan",
        "fastapi",
        "starlette",
        "pydantic",
        "anyio",
        # Transformers
        "transformers",
        "tokenizers",
        "huggingface_hub",
        "safetensors",
        # Utilities
        "PIL",
        "cv2",
        "numpy",
        "yaml",
        "tqdm",
        "requests",
        "psutil",
        "tabulate",
        "regex",
    ],
    "excludes": [
        # GUI frameworks (not needed)
        "PyQt5", "PyQt6", "PySide2", "PySide6",
        "tkinter", "matplotlib",
        # Heavy ML frameworks (using MLX instead)
        "torch", "torchvision", "tensorflow", "keras",
        # Development tools
        "IPython", "jupyter", "notebook", "pytest", "unittest",
        # Web UI (server only)
        "gradio",
        # Not installed
        "sentencepiece",
        # Transformers modules not needed for inference
        "transformers.testing_utils",
        "transformers.trainer",
        "transformers.training_args",
        "transformers.data",
        "transformers.benchmark",
        "transformers.cli",
        "transformers.commands",
    ],
    "include_files": [],
    "optimize": 2,  # -OO optimization
}

setup(
    name="vllm-mlx-server",
    version="0.2.0",
    description="vLLM MLX Server - GPU-accelerated LLM inference on Apple Silicon",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            script="entry_point.py",
            target_name="vllm-mlx-server",
        )
    ],
)
'''
    return setup_content


def build_cxfreeze():
    """Build with cx_Freeze."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    output_dir = "dist-cxfreeze"
    setup_file = "setup_cxfreeze.py"

    # Clean previous build
    if os.path.exists(output_dir):
        print(f"Cleaning previous build in {output_dir}...")
        shutil.rmtree(output_dir)

    build_dir = "build_cxfreeze"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    # Create setup script
    print("Creating cx_Freeze setup script...")
    with open(setup_file, "w") as f:
        f.write(create_setup_script())

    # Run cx_Freeze build
    cmd = [
        sys.executable,
        setup_file,
        "build_exe",
        f"--build-exe={output_dir}",
    ]

    print("\nRunning cx_Freeze build...")
    print("-" * 60)

    try:
        subprocess.check_call(cmd)

        # Find the output binary
        binary_path = os.path.join(output_dir, "vllm-mlx-server")

        print("\n" + "=" * 60)
        print("cx_Freeze build successful!")
        print("=" * 60)

        # Show binary size
        if os.path.exists(binary_path):
            size_mb = os.path.getsize(binary_path) / (1024 * 1024)
            print(f"\nBinary: {binary_path}")
            print(f"Size: {size_mb:.1f} MB")

        # Show total distribution size
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(output_dir)
            for filename in filenames
        )
        print(f"Total distribution size: {total_size / (1024 * 1024):.1f} MB")

        print("\nUsage:")
        print(f"  {binary_path} serve mlx-community/Qwen3-0.6B-8bit --port 8000")
        print(f"  {binary_path} bench mlx-community/Llama-3.2-1B-Instruct-4bit")

        # Clean up setup script
        os.remove(setup_file)

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n Build failed with error code {e.returncode}")
        # Clean up setup script on failure too
        if os.path.exists(setup_file):
            os.remove(setup_file)
        return False


def main():
    print("=" * 60)
    print("Building vllm-mlx with cx_Freeze")
    print("=" * 60)

    check_cxfreeze()

    success = build_cxfreeze()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()