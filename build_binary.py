#!/usr/bin/env python3
"""
Build script for creating a standalone vllm-mlx binary using PyInstaller.

Usage:
    python build_binary.py

Output:
    dist/vllm-mlx-server - Standalone executable for macOS Apple Silicon
"""

import subprocess
import sys
import os


def main():
    # Ensure we're in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("Building vllm-mlx standalone binary")
    print("=" * 60)

    # Check if PyInstaller is installed
    try:
        import PyInstaller

        print(f"✓ PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # PyInstaller command
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "vllm-mlx-server",
        # Clean build
        "--clean",
        "--noconfirm",
        # Add paths for imports
        "--paths",
        ".",
        # Hidden imports - vllm_mlx package (critical for relative imports)
        "--hidden-import=vllm_mlx",
        "--hidden-import=vllm_mlx.cli",
        "--hidden-import=vllm_mlx.server",
        "--hidden-import=vllm_mlx.server_v2",
        "--hidden-import=vllm_mlx.engine",
        "--hidden-import=vllm_mlx.scheduler",
        "--hidden-import=vllm_mlx.request",
        "--hidden-import=vllm_mlx.kv_cache",
        "--hidden-import=vllm_mlx.paged_cache",
        # Exclude conflicting Qt bindings (not needed for server)
        "--exclude-module",
        "PyQt5",
        "--exclude-module",
        "PyQt6",
        "--exclude-module",
        "PySide2",
        "--exclude-module",
        "PySide6",
        # Exclude other unnecessary GUI/heavy modules
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "IPython",
        "--exclude-module",
        "jupyter",
        "--exclude-module",
        "notebook",
        "--exclude-module",
        "gradio",
        # Hidden imports - core MLX
        "--hidden-import=mlx",
        "--hidden-import=mlx.core",
        "--hidden-import=mlx.nn",
        "--hidden-import=mlx.optimizers",
        "--hidden-import=mlx.utils",
        # Hidden imports - mlx-lm
        "--hidden-import=mlx_lm",
        "--hidden-import=mlx_lm.models",
        "--hidden-import=mlx_lm.tuner",
        "--hidden-import=mlx_lm.tokenizer_utils",
        # Hidden imports - mlx-vlm
        "--hidden-import=mlx_vlm",
        "--hidden-import=mlx_vlm.models",
        "--hidden-import=mlx_vlm.utils",
        # Hidden imports - transformers
        "--hidden-import=transformers",
        "--hidden-import=transformers.models",
        "--hidden-import=tokenizers",
        # Hidden imports - server
        "--hidden-import=uvicorn",
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan",
        "--hidden-import=uvicorn.lifespan.on",
        "--hidden-import=fastapi",
        "--hidden-import=starlette",
        "--hidden-import=pydantic",
        "--hidden-import=anyio",
        "--hidden-import=anyio._backends",
        "--hidden-import=anyio._backends._asyncio",
        # Hidden imports - utilities
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=yaml",
        "--hidden-import=tqdm",
        "--hidden-import=requests",
        "--hidden-import=huggingface_hub",
        "--hidden-import=safetensors",
        "--hidden-import=sentencepiece",
        "--hidden-import=regex",
        "--hidden-import=psutil",
        "--hidden-import=tabulate",
        # Collect all data files from these packages
        "--collect-all",
        "vllm_mlx",
        "--collect-all",
        "mlx",
        "--collect-all",
        "mlx_lm",
        "--collect-all",
        "mlx_vlm",
        "--collect-all",
        "transformers",
        "--collect-all",
        "tokenizers",
        "--collect-all",
        "huggingface_hub",
        "--collect-all",
        "safetensors",
        "--collect-all",
        "requests",
        # Collect submodules
        "--collect-submodules",
        "vllm_mlx",
        "--collect-submodules",
        "mlx",
        "--collect-submodules",
        "mlx_lm",
        "--collect-submodules",
        "mlx_vlm",
        "--collect-submodules",
        "transformers",
        "--collect-submodules",
        "uvicorn",
        "--collect-submodules",
        "fastapi",
        "--collect-submodules",
        "starlette",
        # Entry point - use wrapper script
        "entry_point.py",
    ]

    print("\nRunning PyInstaller...")
    print("-" * 60)

    try:
        subprocess.check_call(cmd)
        print("\n" + "=" * 60)
        print("✓ Build successful!")
        print("=" * 60)
        print(f"\nBinary location: {os.path.join(script_dir, 'dist', 'vllm-mlx-server')}")
        print("\nUsage:")
        print(
            "  ./dist/vllm-mlx-server serve mlx-community/Qwen3-0.6B-8bit --port 8000"
        )
        print(
            "  ./dist/vllm-mlx-server bench mlx-community/Llama-3.2-1B-Instruct-4bit"
        )
        print("\nFor Tauri, copy the binary to your app's binaries folder:")
        print("  cp dist/vllm-mlx-server /path/to/tauri-app/src-tauri/binaries/")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed with error code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
