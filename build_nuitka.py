#!/usr/bin/env python3
"""
Build script for creating a high-performance vllm-mlx binary using Nuitka.

Nuitka compiles Python to C and then to native machine code, offering:
- Faster startup times compared to PyInstaller
- Better runtime performance (actual compilation vs bundling)
- Native machine code execution

IMPORTANT: This build takes ~60-90 minutes due to:
- 5000+ C files being generated and compiled
- LTO (Link-Time Optimization) during linking

Known Issues:
- May fail at dependency scan with libb2/blake2 issues on some systems
- If build fails, try: brew install libb2

Usage:
    python build_nuitka.py [--standalone] [--onefile]

Options:
    --standalone  Create standalone distribution (default)
    --onefile     Create single-file executable (slower startup but portable)

Output:
    dist-nuitka/entry_point.app/Contents/MacOS/vllm-mlx-server

BUILD STATUS: FAILED
=====================
Attempted: 2026-01-02
Nuitka version: 2.8.9

Failure reason:
    Nuitka's transformers plugin causes a SyntaxError when processing the
    transformers library. Even with --disable-plugin=transformers, the build
    fails with:

    "transformers: Making changes to 'transformers.cli.add_new_model_like'
     that cause SyntaxError 'f-string: valid expression required before '}'
     (add_new_model_like.py, line 362)'"

    This appears to be an incompatibility between Nuitka 2.8.9 and the
    transformers package's use of complex f-strings that Nuitka cannot
    properly parse/transform.

Possible workarounds (not yet attempted):
    1. Downgrade transformers to an older version
    2. Wait for Nuitka update with better f-string handling
    3. Manually patch the transformers source before building
    4. Use --nofollow-import-to=transformers and handle imports differently

Recommendation:
    Use PyInstaller (build_binary.py) instead - it works successfully.
"""

import subprocess
import sys
import os
import shutil
import argparse


def check_nuitka():
    """Check if Nuitka is installed, install if not."""
    try:
        import nuitka
        print(f"✓ Nuitka found")
        return True
    except ImportError:
        print("Installing Nuitka...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "nuitka", "ordered-set", "zstandard"
        ])
        return True


def check_xcode():
    """Check for Xcode command line tools (required for C compilation)."""
    try:
        result = subprocess.run(
            ["xcode-select", "-p"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Xcode tools found at: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("✗ Xcode command line tools not found")
    print("  Install with: xcode-select --install")
    return False


def build_nuitka(onefile=False):
    """Build with Nuitka."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    output_dir = "dist-nuitka"

    # Clean previous build
    if os.path.exists(output_dir):
        print(f"Cleaning previous build in {output_dir}...")
        shutil.rmtree(output_dir)

    # Base Nuitka command
    cmd = [
        sys.executable, "-m", "nuitka",

        # Output configuration
        "--output-dir=" + output_dir,
        "--output-filename=vllm-mlx-server",

        # Build mode
        "--standalone" if not onefile else "--onefile",

        # pyenv compatibility
        "--static-libpython=no",

        # Performance optimizations
        "--lto=yes",                    # Link-time optimization
        "--jobs=10",                    # Use all CPU cores

        # Python optimization
        "--python-flag=no_site",        # Don't import site module
        "--python-flag=no_warnings",    # Disable warnings module
        "--python-flag=no_asserts",     # Remove assert statements
        "--python-flag=-OO",            # Optimize bytecode (remove docstrings)

        # Follow imports for bundling
        "--follow-imports",

        # Include vllm_mlx package
        "--include-package=vllm_mlx",

        # Include MLX ecosystem
        "--include-package=mlx",
        "--include-package=mlx_lm",
        "--include-package=mlx_vlm",

        # Include server dependencies
        "--include-package=uvicorn",
        "--include-package=fastapi",
        "--include-package=starlette",
        "--include-package=pydantic",
        "--include-package=anyio",

        # Include transformers ecosystem
        "--include-package=transformers",
        "--include-package=tokenizers",
        "--include-package=huggingface_hub",
        "--include-package=safetensors",

        # Include utilities
        "--include-package=PIL",
        "--include-package=cv2",
        "--include-package=numpy",
        "--include-package=yaml",
        "--include-package=tqdm",
        "--include-package=requests",
        "--include-package=psutil",
        "--include-package=tabulate",
        "--include-package=regex",

        # Include data files for packages that need them
        "--include-package-data=transformers",
        "--include-package-data=huggingface_hub",
        "--include-package-data=tokenizers",
        "--include-package-data=mlx",
        "--include-package-data=mlx_lm",
        "--include-package-data=mlx_vlm",
        "--include-package-data=vllm_mlx",

        # Exclude unnecessary modules (GUI, heavy libs)
        "--nofollow-import-to=PyQt5",
        "--nofollow-import-to=PyQt6",
        "--nofollow-import-to=PySide2",
        "--nofollow-import-to=PySide6",
        "--nofollow-import-to=tkinter",
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=jupyter",
        "--nofollow-import-to=notebook",
        "--nofollow-import-to=gradio",
        "--nofollow-import-to=torch",
        "--nofollow-import-to=torchvision",
        "--nofollow-import-to=tensorflow",
        "--nofollow-import-to=keras",
        # Exclude problematic transformers CLI modules (not needed for inference)
        "--nofollow-import-to=transformers.cli",
        "--nofollow-import-to=transformers.commands",
        "--nofollow-import-to=transformers.testing_utils",

        # Disable transformers plugin that causes SyntaxError
        "--disable-plugin=transformers",

        # macOS specific
        "--macos-create-app-bundle",    # Create proper macOS structure

        # Disable console for cleaner output (optional)
        # "--disable-console",

        # Entry point
        "entry_point.py",
    ]

    # Add onefile-specific options
    if onefile:
        cmd.extend([
            "--onefile-tempdir-spec=%TEMP%/vllm-mlx",
        ])

    print("\nRunning Nuitka compilation...")
    print("-" * 60)
    print("This may take several minutes as Nuitka compiles Python to C...")
    print("-" * 60)

    try:
        subprocess.check_call(cmd)

        # Find the output binary
        if onefile:
            binary_path = os.path.join(output_dir, "vllm-mlx-server")
        else:
            # Standalone creates a .dist folder
            dist_folder = os.path.join(output_dir, "entry_point.dist")
            binary_path = os.path.join(dist_folder, "vllm-mlx-server")

            # Also check for .app bundle on macOS
            app_bundle = os.path.join(output_dir, "entry_point.app")
            if os.path.exists(app_bundle):
                binary_path = os.path.join(app_bundle, "Contents", "MacOS", "entry_point")

        print("\n" + "=" * 60)
        print("✓ Nuitka build successful!")
        print("=" * 60)

        # Show binary size
        if os.path.exists(binary_path):
            size_mb = os.path.getsize(binary_path) / (1024 * 1024)
            print(f"\nBinary: {binary_path}")
            print(f"Size: {size_mb:.1f} MB")
        else:
            # List what was created
            print(f"\nOutput directory: {output_dir}")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    # Get directory size
                    total_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(item_path)
                        for filename in filenames
                    )
                    print(f"  {item}/ ({total_size / (1024 * 1024):.1f} MB)")
                else:
                    print(f"  {item} ({os.path.getsize(item_path) / (1024 * 1024):.1f} MB)")

        print("\nUsage:")
        print(f"  {binary_path} serve mlx-community/Qwen3-0.6B-8bit --port 8000")
        print(f"  {binary_path} bench mlx-community/Llama-3.2-1B-Instruct-4bit")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build vllm-mlx-server using Nuitka for high performance"
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Create single-file executable (slower startup but more portable)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Building vllm-mlx with Nuitka (High-Performance Build)")
    print("=" * 60)

    # Check prerequisites
    if not check_xcode():
        sys.exit(1)

    check_nuitka()

    # Build
    success = build_nuitka(onefile=args.onefile)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
