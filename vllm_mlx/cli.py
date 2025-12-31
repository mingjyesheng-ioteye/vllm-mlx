#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for vllm-mlx.

Commands:
    vllm-mlx serve <model> --port 8000    Start OpenAI-compatible server
    vllm-mlx bench <model>                Run benchmark

Usage:
    vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
    vllm-mlx bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10
"""

# Support for PyInstaller frozen executables with multiprocessing
# This must be at the very top, before any other imports that might use multiprocessing
import multiprocessing
multiprocessing.freeze_support()

import argparse
import sys


def serve_command(args):
    """Start the OpenAI-compatible server."""
    import uvicorn

    print(f"Loading model: {args.model}")
    print(f"Default max tokens: {args.max_tokens}")

    if args.continuous_batching:
        # Use server_v2 with continuous batching (for multiple concurrent users)
        from .server_v2 import app, load_model
        from .scheduler import SchedulerConfig

        # Handle prefix cache flags
        enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            prefill_batch_size=args.prefill_batch_size,
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
        )

        print(f"Mode: Continuous batching (for multiple concurrent users)")
        print(f"Stream interval: {args.stream_interval} tokens")
        if args.use_paged_cache:
            print(f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}")
        load_model(
            args.model,
            scheduler_config,
            stream_interval=args.stream_interval,
            max_tokens=args.max_tokens,
        )
    else:
        # Use simple server (maximum throughput for single user)
        from .server import app, load_model

        print(f"Mode: Simple (maximum throughput)")
        load_model(args.model, max_tokens=args.max_tokens)

    # Start server
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def bench_command(args):
    """Run benchmark."""
    from .benchmark import (
        is_mllm_model,
        run_mllm_benchmark,
        print_mllm_summary,
        run_video_benchmark,
        print_video_summary,
    )

    # Check if this is an MLLM model
    run_mllm = args.mllm or is_mllm_model(args.model)

    if args.video:
        # Video Benchmark (MLLM)
        results = run_video_benchmark(
            model_name=args.model,
            video_url=args.video_url,
            video_path=args.video_path,
            quick=args.quick,
            max_tokens=args.max_tokens,
            warmup_runs=1,
        )

        if results:
            print_video_summary(results, args.model)

            # Save to JSON if requested
            if args.output:
                import json
                with open(args.output, "w") as f:
                    json.dump({
                        "type": "video",
                        "model": args.model,
                        "test_video": args.video_url or args.video_path or "Big Buck Bunny 10s",
                        "results": [
                            {
                                "config_name": r.config_name,
                                "fps": r.fps,
                                "max_frames": r.max_frames,
                                "frames_extracted": r.frames_extracted,
                                "video_duration": r.video_duration,
                                "time_seconds": r.time_seconds,
                                "prompt_tokens": r.prompt_tokens,
                                "completion_tokens": r.completion_tokens,
                                "tokens_per_second": r.tokens_per_second,
                            }
                            for r in results
                        ]
                    }, f, indent=2)
                print(f"\nResults saved to: {args.output}")

    elif run_mllm:
        # MLLM Image Benchmark
        results = run_mllm_benchmark(
            model_name=args.model,
            quick=args.quick,
            max_tokens=args.max_tokens,
            warmup_runs=1,
        )

        if results:
            print_mllm_summary(results, args.model)

            # Save to JSON if requested
            if args.output:
                import json
                with open(args.output, "w") as f:
                    json.dump({
                        "type": "mllm_image",
                        "model": args.model,
                        "test_image": "Yellow Labrador (Wikimedia Commons)",
                        "results": [
                            {
                                "resolution": r.resolution,
                                "width": r.width,
                                "height": r.height,
                                "pixels": r.pixels,
                                "time_seconds": r.time_seconds,
                                "tokens_generated": r.tokens_generated,
                                "tokens_per_second": r.tokens_per_second,
                            }
                            for r in results
                        ]
                    }, f, indent=2)
                print(f"\nResults saved to: {args.output}")
    else:
        # LLM Benchmark (original implementation)
        import asyncio
        import time
        from mlx_lm import load
        from .engine import AsyncEngineCore, EngineConfig
        from .request import SamplingParams
        from .scheduler import SchedulerConfig

        # Handle prefix cache flags
        enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

        async def run_benchmark():
            print(f"Loading model: {args.model}")
            model, tokenizer = load(args.model)

            scheduler_config = SchedulerConfig(
                max_num_seqs=args.max_num_seqs,
                prefill_batch_size=args.prefill_batch_size,
                completion_batch_size=args.completion_batch_size,
                enable_prefix_cache=enable_prefix_cache,
                prefix_cache_size=args.prefix_cache_size,
                # Paged cache options
                use_paged_cache=args.use_paged_cache,
                paged_cache_block_size=args.paged_cache_block_size,
                max_cache_blocks=args.max_cache_blocks,
            )
            engine_config = EngineConfig(
                model_name=args.model,
                scheduler_config=scheduler_config,
            )

            if args.use_paged_cache:
                print(f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}")

            # Generate prompts
            prompts = [
                f"Write a short poem about {topic}."
                for topic in ["nature", "love", "technology", "space", "music",
                             "art", "science", "history", "food", "travel"][:args.num_prompts]
            ]

            params = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.7,
            )

            print(f"\nRunning benchmark with {len(prompts)} prompts, max_tokens={args.max_tokens}")
            print("-" * 50)

            total_prompt_tokens = 0
            total_completion_tokens = 0

            async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
                await asyncio.sleep(0.1)  # Warm up

                start_time = time.perf_counter()

                # Add all requests
                request_ids = []
                for prompt in prompts:
                    rid = await engine.add_request(prompt, params)
                    request_ids.append(rid)

                # Collect all outputs
                async def get_output(rid):
                    async for out in engine.stream_outputs(rid, timeout=120):
                        if out.finished:
                            return out
                    return None

                results = await asyncio.gather(*[get_output(r) for r in request_ids])

                total_time = time.perf_counter() - start_time

            # Calculate stats
            for r in results:
                if r:
                    total_prompt_tokens += r.prompt_tokens
                    total_completion_tokens += r.completion_tokens

            total_tokens = total_prompt_tokens + total_completion_tokens

            print(f"\nResults:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Prompts: {len(prompts)}")
            print(f"  Prompts/second: {len(prompts)/total_time:.2f}")
            print(f"  Total prompt tokens: {total_prompt_tokens}")
            print(f"  Total completion tokens: {total_completion_tokens}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Tokens/second: {total_completion_tokens/total_time:.2f}")
            print(f"  Throughput: {total_tokens/total_time:.2f} tok/s")

        asyncio.run(run_benchmark())


def main():
    parser = argparse.ArgumentParser(
        description="vllm-mlx: Apple Silicon MLX backend for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Server
  vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

  # LLM Benchmark
  vllm-mlx bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10

  # MLLM Image Benchmark (auto-detected)
  vllm-mlx bench mlx-community/Qwen3-VL-4B-Instruct-3bit
  vllm-mlx bench mlx-community/Qwen3-VL-4B-Instruct-3bit --quick

  # MLLM Video Benchmark
  vllm-mlx bench mlx-community/Qwen3-VL-4B-Instruct-3bit --video
  vllm-mlx bench mlx-community/Qwen3-VL-4B-Instruct-3bit --video --quick
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible server")
    serve_parser.add_argument("model", type=str, help="Model to serve")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
    )
    serve_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    serve_parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Completion batch size"
    )
    serve_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching for repeated prompts (default: enabled)",
    )
    serve_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    serve_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100)",
    )
    serve_parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=throughput)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation (default: 32768)",
    )
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Enable continuous batching for multiple concurrent users (slower for single user)",
    )
    # Paged cache options (experimental)
    serve_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    serve_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    serve_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )

    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("model", type=str, help="Model to benchmark")
    bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts (LLM only)"
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens per prompt"
    )
    bench_parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="Max concurrent sequences"
    )
    bench_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    bench_parser.add_argument(
        "--completion-batch-size", type=int, default=16, help="Completion batch size"
    )
    bench_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: enabled)",
    )
    bench_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    bench_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100)",
    )
    # Paged cache options (experimental)
    bench_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    bench_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    bench_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # MLLM benchmark options
    bench_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force MLLM benchmark mode (auto-detected by default)",
    )
    bench_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark with fewer configurations",
    )
    bench_parser.add_argument(
        "--video",
        action="store_true",
        help="Run video benchmark instead of image benchmark (for MLLM)",
    )
    bench_parser.add_argument(
        "--video-url",
        type=str,
        default=None,
        help="URL of video to use for benchmark",
    )
    bench_parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Local path to video file for benchmark",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON results",
    )

    args = parser.parse_args()

    if args.command == "serve":
        serve_command(args)
    elif args.command == "bench":
        bench_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
