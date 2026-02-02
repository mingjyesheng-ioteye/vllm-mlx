#!/bin/bash
# vllm-mlx server startup script with multi-session support
# Concurrent HTTP connections are supported; MLX inference is sequential

MODEL="${VLLM_MODEL:-mlx-community/Meta-Llama-3.1-8B-Instruct-4bit}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

echo "Starting vllm-mlx server..."
echo "  Model: $MODEL"
echo "  Host:  $HOST"
echo "  Port:  $PORT"
echo "  Mode:  Multi-session (concurrent connections, sequential inference)"

# Run with uvicorn settings for better concurrent handling
python -m vllm_mlx.server \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL"
