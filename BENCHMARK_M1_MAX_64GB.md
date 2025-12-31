# vLLM-MLX Benchmark Results

## Hardware Configuration

| Spec | Value |
|------|-------|
| **Chip** | Apple M1 Max |
| **Memory** | 64 GB Unified Memory |
| **Memory Bandwidth** | ~400 GB/s |
| **GPU Cores** | 32 |
| **OS** | macOS (Darwin 24.6.0) |
| **Test Date** | December 31, 2025 |

## Benchmark Tool

All benchmarks were run using the **bundled standalone binary** (`dist/vllm-mlx-server`), which packages the entire vLLM-MLX server as a self-contained executable without requiring Python installation.

The binary supports:
- **LLM benchmarks** - Text generation performance
- **MLLM Image benchmarks** - Vision-language models with various image resolutions
- **MLLM Video benchmarks** - Vision-language models with various frame configurations

```bash
# LLM benchmark
./dist/vllm-mlx-server bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10 --max-tokens 256

# MLLM Image benchmark (auto-detected for VL models)
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit --quick

# MLLM Video benchmark
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit --video --quick
```

---

## LLM Benchmark Results

### Text Generation Performance

| Model | Parameters | Quantization | Tokens/sec | Throughput | Prompts/sec |
|-------|------------|--------------|------------|------------|-------------|
| **Qwen3-0.6B-8bit** | 0.6B | 8-bit | 330.4 | 339.4 tok/s | 1.29 |
| **Llama-3.2-1B-Instruct-4bit** | 1.2B | 4-bit | 253.5 | 264.3 tok/s | 1.35 |
| **Qwen2.5-1.5B-Instruct-4bit** | 1.5B | 4-bit | 258.8 | 265.9 tok/s | 1.01 |
| **Phi-3.5-mini-instruct-4bit** | 3.8B | 4-bit | 107.2 | 110.1 tok/s | 0.42 |
| **Llama-3.2-3B-Instruct-4bit** | 3.2B | 4-bit | 113.0 | 117.1 tok/s | 0.52 |
| **Mistral-7B-Instruct-v0.3-4bit** | 7.2B | 4-bit | 55.6 | 57.6 tok/s | 0.26 |
| **Llama-3.1-8B-Instruct-4bit** | 8.0B | 4-bit | 42.1 | 43.5 tok/s | 0.17 |
| **Qwen3-30B-A3B-4bit** (MoE) | 30B (3B active) | 4-bit | 49.6 | 51.0 tok/s | 0.19 |

### Benchmark Details

- **Test Configuration**: 10 prompts, 256 max tokens per prompt (5 prompts for larger models)
- **Tokens/sec**: Output tokens generated per second
- **Throughput**: Total tokens (input + output) processed per second
- **Prompts/sec**: Complete request processing rate

### Performance by Model Size

```
Tokens/sec vs Model Size (M1 Max 64GB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Qwen3-0.6B      ████████████████████████████████████████████████  330 tok/s
Qwen2.5-1.5B    ██████████████████████████████████████             259 tok/s
Llama-3.2-1B    █████████████████████████████████████              254 tok/s
Llama-3.2-3B    ████████████████                                   113 tok/s
Phi-3.5-mini    ███████████████                                    107 tok/s
Mistral-7B      ████████                                            56 tok/s
Qwen3-30B-A3B   ███████                                             50 tok/s
Llama-3.1-8B    ██████                                              42 tok/s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## MLLM (Vision-Language) Benchmark Results

### Model: Qwen3-VL-4B-Instruct-3bit

The bundled binary now supports MLLM benchmarks for both image and video processing.

```bash
# MLLM Image Benchmark (auto-detected for VL models)
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit --quick

# MLLM Video Benchmark
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit --video --quick
```

### Image Processing Performance

| Resolution | Pixels | Time | Tokens | Speed | Memory |
|------------|--------|------|--------|-------|--------|
| 224x224 | 50K | 3.83s | 143 | 37.4 tok/s | 2.64 GB |
| 448x448 | 201K | 4.20s | 126 | 30.0 tok/s | 3.14 GB |
| 768x768 | 590K | 5.82s | 103 | 17.7 tok/s | 3.42 GB |
| 1024x1024 | 1.0M | 8.42s | 101 | 12.0 tok/s | 3.60 GB |

**Summary:**
- **Average Speed**: 21.2 tok/s across all resolutions
- **Peak Memory**: 3.60 GB
- **Fastest**: 224x224 (37.4 tok/s)
- **Slowest**: 1024x1024 (12.0 tok/s)
- **Slowdown Factor**: 2.2x from smallest to largest resolution

### Video Processing Performance

| Configuration | Frames | Time | Tokens | Speed | Memory |
|--------------|--------|------|--------|-------|--------|
| 4 frames @ 1fps | 4 | 11.38s | 256 | 22.5 tok/s | 3.54 GB |
| 8 frames @ 2fps | 8 | 16.68s | 256 | 15.4 tok/s | 4.00 GB |
| 16 frames @ 2fps | 16 | 27.68s | 256 | 9.2 tok/s | 4.67 GB |

**Summary:**
- **Average Speed**: 13.8 tok/s across all configurations
- **Peak Memory**: 4.67 GB
- **Fastest**: 4 frames (22.5 tok/s)
- **Slowest**: 16 frames (9.2 tok/s)

---

## Memory Usage Analysis

### LLM Models

| Model | Estimated VRAM* |
|-------|-----------------|
| Qwen3-0.6B-8bit | ~0.8 GB |
| Llama-3.2-1B-Instruct-4bit | ~0.7 GB |
| Qwen2.5-1.5B-Instruct-4bit | ~0.9 GB |
| Llama-3.2-3B-Instruct-4bit | ~1.8 GB |
| Phi-3.5-mini-instruct-4bit | ~2.2 GB |
| Mistral-7B-Instruct-v0.3-4bit | ~4.2 GB |
| Llama-3.1-8B-Instruct-4bit | ~4.8 GB |
| Qwen3-30B-A3B-4bit (MoE) | ~16 GB |

*Estimated based on model size and quantization. Actual usage varies with context length and KV cache.

### VLM Models

| Model | Base Memory | With 1024x1024 Image | With 16 Video Frames |
|-------|-------------|---------------------|---------------------|
| Qwen3-VL-4B-Instruct-3bit | ~2.5 GB | ~3.6 GB | ~4.7 GB |

---

## Comparison with M4 Max 128GB

Based on the benchmark results in the main README, here's how M1 Max 64GB compares to M4 Max 128GB:

| Model | M1 Max 64GB | M4 Max 128GB | Ratio |
|-------|-------------|--------------|-------|
| Qwen3-0.6B-8bit | 330.4 tok/s | 395.4 tok/s | 0.84x |
| Llama-3.2-1B-Instruct-4bit | 253.5 tok/s | 463.4 tok/s | 0.55x |
| Llama-3.2-3B-Instruct-4bit | 113.0 tok/s | 200.1 tok/s | 0.56x |
| Qwen3-30B-A3B-4bit | 49.6 tok/s | 123.9 tok/s | 0.40x |

**Analysis:**
- M1 Max achieves approximately **40-85%** of M4 Max performance depending on model size
- Smaller models (0.6B-1.5B) show better relative performance due to lower memory bandwidth requirements
- Larger models (7B+) are more bandwidth-bound and show greater performance gap
- The M4 Max's higher memory bandwidth (~546 GB/s vs ~400 GB/s) provides significant advantage for larger models

---

## Recommendations for M1 Max 64GB

### Best Models for Interactive Use (>100 tok/s)
- **Qwen3-0.6B-8bit** - Best balance of speed and capability
- **Llama-3.2-1B-Instruct-4bit** - Fast with good instruction following
- **Qwen2.5-1.5B-Instruct-4bit** - Excellent multilingual support
- **Phi-3.5-mini-instruct-4bit** - Great reasoning for its size
- **Llama-3.2-3B-Instruct-4bit** - Solid all-around performance

### Best Models for Quality (Lower Priority on Speed)
- **Mistral-7B-Instruct-v0.3-4bit** - Strong 7B performance at ~56 tok/s
- **Llama-3.1-8B-Instruct-4bit** - Excellent capabilities at ~42 tok/s
- **Qwen3-30B-A3B-4bit** - MoE model with 30B parameters at ~50 tok/s

### Vision-Language Tasks
- **Qwen3-VL-4B-Instruct-3bit** - Best VLM option with good balance
- Use 224x224 to 448x448 for best speed (30-37 tok/s)
- Use 768x768+ for detailed image analysis (12-18 tok/s)
- Video: 4-8 frames optimal for most tasks (15-22 tok/s)

---

## Running Your Own Benchmarks

### Using the Bundled Binary

The bundled binary supports LLM, MLLM image, and MLLM video benchmarks without requiring Python installation.

```bash
# LLM Benchmark
./dist/vllm-mlx-server bench mlx-community/Llama-3.2-1B-Instruct-4bit \
    --num-prompts 10 \
    --max-tokens 256

# With more prompts for accurate measurements
./dist/vllm-mlx-server bench mlx-community/Qwen3-0.6B-8bit \
    --num-prompts 20 \
    --max-tokens 512

# MLLM Image Benchmark (auto-detected for VL models)
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit --quick

# MLLM Image Benchmark (full - 10 resolutions)
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit

# MLLM Video Benchmark (quick - 3 configurations)
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit --video --quick

# MLLM Video Benchmark (full - 8 configurations)
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit --video

# Save results to JSON
./dist/vllm-mlx-server bench mlx-community/Qwen3-VL-4B-Instruct-3bit --quick --output results.json
```

### Benchmark CLI Options

| Option | Description |
|--------|-------------|
| `--num-prompts N` | Number of prompts for LLM benchmark (default: 10) |
| `--max-tokens N` | Max tokens per generation (default: 256) |
| `--mllm` | Force MLLM mode (auto-detected for VL models) |
| `--quick` | Quick benchmark with fewer configurations |
| `--video` | Run video benchmark instead of image |
| `--video-url URL` | Custom video URL for benchmark |
| `--video-path PATH` | Local video file for benchmark |
| `--output FILE` | Save results to JSON file |

---

## Notes

1. **Memory Bandwidth**: LLM inference on Apple Silicon is largely memory bandwidth-bound. The M1 Max's ~400 GB/s provides excellent performance for models up to ~8B parameters.

2. **Quantization Impact**: 4-bit quantization reduces memory requirements by ~75% compared to FP16, enabling larger models to run efficiently.

3. **Unified Memory**: Apple's unified memory architecture eliminates CPU-GPU data transfers, providing latency benefits for inference workloads.

4. **MoE Models**: Mixture-of-Experts models like Qwen3-30B-A3B activate only a fraction of parameters per token, offering better quality-to-speed ratios.

5. **VLM Memory Scaling**: Vision-language models use additional memory proportional to image resolution and video frame count.

---

## Methodology

- **Warm-up**: 1 warm-up run before measurements
- **Averaging**: Results from multiple prompts averaged
- **Prompts**: Standard benchmark prompts testing various capabilities
- **Max Tokens**: 256 tokens per generation (consistent across tests)
- **Temperature**: Default settings (typically 0.7)
- **No Prefix Cache**: Tests run without prefix caching for consistent baseline

---

*Generated on December 31, 2025 using vLLM-MLX bundled binary*
