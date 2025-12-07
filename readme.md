# 🚀 TensorRT-LLM Deployment on B200 GPUs

<div align="center">

![NVIDIA](https://img.shields.io/badge/NVIDIA-AI-blue?style=for-the-badge&logo=nvidia)
![TensorRT](https://img.shields.io/badge/TensorRT--LLM-2.0-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Containers-blue?style=for-the-badge&logo=docker)

**⚡ Production-Ready LLM Deployment on 8x NVIDIA B200 GPUs ⚡**

*Optimized for GPT-OSS-120B with 2-10x faster inference than PyTorch*

[📖 Quick Start](#-one-command-deployment-recommended) • [🔧 Manual Setup](#manual-deployment) • [📊 Benchmarks](#benchmarking) • [🐛 Troubleshooting](#troubleshooting)

</div>

---

## 📋 Table of Contents

- [🚀 One-Command Deployment (Recommended)](#-one-command-deployment-recommended)
- [🔧 Manual Deployment](#manual-deployment)
- [⚙️ Configuration](#configuration)
- [🏗️ Architecture](#architecture)
- [📜 Scripts Overview](#scripts-overview)
- [📁 Directory Structure](#directory-structure)
- [🤖 Model Configuration](#model-configuration)
- [⚡ Performance Tuning](#performance-tuning)
- [📊 Benchmarking](#benchmarking)
- [🔍 Troubleshooting](#troubleshooting)
- [💻 API Usage Examples](#api-usage-examples)
- [🤝 Contributing](#contributing)

## Configuration

### Environment Setup
```bash
# 1. Copy and edit configuration
cp .env.example .env
# Edit with your NGC API key and preferences

# 2. Setup standard directory structure
bash scripts/setup/setup_directories.sh

# 3. Auto-download model (if configured)
bash scripts/download/download_model_auto.sh
```

### Key Configuration Options
- `NGC_API_KEY`: Your NVIDIA NGC API key
- `MODEL_NAME`: Model to deploy (default: nvidia/gpt-oss-120b)
- `FRAMEWORK`: tensorrt-llm or vllm
- `AUTO_DOWNLOAD_MODEL`: true/false for automatic Hugging Face downloads
- `PORT`: Server port (default: 8000)

### 🚀 One-Command Deployment (Recommended)

**✨ The easiest way to deploy is using the interactive `main.sh` script:**

```bash
🎯 bash main.sh
```

**What it does:**
- 🔧 **Guides** you through configuration setup
- 🎛️ **Lets you choose** between TensorRT-LLM or vLLM
- ⚙️ **Handles all setup** steps automatically
- 🚀 **Starts your LLM server** with one command
- 🧪 **Provides testing** and benchmarking options

### 🎮 Interactive Menu Options

1. **🚀 Full deployment** - Complete setup and deployment in one go
2. **🔧 Setup only** - Configure environment without starting server
3. **▶️ Deploy only** - Start server using existing setup
4. **🧪 Test only** - Test a running server
5. **📊 Benchmark only** - Run performance benchmarks

### 🎯 Why Use main.sh?

- ✅ **Beginner-friendly** - No need to remember commands
- ✅ **Error handling** - Guides you through issues
- ✅ **Framework choice** - Easy switching between TensorRT-LLM and vLLM
- ✅ **Status updates** - Real-time progress with emojis
- ✅ **One command** to live LLM! 🎉

### 🔧 Manual Deployment

**If you prefer manual control, follow these steps:**

#### 📋 Prerequisites
- 🖥️ **Access** to compute node: `exp-blr-dgxb200-01`
- 🔑 **NGC API key** from https://ngc.nvidia.com/setup/api-key
- 🔐 **SSH access** to the cluster

#### 🎯 Choose Your Framework

**🏆 Option 1: TensorRT-LLM (Recommended - 2-10x faster)**
```bash
# Setup TensorRT-LLM container
bash deploy/02_setup_container.sh tensorrt-llm
```

**🔄 Option 2: vLLM (Easier setup, more flexible)**
```bash
# Setup vLLM container
bash deploy/02_setup_container.sh vllm
```

#### 🚀 Deploy GPT-OSS-120B

**⭐ Option 1: Setup Both Frameworks (Recommended)**
```bash
# One-time setup for both containers
bash scripts/setup/setup_both_frameworks.sh
```

**🎯 Option 2: Setup Individual Frameworks**
```bash
# Setup only TensorRT-LLM
bash deploy/02_setup_container.sh tensorrt-llm

# Setup only vLLM
bash deploy/02_setup_container.sh vllm
```

#### ▶️ Run Frameworks One by One

```bash
# 1. SSH to compute node
ssh exp-blr-dgxb200-01

# 2. Verify hardware
bash deploy/01_verify_hardware.sh

# 3. Start container (auto-detects available container)
bash deploy/03_start_container.sh

# 4. Inside container - Choose your framework:
# For TensorRT-LLM (2-6 hours engine building)
bash /workspace/deploy/04_deploy_tensorrt_llm.sh

# For vLLM (much faster startup)
bash /workspace/deploy/04_deploy_vllm.sh
```

#### 🔄 Switch Between Frameworks

**To switch between frameworks, exit the current container and start the other one:**

```bash
# Exit current container (type 'exit' or Ctrl+D)
exit

# Start the other container
bash deploy/03_start_container.sh tensorrt-llm  # or vllm-pytorch
```

## 🏗️ Architecture

### 🎯 Framework Options

#### 🏆 TensorRT-LLM (Recommended)
- ⚡ **Performance:** 2-10x faster inference than PyTorch/vLLM
- 🧠 **Memory Efficiency:** Optimized KV cache management with PagedAttention
- 🔗 **Multi-GPU:** Native support for tensor parallelism across 8 GPUs
- 🏢 **Production Ready:** NVIDIA's enterprise-grade inference framework
- ⏱️ **Setup Time:** 2-6 hours for engine building, then very fast inference

#### 🔄 vLLM
- 📈 **Performance:** Good performance with PagedAttention
- 🎛️ **Flexibility:** Supports more model architectures and configurations
- 🚀 **Ease of Use:** Much faster setup and startup times
- 🌐 **Community:** Large open-source community and active development
- ⚡ **Setup Time:** Minutes to install and start serving

### 💻 Hardware Configuration
- 🎮 **GPUs:** 8x NVIDIA B200 (183GB HBM3e each)
- 🌐 **Interconnect:** NVSwitch for high-speed GPU-to-GPU communication
- 💾 **Memory:** 1.4TB total VRAM
- 🐍 **CUDA:** Version 13.0
- 🐧 **OS:** Ubuntu 24.04

## 📜 Scripts Overview

| 🎯 Script | 📝 Purpose |
|-----------|------------|
| `main.sh` | 🎮 **Interactive deployment script (recommended)** |
| `.env` | ⚙️ **Configuration file (created from .env.example)** |
| `.env.example` | 📋 **Configuration template** |
| `scripts/setup/load_config.sh` | 🔄 **Load .env configuration** |
| `scripts/setup/setup_directories.sh` | 📁 **Create standard directory structure** |
| `scripts/download/download_model_auto.sh` | ⬇️ **Auto-download models from Hugging Face** |
| `scripts/setup/setup_both_frameworks.sh` | 🔧 **Setup both TensorRT-LLM and vLLM containers** |
| `deploy/01_verify_hardware.sh` | ✅ **Verify 8x B200 GPUs and NVSwitch connectivity** |
| `deploy/02_setup_container.sh` | 🐳 **Setup individual Enroot container (TensorRT-LLM or vLLM)** |
| `deploy/03_start_container.sh` | ▶️ **Start container with GPU access and workspace mounting** |
| `deploy/04_deploy_tensorrt_llm.sh` | 🚀 **Build TensorRT engine and start OpenAI-compatible API server** |
| `deploy/04_deploy_vllm.sh` | ⚡ **Install vLLM and start OpenAI-compatible API server** |
| `scripts/test/05_test_server.sh` | 🧪 **Comprehensive vLLM server testing** |
| `scripts/test/quick_test_server.sh` | ⚡ **Quick TensorRT-LLM server health check** |
| `scripts/benchmark/benchmark_tensorrt_llm.py` | 📊 **Comprehensive performance benchmarking** |
| `scripts/benchmark/install_benchmark_deps.sh` | 📦 **Install benchmarking dependencies** |
| `scripts/benchmark/requirements_benchmark.txt` | 🐍 **Python dependencies for benchmarking** |
| `scripts/benchmark/BENCHMARK_README.md` | 📖 **Detailed benchmarking guide** |

## 📁 Directory Structure

**After running `setup_directories.sh`, you'll have this standard structure:**

```
workspace/
├── ⚙️ .env                    # Configuration file
├── 📝 logs/                   # All log files
│   ├── 🌐 server/            # Server logs
│   ├── 📊 benchmark/         # Benchmark logs
│   └── 🔧 setup/             # Setup logs
├── ⚙️ config/                # Configuration files
├── 💾 data/                  # Data files
├── 🤖 models/                # Model files
│   └── nvidia--gpt-oss-120b/
│       ├── 💾 checkpoints/   # Model checkpoints
│       ├── ⚙️ config/        # Model config
│       └── 🔤 tokenizer/     # Tokenizer files
├── ⚙️ engines/               # TensorRT engines
│   └── nvidia--gpt-oss-120b/
│       ├── 🔢 fp16/          # FP16 engines
│       └── 🔢 int8/          # INT8 engines
└── 📊 benchmarks/            # Benchmark results
    ├── 📈 results/           # Test results
    ├── ⚙️ configs/           # Benchmark configs
    └── 📊 plots/             # Performance plots
```

## 🤖 Model Configuration

### 🎯 GPT-OSS-120B Setup
- 🤖 **Model:** `nvidia/gpt-oss-120b`
- 🔢 **Precision:** FP16 (optimized for B200 GPUs)
- 🔗 **Tensor Parallelism:** 8 GPUs
- 🧠 **Paged Attention:** Enabled for memory efficiency
- 📏 **Max Sequence Length:** 4096 tokens
- 💾 **Storage:** ~183GB downloaded model

### 🎨 Supported Models
**TensorRT-LLM supports major model architectures:**
- 🧠 GPT-style models (GPT-2, GPT-J, GPT-NeoX)
- 🦙 LLaMA models (LLaMA 2, LLaMA 3, Code Llama)
- 🦅 Falcon models
- 🐦 MPT models
- ➕ And more...

**Full list:** [TensorRT-LLM Supported Models](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples)

### ⬇️ Model Download Process

**⭐ Automatic (Recommended):**
- 🤖 TensorRT-LLM automatically downloads models from Hugging Face during engine building
- 💾 Models are cached at `~/.cache/huggingface/hub/`
- 🔄 Subsequent builds use the cached version

**🔧 Manual Pre-download:**
```bash
# Inside the container
export MODEL="nvidia/gpt-oss-120b"
bash /workspace/download_model.sh
```

### ⏱️ Download Time Estimates:
- 🤖 **GPT-OSS-120B:** 30-60 minutes (183GB)
- 🦙 **Llama-3-70B:** 15-30 minutes
- 🦙 **Llama-2-7B:** 5-10 minutes

## ⚡ Performance Tuning

### 🔧 TensorRT Engine Configuration
```bash
# Inside container - GPT-OSS-120B optimized settings
export MODEL_NAME="nvidia/gpt-oss-120b"
export TENSOR_PARALLEL_SIZE=8
export PIPELINE_PARALLEL_SIZE=1
export PRECISION="float16"
export MAX_BATCH_SIZE=8
export MAX_INPUT_LEN=4096
export MAX_OUTPUT_LEN=1024

# Build engine with paged attention
trtllm-build --checkpoint_dir /workspace/models/${MODEL_NAME} \
             --output_dir /workspace/engines/${MODEL_NAME} \
             --gemm_plugin float16 \
             --paged_kv_cache enable \
             --max_batch_size ${MAX_BATCH_SIZE} \
             --max_input_len ${MAX_INPUT_LEN} \
             --max_output_len ${MAX_OUTPUT_LEN}
```

### 🚀 Multi-GPU Optimization
- 🔗 **Tensor Parallelism:** Distributes model weights across 8 GPUs
- 📦 **Pipeline Parallelism:** Splits model layers across GPUs
- 🧠 **Paged KV Cache:** Efficient memory management for long contexts
- 🔄 **In-flight Batching:** Concurrent request processing

## Container Management

### List Containers
```bash
enroot list
```

### Remove Container
```bash
enroot remove tensorrt-llm
```

### Recreate Container
```bash
bash deploy/02_setup_container.sh
```

## 📊 Benchmarking

**Run benchmarks after starting the server:**

```bash
# Quick server test (recommended first)
bash scripts/test/quick_test_server.sh

# Full comprehensive benchmark
python3 scripts/benchmark/benchmark_tensorrt_llm.py
```

**This will test:**
- 🚀 **Token generation throughput**
- ⚡ **Latency measurements**
- 🔄 **Concurrent request handling**
- 💾 **Memory utilization**

### 🔬 Advanced Benchmarking

**For production benchmarking, use NVIDIA tools:**

- 📈 **GenAI Perf Analyzer:** Inference server performance
- 🧮 **TensorRT-LLM Bench:** Engine-level performance metrics

**📚 References:**
- [TensorRT-LLM Benchmarking](https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks)
- [GenAI Perf Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)

## 🔍 Troubleshooting

### 🐳 Container Creation Issues

```bash
# Check NGC credentials
cat ~/.config/enroot/.credentials

# Verify NGC API key
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.ngc.nvidia.com/v2/org/nvidia/containers/tensorrt_llm
```

### 🎮 GPU Not Detected Inside Container

```bash
# Verify GPUs are visible
nvidia-smi

# Check CUDA installation
nvcc --version
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 🔧 Engine Building Fails

```bash
# Check available disk space (need ~400GB for engine)
df -h /workspace

# Verify model download completed
ls -la /workspace/models/nvidia/gpt-oss-120b/

# Check TensorRT-LLM installation
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

### 💾 Out of Memory During Engine Build

```bash
# Reduce batch size or sequence length
export MAX_BATCH_SIZE=4
export MAX_INPUT_LEN=2048
export MAX_OUTPUT_LEN=512

# Or use fewer GPUs for building (then load on all 8)
export TENSOR_PARALLEL_SIZE=4
```

### 🚫 Server Won't Start

```bash
# Check if port is already in use
netstat -tuln | grep 8000

# Use a different port
export PORT=8001
python3 -m tensorrt_llm.server --model_path /workspace/engines/nvidia/gpt-oss-120b --port ${PORT}
```

## Files

| File | Purpose |
|------|---------|
| `01_verify_hardware.sh` | Verify 8x B200 GPUs and NVSwitch |
| `02_setup_container.sh` | Setup Enroot container with NGC auth |
| `03_start_container.sh` | Start container with GPU access |
| `04_deploy_tensorrt_llm.sh` | Build TensorRT engine and start server |
| `05_test_server.sh` | Test the running server |
| `download_model.sh` | Pre-download models |
| `benchmark_tensorrt_llm.py` | Comprehensive performance benchmarking |
| `quick_test_server.sh` | Quick server health check |
| `install_benchmark_deps.sh` | Install benchmarking dependencies |
| `requirements_benchmark.txt` | Python dependencies for benchmarking |
| `BENCHMARK_README.md` | Detailed benchmarking guide |

## Workflow Summary

### One-Time Setup (Do This First)
```bash
# Setup both containers (recommended)
bash scripts/setup/setup_both_frameworks.sh

# Or setup individually
bash deploy/02_setup_container.sh tensorrt-llm
bash deploy/02_setup_container.sh vllm
```

## 💻 API Usage Examples

### 🐍 Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "nvidia/gpt-oss-120b",
        "prompt": "Write a Python function to calculate fibonacci:",
        "max_tokens": 200,
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["text"])
```

### 🌐 cURL

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/gpt-oss-120b",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

## 🤝 Contributing

**We welcome contributions!** 🚀

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. 🔧 **Make** your changes
4. ✅ **Test** thoroughly on B200 hardware
5. 📤 **Submit** a pull request

### 📋 Development Guidelines
- 🧪 Test all changes on actual B200 hardware
- 📚 Update documentation for any new features
- 🔄 Follow existing code style and patterns
- 🏷️ Use clear commit messages

## 📄 License

**This project is licensed under the Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.

## 👥 Authors



## 🆘 Support

**For issues and questions:**
- 🐛 **Create an issue** in this repository
- 👨‍💼 **Contact** the cluster administrators
- 📖 **Check** NVIDIA documentation for TensorRT-LLM and vLLM

---

<div align="center">

**Made with ❤️ by the NVIDIA AI Team**

[📖 TensorRT-LLM Docs](https://github.com/NVIDIA/TensorRT-LLM) • [🌐 NGC Catalog](https://catalog.ngc.nvidia.com/containers) • [🤖 Supported Models](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples)

</div>
