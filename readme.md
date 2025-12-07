TensorRT-LLM installation
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