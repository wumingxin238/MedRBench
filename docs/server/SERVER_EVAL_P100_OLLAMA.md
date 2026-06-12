# P100 + Python 3.6 环境说明

- **vLLM**：需要 Python ≥3.8，且 GPU 算力 ≥ sm_70（V100/A100 等）。**P100 (sm_60) 不能用 vLLM**。
- **系统 pip 默认 Python 3.6** 时，必须用 **conda 环境 Python 3.10** 跑评估脚本。

推荐：**Ollama + qwen2.5:14b-instruct** 作 Judge。
