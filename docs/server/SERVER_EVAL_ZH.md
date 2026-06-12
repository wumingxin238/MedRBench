# 服务器上用本地 Qwen 评估 Gemini 推理结果（35 样本）

> **A800 单卡 + XShell**：见专用文档 [`SERVER_EVAL_A800_XSHELL.md`](SERVER_EVAL_A800_XSHELL.md)（含 tmux、上传、vLLM 启动命令）。

## 目标

- **被评估对象**：`gemini2-ft` 在 oracle diagnose 任务上的推理结果（35 例）
- **评判模型（Judge）**：服务器本地 **Qwen**（Ollama 或 vLLM），替代原先评估流程里的 GPT-4o API

数据文件：

| 文件 | 说明 |
|------|------|
| `data/MedRBench/test_cases.json` | 35 个病例 ground truth |
| `src/Inference/oracle_diagnosis_gemini.json` | Gemini 推理结果（嵌入格式，含 `gemini2-ft` 字段） |

## 一、服务器部署 Qwen

### 方案 A：Ollama（推荐，单卡即可）

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve   # 若未自动启动
ollama pull qwen2.5:14b-instruct
```

### 方案 B：vLLM（多卡 / 高吞吐）

```bash
pip install vllm
tmux new -s qwen
vllm serve Qwen/Qwen2.5-14B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 8192
```

### 一键环境（conda + Python 依赖 + 生成 `eval_config.env`）

```bash
cd /path/to/MedRBench
bash scripts/server/setup/server_setup.sh          # 默认 Ollama
# 或
USE_OLLAMA=0 bash scripts/server/setup/server_setup.sh   # 使用 vLLM 配置模板
```

复制并检查评判模型配置：

```bash
cp scripts/server/config/eval_config.env.example scripts/server/config/eval_config.env
# 编辑 EVAL_MODEL / EVAL_BASE_URL
source scripts/server/config/eval_config.env
curl -s $EVAL_BASE_URL/models | head
```

## 二、安装评估依赖

```bash
conda activate medrbench   # 或你的环境名
cd src/Evaluation
pip install openai selenium beautifulsoup4 fake-useragent duckduckgo-search
```

**推理质量评估**默认已 **关闭在线搜索**（`--no-web-search` 或 `export EVAL_DISABLE_WEB_SEARCH=1`），只需本地 Qwen Judge，**不需要 Chrome**。

若需恢复 Bing 搜索：去掉上述设置，并安装 `chromium-browser`、`chromium-chromedriver`。

准确率评估（`oracle_diagnose_accuracy.py`）本身也不使用网页搜索。

## 三、运行评估

```bash
source scripts/server/config/eval_config.env
bash scripts/eval/run_eval_gemini35.sh
```

或手动分步（顺序执行，适合单卡本地模型）：

```bash
cd src/Evaluation
source ../../scripts/server/config/eval_config.env

# 1) 诊断准确率
python oracle_diagnose_accuracy.py \
  --model gemini2-ft \
  --sequential \
  --embedded-outputs \
  --patient-cases ../../data/MedRBench/test_cases.json \
  --model-outputs ../Inference/oracle_diagnosis_gemini.json \
  --output-dir ./acc_results_qwen_judge

# 2) 推理效率 / 真实性 / 完整性（仅 Judge，无网页搜索）
python oracle_diagnose_reasoning.py \
  --model gemini2-ft \
  --sequential \
  --no-web-search \
  --embedded-outputs \
  --patient-cases ../../data/MedRBench/test_cases.json \
  --model-outputs ../Inference/oracle_diagnosis_gemini.json \
  --output-dir ./reasoning_results_qwen_judge
```

结果目录（与原先 GPT 评判结果分开，避免覆盖）：

- `src/Evaluation/acc_results_qwen_judge/gemini2-ft/*.json`
- `src/Evaluation/reasoning_results_qwen_judge/gemini2-ft/*.json`

## 四、环境变量说明

| 变量 | 含义 |
|------|------|
| `EVAL_BACKEND` | `ollama` / `vllm` / `openai` |
| `EVAL_BASE_URL` | OpenAI 兼容 API 地址 |
| `EVAL_API_KEY` | 本地可填 `ollama` 或 `local` |
| `EVAL_MODEL` | Ollama 模型名或 vLLM 模型 id |
| `EVAL_TEMPERATURE` | 建议 `0` |

仍用远程 GPT 评判时：不设置 `EVAL_BACKEND`，在 `src/Evaluation/metrics/gpt_key.txt` 中配置 API key。

## 五、常见问题

1. **`Total cases to evaluate: 0`**  
   该病例结果已存在于 `--output-dir` 下。删除对应 JSON 或换一个新输出目录。

2. **嵌入格式**  
   `oracle_diagnosis_gemini.json` 需加 `--embedded-outputs`；也可用  
   `python scripts/inference/prepare_gemini_outputs.py` 转为标准格式后去掉该参数。

3. **本地模型并发**  
   务必加 `--sequential`，且 reasoning 脚本默认 `NUM_WORKERS=1`，避免压垮 Ollama。

4. **已有 32/34 条旧结果**  
   在 `acc_results/gemini2-ft` 等目录是之前 GPT 评判的；用 `*_qwen_judge` 目录重新跑完整 35 条即可对比。
