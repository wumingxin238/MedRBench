# A800 单卡 + XShell 部署指南（35 例 Gemini 评估）

**机器**：1× NVIDIA A800 80GB · 32 核 CPU · 1TB SSD  
**连接**：XShell SSH（断线后用 `tmux` 保持任务）

---

## 0. XShell 连接与上传文件

### SSH 会话

1. XShell → 新建会话 → 协议 **SSH**，填服务器 IP、端口（通常 22）、用户名。
2. 用户认证：密码或私钥（`.pem` / PuTTY `.ppk` 需在 XShell 里导入）。
3. 连接后终端即为 Linux shell。

### 上传项目（任选一种）

**方式 A：Xftp（随 XShell 配套）**  
会话窗口 → 文件 → 传输 → 把整个 `MedRBench` 目录传到例如 `~/MedRBench`。

**方式 B：本机压缩后 scp（在 Windows PowerShell，非 XShell 内）**

```powershell
scp -r "E:\WMX\uni study\summer project\MedRBench - 副本" user@SERVER_IP:~/MedRBench
```

**必须存在的文件：**

- `data/MedRBench/test_cases.json`
- `src/Inference/oracle_diagnosis_gemini.json`

首次运行会从 Hugging Face 拉取 Qwen 权重（约 60GB+ for 32B），确保 `~` 或 `HF_HOME` 所在盘空间充足（1TB SSD 足够）。

---

## 1. 一次性环境安装（在 XShell 里执行）

```bash
cd ~/MedRBench
bash scripts/server/setup/server_setup_a800.sh
```

会创建 conda 环境 `medrbench`、安装 vLLM、写入 `scripts/server/config/eval_config.env`（默认 **Qwen2.5-32B** 作 Judge）。

```bash
conda activate medrbench
source scripts/server/config/eval_config.env
```

**在线搜索（可选）**：默认已关闭（`EVAL_DISABLE_WEB_SEARCH=1` / `--no-web-search`），**无需安装 Chrome**。若以后要恢复 Bing 搜索，去掉该环境变量并安装 `chromium-browser`。

---

## 2. 启动 vLLM（必须用 tmux，防止 XShell 断开杀进程）

```bash
conda activate medrbench
tmux new -s vllm
```

在 tmux 窗口内：

```bash
cd ~/MedRBench
bash scripts/server/judge/start_vllm_a800.sh
```

首次启动会下载模型，等到出现 `Uvicorn running` / `Application startup complete`。

- **脱离 tmux（任务继续跑）**：`Ctrl+B` 然后按 `D`
- **重新 attach**：`tmux attach -t vllm`

**另开一个 XShell 标签页** 测 API：

```bash
curl -s http://127.0.0.1:8000/v1/models | head
```

若 32B 显存不足（少见），改用 14B：

```bash
QWEN_VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct bash scripts/server/judge/start_vllm_a800.sh
# 并改 scripts/server/config/eval_config.env 里 EVAL_MODEL 为同一名称
```

---

## 3. 跑 35 例评估（建议再开一个 tmux）

```bash
tmux new -s eval
conda activate medrbench
cd ~/MedRBench
source scripts/server/config/eval_config.env
bash scripts/eval/run_eval_gemini35.sh
```

| 步骤 | 脚本 | 耗时（粗估） | 说明 |
|------|------|--------------|------|
| 准确率 | `oracle_diagnose_accuracy.py` | ~10–30 分钟 | 35 次 Judge 调用，无网页搜索 |
| 推理质量 | `oracle_diagnose_reasoning.py` | 1–3 小时 | 仅本地 Judge，无网页搜索 |

结果目录：

- `src/Evaluation/acc_results_qwen_judge/gemini2-ft/`
- `src/Evaluation/reasoning_results_qwen_judge/gemini2-ft/`

用 Xftp 把上述目录拉回本机分析。

---

## 4. A800 推荐参数说明

| 项 | 推荐值 | 原因 |
|----|--------|------|
| 推理框架 | **vLLM** | A800 上吞吐远高于 Ollama，适合成百上千次 Judge 调用 |
| Judge 模型 | **Qwen2.5-32B-Instruct** | 80GB 可 bf16 满血加载，评判质量更好 |
| 并发 | `--sequential` | 评估脚本已默认顺序；vLLM 侧可自行批处理 |
| `max-model-len` | 16384 | 医学 prompt + 推理步骤足够 |

环境变量（已在 `eval_config.env`）：

```bash
export EVAL_BACKEND=vllm
export EVAL_BASE_URL=http://127.0.0.1:8000/v1
export EVAL_MODEL=Qwen/Qwen2.5-32B-Instruct
```

---

## 5. 常用运维命令

```bash
nvidia-smi                    # 看显存
tmux ls                       # 列出后台会话
tmux attach -t eval           # 回到评估任务
tail -f src/Evaluation/gemini2-ft_error.log   # 推理评估错误日志
```

**端口**：Judge 只监听 `127.0.0.1:8000`，无需在 XShell 里做端口转发；若多机访问需自行改 `start_vllm_a800.sh` 的 `--host` 并注意防火墙。

---

## 6. 最小命令清单（复制用）

```bash
# === 终端 1：vLLM ===
conda activate medrbench
tmux new -s vllm
cd ~/MedRBench && bash scripts/server/judge/start_vllm_a800.sh

# === 终端 2：评估（等 vLLM 就绪后）===
conda activate medrbench
tmux new -s eval
cd ~/MedRBench && source scripts/server/config/eval_config.env && bash scripts/eval/run_eval_gemini35.sh
```

更通用的说明见 [`SERVER_EVAL_ZH.md`](SERVER_EVAL_ZH.md)。
