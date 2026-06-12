# Scripts 目录说明

按用途分类，从项目根目录调用（示例：`bash scripts/eval/run_eval_gemini35.sh`）。

## `server/` — 服务器部署与 Judge 服务

| 子目录 | 内容 |
|--------|------|
| `server/setup/` | 环境安装：`server_setup.sh`、`server_setup_a800.sh`、`server_setup_p100.sh`、`install_*.sh`、`fix_scripts.sh` |
| `server/judge/` | 启动 Judge：`start_vllm_a800.sh`、`start_vllm_p100.sh`、`start_ollama_judge.sh`、`judge_server_transformers.py` |
| `server/config/` | `eval_config.env.example`（复制为 `eval_config.env` 后 `source`） |

**常用：**

```bash
bash scripts/server/setup/server_setup_a800.sh
source scripts/server/config/eval_config.env
bash scripts/server/judge/start_vllm_a800.sh
```

## `eval/` — 评估运行与分析

| 文件 | 用途 |
|------|------|
| `run_eval_gemini35.sh` | 35 例 Gemini + 本地 Qwen Judge 一键评估 |
| `run_judge_variance_p100.sh` | Judge 稳定性探测（P100） |
| `compare_oracle_957_eval.py` | 957 例三模型 vs 论文对比 |
| `compare_qwq_eval.py` | 同上（兼容入口） |
| `compare_gemini_35.py` | 35 例 Gemini 对比 |
| `compare_gemini_vs_paper.py` | Gemini 本地 vs 论文 |
| `analyze_reasoning_paper_957.py` | 957 例推理指标统计 |
| `analyze_completeness_gap_by_case.py` | 按病例 completeness 差距分析 |
| `judge_completeness_variance.py` | Judge 方差复测 |
| `summarize_judge_variance.py` | 方差结果汇总 |

## `inference/` — 推理辅助

| 文件 | 用途 |
|------|------|
| `run_gemini_oracle_35.py` | 重跑 35 例 Gemini oracle 诊断 |
| `prepare_gemini_outputs.py` | 嵌入格式 → 标准 oracle 格式 |

## `data/` — 数据集构建（Stage 1）

| 文件 | 用途 |
|------|------|
| `build_demo_subset.py` | 生成 demo 100 诊断 + 100 治疗子集 |
| `slice_model_outputs.py` | 从全量 oracle 输出切 demo 子集 |
