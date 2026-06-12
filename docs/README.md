# Docs 目录说明

## `reference/` — 稳定参考文档

| 文件 | 内容 |
|------|------|
| `DATA_FIELDS_AND_EVAL_IO.md` | GT / 模型输出 / 评估结果字段说明 |

## `server/` — 服务器部署指南

| 文件 | 内容 |
|------|------|
| `SERVER_EVAL_ZH.md` | 通用：本地 Qwen Judge 评估流程 |
| `SERVER_EVAL_A800_XSHELL.md` | A800 + XShell 专用（推荐） |
| `SERVER_EVAL_P100_OLLAMA.md` | P100 限制与 Ollama 说明 |

## `reports/` — 分析报告（Markdown）

| 文件 | 内容 |
|------|------|
| `STAGE1_EXPERIMENT_REPORT.md` | **Stage 1** demo 100 例：流程、结果汇总、I/O 示例、结论与待办 |
| `EVAL_REPORT_gemini2ft_oracle_957_qwen_judge.md` | 957 例 gemini2-ft 评估报告 |
| `COMPLETENESS_GAP_CASE_ANALYSIS.md` | qwq vs gemini completeness 差距分析 |

## `artifacts/` — 脚本生成的数据文件

| 文件 | 来源脚本 |
|------|----------|
| `completeness_gap_by_case.csv` | `scripts/eval/analyze_completeness_gap_by_case.py` |
| `judge_completeness_variance.json` | `scripts/eval/judge_completeness_variance.py` |
| `judge_variance_*.json` | `scripts/eval/judge_completeness_variance.py` |
