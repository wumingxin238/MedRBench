# Stage 2 实验报告：Qwen3-14B-thinking × 400

> **数据**：100 demo + 300 hard · **模型**：qwen3-14b-thinking（本地 fp16）  
> **Judge**：Gemma-9B（reasoning）· GPT-4o（accuracy）· **状态**：400/400 完成（2026-06-24）  
> 详表/复现：`python scripts/stage1/analyze_stage2_results.py`

## 目录

- [400 例总览（主结果）](#sec-summary)
- [1. 实验概要](#sec-1)
- [2. 分项解读](#sec-2)
- [3. 与 Stage-1 / 论文](#sec-3)
- [4. 后续方向](#sec-4)
- [5. 产出索引](#sec-5)

---

<a id="sec-summary"></a>
## 400 例总览（主结果）

**Subject**：qwen3-14b-thinking · **任务**：oracle diagnosis · **n = 400**

### Outcome：Diagnosis Accuracy（GPT-4o）

| 指标 | 400 总计 |
|------|----------|
| 正确 / 总数 | **311 / 400** |
| **Accuracy** | **77.8%** |

### Process：Gemma-9B reasoning_eval（400 例均值）

| 组别 | Efficiency | Factuality | Completeness |
|------|------------|------------|--------------|
| **direct** | **97.8%** | 93.2% | 89.2% |
| **inference_augmented** | 88.4% | 91.8% | **91.0%** |
| **aug − direct** | **−8.7 pp** | −1.4 pp | **+2.4 pp** |

### Accuracy × reasoning（400 例，按 GPT-4o 对错分层）

| 组别 | Rec 差（对−错） | Eff 差（对−错） | acc×Rec 相关 r |
|------|----------------|----------------|----------------|
| direct | **+9.0 pp** | +1.1 pp | **0.26** |
| aug | +6.6 pp | **+4.7 pp** | 0.21 |

**三句话结论**：

1. **77.8%** 是 400 混合集 headline；hard 子集把 Acc 从 demo 的 91% 拉到 73%。
2. Gemma 三指标在 400 上 **不反映 case 难度**（demo≈hard），与 Acc **不同步**。
3. **14B 比 8B 更准（demo +5 pp Acc），但 Gemma Rec 更低（−7 pp）**——outcome 与 process 仍背离。

---

<a id="sec-1"></a>
## 1. 实验概要

| 项 | 内容 |
|----|------|
| 目的 | Stage-1 demo 100 上复用 strong 输出；本地跑 **qwen3-14b-thinking** 全 400；Gemma-9B 评 reasoning；GPT-4o 评 Acc |
| 数据 | `diagnosis_400.json` = demo 100 + hard 300（seed=42，hard 按 Stage-1 难度加权抽样） |
| 流水线 | Qwen 4 卡 infer → Gemma 3 卡 eval → GPT API acc（`run_stage2_a800_parallel.sh`） |
| 两组 | **direct** = 仅推理步；**aug** = 推理步 + `Final model inference: {答案}`（**不改变 Acc**） |

---

<a id="sec-2"></a>
## 2. 分项解读

### 2.1 Accuracy：400 内部分解

| 子集 | Acc | 说明 |
|------|-----|------|
| **400 总计** | **77.8%** | 主报告数字 |
| Demo 100 | 91.0% | 与 Stage-1 同批 case |
| Hard 300 | 73.3% | 89 错例中 **80 例（90%）** 来自此处 |

### 2.2 Gemma：为何 demo / hard 几乎一样？

400 上 direct 的 Eff/Fact/Rec 在 demo 与 hard 间 **差 < 1 pp**；Acc 却差 **17.7 pp**。  
→ **process 分数不能代替 outcome**；读 77.8% 时必须知道含 75% hard case。

### 2.3 aug 效应（400 paired）

| 现象 | 原因（简述） |
|------|--------------|
| Eff **−8.7 pp** | 多一步 Final inference，Gemma 常标冗余（201/400 单例 Eff 降 ≥10 pp） |
| Rec **+2.4 pp** | 显式最终诊断更易 hit GT 结论段 |
| 猜对错 | **direct** 靠 Rec/Fact；**aug** 靠 Eff 略强、Rec 略弱——**评对错仍看 Acc** |

### 2.4 错例（89 例）

- **多数**：Rec 中等偏高但诊断错（仅 6 例 Rec < 0.5）。
- **典型**：PMC11489070 等——direct Rec=1.0 仍错（推理满分工结论偏）。

---

<a id="sec-3"></a>
## 3. 与 Stage-1 / 论文

### Demo 100 对照（同 Judge、同 case）

| 模型 | Acc |
|------|-----|
| deepseek-r1 | 95% |
| o3-mini | 92% |
| **qwen3-14b-thinking** | **91%** |
| qwen3-8b | 86% |

| 对比 | Acc | Gemma Rec（direct） |
|------|-----|---------------------|
| 14B vs 8B（demo） | **+5 pp** | **−7 pp** |

### 论文 Oracle Acc（957 例，GPT-4o，**不同模型/全集**）

| 模型 | 论文 Acc |
|------|----------|
| DeepSeek-R1 | 89.8% |
| Gemini-2.0-FT | 86.8% |
| QwQ | 85.1% |

**可比性**：论文无 Qwen3-14B；我们的 **91%（demo）** 落在论文 oracle 中段，**77.8%（400）** 因 hard 加权 **不宜**直接对标 957 headline。

---

<a id="sec-4"></a>
## 4. 后续方向

按 **投入 / 价值** 排序：

### A. 补全同 setting 对比（推荐，成本低）

| 任务 | 做法 | 预期产出 |
|------|------|----------|
| Strong 在 **同一 400** 上的 Acc | 对 o3 / deepseek 跑 `run_stage2_diagnosis_accuracy.py`（subjects 已有） | 400 上 Strong vs Weak 完整排序 |
| Hard 300 单独报告 | 三模型仅 hard 子集 Acc + Gemma 分层 | 验证 hard 是否 uniformly 难 |

→ 回答：**14B 在难例上离 strong 还有多远**。

### B. 错例与 mechanism（推荐，写论文用）

| 任务 | 做法 |
|------|------|
| 89 错例分型 | 「高 Rec 错结论」vs「低 Rec 错」case study（各 5–10 例） |
| thinking 链分析 | 14B thinking 长度 / 结构 vs 8B，解释 Rec↓ Acc↑ |
| aug 消融 | 已有数据；可补充 Final 步被标 Redundancy 的比例 |

→ 回答：**为何 process 与 outcome 背离**（Stage-1/2 核心 story）。

### C. 与论文对齐（可选，成本高）

| 任务 | 做法 |
|------|------|
| 14B 在 **957 oracle** 上推理 + Acc | 全量跑通后可与 Extended Table 3 直接比 |
| Judge 一致性 | 子集上 GPT-4o vs Gemma-9B reasoning（或论文 Reasoning Evaluator） |

→ 回答：**本地 pipeline 与论文数值差多少**。

### D. 延伸实验（视时间）

| 方向 | 说明 |
|------|------|
| Treatment Stage-2 | 复用 hard 抽样思路到 treatment 496 |
| 1-turn / free-turn | 论文 Acc 在 oracle 外更低；可看 14B 在非 oracle 设置下降幅度 |
| 改进 weak 模型 | AWQ/量化/ prompt / thinking 截断对 Acc–Rec 权衡的影响 |

### 建议优先级

```
1. o3/deepseek × 400 Acc     （1–2h API，表格立刻完整）
2. 错例 case study           （支撑讨论 section）
3. 957 全量 14B              （仅当需要与 Nature 表逐数字对齐）
```

---

<a id="sec-5"></a>
## 5. 产出索引

| 路径 | 说明 |
|------|------|
| `data/Stage2/inference/qwen3-14b-thinking_diagnosis.json` | 推理 |
| `data/Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-14b-thinking_*.json` | Gemma direct / aug |
| `data/Stage2/acc_results_gpt/qwen3-14b-thinking/` | GPT-4o Acc |
| `data/MedRBench/stage2_manifest.json` | demo / hard 划分 |
| `scripts/stage1/analyze_stage2_results.py` | 统计复现 |
| `docs/reports/STAGE1_EXPERIMENT_REPORT.md` | Stage-1 对照 |

---

*主数字均为 qwen3-14b-thinking × **400 总计**；子集拆解见 §2.1。*
