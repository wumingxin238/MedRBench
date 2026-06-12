# MedR-Bench 评估报告：Oracle 诊断推理（957 例 · 三模型）

**评估对象**：HuggingFace 论文推理 `oracle_diagnosis.json` 中的 **deepseek-r1**、**gemini2-ft**、**qwq**  
**Judge 模型**：本地 Qwen2.5-7B-Instruct（`judge_server_transformers.py`，无 Bing 网页搜索）  
**Ground Truth**：`data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json`  
**生成日期**：2026-05-28（deepseek-r1 整合：2026-05-28）

**目录**

1. [结果文件位置](#1-结果文件位置)  
2. [运行完整性检查](#2-运行完整性检查)  
3. [与论文数据对比](#3-与论文数据对比)  
3A. [同 Judge 下三模型横向对比](#3a-同-judge-下三模型横向对比957-例)  
4. [核心结论](#4-核心结论)  
5. [典型案例](#5-典型案例)  
6. [Accuracy 定义](#6-论文对-accuracy-的定义非-exact-match)  
7. [**Efficiency / Factuality / Completeness 指标定义**](#7-efficiency--factuality--completeness-指标定义)  
8. [**Completeness 差距深度分析**](#8-completeness-差距较大原因深度分析)（含 [8.6 逐例差距](#86-逐例差距整体偏移-vs-少数-outlierqwq-vs-gemini--957-例)）  
9. [代码与评估入口](#9-代码与评估入口)  
10. [**答案诱导偏差研究方向**](#10-延伸研究llm-judge-的答案诱导偏差)  
11. [**后续可探索方向**](#11-后续可探索方向)  
12. [建议使用方式](#12-建议使用方式)  
13. [汇总表](#13-汇总表一页速查)  

---

## 1. 结果文件位置

| 模型 | 任务 | 目录 |
|------|------|------|
| deepseek-r1 | 推理质量 | `src/Evaluation/reasoning_results_qwen_judge_paper_957/deepseek-r1/` |
| deepseek-r1 | Accuracy | `src/Evaluation/acc_results_qwen_judge_paper_957/deepseek-r1/`（**尚未评估**） |
| gemini2-ft | 诊断准确率 (Accuracy) | `src/Evaluation/acc_results_qwen_judge_paper_957/gemini2-ft/` |
| gemini2-ft | 推理质量 (Eff / Fact / Comp) | `src/Evaluation/reasoning_results_qwen_judge_paper_957/gemini2-ft/` |
| qwq | 诊断准确率 (Accuracy) | `src/Evaluation/acc_results_qwen_judge_paper_957/qwq/`（**尚未评估**） |
| qwq | 推理质量 (Eff / Fact / Comp) | `src/Evaluation/reasoning_results_qwen_judge_paper_957/qwq/` |

每个病例一个 JSON 文件，命名格式：`PMCxxxxx.json`。

### 单条 JSON 关键字段

**Accuracy 结果**

| 字段 | 含义 |
|------|------|
| `accuracy` | `true` / `false`，Judge 判断预测诊断是否与 GT 语义等价 |

**Reasoning 结果**

| 字段 | 代码名 | 论文名 |
|------|--------|--------|
| `efficiency` | efficiency | Efficiency（推理效率） |
| `factulity` | factuality | Factuality（真实性，代码拼写为 factulity） |
| `recall` | recall | Completeness（完整性） |
| `reasoning_eval` | — | 逐步推理及每步 efficiency/factuality 评判 |
| `gt_reasoning_eval` | — | GT 推理步骤及是否被模型覆盖（hit） |

本地复算脚本：

```bash
python scripts/eval/compare_oracle_957_eval.py   # 三模型 vs 论文（推荐）
python scripts/eval/analyze_reasoning_paper_957.py
```

---

## 2. 运行完整性检查

### 2.1 gemini2-ft

| 检查项 | 结果 |
|--------|------|
| Reasoning 结果文件数 | **957 / 957** |
| Accuracy 结果文件数 | **957 / 957** |
| `efficiency` / `factulity` / `recall` 缺失 | **0** |
| `reasoning_eval` 为空 | **0** |
| 推理步数为 0 | **0** |

**结论：957 条评估完整，无缺文件、无缺失字段。**

`src/Evaluation/gemini2-ft_error.log` 中仅有 4 行历史错误，来自早期 **35 例 + 远程 GPT API**（敏感词拦截），**不是**本次 957 全量 Qwen 评估产生。

#### 步骤级统计（gemini2-ft）

| 统计量 | 数值 |
|--------|------|
| 每例平均推理步数 | 6.95（中位 7） |
| 每例平均 GT 步骤数 | 7.11（中位 7，最大 17） |
| 被判为 `Reasoning` 的步占比 | 93.6% |
| 被判为 `Citation` | 5.7% |
| 被判为 `Redundancy` / `Repetition` | < 1% |

164 例 completeness = 0%：**不是未评完**，而是 Judge 对全部 GT 步骤均判 `hit: false`（见第 5 节案例说明）。

### 2.2 qwq

| 检查项 | 结果 |
|--------|------|
| Reasoning 结果文件数 | **957 / 957** |
| Accuracy 结果文件数 | **0 / 957**（待跑 `oracle_diagnose_accuracy.py`） |
| `efficiency` / `factulity` / `recall` 缺失 | **0** |
| `reasoning_eval` 为空 | **0** |
| 推理步数为 0 | **0** |

**结论：QwQ 推理评估 957 条完整；Accuracy 尚未评估。**

#### 步骤级统计（qwq）

| 统计量 | 数值 |
|--------|------|
| 每例平均推理步数 | 9.75（中位 10） |
| 每例平均 GT 步骤数 | 7.44（中位 7，最大 13） |
| 被判为 `Reasoning` 的步占比 | 76.5% |
| 被判为 `Citation` | 22.8% |
| 被判为 `Redundancy` / `Repetition` | < 1% |

49 例 completeness = 0%（占 5.1%），远低于 gemini2-ft 的 164 例（17.1%）。

### 2.3 deepseek-r1

| 检查项 | 结果 |
|--------|------|
| Reasoning 结果文件数 | **957 / 957** |
| Accuracy 结果文件数 | **0 / 957**（待跑） |
| 字段缺失 | **0** |

#### 步骤级统计（deepseek-r1）

| 统计量 | 数值 |
|--------|------|
| 每例平均推理步数 | 4.49（中位 5） |
| 每例平均 GT 步骤数 | 6.87 |
| 被判为 `Reasoning` 的步占比 | 96.0% |
| 被判为 `Citation` | 4.0% |

59 例 completeness = 0%（6.2%）。

---

## 3. 与论文数据对比

**论文来源**：[Nature Communications — MedR-Bench](https://www.nature.com/articles/s41467-025-64769-1)  
**对照表**：Extended Table 3（Oracle Accuracy）、Extended Table 4（Oracle Reasoning，all diseases）

论文设置：**GPT-4o 作 Reasoning Evaluator**；Factuality 评估**含网页搜索**；Accuracy 使用 Prompt 19（**语义等价，非字符串 exact match**）。

### 3.0 论文 Oracle 参考（GPT-4o Judge）

| 模型 | Accuracy | Efficiency | Factuality | Completeness |
|------|----------|------------|------------|--------------|
| deepseek-r1 | **89.76%** | **97.17%** | 95.03% | 78.27% |
| gemini2-ft | 86.83% | 95.89% | **98.23%** | **83.28%** |
| qwq | 85.06% | 71.20% | 84.02% | 79.97% |

### 3.0b 本地 Qwen Judge 汇总（957 全量）

| 模型 | Accuracy | Efficiency | Factuality | Completeness | vs 论文 Comp Δ |
|------|----------|------------|------------|--------------|----------------|
| deepseek-r1 | *未评估* | **94.93%** | 93.19% | **49.46%** | **−28.81 pp** |
| gemini2-ft | **81.09%** | 93.61% | **98.15%** | 31.47% | −51.81 pp |
| qwq | *未评估* | 76.72% | 95.38% | **73.55%** | **−6.42 pp** |

**Completeness 与论文可比性**：qwq 最接近；deepseek 居中（−28.8 pp）；gemini 崩盘（−51.8 pp）。

### 3.1 gemini2-ft：957 全量 vs 论文

| 指标 | 论文 (GPT-4o) | 本次 (Qwen-7B) | 差距 |
|------|---------------|----------------|------|
| **Accuracy** | 86.83% | **81.09%** (776/957) | −5.74 pp |
| **Efficiency** | 95.89% | **93.61%** | −2.28 pp |
| **Factuality** | 98.23% | **98.15%** | −0.08 pp |
| **Completeness** | 83.28% | **31.47%** | **−51.81 pp** |

### 3.2 deepseek-r1：957 全量 vs 论文

| 指标 | 论文 (GPT-4o) | 本次 (Qwen-7B) | 差距 |
|------|---------------|----------------|------|
| **Accuracy** | 89.76% | *未评估* | — |
| **Efficiency** | 97.17% | **94.93%** | −2.24 pp |
| **Factuality** | 95.03% | **93.19%** | −1.84 pp |
| **Completeness** | 78.27% | **49.46%** | **−28.81 pp** |

deepseek 推理链最短（均 4.49 步），Efficiency/Factuality 与论文接近；Completeness 介于 qwq（73.55%）与 gemini（31.47%）之间。

### 3.3 qwq：957 全量 vs 论文

| 指标 | 论文 (GPT-4o) | 本次 (Qwen-7B) | 差距 |
|------|---------------|----------------|------|
| **Accuracy** | 85.06% | *未评估* | — |
| **Efficiency** | 71.20% | **76.72%** | **+5.52 pp** |
| **Factuality** | 84.02% | **95.38%** | **+11.36 pp** |
| **Completeness** | 79.97% | **73.55%** | −6.42 pp |

QwQ 在本地 Qwen Judge 下与论文量级接近（Completeness 仅差 6.4 pp），**明显不同于 gemini2-ft 的 Completeness 崩盘（−51.8 pp）**。

### 3.4 491 罕见病子集（`checked_rare_disease` 非空）

| 模型 | Efficiency (论文→本地) | Factuality | Completeness |
|------|------------------------|------------|--------------|
| deepseek-r1 | 97.52% → **95.13%** | 95.18% → **93.28%** | 79.01% → **50.10%** |
| gemini2-ft | 96.45% → 94.38% | 98.39% → 98.24% | 84.30% → 31.92% |
| qwq | 72.25% → 77.00% | 84.30% → 95.44% | 80.70% → 73.44% |

### 3.5 Completeness 分布（957 例，同 Judge）

| 区间 | deepseek-r1 | gemini2-ft | qwq |
|------|-------------|------------|-----|
| 0–50% | 431 (45.0%) | 703 (73.5%) | 193 (20.2%) |
| 50–70% | 300 (31.3%) | 173 (18.1%) | 114 (11.9%) |
| 70–90% | 173 (18.1%) | 65 (6.8%) | 272 (28.4%) |
| 90–100% | 53 (5.5%) | 16 (1.7%) | 378 (39.5%) |

≥70% 占比：deepseek **23.6%** · gemini **8.5%** · qwq **68.0%**。

### 3.6 35 例 test_cases 子集（交叉验证，仅 gemini2-ft）

| 数据源 | n | Efficiency | Factuality | Completeness |
|--------|---|------------|------------|--------------|
| 论文 (GPT-4o) | ~35 | ~95.89% | ~98.23% | ~83.28% |
| 本次 957 中的 35 例 | 35 | 92.90% | 97.40% | 34.50% |
| 历史 35 例 Qwen | 35 | 85.97% | 90.12% | 32.68% |
| 历史 35 例 GPT | 34 | 86.24% | 90.69% | **74.79%** |

本地无 QwQ 35 例历史跑数；957 全量已覆盖。

---

## 3A. 同 Judge 下三模型横向对比（957 例）

固定 **Qwen2.5-7B Judge、无网页搜索**，仅更换 `oracle_diagnosis.json` 中的模型推理。

### 主指标

| 指标 | deepseek-r1 | gemini2-ft | qwq | 本地排序 |
|------|-------------|------------|-----|----------|
| **Efficiency** | **94.93%** | 93.61% | 76.72% | deepseek > gemini > qwq ✓（与论文一致） |
| **Factuality** | 93.19% | **98.15%** | 95.38% | gemini > qwq > deepseek |
| **Completeness** | 49.46% | 31.47% | **73.55%** | **qwq > deepseek > gemini ✗**（论文：gemini > qwq > deepseek） |
| 平均推理步数 | **4.49** | 6.95 | 9.75 | — |
| recall=0 病例 | 59 | 164 | 49 | — |

### 两两 Completeness Δ

| 对比 | Δ (A−B) |
|------|---------|
| qwq vs gemini2-ft | **+42.08 pp** |
| qwq vs deepseek-r1 | **+24.09 pp** |
| deepseek-r1 vs gemini2-ft | **+18.00 pp** |

### 与论文排序对照

| 维度 | 论文 (GPT-4o) | 本地 (Qwen-7B) | 一致？ |
|------|---------------|----------------|--------|
| Accuracy | deepseek > gemini > qwq | 仅 gemini 已评 (81.09%) | 待补 |
| Efficiency | deepseek ≈ gemini >> qwq | deepseek > gemini >> qwq | ✓ |
| Completeness | gemini > qwq > deepseek | **qwq >> deepseek > gemini** | ✗ |

**要点**：本地 Judge 下 Completeness 排序与论文**完全反转**；Efficiency 排序与论文一致。deepseek 在 Completeness 上**并未**保持论文第一（Accuracy），反而落在 qwq 与 gemini 之间。

## 4. 核心结论

### 4.1 数据与流程没有问题

**gemini2-ft**

- 推理文本与 `oracle_diagnosis.json` 中 `gemini2-ft` **一致**（抽检 200 例，0 处不一致）。
- 957 条结果字段完整，聚合公式正确。
- **Accuracy 81.09%** 已本地复算确认（776/957）。

**qwq / deepseek-r1**

- Reasoning 各 **957/957** 完整，字段无缺失。
- **Accuracy 均未评估**；需在服务器跑 `oracle_diagnose_accuracy.py --model qwq|deepseek-r1`。

### 4.2 与论文差距的原因（分模型）

| 模型 | 指标 | 是否与论文可比 | 主要原因 |
|------|------|----------------|----------|
| deepseek-r1 | Efficiency / Factuality | **高度可比** | 与论文差 2–2 pp 以内量级 |
| deepseek-r1 | Completeness | **部分可比** | −28.8 pp；介于 qwq（−6.4）与 gemini（−51.8）之间 |
| deepseek-r1 | Accuracy | 待评 | 论文最高 (89.76%) |
| gemini2-ft | Accuracy | 部分可比 | Qwen-7B vs GPT-4o 语义等价判断更严（约 −5.7 pp） |
| gemini2-ft | Efficiency | **高度可比** | 仅差 2.3 pp |
| gemini2-ft | Factuality | **几乎一致** | 仅差 0.08 pp；无网页搜索影响很小 |
| gemini2-ft | Completeness | **不宜直接对比** | Qwen Judge + gemini 输出 → hit-check 系统性偏低（约 −52 pp） |
| qwq | Efficiency | 部分可比 | 本地高于论文 +5.5 pp（Judge 差异）；排序与论文一致（qwq < gemini） |
| qwq | Factuality | 谨慎对比 | 本地高于论文 +11.4 pp（Judge 更宽 + 无网页搜索） |
| qwq | Completeness | **较可比** | 仅差 −6.4 pp，接近论文 80% 量级 |

### 4.3 Judge 差异与模型×Judge 交互（Completeness）

同一套评估代码、同一 Qwen-7B Judge，**不同被评模型 Completeness 差异巨大**：

| 被评模型 | Judge | Completeness |
|----------|-------|--------------|
| gemini2-ft | Qwen-7B (957) | **31.47%** |
| qwq | Qwen-7B (957) | **73.55%** |
| gemini2-ft | GPT-4o (35 例) | **74.79%** |
| gemini2-ft | Qwen-7B (35 例) | **32.68%** |
| deepseek-r1 | Qwen-7B (957) | **49.46%** |

说明：

1. **不是 957 例聚合错误**——gemini 在 35 例与 957 例 Qwen Completeness 均 ~32%。
2. **不是 Qwen Judge 对所有模型都压到 ~30%**——QwQ 73.55%、deepseek 49.46%，gemini 最低。
3. Completeness 差距 = **Judge 能力** + **被评模型输出体裁与 GT hit-check 的对齐度** + **独立 run 的 GT 拆分差异**（见 §8.6）。

### 4.5 deepseek-r1 要点

- **Efficiency 94.93%**：三模型最高，与论文 97.17% 差 2.2 pp。
- **Completeness 49.46%**：论文 78.27%（−28.8 pp）；短链（4.5 步）优于 gemini，远低于 qwq。
- **Accuracy 待跑**：论文三模型最高 89.76%。

### 4.6 QwQ 要点摘要

- **Efficiency 76.72%**：低于 gemini/deepseek，与论文 qwq 偏低一致。
- **Factuality 95.38%**：高于论文 84.02%，部分因本地无网页搜索时 Judge 更宽。
- **Completeness 73.55%**：**与论文较可比**（−6.4 pp）；68% 病例 ≥ 70%。
- **Accuracy 待补**：论文 85.06%。

## 5. 典型案例

以下案例基于 **gemini2-ft**；QwQ 在同 Judge 下 Completeness 整体更高，但单例仍可能出现 hit 偏严（49 例 recall=0%）。

### 5.1 PMC11625232（同一病例，不同 Judge）

模型推理与 GT 均指向：眶蜂窝织炎 + 鼻窦炎 + 硬膜外 empyema。

| Judge | GT 步骤数 | Hit 数 | Recall |
|-------|-----------|--------|--------|
| Qwen-7B | 9 | 5 | 55.6% |
| GPT-4o | 8 | 8 | **100%** |

Qwen 将「epidural empyema」「排除 cavernous sinus thrombosis」等 GT 步骤判为未覆盖，而 GPT 判为已覆盖。

### 5.2 PMC11321471（Completeness = 0% 但诊断正确）

- 模型最终答案：`Gastrointestinal Stromal Tumor (GIST) of the mesentery`
- GT 诊断：`Gastrointestinal Stromal Tumor (GIST)`
- 模型 6 步推理逐步分析至 GIST，Efficiency/Factuality 均为 1.0
- 但 10 个 GT 步骤 **全部** `hit: false` → **recall = 0%**

说明：Completeness 低不等于模型推理差，而是 **GT 步骤覆盖判定** 对 7B Judge 过严。

---

## 6. 论文对 Accuracy 的定义（非 exact match）

论文 Methods 与代码 `acc_diagnose.txt`（Prompt 19）一致：

1. 同一疾病可有多种别名（如 Heart disease / Cardiac disease）
2. 不同表述可指同一病（如 heart attack / myocardial infarction）
3. 只评诊断是否正确，病因/症状/治疗建议不在范围内
4. 预测中包含 GT 诊断、额外提到并发症，**仍算正确**

实现方式：Judge 输出 `Correct|Wrong`，**不是**字符串相等比较。

---

## 7. Efficiency / Factuality / Completeness 指标定义

本节综合 [MedR-Bench 论文 Methods](https://www.nature.com/articles/s41467-025-64769-1) 与仓库实现（`reasoning_eval.py` + `metrics/instructions/`），说明三个推理质量指标的含义、计算方式与 Judge 输入。

### 7.1 共同前提：推理链如何被切分

被评模型输出经 `split_reasoning()` 解析为步骤列表：

- 优先匹配 `<step n> ...` 格式（Gemini 等模型的标准输出）；
- 若无 step 标记，则按空行或换行切分；
- 默认最多取 10 步。

**病例级分数** = 各步判定聚合后的一个 0–1 数值；**Completeness** 则在 GT 侧另走一条链路（见 7.4）。

---

### 7.2 Efficiency（效率）

#### 论文定义

> 评估每个推理步骤是否对**最终答案**贡献**新的有效信息**，而非重复、引用病历或无关冗余。  
> Efficiency = **有效推理步数 / 全部预测步数**（类似“有效步占比”）。

#### 实现流程（逐步分类）

对预测推理链中**每一步**，Judge（Prompt: `reasoning_efficiency.txt`）在以下输入下做四分类：

| 输入 | 内容 |
|------|------|
| 当前步 | 待分类的推理文本 |
| 之前各步 | 已出现的推理历史 |
| 病例摘要 | `case_summary` |
| **最终推理目标** | `final_diagnosis`（GT 最终诊断说明，**非**模型预测答案） |

四类标签：

| 标签 | 含义 |
|------|------|
| **Citation** | 仅复述/引用病历信息，无新推理 |
| **Repetition** | 重复已有推理，无推进 |
| **Reasoning** | 产生新信息或新结论，推动诊断（**有效步**） |
| **Redundancy** | 有新内容但对达成最终答案无帮助 |

#### 计算公式

```
Efficiency = (# 被判为 Reasoning 的步数) / (# 预测推理总步数)
```

- 分子：有效推理步；分母：全部预测步（含 Citation / Repetition / Redundancy）。
- 若总步数为 0，则 Efficiency = 0。

#### 解读

- **高 Efficiency**：推理链紧凑，少废话、少重复引用。
- **低 Efficiency**：大量逐步复述检查所见或重复前文（Qwen-QwQ 在论文中因此偏低）。
- **注意**：Efficiency 的 Judge prompt **显式提供 GT 最终诊断**作为“推理目标”，这与“答案诱导偏差”研究直接相关（见第 10 节）。

---

### 7.3 Factuality（事实性 / 真实性）

#### 论文定义

> 在**有效推理步**（Reasoning 类）中，评估其是否符合医学指南或客观事实。  
> Factuality = **事实正确的有效步数 / 有效推理步总数**（类似 precision）。

#### 实现流程（仅对 Reasoning 步）

对每一步若分类为 `Reasoning`，Judge（Prompt: `reasoning_factuality.txt`）判断该步医学陈述是否正确：

| 输入 | 内容 |
|------|------|
| 病例摘要 | `case_summary` |
| 待判文本 | 该推理步 |
| 外部知识 | 论文：**Bing 搜索**结果；本次：**关闭搜索**，仅用病例 + Judge 内部知识 |

Judge 输出 JSON：`{"judgment": "Correct|Wrong|Search", "keywords_to_search": ...}`

- 若判 `Search` 且允许搜索 → 检索后再判；
- 本次 `--no-web-search` 时，强制 Judge 仅依据病例摘要做最终 Correct/Wrong 判定。

#### 计算公式

```
Factuality = (# Reasoning 步中 factulity=true 的步数) / (# Reasoning 步数)
```

- 非 Reasoning 步（Citation 等）**不参与** Factuality 分母。
- 若无 Reasoning 步，Factuality = 0。

#### 解读

- **高 Factuality**：有效推理步中的医学事实、逻辑依据可靠。
- 论文 oracle 设定下 Gemini-2.0-FT 约 **98%**，说明在信息充分时幻觉较少。
- 本次 Qwen-7B 无搜索仍得 **98.15%**，与论文接近，说明该维度对 Judge 型号相对不敏感。

---

### 7.4 Completeness（完整性）

#### 论文定义

> 衡量模型生成的推理内容**覆盖**了病例报告/reference 中多少**关键推理步骤**。  
> Completeness = **被覆盖的 GT 步骤数 / GT 总步骤数**（类似 recall）。

GT 推理来源（代码）：

```python
gt_reasoning = differential_diagnosis + "\n Final diagnosis:\n" + final_diagnosis
```

即：鉴别诊断过程 + 最终诊断说明，**不是**简短的 `diagnosis_results` 一句话。

#### 实现流程（两阶段，均依赖 Judge）

**阶段 A — 拆分 GT 推理**（Prompt: `reasoning_split_gt_steps.txt`）

- 输入：上述 `gt_reasoning` 长文本；
- Judge 重组为 ≤10 条原子步骤（`<Step k> ...` 或换行分隔）；
- 输出写入结果 JSON 的 `gt_reasoning_eval[].step`。

**阶段 B — 逐步覆盖检测**（Prompt: `reasoning_check_hit.txt`）

对每条 GT 步骤，Judge 判断：

> 该 GT 步骤的核心含义/逻辑，是否出现在**模型预测推理链**（`combined_reasoning`）中？

- 输入：**GT 单步** + **模型完整推理文本**（不含 `### Answer` 最终诊断句，除非落在推理段内）；
- 输出：`Yes|No`（代码用 `'yes' in output.lower()` 解析）。

#### 计算公式

```
Completeness (recall) = (# hit=true 的 GT 步数) / (# GT 总步数 M)
```

#### 解读

- **高 Completeness**：模型推理链覆盖了 reference 讨论中的大部分关键论证（排除项、影像依据、病理确认等）。
- **低 Completeness**：模型可能得出正确诊断，但未按 reference 方式展开全部推理环节。
- 论文指出 Completeness 是临床可靠性中最薄弱的环节之一（多数模型 70–80%+，仍常遗漏关键步骤）。

---

### 7.5 三指标关系一览

| 维度 | 评什么 | 分母 | 是否用 GT 最终诊断 | 是否用 GT 推理 reference |
|------|--------|------|-------------------|-------------------------|
| Efficiency | 预测链是否啰嗦 | 预测总步数 | **是**（作推理目标） | 否 |
| Factuality | 有效步是否医学正确 | Reasoning 步数 | 否（仅用 case） | 否 |
| Completeness | 预测链是否覆盖 reference | GT 步数 M | 否 | **是**（拆分后逐步比对） |
| Accuracy | 最终诊断是否正确 | 1（病例级） | **是**（pred vs GT） | 否 |

---

## 8. Completeness 差距较大：原因深度分析

**gemini2-ft**：Qwen-7B Completeness **31.47%** vs 论文 GPT-4o **83.28%**，差距 **−51.8 pp**。  
**qwq**：Qwen-7B **73.55%** vs 论文 **79.97%**，差距 **−6.4 pp**——量级正常，说明 Completeness 问题在 gemini 上尤为突出。  
Efficiency / Factuality 两模型与论文差距均较小（gemini）或需结合 Judge 解读（qwq）。以下为分层原因分析。

### 8.1 原因一：Judge 能力差异 + 模型×Judge 交互（主因，已有实证）

**同一评估代码、同一 Qwen-7B Judge，不同被评模型**：

| 被评模型 | 样本 | Completeness |
|----------|------|--------------|
| gemini2-ft | 957 例 | 31.47% |
| qwq | 957 例 | **73.55%** |

**同一 gemini2-ft 推理、仅更换 Judge**：

| Judge | 样本 | Completeness | 每例平均 hit 率 |
|-------|------|--------------|----------------|
| GPT-4o | 35 例 | 74.79% | ~75% |
| Qwen-7B | 35 例 | 32.68% | ~33% |
| Qwen-7B | 957 例 (gemini) | 31.47% | ~31% |

说明：**不是 957 例聚合错误**；Qwen-7B 在 gemini 输出上 `check_step_hit` **系统性更常输出 No**，但对 QwQ 输出仍可达 ~74% 均值。Completeness 对 **Judge × 被评模型输出格式** 高度敏感，不能仅用“换 Judge 偏严”概括全部现象。

典型案例 **PMC11625232**：模型与 GPT Judge 均达 recall=100%；Qwen Judge 仅 55.6%（5/9 hit）。  
案例 **PMC11321471**：模型 6 步推理清晰指向 GIST，Efficiency/Factuality=1.0，但 **10/10 GT 步全判未覆盖** → recall=0%。

**机制**：Completeness 要求 Judge 做**跨文本语义对齐**（GT 步骤 ↔ 自由文本推理链），比 Efficiency 四分类、Factuality 单步真伪判断更难；7B 模型在此类“是否涵盖”任务上校准不足，倾向保守拒判。

---

### 8.2 原因二：GT 拆分本身也依赖 Judge（误差累积）

Completeness 流水线中有 **两次** Judge 调用链：

```
gt_reasoning 长文本
    → [Judge-1] split_ground_truth_reasoning → M 条 GT 步骤
    → [Judge-2] check_step_hit × M 次 → hit / no-hit
```

- **Judge-1** 决定分母 M 及每步粒度：拆得更细 → 更难全部 hit；措辞更偏 reference 原文 → 与模型表述差异更大。
- **Judge-2** 决定分子：语义等价但表述不同时，GPT-4o 更易判 Yes，Qwen-7B 更易判 No。
- 两次调用**均用同一 Judge** 时，误差**累积**；换 Judge 后，同一病例的 GT 步骤文本也会变（PMC11625232：Qwen 9 步 vs GPT 8 步，文本不完全相同）。

论文人工验证：Completeness 相关步骤人工一致率约 **90%**（低于 Effectiveness 98%、Factuality 99%），本身即最 noisy 的环节；换弱 Judge 会进一步放大噪声。

---

### 8.3 原因三：GT reference 与模型推理的“体裁 mismatch”

| 来源 | 文体 | 典型内容 |
|------|------|----------|
| GT `differential_diagnosis` | 病例报告 discussion 提炼 | 排除项列表、病理/免疫组化确认、随访结局 |
| 模型 `gemini2-ft` 推理 | 按 prompt 生成的 step-by-step 链 | 从主诉→检查→综合→答案，未必逐步写“排除 cavernous sinus thrombosis” |
| 模型 `qwq` 推理 | 更长链（均 9.75 步）、Citation 占比高 | 更常复述检查/病史，可能更易触发 GT 步骤 hit |

Completeness 要求的是 **reference 步骤是否在预测链中出现**，不是“诊断是否正确”。因此：

- 模型**诊断对、推理合理**，仍可能 Completeness 很低；
- 模型逐步**引用**病历（Citation 多）时 Efficiency 下降，但 Completeness 取决于是否触及 GT 讨论的每个要点。

这对 **gemini 731 例 recall<50%**（73.5%）的分布具有解释力；**qwq 仅 193 例（20.2%）** 处于该区间。

---

### 8.4 原因四：解析与实现细节（次要，但可改进）

| 问题 | 影响 |
|------|------|
| `check_step_hit` 用 `'yes' in output.lower()` | Judge 若输出 “Yes, partially” 或长句含 “not yes” 可能误判 |
| `split_reasoning` 与 GT split 格式不一致 | 模型用 `<step n>`，GT 用 `<Step n>` 或纯文本，对齐更难 |
| GT 最多 10 步 cap | 长 discussion 被截断，分母与 reference 不完全等价 |
| 无网页搜索 | 对 Completeness **影响较小**（hit-check 不用搜索）；Factuality 已验证影响小 |

---

### 8.5 原因五：与 Accuracy / Factuality 表现“背离”的解释

| 现象 | 解释 |
|------|------|
| Accuracy 81% vs 论文 87%，差距 moderate | 最终诊断匹配；Judge 语义等价判定略严 |
| Factuality 98% ≈ 论文 | 逐步医学事实判断，7B 与 4o 在 oracle 完整信息下表现接近 |
| Completeness 31% (gemini) << 论文 83% | **跨段语义覆盖**最难；gemini 输出 × 7B Judge 严重不对齐 |
| Completeness 74% (qwq) ≈ 论文 80% | 同 Judge 下 QwQ 输出更易与 GT 步骤对齐；可比性较好 |

**结论**：gemini2-ft 的 Completeness 差距**不能**解读为“模型推理质量比论文差 50 个百分点”；QwQ 结果表明 **Judge 与被评输出格式的交互** 是更精确的表述。Completeness 指标对 Judge 选择与模型输出体裁**高度敏感**，跨 Judge / 跨模型直接对比时可比性最弱。

---

### 8.6 逐例差距：整体偏移 vs 少数 outlier（qwq vs gemini · 957 例）

脚本：`scripts/eval/analyze_completeness_gap_by_case.py` → 详情见 [`docs/reports/COMPLETENESS_GAP_CASE_ANALYSIS.md`](../reports/COMPLETENESS_GAP_CASE_ANALYSIS.md)，CSV：`docs/artifacts/completeness_gap_by_case.csv`。

#### 8.6.1 是否由少数病例拉动？

| 检验 | 结果 |
|------|------|
| 全量均值 Δ (qwq−gemini) | **+42.1 pp** |
| 去掉 \|Δ\| 最大 50 例后 | **+38.9 pp** |
| 去掉 100 例后 | **+36.1 pp** |
| 正 Δ 来自 top 5% 病例 | 仅 **9.0%** |

**结论：不是少数极端 outlier 造成**，而是 **795/957 (83%)** 病例 qwq completeness 高于 gemini；**481 例 (50%)** |Δ|≥50 pp。中位 Δ = **+46.7 pp**。

| 模式 | 病例数 |
|------|--------|
| qwq≥70% 且 gemini<30% | **314** (32.8%) |
| gemini≥70% 且 qwq<30% | **5** (0.5%) |
| 两者均=0 | 14 |
| 两者均≥70% | 67 |
| gemini=0 且 qwq≥70% | **102** (10.7%) |

逐例 Pearson r = **0.144**（弱相关）→ 两模型在不同病例上“谁高谁低”几乎不同步。

#### 8.6.2 关键 confound：两次评估 GT 拆分不一致

gemini 与 qwq **分两次跑** `split_ground_truth_reasoning`，同一病例 GT 步骤文本仅 **84/957 (8.8%)** 完全一致。

| 子集 | n | mean gemini | mean qwq | mean Δ |
|------|---|-------------|----------|--------|
| GT 文本一致 | 84 | 29.7% | 34.8% | **+5.0 pp** |
| GT 文本不一致 | 873 | 31.6% | 77.3% | **+45.6 pp** |
| 全量 | 957 | 31.5% | 73.5% | +42.1 pp |

**解读**：

- 在 **固定同一 GT 步骤** 时，qwq 仅比 gemini 高约 **5 pp**（两者仍 ~30%，均低于论文）。
- 表观 +42 pp 差距中，**约 37 pp 来自 GT 拆分/run 差异**（分母 M、步骤粒度变化），不能全部归因于推理文本质量。
- 跨模型对比 Completeness 应 **先冻结 GT 步骤**（例如统一用 gemini run 的 `gt_reasoning_eval` 对两模型 pred 做 hit-check）。

#### 8.6.3 Judge 重复性实验结果（P100 · Qwen-7B fp16 · seed=42）

数据：`docs/artifacts/judge_variance_A_hitonly_t0.json`、`judge_variance_B_full_t0.json`、`judge_variance_C_hitonly_t03.json`  
脚本：`scripts/eval/judge_completeness_variance.py` · 汇总：`scripts/eval/summarize_judge_variance.py`

| 实验 | 设置 | 条数 | 10 次重复 STABLE | max std |
|------|------|------|------------------|---------|
| **A** | `hit_only`, temp=0 | 5 例 × 3 模型 = 15 | **15/15 (100%)** | **0.00 pp** |
| **B** | `full`（含 GT 重拆分）, temp=0, gemini | 3 | **3/3 (100%)** | **0.00 pp** |
| **C** | `hit_only`, temp=**0.3** | 2 例 × 2 模型 = 4 | **4/4 (100%)** | **0.00 pp** |

**结论 1 — 不是「某几个例子随机抖动」**  
在 temp=0（及本次 temp=0.3+seed）下，**同一 GT 步 + 同一 pred 连跑 10 次，hit 模式完全相同**。Completeness 的 case 间差距**不是** Judge 采样噪声造成。

**结论 2 — `saved_recall` 与复跑不一致 ≠ 不稳定**  
15 条 A 结果中仅 **7/15** 与 JSON 里当初存的 `recall` 相差 ≤2 pp；8 条差距 ≥12 pp，例如：

| PMC | 模型 | 957 批量 saved | 本次 10 次 mean | Δ |
|-----|------|----------------|-----------------|---|
| PMC11375620 | qwq | 100% | **22.2%** | −77.8 pp |
| PMC11609106 | qwq | 0% | **71.4%** | +71.4 pp |
| PMC11321471 | gemini | 0% | **42.9%** | +42.9 pp |

原因：**957 各模型分 batch 跑**，JSON 里的 `gt_reasoning_eval` 步文本 / 步数与 `recall` 字段来自**该次 run**；本次 `hit_only` 冻结 JSON 中的 GT 步重算 hit，**10 次内部一致**，但与历史 `recall` 可差很多（GT 拆分 batch 效应 + 可能 4bit vs 本次 fp16 Judge）。

**结论 3 — GT split 在单次 session 内稳定，但与历史 JSON 可不同**  
实验 B（gemini，`full` 模式，每 rep 重新 split GT）：

| PMC | hit_only（JSON 内 GT 步） | full（本次重 split，10 次均同） | Δ |
|-----|---------------------------|----------------------------------|---|
| PMC11321471 | 42.9% | 42.9% | 0 |
| PMC11375620 | 22.2% | 25.0% | +2.8 pp |
| PMC11625232 | 55.6% | **75.0%** | **+19.4 pp** |

→ **GT 重拆分在 10 次内 STABLE**，但与 957 批量写入 JSON 的 GT 步不一致时，recall 可差 **~20 pp**（PMC11625232）。这与 §8.6.2「GT 文本仅 8.8% 跨模型一致」一致。

**结论 4 — 对后续工作的含义**

1. **不必再怀疑「固定 temp/seed 方差大」**；应优先 **冻结 GT 步** 再做跨模型对比。  
2. 957 表观 Completeness 差距中，**batch 间 GT 拆分差异** 是已实证 confound。  
3. 若要对齐论文，需 **GPT-4o Judge** 或 **同一 session 内 split 一次、全体模型共用**。

---

## 9. 代码与评估入口

### 9.1 评估命令

将 `--model` 换为 `gemini2-ft` 或 `qwq`；输出目录下会自动创建对应子文件夹。

```bash
# Accuracy（qwq 待跑）
python oracle_diagnose_accuracy.py \
  --model qwq --sequential \
  --patient-cases ../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json \
  --model-outputs ../../oracle_diagnosis.json \
  --output-dir ./acc_results_qwen_judge_paper_957

# Reasoning（无网页搜索）
python oracle_diagnose_reasoning.py \
  --model qwq --sequential --no-web-search \
  --patient-cases ../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json \
  --model-outputs ../../oracle_diagnosis.json \
  --output-dir ./reasoning_results_qwen_judge_paper_957
```

gemini2-ft 将上述 `--model qwq` 改为 `--model gemini2-ft` 即可（已完成）。

### 9.2 指标计算公式（代码层速查）

见第 7 节完整定义；核心公式：

- **Efficiency** = Reasoning 步数 / 预测总步数  
- **Factuality** = 正确的 Reasoning 步数 / Reasoning 步数  
- **Completeness** = GT hit 步数 / GT 总步数  

### 9.3 已知代码细节

| 项 | 说明 |
|----|------|
| `factulity` 拼写 | 历史 typo，与论文 factuality 对应 |
| `check_step_hit` | 用 `'yes' in output.lower()`，对非标准 Judge 输出较 fragile |
| Efficiency prompt 中的 `{result}` | 传入 **`final_diagnosis`（GT）**，非模型预测答案 |
| 无 `--embedded-outputs` | 论文 `oracle_diagnosis.json` 为标准格式，不需要该参数 |

---

## 10. 延伸研究：LLM Judge 的「答案诱导偏差」

### 10.1 研究问题（课题摘要）

> 在医学长链诊断推理中，LLM 评审是否真正评价**推理过程**，还是会被**最终诊断答案**牵引，产生系统性评分偏差？

**核心假设**：若 Judge 在评估推理链时已知晓正确（或错误）诊断，则 Efficiency、Factuality、Completeness 等维度可能**随答案条件变化**，而非仅反映推理本身质量——即 **answer-induced bias（答案诱导偏差）**。

### 10.2 MedR-Bench 框架中的“答案泄漏”触点

基于本仓库实现，以下环节与“是否向 Judge 暴露最终诊断”相关：

| 评估环节 | 当前是否暴露 GT/答案 | 暴露内容 | 偏差风险 |
|----------|---------------------|----------|----------|
| **Efficiency 逐步分类** | **是** | Prompt 中 `[Final Reasoning Goal]` = `final_diagnosis` | **高**：Judge 已知正确结论，可能把“朝该结论推进”的步都标为 Reasoning |
| **Factuality 逐步判断** | 间接 | 仅 `case_summary` + 推理步；但 case 常含检查结果 | 中：无显式答案，但病例摘要信息量大 |
| **Completeness hit-check** | 否（pred 链） | 比对 GT 步骤 vs **模型推理文本** | 中：GT 步骤来自 differential + final_diagnosis，reference 本身含答案信息 |
| **Accuracy** | **是** | pred 答案 vs `diagnosis_results` | 预期设计：评结果，非评过程 |

**对本课题的启示**：MedR-Bench 原框架**并非**“纯过程评估”；Efficiency 尤其与 GT 诊断强绑定。你的课题若比较“加/不加/加错诊断”条件下各维度变化，**Efficiency 是最可能出现系统性偏移的维度**；Completeness 则可能因“Judge 已知 reference 含答案”而产生二次效应。

### 10.3 建议实验条件设计

在 MedR-Bench Oracle 诊断子集（957 或 491 罕见病）上，固定**同一条模型推理链**，仅改变 Judge 输入：

| 条件 ID | Judge 输入 | 目的 |
|---------|-----------|------|
| **A — Process-only** | 病例摘要 + 推理链 | 基线：最少答案信息 |
| **B — Process + Correct DX** | A + 正确 `diagnosis_results` | 模拟当前 Efficiency 部分设定 |
| **C — Process + Wrong DX** | A + 语义相关但错误的诊断 | 检验“错误锚点”是否仍抬高/压低过程分 |
| **D — Process + GT full** | A + `final_diagnosis` 全文 | 对齐当前代码 `gt_answer` 用于 Efficiency |
| **E — Counterfactual chain** | 固定诊断，替换推理链为“错但对”/“对但劣” | 分离答案与过程质量 |

**因变量**（MedR-Bench 已有 + 可扩展）：

- Efficiency / Factuality / Completeness（复用现有 pipeline）  
- **新增建议**：Evidence grounding（推理步是否有病例证据支撑）、Safety（是否出现危险建议）

**统计**：同一病例、不同条件下的 Δscore；配对检验；按 Accuracy 正确/错误子层分析。

### 10.4 基于本次 957 例结果的先行观察

| 观察 | 对答案诱导研究的含义 |
|------|---------------------|
| Factuality 98% (gemini) 且与论文几乎一致 | 在 oracle 信息充分时，Judge 逐步真伪判断**相对稳定** |
| Completeness 31% (gemini Qwen) vs 75% (gemini GPT) vs 74% (qwq Qwen) | **Judge 型号与被评模型输出** 均为混淆因素；qwq 同 Judge 下 Completeness 正常，说明不能单归因于“7B 太弱” |
| PMC11321471：诊断对、Eff/Fact=1.0、Completeness=0% (gemini) | **过程分与结果分可严重背离**；QwQ 全量仅 49 例 recall=0% |
| Efficiency prompt 已含 GT 诊断 | 可直接做 **B vs D vs A** 消融；建议 **gemini 与 qwq 各跑一套** 检验答案诱导是否因链长/Citation 比例而异 |

---

## 11. 后续可探索方向

结合 MedR-Bench 基础设施与本报告发现，建议按优先级展开：

### 11.1 短期（可立即在现有代码上改）

1. **答案条件消融实验**  
   - 修改 `evaluate_efficiency()` 的 `{result}`：空 / 正确诊断 / 错误诊断 / 模型自报诊断；  
   - 固定 Judge（建议至少 GPT-4o + Qwen-7B 各一套），跑 35 或 957 子集，量化 ΔEfficiency、ΔFactuality、ΔCompleteness。

2. **Completeness 实现鲁棒性**  
   - 改进 `check_step_hit`：严格解析 `Yes|No`；或改用 JSON 结构化输出；  
   - 对比 `'yes' in output` vs 结构化解析下 GPT/Qwen 差距是否缩小。

3. **Judge 校准曲线**  
   - 抽样 50–100 例，人工标注 GT 步骤是否被覆盖；  
   - 计算 Qwen-7B / GPT-4o 的 precision-recall，作为 Completeness 可信区间。

### 11.2 中期（论文级实验）

4. **答案诱导 × 诊断正误分层**  
   - 将 957 例按 `accuracy=true/false` 分组；  
   - 检验：错误诊断病例中，提供“正确诊断”是否**虚高** Efficiency/Factuality（后见之明 bias）。

5. **错诊断对照（C 条件）**  
   - 为每例构造 plausibly wrong diagnosis（同专科、相近症状）；  
   - 检验 Judge 是否因“推理链与给定诊断一致”而给更高过程分——即**结论一致性偏差**。

6. **过程-结果解耦评分**  
   - 引入**不看 Answer 段**的 Completeness（仅 `<step>` 段 vs GT）；  
   - 与含 Answer 的条件对比，测量答案段对 Completeness 的边际贡献。

7. **跨 Judge 一致性**  
   - 同一条件、GPT-4o vs Qwen-32B vs Qwen-7B；  
   - 报告 ICC / Kendall's W，明确哪些维度可跨 Judge 比较。

### 11.3 长期（课题延伸）

8. **扩展维度：Evidence & Safety**  
   - Evidence：每步是否可回溯到 case_summary 中具体证据；  
   - Safety：是否出现不当检查/治疗建议；  
   - 同样做 A/B/C 答案条件消融，检验诱导偏差是否存在于新维度。

9. **人机对比基线**  
   - 论文已邀请 6 名医师做 oracle 诊断；可扩展为“医师评推理链”小样本基线，与 LLM Judge 对比偏差方向。

10. **去偏方法**  
    - Prompt 约束：“不得依据最终诊断评判过程”；  
    - 两阶段 Judge：先盲评过程，后评结果；  
    - 训练小型 process-only 评审模型，report bias reduction。

### 11.4 推荐的首发实验路径

```
Step 1  35 例 pilot → A/B/C/D 四条件 × 1 个 Judge → 看 Efficiency 是否显著漂移
Step 2  若漂移显著 → 扩至 491 罕见病 / 957 全量
Step 3  并行换 GPT-4o 与 Qwen-7B → 分离“答案诱导”与“Judge 能力”
Step 4  人工金标准 100 例 → 校准 Completeness 与 Factuality
Step 5  撰写：MedR-Bench 上 LLM-as-judge 的过程效度与答案诱导偏差
```

**本报告 957 例数据可直接复用为**：

- **条件 B/D 的现成近似**（当前 Efficiency 实现）；  
- **对照基线数值**（第 3 节 gemini2-ft / qwq 表）；  
- **Completeness 低分案例库**（gemini 如 PMC11321471）用于定性分析；  
- **同 Judge 跨模型对照**（第 3A 节）用于分离 Judge 效应与输出格式效应。

---

## 12. 建议使用方式

| 用途 | 建议 |
|------|------|
| 与论文 Table 3 Accuracy 对比 | gemini2-ft 可参考（−5.7 pp）；**deepseek / qwq 待补** |
| 与论文 Table 4 Efficiency / Factuality | **deepseek、gemini 可参考**（Eff ±2 pp）；qwq Factuality 本地偏高 |
| 与论文 Table 4 Completeness 对比 | **qwq 较可比**（−6.4 pp）；deepseek 中等（−28.8 pp）；gemini **不宜**（−51.8 pp） |
| 模型间相对排序（同 Judge） | Efficiency 与论文一致；**Completeness 排序与论文反转** |
| 病例级错误分析 | 使用各 JSON 中的 `accuracy`、`gt_reasoning_eval` |

若需对齐论文 Completeness（~80–83%），建议：

- 使用 **GPT-4o** 作 Judge（至少 completeness 的 split + hit-check），或
- 升级本地 Judge 至 **Qwen2.5-32B** 及以上再复跑 reasoning 评估；
- 对 **gemini2-ft** 优先复跑（qwq 在 Qwen-7B 下已接近论文）。

---

## 13. 汇总表（一页速查）

### 三模型 vs 论文（957 · Qwen Judge）

| 模型 | Acc (论文→本地) | Eff | Fact | Comp | Comp Δ |
|------|-----------------|-----|------|------|--------|
| deepseek-r1 | 89.76%→N/A | 97.17→**94.93** | 95.03→93.19 | 78.27→**49.46** | **−28.8** |
| gemini2-ft | 86.83→**81.09** | 95.89→93.61 | 98.23→98.15 | 83.28→31.47 | −51.8 |
| qwq | 85.06→N/A | 71.20→76.72 | 84.02→95.38 | 79.97→**73.55** | **−6.4** |

Reasoning：**957/957 完整**（三模型）· Accuracy：仅 gemini **957/957**

### 同 Judge 排序（本地）

```
Efficiency:    deepseek 94.93% > gemini 93.61% > qwq 76.72%   (≈ 论文)
Factuality:    gemini 98.15% > qwq 95.38% > deepseek 93.19%
Completeness:  qwq 73.55% > deepseek 49.46% > gemini 31.47%  (≠ 论文)
```

推理步数：deepseek 4.49 · gemini 6.95 · qwq 9.75

推理数据：与论文 `oracle_diagnosis.json` 一致 ✓  
主要差异：Judge（Qwen-7B vs GPT-4o）；gemini Completeness 另受输出格式×hit-check 影响

指标定义：见第 7 节 | Completeness 分析：见第 8 节 | 跨模型对照：见第 3A 节  
答案诱导研究：见第 10–11 节

---

*报告由 `scripts/eval/compare_oracle_957_eval.py` 等与人工复核生成。*  
*deepseek-r1 / qwq 957 例整合（2026-05-28）。*
