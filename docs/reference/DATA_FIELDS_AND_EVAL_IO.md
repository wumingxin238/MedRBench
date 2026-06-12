# MedR-Bench Oracle 诊断：数据结构与评估 IO

## 1. 文件关系

```
diagnosis_957_*.json          GT，键为 PMC ID
oracle_diagnosis.json         模型输出：{ "PMCxxxxx": { "模型名": results对象 } }

评估合并：
  case_data = GT[case_id].copy()
  case_data["id"] = case_id
  case_data["results"] = oracle[case_id][model_name]

输出：
  acc_results/.../PMCxxxxx.json
  reasoning_results/.../PMCxxxxx.json
```

---

## 2. GT 单病例（`diagnosis_957_*.json` 中每个 PMC 值）

```json
{
  "raw_case": "原始病例全文",
  "generate_case": {
    "case_summary": "结构化病例描述",
    "differential_diagnosis": "GT 鉴别/推理讨论",
    "final_diagnosis": "GT 最终诊断说明（段落）",
    "diagnosis_results": "GT 标准答案（短句）"
  },
  "body_category": ["..."],
  "disorder_category": ["..."],
  "receive_time": "...",
  "checked_rare_disease": []
}
```

| 字段 | 推理 | 评估 |
|------|------|------|
| `generate_case.case_summary` | 模型输入 | Efficiency、Factuality |
| `generate_case.final_diagnosis` | — | Efficiency 的推理目标 |
| `generate_case.differential_diagnosis` | — | Completeness GT 拆分（前半） |
| `generate_case.final_diagnosis` | — | Completeness GT 拆分（后半） |
| `generate_case.diagnosis_results` | — | Accuracy |
| `raw_case` | — | 不用 |

---

## 3. 模型输出 `results`（`oracle_diagnosis.json`）

| 模型 | 字段 | 评估读取 |
|------|------|----------|
| gemini2-ft / qwq | `content` | ✅ 推理步 + 答案 |
| deepseek-r1 | `thinking_process` | ❌（仅 `--model deepseek-r1-thinkingprocess` 时读） |
| deepseek-r1 | `content` | ✅ 默认读此 |
| 本地 gemini/qwq | `input`, `content` | 仅 `content` |
| 本地 deepseek | `input`, `out_reasoning`, `out_answer` | 需转成论文格式后再评 |

**`content` 格式**：

```
### Resoning:
<step 1> ...
<step 2> ...
### Answer:
最终诊断
```

**deepseek 论文格式**：

```json
{
  "thinking_process": "原始 CoT 自由文本",
  "content": "### Resoning:\n<step 1> ...\n### Answer:\n..."
}
```

**推理输入**：`generate_case.case_summary` → prompt 模板。

**`split_reasoning(text)`**（从 `content` 或 `thinking_process` 切预测步）：

1. 去 ` ``` `、取 `### Resoning:` 等标记后内容
2. 截断 `### Answer:` 之前
3. 优先 `<step n>`；否则按空行/换行切
4. 最多 10 步

| 用途 | 读取 |
|------|------|
| Eff / Fact / Comp 预测步 | `content`（默认）或 `thinking_process` |
| Accuracy 预测答案 | 始终 `content` 中 `### Answer` 后 |

---

## 4. 评估结果 JSON

评估结果 = **GT 原对象完整 copy**（`raw_case` + `generate_case` + metadata）+ `id` + `results` + 指标。

### 4.1 Reasoning 结果

```json
{
  "raw_case": "...",
  "generate_case": {
    "case_summary": "...",
    "differential_diagnosis": "...",
    "final_diagnosis": "...",
    "diagnosis_results": "..."
  },
  "body_category": ["..."],
  "disorder_category": ["..."],
  "receive_time": "...",
  "checked_rare_disease": [],
  "id": "PMC11625232",
  "results": {
    "thinking_process": "...",
    "content": "..."
  },
  "reasoning_eval": [
    {
      "step": "预测步原文",
      "efficiency": "Reasoning | Citation | Repetition | Redundancy",
      "factulity": true,
      "judgment_path": [{ "judgment": "Correct", "keywords_to_search": "None" }]
    }
  ],
  "gt_reasoning_eval": [
    { "step": "Step 1: ...", "hit": true }
  ],
  "efficiency": 0.85,
  "factulity": 0.92,
  "recall": 0.375
}
```

| 追加字段 | 含义 |
|----------|------|
| `reasoning_eval[]` | 每预测步 Eff + Fact |
| `gt_reasoning_eval[]` | 每 GT 步 + 是否被覆盖 |
| `efficiency` | Reasoning 步占比 |
| `factulity` | Reasoning 步中事实正确占比 |
| `recall` | Completeness |

非 Reasoning 步：`factulity: null`，`judgment_path: []`。

### 4.2 Accuracy 结果

同上 GT 结构 + `results` + 顶层 `accuracy`，**无** reasoning 相关字段。

```json
{
  "...GT 字段同上...",
  "id": "PMC11625232",
  "results": { "content": "..." },
  "accuracy": true
}
```

---

## 5. 各评估步骤 IO

| 指标 | 输入 | 输出 |
|------|------|------|
| **Accuracy** | `generate_case.diagnosis_results`<br>`results.content` → Answer 段 | `accuracy` |
| **Efficiency** | 每预测步 + `generate_case.case_summary` + `generate_case.final_diagnosis` | `reasoning_eval[].efficiency`<br>`efficiency` |
| **Factuality** | 仅 Reasoning 步 + `generate_case.case_summary` + 搜索摘要 | `reasoning_eval[].factulity`<br>`factulity` |
| **Completeness** | GT：`differential_diagnosis + "\n Final diagnosis:\n" + final_diagnosis` → Judge 切步<br>Pred：`"\n".join(预测步)` | `gt_reasoning_eval[]`<br>`recall` |

### Prompt 占位符

**Accuracy**（`acc_diagnose.txt`）

| 占位符 | 来源 |
|--------|------|
| `{pred_diagnose}` | `results.content` Answer 段 |
| `{gt_diagnose}` | `generate_case.diagnosis_results` |

**Efficiency**（每预测步 1 次）

| 占位符 | 来源 |
|--------|------|
| `{current_step}` | 当前预测步 |
| `{previous_steps}` | 之前各步拼接 |
| `{case}` | `generate_case.case_summary` |
| `{result}` | `generate_case.final_diagnosis` |

**Factuality**（仅 Reasoning 步）

| 占位符 | 来源 |
|--------|------|
| `{case}` | `generate_case.case_summary` |
| `{reasoning_step}` | 当前预测步 |
| `{info}` | Bing 摘要或关搜索时的固定占位文本 |

**Completeness — GT 拆分**（1 次/例）

| 占位符 | 来源 |
|--------|------|
| `{gt_reasoning}` | `differential_diagnosis + "\n Final diagnosis:\n" + final_diagnosis` |

**Completeness — hit-check**（每 GT 步 1 次）

| 占位符 | 来源 |
|--------|------|
| `{a_reasoning_step}` | GT 单步 |
| `{out_reasoning}` | 预测步 join 后的字符串 |

---

## 6. 字段 → 指标

```
generate_case.case_summary      → 推理输入；Eff/Fact {case}
generate_case.final_diagnosis   → Eff {result}
generate_case.differential_diagnosis ┐
generate_case.final_diagnosis        ├→ Completeness GT 步
results.content                 → split → Eff/Fact/Comp 预测链
results.thinking_process        → 仅 deepseek-r1-thinkingprocess 模式
generate_case.diagnosis_results ← results.content Answer → Accuracy
```

---

## 7. 完整示例：PMC11625232（Reasoning: deepseek-r1 · Accuracy: gemini2-ft）

以真实数据走一遍：**GT → 推理 → 合并 → 四项评估 → 落盘**。

### Step 1 · GT（`diagnosis_957_*.json["PMC11625232"]`）

```json
{
  "raw_case": "Neurological and orbital complication of acute sinusitis...（原文，评估不用）",
  "generate_case": {
    "case_summary": "- **Patient Information:** 13-year-old male\n- **Chief Complaint:** Severe left eye pain\n- **History of Present Illness:** Eyelid edema, erythema... recent nasal congestion and headaches.\n- **Physical Examination:** Febrile (39.5°C)...\n- **Ancillary Tests:** Elevated CRP 306.9 mg/L; NCCT/NCMRI: maxillary sinusitis, orbital cellulitis, frontal epidural empyema.",
    "differential_diagnosis": "1. **Orbital Cellulitis:** ... 2. **Epidural Empyema:** ... 3. **Acute Sinusitis:** ... 4. **Other Conditions:** excluded ...",
    "final_diagnosis": "The patient was diagnosed with **orbital cellulitis secondary to acute sinusitis**, complicated by a small **frontal epidural empyema**.",
    "diagnosis_results": "Orbital cellulitis secondary to acute sinusitis with frontal epidural empyema."
  },
  "body_category": ["Brain and Nerves"],
  "disorder_category": ["Infections"],
  "receive_time": "2024-10-9",
  "checked_rare_disease": []
}
```

### Step 2 · 推理（模型只看 `case_summary`）

```
输入  →  generate_case.case_summary  （填入诊断 prompt）
输出  →  oracle_diagnosis.json["PMC11625232"]["deepseek-r1"]
```

```json
{
  "thinking_process": "Okay, let's start by looking at the patient's information. He's a 13-year-old male with severe left eye pain... Putting it all together: orbital cellulitis secondary to acute bacterial sinusitis, complicated by frontal epidural empyema.",
  "content": "### Resoning:\n<step 1> The patient presents with severe left eye pain, eyelid edema, erythema... severe bacterial infection.\n<step 2> ... imaging-confirmed left maxillary sinusitis ... sinus origin.\n<step 3> NCCT and NCMRI reveal ... frontal epidural empyema.\n<step 4> The progression from sinusitis to orbital cellulitis ... bacterial sinusitis.\n<step 5> ... ruling out isolated preseptal cellulitis or non-infectious causes.\n\n### Answer:\nOrbital cellulitis secondary to acute bacterial sinusitis complicated by frontal epidural empyema."
}
```

`split_reasoning(content)` → **5 个预测步**（评估读 `content`，不读 `thinking_process`）。

### Step 3 · 评估前合并

```python
case_data = GT["PMC11625232"].copy()
case_data["id"] = "PMC11625232"
case_data["results"] = oracle["PMC11625232"]["deepseek-r1"]
```

### Step 4 · 各指标读什么

| 指标 | 本例实际取值 |
|------|-------------|
| **Accuracy** | GT: `"Orbital cellulitis secondary to acute sinusitis with frontal epidural empyema."`<br>Pred（Step 6 · gemini2-ft 真跑）: `"Orbital Cellulitis with Frontal Epidural Empyema secondary to Maxillary Sinusitis."` |
| **Efficiency ×5** | 每步 + `case_summary` + `final_diagnosis`（段落） |
| **Factuality** | 5 步均判 Reasoning → 逐步评事实性 |
| **Completeness** | GT 拼接: `differential_diagnosis + "\n Final diagnosis:\n" + final_diagnosis`<br>Pred 链: 5 步 join 成一段字符串 |

Completeness GT 拆分 → **8 条 GT 步**，逐条与预测链比 hit。

### Step 5 · Reasoning 评估落盘

`reasoning_results_qwen_judge_paper_957/deepseek-r1/PMC11625232.json`（截断）：

评估结果是 **GT 整条原样保留**（含 `raw_case`、metadata）+ 追加字段：

```json
{
  "raw_case": "Neurological and orbital complication of acute sinusitis...（原文全文，评估不读）",
  "generate_case": {
    "case_summary": "- **Patient Information:** 13-year-old male ...",
    "differential_diagnosis": "1. **Orbital Cellulitis:** ...",
    "final_diagnosis": "The patient was diagnosed with **orbital cellulitis secondary to acute sinusitis** ...",
    "diagnosis_results": "Orbital cellulitis secondary to acute sinusitis with frontal epidural empyema."
  },
  "body_category": ["Brain and Nerves"],
  "disorder_category": ["Infections"],
  "receive_time": "2024-10-9",
  "checked_rare_disease": [],
  "id": "PMC11625232",
  "results": { "thinking_process": "...", "content": "### Resoning:\n<step 1> ..." },
  "reasoning_eval": [
    { "step": "The patient presents with severe left eye pain...", "efficiency": "Reasoning", "factulity": true, "judgment_path": [{"judgment": "Correct", "keywords_to_search": "None"}] },
    { "step": "The recent severe nasal congestion...", "efficiency": "Reasoning", "factulity": true, "judgment_path": [...] },
    "... 共 5 步，均为 Reasoning + factulity=true ..."
  ],
  "gt_reasoning_eval": [
    { "step": "Step 1: Orbital Cellulitis was strongly suspected ...", "hit": true },
    { "step": "Step 2: Acute Maxillary Sinusitis ...", "hit": false },
    { "step": "Step 3: Small Epidural Empyema ...", "hit": false },
    { "step": "Step 4: Other conditions ... excluded ...", "hit": false },
    { "step": "Step 5: ... orbital cellulitis and sinusitis ...", "hit": true },
    { "step": "Step 6: Imaging findings matched ...", "hit": false },
    { "step": "Step 7: No evidence ... thrombotic or meningeal ...", "hit": false },
    { "step": "Final Diagnosis: ... orbital cellulitis secondary to acute sinusitis ...", "hit": true }
  ],
  "efficiency": 1.0,
  "factulity": 1.0,
  "recall": 0.375
}
```

`recall = 3/8 = 0.375`（8 条 GT 步中 3 条 hit）。

### Step 6 · gemini2-ft 评估落盘（957 batch 真跑）

同一病例 PMC11625232，换模型 **`gemini2-ft`**；GT 不变，合并时取：

```python
case_data["results"] = oracle["PMC11625232"]["gemini2-ft"]
```

Judge 比较 `generate_case.diagnosis_results` 与 `results.content` 的 `### Answer:` 段。

| | 文本 |
|--|------|
| **GT** | Orbital cellulitis secondary to acute sinusitis with frontal epidural empyema. |
| **Pred** | Orbital Cellulitis with Frontal Epidural Empyema secondary to Maxillary Sinusitis |

`reasoning_results_qwen_judge_paper_957/gemini2-ft/PMC11625232.json`（截断；Accuracy 另存 `acc_results_.../`，仅多 `accuracy` 字段）：

```json
{
  "raw_case": "Neurological and orbital complication of acute sinusitis...（原文全文，评估不读）",
  "generate_case": {
    "case_summary": "- **Patient Information:** 13-year-old male ...",
    "differential_diagnosis": "1. **Orbital Cellulitis:** ...",
    "final_diagnosis": "The patient was diagnosed with **orbital cellulitis secondary to acute sinusitis** ...",
    "diagnosis_results": "Orbital cellulitis secondary to acute sinusitis with frontal epidural empyema."
  },
  "body_category": ["Brain and Nerves"],
  "disorder_category": ["Infections"],
  "receive_time": "2024-10-9",
  "checked_rare_disease": [],
  "id": "PMC11625232",
  "results": {
    "content": "### Resoning:\n<step 1> The patient is a 13-year-old male presenting with severe left eye pain...\n...<step 7> ... Maxillary Sinusitis.\n\n### Answer:\nOrbital Cellulitis with Frontal Epidural Empyema secondary to Maxillary Sinusitis"
  },
  "reasoning_eval": [
    {
      "step": "The patient is a 13-year-old male presenting with severe left eye pain, eyelid edema, erythema, and localized warmth in the left eye. These are classic signs of orbital inflammation or infection.",
      "efficiency": "Reasoning",
      "factulity": true,
      "judgment_path": [{ "judgment": "Correct", "keywords_to_search": "None" }]
    },
    {
      "step": "The history of recent severe nasal congestion and headaches suggests a possible link between the eye symptoms and a sinus infection...",
      "efficiency": "Reasoning",
      "factulity": true,
      "judgment_path": [{ "judgment": "Correct", "keywords_to_search": "None" }]
    },
    "... 共 7 步，均为 Reasoning + factulity=true ..."
  ],
  "gt_reasoning_eval": [
    { "step": "<Step 1> Orbital Cellulitis was strongly suspected given the patient's symptoms of severe eye pain, erythema, eyelid swelling...", "hit": true },
    { "step": "<Step 2> Acute Sinusitis was evident on imaging with inflammatory obliteration of the maxillary sinus...", "hit": true },
    { "step": "<Step 3> A small epidural empyema was identified by NCCT and confirmed by NCMRI...", "hit": false },
    { "step": "<Step 4> Other conditions such as cavernous sinus thrombosis or meningitis were excluded...", "hit": false },
    "... 共 9 步 ..."
  ],
  "efficiency": 1.0,
  "factulity": 1.0,
  "recall": 0.556
}
```

`recall = 5/9 ≈ 0.556`（9 条 GT 步中 5 条 hit）。

**Accuracy 落盘**（`acc_results_qwen_judge_paper_957/gemini2-ft/PMC11625232.json`）：结构同上 GT + `results`，**无** `reasoning_eval` / `gt_reasoning_eval`，仅追加：

```json
{
  "...GT + results 同上...",
  "accuracy": true
}
```

Judge 认为 pred 与 `diagnosis_results` 语义等价 → `accuracy: true`。

### 数据流小结

```
case_summary ──► 模型推理 ──► results.content (+ deepseek 另有 thinking_process)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
         deepseek: split 5 步      Answer 段          (thinking 默认不用)
                    │                   │
         Eff/Fact/Comp              Accuracy（本例 Step 6 用 gemini2-ft）
                    │                   │
    reasoning_eval[] +          acc_results/.../gemini2-ft/
    gt_reasoning_eval[] +       accuracy: true
    efficiency/factulity/recall
```

---

## 8. 已知拼写 typo

MedR-Bench 原始 benchmark / 评估代码中有两处历史拼写错误，读数据或写脚本时需按**实际字段名**处理，不要自行“纠正”。

| 实际写法 | 正确拼写 | 来源 | 说明 |
|----------|----------|------|------|
| **`factulity`** | factuality | **评估 code** | 落盘 JSON 字段名；`reasoning_eval.py`、`oracle_diagnose_reasoning.py` 等均使用此拼写。论文指标名 **Factuality** 本身无误。 |
| **`### Resoning:`** | `### Reasoning:` | **推理 prompt 模板** | `src/Inference/instructions/oracle_diagnose.txt` 及 HuggingFace 论文推理输出 `oracle_diagnosis.json` 均沿用此格式。评估 code 在 `split_reasoning` 中**同时兼容** `Reasoning` 与 `Resoning` 两种标记（见 `src/Evaluation/utils.py`）。 |
