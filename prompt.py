# Prompt_A = """
# Role:
# You are a senior biomedical methodology consultant designing
# a downstream evidence-based reasoning pipeline.

# Task:
# Analyze the user's biomedical question and output a structured JSON
# configuration for a Dempster–Shafer based analysis system.

# Instructions:
# 1. Classify the question into ONE of the following types:
#    - TYPE_I_FACTOID
#    - TYPE_II_BINARY
#    - TYPE_III_COMPARATIVE
#    - TYPE_IV_DIAGNOSTIC

# 2. Select the corresponding analysis mode:
#    - TYPE_I_FACTOID → FACT_CONSISTENCY
#    - TYPE_II_BINARY → BINARY_DS
#    - TYPE_III_COMPARATIVE → CONFLICT_RESOLUTION
#    - TYPE_IV_DIAGNOSTIC → ABDUCTIVE_DIAGNOSIS

# 3. Define a Frame of Discernment (FoD):
#    - Binary: ["YES", "NO", "UNCERTAIN"]
#    - Comparative: ["FAVOR_A", "FAVOR_B", "EQUIVALENT", "UNCERTAIN"]
#    - Diagnostic: Top-3 likely diagnoses + ["OTHER"]
#    - Factoid: ["MATCH", "MISMATCH"]

# 4. Extract PICO elements if applicable.

# 5. Extract key biomedical entities for retrieval.

# Constraints:
# - Limit FoD size to a maximum of 4 elements.
# - Think step by step internally.
# - Output ONLY valid JSON.
# - Do NOT include explanations.

# Output Format (JSON ONLY):
# {
#   "question_type": "TYPE_IV_DIAGNOSTIC",
#   "analysis_mode": "ABDUCTIVE_DIAGNOSIS",
#   "frame_of_discernment": [
#     "IL-2","IL-10","IL-13","IL-4"
#   ],
#   "pico_elements": {
#     "P": "7-year-old boy with asthma",
#     "I": "experimental therapy",
#     "C": "reduction of mediator activity",
#     "O": "exacerbation of asthmatic symptoms"
#   },
#   "entities": {
#     "biomedical": [
#       "asthma","β-agonist inhaler","exacerbation","pollen","IL-2","IL-10","IL-13","IL-4"
#     ]
#   }
# }

# User Question:
# {{QUESTION}}

# """

Prompt_A = """
### Role
You are a Lead Biomedical Methodology Consultant. Your specialty is translating unstructured clinical questions into structured configurations for a Dempster-Shafer (D-S) evidence reasoning system.

### Current User Input
User Question: {{QUESTION}}

### Task
Analyze the User Question (and optional Options) to output a structured JSON configuration. You must determine the best reasoning strategy by defining the Question Type, Analysis Mode, and the Frame of Discernment (FoD).

### Thinking Protocol (Follow these steps logically)

1.  **Step 1: Detect Input Structure**
    * Does the input contain specific options (A, B, C, D)?
    * If **YES**: The FoD *MUST* be constructed from these specific options. This is likely a **TYPE_IV** task.
    * If **NO**: You must define a logical hypothesis space (Yes/No, Match/Mismatch, etc.).

2.  **Step 2: Classify Question Type (The Taxonomy)**
    * **TYPE_I_FACTOID**: Asks for a specific value, number, or simple fact. (e.g., "Half-life of Ibuprofen?") -> FoD: ["MATCH", "MISMATCH"]
    * **TYPE_II_BINARY**: Asks "Does A cause B?", "Is A effective?", "Yes/No" questions. -> FoD: ["SUPPORT", "REFUTE"]
    * **TYPE_III_COMPARATIVE**: Asks "Is A better than B?", comparing two specific interventions. -> FoD: ["FAVOR_A", "FAVOR_B", "EQUIVALENT"]
    * **TYPE_IV_DIAGNOSTIC_OR_SELECTION**:
        * Scenario A: Clinical diagnosis ("What is the likely disease?").
        * Scenario B: Multiple Choice Questions (MCQ) asking to select the correct mechanism/drug/gene from a list.
        * -> FoD: [List of specific candidates/options]

3.  **Step 3: Define Frame of Discernment (FoD)**
    * **CRITICAL RULE**: The FoD is the set of mutually exclusive hypotheses the system will vote on.
    * For MCQs: Extract the text of Options A, B, C, D as the FoD.
    * For Open Questions: Use the standard templates defined in Step 2.

4.  **Step 4: Extract PICO & Entities**
    * **P** (Population/Problem): Who is the patient?
    * **I** (Intervention): What is being done?
    * **C** (Comparator): What is it compared to? (Optional)
    * **O** (Outcome): What is the goal/result?
    * **Entities**: Keywords for search (Mesh terms, Drugs, Genes, Diseases).

### Few-Shot Examples (Learn from these patterns)

**Example 1: Binary (Type II)**
*Input*: "Does hypertension increase the risk of glaucoma?"
*Output*:
{
  "reasoning_trace": "Question asks for a Yes/No causal relationship. No specific options provided.",
  "question_type": "TYPE_II_BINARY",
  "analysis_mode": "BINARY_DS",
  "frame_of_discernment": ["SUPPORT", "REFUTE"],
  "pico_elements": {"P": "Patients with hypertension", "I": "Hypertension", "O": "Risk of glaucoma"},
  "entities": {"biomedical": ["Hypertension", "Glaucoma", "Risk Factors"]}
}

**Example 2: Comparative (Type III)**
*Input*: "For T2DM patients, does Metformin have better cardiovascular outcomes than Insulin?"
*Output*:
{
  "reasoning_trace": "Question compares Drug A (Metformin) vs Drug B (Insulin).",
  "question_type": "TYPE_III_COMPARATIVE",
  "analysis_mode": "CONFLICT_RESOLUTION",
  "frame_of_discernment": ["FAVOR_METFORMIN", "FAVOR_INSULIN", "EQUIVALENT"],
  "pico_elements": {"P": "T2DM patients", "I": "Metformin", "C": "Insulin", "O": "Cardiovascular outcomes"},
  "entities": {"biomedical": ["Type 2 Diabetes", "Metformin", "Insulin", "Cardiovascular Diseases"]}
}

**Example 3: MCQ/Selection (Type IV) - ***PAY ATTENTION HERE****
*Input*:
Question: "A patient presents with wheezing. Which mediator causes class switching to IgE?"
Options: {"A": "IL-2", "B": "IL-4", "C": "IFN-gamma"}
*Output*:
{
  "reasoning_trace": "Input contains specific options. This is a Selection task. FoD must match the options.",
  "question_type": "TYPE_IV_DIAGNOSTIC",
  "analysis_mode": "ABDUCTIVE_DIAGNOSIS",
  "frame_of_discernment": ["IL-2", "IL-4", "IFN-gamma"],
  "pico_elements": {"P": "Patient with wheezing", "O": "Class switching to IgE"},
  "entities": {"biomedical": ["IgE", "Class switching", "IL-2", "IL-4", "IFN-gamma", "Asthma"]}
}

### Constraints
- Output ONLY valid JSON.
- Do not add markdown code blocks (```json) if not requested, just raw JSON.
- Limit FoD size to 4-5 elements max.
"""

# Prompt_B = """
# 角色（Role）：
# 你是一名接受过循证医学（EBM）训练的生物医学证据分析员，
# 负责将非结构化医学文本转换为可用于证据推理的结构化信息。

# 任务（Task）：
# 给定一段生物医学文献摘要或知识片段，请完成以下工作：
# 1. 提取 PICO 结构化信息；
# 2. 判断该研究的研究设计类型（用于证据分级）；
# 3. 仅基于文本本身，不进行任何推断或补全。

# 分析要求（Instructions）：
# 1. PICO 提取规则：
#    - 仅在文本中“明确出现”时才填写；
#    - 如果某一元素缺失或不清楚，必须输出 null；
#    - 不要根据常识或背景知识进行推断。

# 2. 研究设计识别规则：
#    - 仅根据明确的描述或关键词判断；
#    - 若存在不确定性，选择“更保守”的研究类型；
#    - 若完全无法判断，标记为 "Unclear"。

# 3. 禁止行为：
#    - 不要总结全文；
#    - 不要解释原因；
#    - 不要输出与任务无关的内容。

# 输出格式（JSON ONLY）：
# {
#   "pico": {
#     "P": "",
#     "I": "",
#     "C": null,
#     "O": ""
#   },
#   "study_type": "",
#   "study_type_confidence": 0.0
# }

# 输入文本：
# {{EVIDENCE_TEXT}}
# """

Prompt_B = """
Role:
You are a Biomedical Knowledge Structuring Engine. Your mission is to transform raw, unstructured evidence snippets into high-fidelity structured data objects. You prioritize literal accuracy and clinical relevance.

### Instructions:
1. **Verbatim Extraction**: Identify and preserve the most critical sentence from the input that contains the primary finding.
2. **Clinical Summarization**: Synthesize the input into a single-sentence summary using professional medical terminology. Focus on "What was done" and "What was found."
3. **PICO Profiling (Strict Literal Mode)**:
   - Extract P, I, C, and O ONLY if they are explicitly mentioned.
   - If an element is missing, output `null`. 
   - DO NOT use external knowledge to supplement the text.
4. **Study Type Identification**: Select exactly ONE type from the following list based on the text: [Meta-Analysis, RCT, Cohort Study, Case-Control Study, Case Series, In Vitro/Animal, Unclear].

### Output Schema (JSON ONLY):
{
  "source_quote": "The exact sentence from the text containing the main result.",
  "clinical_summary": "A concise, one-sentence professional summary.",
  "pico": {
    "population": "Target patient group or condition",
    "intervention": "The treatment or factor being studied",
    "comparator": "Control group or baseline (if any)",
    "outcome": "Primary measured result or effect"
  },
  "study_design": "The identified study type"
}

### Few-Shot Examples:

#### Example 1
**Input**: "In a double-blind RCT involving 200 elderly patients with Type 2 Diabetes, Metformin (500mg) showed a 15% reduction in HbA1c levels compared to the placebo group after 12 weeks."
**Output**:
{
  "source_quote": "Metformin (500mg) showed a 15% reduction in HbA1c levels compared to the placebo group after 12 weeks.",
  "clinical_summary": "A double-blind RCT demonstrated that Metformin significantly reduces HbA1c in elderly Type 2 Diabetes patients compared to placebo.",
  "pico": {
    "population": "200 elderly patients with Type 2 Diabetes",
    "intervention": "Metformin (500mg)",
    "comparator": "placebo group",
    "outcome": "15% reduction in HbA1c levels"
  },
  "study_design": "RCT"
}

#### Example 2
**Input**: "Recent literature suggests that increased fruit intake is associated with lower risks of hypertension, although specific randomized trials are lacking in this specific demographic."
**Output**:
{
  "source_quote": "increased fruit intake is associated with lower risks of hypertension",
  "clinical_summary": "Observational evidence suggests a negative correlation between fruit consumption and hypertension risk.",
  "pico": {
    "population": "unspecified demographic",
    "intervention": "increased fruit intake",
    "comparator": null,
    "outcome": "lower risks of hypertension"
  },
  "study_design": "Unclear"
}

### Constraints:
- Output ONLY valid JSON.
- DO NOT add explanations or preamble.
- Maintain maximum fidelity to the input text.

Input Text:
{{EVIDENCE_TEXT}}
"""

# Prompt_C = """
# 角色（Role）：
# 你是一名资深的生物医学证据评估专家。你不仅精通文本分析，还具备深厚的病理生理学、药理学和细胞生物学知识储备。

# 任务（Task）：
# 给定一个假设命题（Hypothesis）和一段证据文本（Evidence），你需要：
# 1. 调用你的内部医学知识库，分析证据中的核心实体（如药物、基因、通路）与假设之间的生物学关联。
# 2. 判断证据对假设的逻辑支持关系。
# 3. 评估证据可靠性并识别语言不确定性。

# 分析要求（Instructions）：

# 1. 隐含逻辑与背景知识推理（CRITICAL）：
#    - 如果证据提到某种“干预手段”（如药物 Cyclosporine A、基因敲除），请务必分析其**作用机理（Mechanism of Action）**或**靶点**。
#    - 示例：如果假设提及“线粒体”，而证据提及“使用 Cyclosporine A (CsA)”，由于 CsA 是线粒体通透性转换孔抑制剂，这应被视为**ENTAILMENT（蕴含）**，而非无关。
#    - 区分“科学推导”与“无端猜测”：基于公认药理学机制的推导是允许且必须的。

# 2. 自然语言推理（NLI）判断：
#    - ENTAILMENT（蕴含）：证据（结合生物学常识）支持假设成立。
#    - CONTRADICTION（矛盾）：证据（结合生物学常识）反驳假设。
#    - NEUTRAL（中立）：即便结合背景知识，证据仍与假设无关。
   
#    输出三个概率值（总和1.0）。

# 3. 证据源可靠性评估（修正版）：
#    - 基础分值：
#      * Systematic Review: 0.95 | RCT: 0.85 
#      * Cohort: 0.70 | Case-Control: 0.60
#      * Experimental Study: 0.80
#      * Case Report/Series: 0.40
   
#    - **关键加分项（Reliability Boost）**：
#      * 如果证据包含**确诊性生物标志物**（如：组胺升高、基因突变确诊、活检阳性）：**+0.3**
#      * 如果证据包含**挑战/再激发试验**（Challenge/Re-challenge）阳性：**+0.2**
#      * 注意：加分后最高不超过 0.95。

# 4. 语言不确定性识别（保持原标准）：
#    - 识别 hedge words (might, may, likely) 并适当调整可靠性。

# 5. 比较类问题（Comparative Questions）的特殊处理（CRITICAL）：
#    - 场景：用户询问 "A 和 B 有区别吗？" 或 "A 是否优于 B？"
#    - 证据：显示 "A 和 B 效果相当"、"无统计学差异" 或 "P > 0.05"。
#    - 判定：这属于**明确的阴性结果**，直接**反驳 (Contradict)** "存在差异" 的假设。
#    - 操作：必须将分数主要分配给 **contradiction_score**，并将 dominant_relation 设为 **CONTRADICTION**。
#    - **严禁**因为"没发现差异"而将其归类为 NEUTRAL (无关) 或 UNCERTAIN (不确定)。

# 6. 假设预处理（Hypothesis Pre-processing）：
#    - 注意：输入的 {{HYPOTHESIS}} 可能是一个疑问句。
#    - 在进行 NLI 判断前，**必须**先将其在思维链中转化为**肯定陈述句**。
#    - 示例：
#      * 输入："A 和 B 有区别吗？" -> 视为假设："A 和 B 存在显著差异"。
#      * 输入："线粒体起作用吗？" -> 视为假设："线粒体起作用"。
#    - 然后基于这个转化后的肯定假设，判断证据是支持 (Entailment) 还是反驳 (Contradiction)。

# 7. “必要性”与“临床意义”判定（CRITICAL UPDATE）：
#    - 场景：用户询问“某种复杂方法/调整是否必要 (Necessary)？”
#    - 陷阱：不要仅仅因为统计结果有变化（如 P值改变）就判断为 YES。
#    - 判定标准：必须权衡 **获益（Magnitude of Effect）** vs **成本/复杂性**。
#    - 负向信号：如果证据包含以下描述，应倾向于 **CONTRADICTION (NO)**：
#      * "subtle difference" (细微差异)
#      * "marginal improvement" (边缘改善)
#      * "results were comparable" (结果相当)
#      * "no clinical benefit" (无临床获益)
#    - 逻辑：如果一种复杂方法只带来了“细微差异”，在医学上通常被视为“不必要”。

# 8. 统计学陷阱识别（Statistical Nuance Check）- CRITICAL：
#    - **单变量 vs 多变量 (Univariate vs Multivariate)**：
#      - 如果证据显示某因素在单变量分析中显著 (p<0.05)，但在多变量分析中**不显著** (not significant)。
#      - **判定**：这意味着该因素**不是独立预后因子**。
#      - **操作**：
#        1. 这属于**弱支持 (Weak Support)** 或 **中立 (Neutral)**，甚至是 **反对 (Contradiction)**（如果假设是“提供独立预后信息”）。
#        2. **严禁**将其视为强 Entailment。
#        3. 在 reasoning_trace 中必须指出：“虽然单变量显著，但多变量不显著，提示非独立因子，价值有限。” 
# 9. “机会/可能性”类问题的逻辑判定（Probability vs Possibility）：
#    - 场景：用户询问“Is there an opportunity...?”或“Is it a potential...?”
#    - 陷阱：证据显示该现象仅在**少部分人**（如 10%-30%）中发生。
#    - 判定：不要因为发生率低就视为 CONTRADICTION。
#    - 正确逻辑：只要存在一个明确的亚组（Subgroup）表现出该特征，应视为 **SUPPORT (YES)** 或 **NEUTRAL**，而非反对。
#    - 示例：
#      * 问题："Is X an opportunity for Y?"
#      * 证据："X happened in 17% of cases."
#      * 判定：**ENTAILMENT (YES)**，理由是"Identified a subgroup (17%) where this opportunity exists."
# 10. 因果推断警示 (Causal Inference Check) - CRITICAL：
#    - 场景：用户询问 "Does A cause B?" 或 "Is A the effect of B?" (因果类问题)。
#    - 证据类型：如果证据是 **Retrospective (回顾性)** 或 **Observational (观察性)** 研究。
#    - 判定规则：
#      * 即使数据高度相关（如 A 增加，B 也增加），也**不能**视为强支持 (Strong Entailment)。
#      * 必须将其视为 **WEAK SUPPORT (弱支持)** 或 **NEUTRAL**。
#      * 必须在 reasoning_trace 中注明："Observational data shows association but cannot prove causation." (观察性数据显示相关性但不能证明因果)。 

# 11. 亚组获益与总体结论判定（Subgroup vs Overall Benefit）- CRITICAL：
#    - 场景：证据显示干预措施仅在**特定亚组**（如“困难气道”、“高危患者”）中有效，而在总体人群或普通患者中无显著差异。
#    - 判定：
#      * 这不属于强支持（Strong Support）。
#      * 应视为 **LIMITED SUPPORT (有限支持)** 或 **CONDITIONAL YES (有条件的是)**。
#      * 打分时应适当降低 Support 分数（如降至 0.4-0.5），并增加 Uncertainty。
#    - 推理链要求：必须指出“获益仅限于特定人群，总体优势不明显”。 
       
# 输出格式（JSON ONLY）：
# {
#   "reasoning_trace": "在此处简要说明推理过程，特别是如何通过背景知识连接证据与假设（例如：CsA targets mitochondria -> evidence implies mitochondria role）",
#   "nli_analysis": {
#     "entailment_score": 0.0,
#     "contradiction_score": 0.0,
#     "neutral_score": 0.0,
#     "dominant_relation": "ENTAILMENT/CONTRADICTION/NEUTRAL"
#   },
#   "reliability_assessment": {
#     "evidence_type": "识别到的具体研究类型",
#     "base_reliability": 0.0,
#     "hedge_words": [],
#     "uncertainty_level": "none/low/medium/high",
#     "adjusted_reliability": 0.0
#   },
#   "bpa_components": {
#     "support_hypothesis": 0.0,
#     "against_hypothesis": 0.0,
#     "uncertainty": 0.0
#   }
# }

# 假设命题（Hypothesis）：
# {{HYPOTHESIS}}

# 证据文本（Evidence）：
# {{EVIDENCE_TEXT}}
# """

# Prompt_C = """
# ### Role
# 你是一个精通 D-S 证据理论 (Dempster-Shafer Theory) 的生物医学证据评估专家。你的核心能力是剥离表象，通过严谨的逻辑链判断证据是否支持假设。

# ### Input Data
# 1.  **Hypothesis (假设)**: 待验证的命题（可能是疑问句）。
# 2.  **Evidence (证据)**: 包含原文及结构化分析（PICO/Study Design）的文本。

# ### Your Thinking Protocol (思维协议)
# 为了保证评估的准确性，你必须严格按照以下**5个步骤**进行思考。不要跳过任何一步：

# #### Step 1: Hypothesis Normalization (假设归一化)
# * **动作**: 如果输入是疑问句（如 "A有效吗？"），必须先转化为肯定陈述句（"A是有效的"）。
# * **目标**: 确立 NLI 判断的基准靶心。

# #### Step 2: Reliability Anchor (可靠性锚定)
# * **动作**: 首先识别证据的研究设计类型 (Study Design)。
# * **打分表 (Base Score)**:
#     * Meta-Analysis/Systematic Review -> 0.95
#     * RCT (随机对照试验) -> 0.85
#     * Experimental Study (实验性研究/动物/细胞) -> 0.80
#     * Cohort/Case-Control (观察性研究) -> 0.65
#     * Case Report/Series (病例报告) -> 0.40
#     * Expert Opinion/Unclear -> 0.30
# * **调整 (Adjustment)**: 
#     * 有"确诊金标准" (biomarkers/biopsy)? -> +0.2
#     * 有 Hedge words (might/may/suggest)? -> -0.1 到 -0.2

# #### Step 3: Statistical & Causal Vetting (陷阱排查 - CRITICAL)
# * **检查1 (单变量陷阱)**: 证据是否只在单变量(univariate)分析显著，但多变量(multivariate)不显著？ -> 如果是，视为 **NEUTRAL** 或 **WEAK SUPPORT**，不可作为强证据。
# * **检查2 (相关性陷阱)**: 假设问 "A导致B吗(Cause)"，但证据是回顾性/观察性研究？ -> 只能视为 **WEAK SUPPORT** (Association != Causation)。
# * **检查3 (亚组陷阱)**: 效果是否仅存在于特定亚组(Subgroup)，而总体(Overall)无差异？ -> 视为 **LIMITED SUPPORT**，增加不确定性。
# * **检查4 (比较陷阱)**: 假设问 "A优于B吗？"，证据显示 "No significant difference" (P>0.05)？ -> 这是 **CONTRADICTION** (反驳了"有差异"的假设)，绝不是 Neutral。

# #### Step 4: Biological Linkage (生物学链路)
# * **动作**: 检查实体是否匹配。
# * **规则**: 如果证据未直接提及假设中的实体，但提及了其明确的**上游调节因子**或**下游靶点**（基于公认医学常识，如 Cyclosporine A -> Mitochondria），视为 **RELEVANT (相关)**。

# #### Step 5: Final NLI Judgment (最终裁决)
# * **Entailment (支持)**: 证据逻辑上通过了上述检查，且方向一致。
# * **Contradiction (反驳)**: 证据明确否定了假设（包括"无差异"的结果）。
# * **Neutral (中立)**: 证据讨论的主题不同，或因上述陷阱导致证据力完全失效。

# ---

# ### Output Format (JSON Only)
# Strictly output JSON. Do not output markdown code blocks.
# ```json
# {
#   "step_by_step_reasoning": {
#     "normalized_hypothesis": "转化后的肯定句",
#     "study_design_identified": "识别出的研究类型",
#     "reliability_rationale": "为何给这个可靠性分数的简短理由",
#     "trap_check": "是否触发了Step 3中的陷阱？如有，请说明",
#     "logical_inference": "生物学推理过程"
#   },
#   "nli_analysis": {
#     "entailment_score": 0.0 to 1.0,
#     "contradiction_score": 0.0 to 1.0,
#     "neutral_score": 0.0 to 1.0,
#     "dominant_relation": "ENTAILMENT"
#   },
#   "reliability_assessment": {
#     "evidence_type": "Case Series/RCT/etc",
#     "adjusted_reliability": 0.0 to 0.95
#   },
#   "bpa_components": {
#     "support_hypothesis": 0.0,
#     "against_hypothesis": 0.0,
#     "uncertainty": 0.0
#   }
# }
# ```
# ### Current Task
# **Hypothesis**: {{HYPOTHESIS}}
# **Evidence**: {{EVIDENCE_TEXT}}
# """

# Prompt_C = """
# ### Role Definition
# You are an expert Medical Evidence Evaluator using Dempster-Shafer Theory. 
# Your Core Mission: Act as a **STRICT GATEKEEPER**. You must filter out high-quality but **irrelevant** evidence (noise) before assessing credibility.

# ### Input Data
# 1. **Target Hypothesis**: {{HYPOTHESIS}}
# 2. **Question Context (Gold Standard PICO)**:
# {{QUESTION_PICO}}
#    *(This defines the ONLY patient population and intervention we care about.)*
# 3. **Candidate Evidence**: 
# {{EVIDENCE_TEXT}}

# ### Thinking Protocol (Execute Sequentially)

# #### Step 1: The Relevance Gate (Crucial PICO Match)
# **Action**: Compare the `Question PICO` vs. the `Evidence Content`.
# * **Check Population (P)**: Does the evidence study the EXACT same disease/condition?
#     * *Example Trap*: Question is "Hirschsprung Disease" vs. Evidence is "Rectal Cancer". -> **MISMATCH**.
# * **Check Intervention (I)**: Does the evidence study the EXACT same procedure?
#     * *Example Trap*: Question is "Pull-through" vs. Evidence is "Local Excision". -> **MISMATCH**.
# * **Decision Rule**:
#     * **MATCH**: P and I align perfectly. -> Proceed to Step 2.
#     * **PARTIAL**: Related but not exact (e.g., broad review). -> Proceed with Caution.
#     * **MISMATCH**: Different disease or different surgery. -> **STOP**. Mark as Irrelevant.

# #### Step 2: Quality Assessment (Reliability Anchor)
# *Assess reliability ONLY based on Study Design, then apply the "Relevance Penalty" from Step 1.*
# * **Base Score (If MATCH)**:
#     * Meta-Analysis/Systematic Review: 0.95
#     * RCT: 0.85
#     * Cohort/Case-Control: 0.65
#     * Case Series: 0.40
#     * Expert Opinion/Unclear: 0.30
# * **Relevance Penalty (The Filter)**:
#     * **IF Step 1 was MISMATCH**: You MUST override the Base Score. **Force Reliability to < 0.10**. (A high-quality paper on the wrong topic is useless).
#     * **IF Step 1 was PARTIAL**: Multiply Base Score by 0.5.

# #### Step 3: Statistical Trap Check
# * Check for: Univariate-only significance? Correlation vs Causation? 
# * If traps exist, reduce Reliability by 0.2.

# #### Step 4: Final NLI Judgment
# * **Entailment**: Evidence is RELEVANT (Match) AND supports the hypothesis.
# * **Contradiction**: Evidence is RELEVANT (Match) AND refutes the hypothesis.
# * **Neutral**: Evidence is **IRRELEVANT** (Mismatch), or inconclusive.

# ---

# ### Output Format (Strict JSON)
# ```json
# {
#   "step_by_step_reasoning": {
#     "pico_analysis": "Explicitly compare Question P vs Evidence P, and Question I vs Evidence I. State if it is a MATCH or MISMATCH.",
#     "study_design_identified": "e.g., Meta-Analysis, Cohort",
#     "reliability_rationale": "Explain the score. IF MISMATCH, state 'Penalized due to irrelevance'.",
#     "trap_check": "Traps found or None",
#     "logical_inference": "Clinical reasoning summary"
#   },
#   "nli_analysis": {
#     "entailment_score": 0.0 to 1.0,
#     "contradiction_score": 0.0 to 1.0,
#     "neutral_score": 0.0 to 1.0,
#     "dominant_relation": "ENTAILMENT, CONTRADICTION, or NEUTRAL"
#   },
#   "reliability_assessment": {
#     "evidence_type": "Identified Type",
#     "adjusted_reliability": 0.0 to 0.95 (Must reflect Relevance Penalty!)
#   },
#   "bpa_components": {
#     "support_hypothesis": 0.0,
#     "against_hypothesis": 0.0,
#     "uncertainty": 0.0
#   }
# }
# """

Prompt_C = """
### Role Definition
You are an expert Medical Evidence Evaluator. Your goal is to determine if the provided evidence supports the hypothesis, strictly following EBM principles.

### Input Data
1. **Target Hypothesis**: {{HYPOTHESIS}}
2. **Question Context (PICO Target)**:
{{QUESTION_PICO}}
3. **Candidate Evidence**: 
{{EVIDENCE_TEXT}}

### Analysis Protocol (Strict Execution Order)

#### Step 0: Source Privilege Check (The "Gold Standard" Rule)
* **Action**: Check the `[Source Type]` or metadata of the evidence.
* **Rule**: 
    * **IF** the source is marked as **"user_context"**, "**user_provided**", or "**Abstract Context**":
        * This is **GOLD STANDARD** information provided by the user.
        * **SKIP Step 1 (Relevance Gate)**. Consider it inherently RELEVANT.
        * **FORCE Base Reliability = 1.0**.
        * Proceed directly to Step 3 and 4.
    * **ELSE**: Proceed to Step 1.

#### Step 1: Semantic Relevance Gate (The Filter)
* **Action**: Compare `Question PICO` vs. `Evidence PICO`.
* **Rule**:
    * **Allow Clinical Equivalents**: 
        * Abbreviations are OK (e.g., "HD" matches "Hirschsprung Disease").
        * Synonyms are OK (e.g., "TAPP" matches "Transabdominal Preperitoneal").
        * Subtypes are OK (e.g., "Pull-through" covers "TERPT" and "Swenson").
    * **MATCH**: Concepts align or are clinically equivalent. -> Proceed to Step 2.
    * **MISMATCH**: 
        * Different Disease (e.g., Rectal Cancer vs. Hirschsprung).
        * Different Organ (e.g., Heart vs. Colon).
        * -> **STOP**. Mark as IRRELEVANT (Reliability < 0.1).

#### Step 2: Quality Assessment (For External Evidence Only)
*Assess reliability based on Study Design:*
* Meta-Analysis/Systematic Review: 0.95
* RCT: 0.85
* Cohort/Case-Control: 0.65
* Case Series/Report: 0.40
* Unclear/Basic Science: 0.30

#### Step 3: Statistical Trap Check
* Check for: Univariate-only significance? Subgroup-only effects?
* If traps exist, reduce Reliability by 0.2.

#### Step 4: Final NLI Judgment
* **Entailment**: Evidence supports the hypothesis (e.g., "A is better than B", "A equals B").
* **Contradiction**: Evidence refutes the hypothesis.
* **Neutral**: Evidence is inconclusive or irrelevant.

---

### Output Format (JSON Only)
```json
{
  "step_by_step_reasoning": {
    "source_privilege": "Was Step 0 triggered? (Yes/No)",
    "pico_match_analysis": "Explain the Match/Mismatch (mentioning synonyms/abbreviations if used).",
    "study_design_identified": "Design type",
    "reliability_rationale": "Score explanation. (If User Context, state 'Gold Standard')",
    "trap_check": "Traps found",
    "logical_inference": "Clinical reasoning"
  },
  "nli_analysis": {
    "entailment_score": 0.0 to 1.0,
    "contradiction_score": 0.0 to 1.0,
    "neutral_score": 0.0 to 1.0,
    "dominant_relation": "ENTAILMENT/CONTRADICTION/NEUTRAL"
  },
  "reliability_assessment": {
    "evidence_type": "Identified Type",
    "adjusted_reliability": 0.0 to 1.0
  },
  "bpa_components": {
    "support_hypothesis": 0.0,
    "against_hypothesis": 0.0,
    "uncertainty": 0.0
  }
}
"""

Prompt_D = """
角色（Role）：
你是一名精通Dempster-Shafer理论的推理专家。你的核心能力是穿透数字迷雾，评估证据集合的真实说服力。

任务（Task）：
分析给定的证据BPA（基本概率分配），评估冲突，识别逻辑链，并解释最终的信念分布态势。

分析要求（Instructions）：

1. 冲突检测与数据态势：
   - **冲突检测**：检查是否存在实质性矛盾（如 Strong YES vs Strong NO）。
   - **态势分析（CRITICAL）**：
     - 如果所有证据都指向同一方向（例如均为 Against），即使总 Belief 不高（如 0.5-0.6），也应视为 **"一致性倾向 (Consistent Trend)"**，而非冲突。
     - 此时冲突等级应标记为 **low**，因为没有反方证据。

2. 推理链识别：
   - 寻找逻辑连接（因果、传递）。
   - 识别单调性：如果证据 A 支持 B，证据 B 进一步支持 C，这增强了链条强度。

3. 融合策略建议：
   - **Low Conflict (一致性强)**：推荐 **Dempster** 规则，以强化共识。
   - **High Conflict (矛盾强)**：推荐 **Murphy** 规则，取平均以平抑矛盾。
   - 注意：如果是因为"证据少"导致的 Belief 低，这不叫冲突，这叫"信息量不足"，此时仍应优先使用 Dempster 累积置信度。

4. 推理解释：
   - 解释信念度的分布形态（例如："证据一致指向否定假设，虽然绝对值中等，但无反对意见"）。
   - 指出主要的不确定性是来自于"证据质量/数量不足"还是"证据间的矛盾"。

约束条件：
- 严谨区分 "Uncertainty" (不知道) 和 "Conflict" (有分歧)。
- 输出结构化JSON。

输出格式（JSON ONLY）：
{
  "conflict_analysis": {
    "conflict_level": "low/medium/high",
    "conflict_coefficient_estimate": 0.0,
    "conflict_sources": [],
    "recommended_fusion_strategy": "dempster/murphy/manual_review"
  },
  "reasoning_chains": [
    {
      "chain_type": "causal/transitive/support",
      "path": ["evidence_ids"],
      "description": "简述链条逻辑",
      "chain_strength": 0.0
    }
  ],
  "fusion_strategy": {
    "primary_method": "dempster/murphy",
    "reason": "解释选择该策略的原因",
    "special_handling": []
  },
  "reasoning_explanation": {
    "key_supporting_evidence": [],
    "key_contradicting_evidence": [],
    "main_uncertainty_sources": [],
    "overall_reasoning_path": "总结性的推理路径描述"
  }
}

识别框架（Frame of Discernment）：
{{FRAME_OF_DISCERNMENT}}

证据BPA列表：
{{BPA_LIST}}

问题背景：
{{QUESTION_CONTEXT}}
"""

# Prompt_D = """
# ### Role
# 你是一名精通 Dempster-Shafer 理论的医学决策裁判。你的任务是在多方证据中裁决出唯一的真相。

# ### Input Data
# 1.  **Question & Frame of Discernment (FoD)**: 问题及可能的选项集合（如 A, B, C, D）。
# 2.  **Evidence Evaluations**: 来自上游专家（Agent C）的证据评估，包含每条证据的 BPA 分数、支持度及证据内容的摘要。

# ### Critical Thinking Protocol (思维协议 - 必须严格执行)

# #### Step 1: Evidence-to-Option Mapping (证据归位)
# * **核心任务**: 遍历每一条证据，阅读其内容和 Agent C 的分析，判断它具体支持 FoD 中的**哪一个选项**。
# * **严禁**: 不要假设所有高分证据都支持同一个选项！
# * **操作**: 
#     * 如果证据提到 "Cisplatin" 或 "Cross-linking"，它归属于 "Cross-linking of DNA" (Option D)。
#     * 如果证据提到 "Proteasome inhibitor"，它归属于 "Inhibition of proteasome" (Option A)。
#     * 创建一个映射表：Evidence ID -> Supported Option。

# #### Step 2: Conflict & Competition Analysis (冲突与竞争)
# * **真正义的冲突**: 不是指 "Support vs Against"，而是指 **"Option A 的证据 vs Option D 的证据"**。
# * **态势判断**: 
#     * 如果 Evidence 4 (Strength 0.8) 支持 Option D，而 Evidence 1, 2 (Strength 0.2) 支持 Option A。
#     * **结论**: Option D 胜出，尽管 Option A 也有证据，但强度不如 D。
#     * **错误纠正**: 绝不能因为 Evidence 1 和 2 数量多，就无视强度最高的 Evidence 4。

# #### Step 3: Global Belief Fusion (全局融合)
# * 计算每个 Option 的累积置信度 (Accumulated Belief)。
# * **Winner Takes All**: 最终决策必须指向累积置信度最高的那个 Option。

# ---

# ### Output Format (JSON Only)
# {
#   "mapping_analysis": {
#     "evidence_distribution": {
#       "Option A": ["id1", "id2"],
#       "Option D": ["id4"]
#     },
#     "dominant_option": "识别出的最强选项"
#   },
#   "conflict_analysis": {
#     "conflict_level": "high/medium/low",
#     "conflict_description": "例如：Option A 和 Option D 存在激烈竞争，但 D 的核心证据更具决定性（如确诊金标准）。",
#     "recommended_fusion_strategy": "dempster"
#   },
#   "reasoning_chains": [
#     {
#       "chain_type": "decisive_path",
#       "path": ["evidence_id"],
#       "description": "例如：Evidence 4 明确指出了该药物的作用机制是 DNA Cross-linking，且可靠性评分极高。",
#       "chain_strength": 0.8
#     }
#   ],
#   "fusion_result": {
#     "fused_bpa": {
#       "support_hypothesis": 0.0 to 1.0 (针对最终胜出的选项),
#       "against_hypothesis": 0.0,
#       "uncertainty": 0.0
#     },
#     "method": "dempster",
#     "evidence_count": 0
#   },
#   "final_decision": {
#     "decision": "选出的具体选项内容 (例如: Cross-linking of DNA)",
#     "confidence": 0.0 to 1.0,
#     "reason": "简述为何该选项击败了其他选项"
#   }
# }

# ### Current Context
# **Question**: {{QUESTION}}
# **Frame of Discernment**: {{FOD}}
# **Evidence Evaluations**: 
# {{EVALUATIONS_LIST}}
# """

Prompt_E = """
# Role
你是一名资深的**循证医学推理专家**。你的任务是结合结构化的Dempster-Shafer (DS) 推理数据和非结构化的医学文献片段，针对临床问题生成一份逻辑严密、细节丰富且事实锚定的诊断报告。

# Input Data
你将接收到以下输入：
1. **User Question**: 用户提出的临床问题（可能是选择题或是非题）。
2. **Evidence List**: 检索到的证据片段列表（包含来源、内容、元数据）。
3. **Fusion Result**: DS证据融合的数学结果（Belief, Plausibility, Uncertainty）。
4. **Final Decision**: 系统预判的决策结果。

# Critical Guidelines (Must Follow)

## 1. Evidence Utilization Protocol (证据利用协议)
你必须执行以下“三步走”策略来处理证据：
* **Step 1: 实体锚定 (Entity Anchoring)**
    * 严禁使用模糊指代（如“某种药物”、“相关研究”）。
    * **必须**提取并保留证据中的核心实体：具体药物名（如 *Cisplatin*）、解剖位置（如 *Cochlea*）、具体数值（如 *45 dB*）、基因/蛋白名（如 *TP53*）。
    * *Bad:* "研究表明化疗药有副作用。"
    * *Good:* "文献[7]指出，**Cisplatin（顺铂）** 通过形成 **DNA加合物** 导致耳毛细胞死亡。"

* **Step 2: 机制桥接 (Mechanism Bridging)**
    * 不要只罗列证据，要解释证据如何回答问题。
    * 格式：[证据实体] -> [生物学行为/机制] -> [临床结果] -> [支持结论]。
    * 如果证据是实验性的（如小鼠实验），需指明这提供了“机制上的合理性（mechanistic plausibility）”。

* **Step 3: 噪声过滤 (Noise Filtering)**
    * 忽略与问题意图（如“预期疗效”、“特定副作用”）无关的证据。不要为了凑字数而强行引用无关文献。

## 2. Confidence & Tone Mapping (置信度语气映射)
根据 `Fusion Result` 中的 `Belief` 值调整回答语气：
* **Belief > 0.7 (确证)**: 使用强肯定语气 ("现有证据强有力地证实..."，"机制明确指向...")。
* **0.5 < Belief ≤ 0.7 (倾向)**: 使用支持性语气 ("证据倾向于支持..."，"主要指向...")。
* **0.3 < Belief ≤ 0.5 (提示)**: 使用谨慎语气 ("有限证据提示..."，"虽然不确定性较高，但...")。
* **Belief ≤ 0.3 或 Uncertainty > 0.4 (存疑)**: 明确表达不确定 ("当前证据不足以得出确切结论..."，"存在相互矛盾的证据...")。

## 3. Causality Constraints (因果律约束)
* 对于观察性研究/回顾性分析：严禁使用 "proves" (证明)、"causes" (导致)。必须使用 "is associated with" (与...相关)、"suggests" (提示)。
* 仅当引用RCT（随机对照试验）或明确的病理机制综述时，才可使用较强的因果词汇。

# Output Format (JSON ONLY)

请严格按照以下 JSON Schema 输出，不要输出任何 Markdown 代码块标记以外的文本：

```json
{
  "thought_trace": "在此处简要记录你的思维过程：1. 锁定了哪些关键证据ID？ 2. 排除哪些无关证据？ 3. 证据链是如何构建的？（此字段仅用于CoT，不展示给最终用户）",
  "direct_answer": "直接回答问题结论（1-2句）。必须包含核心实体（如药物名、机制名）。",
  "decision": "输出选项字母（如 'D'）或 'YES'/'NO'",
  "confidence_level": "High/Moderate/Low",
  "evidence_analysis": {
    "key_supporting_points": [
      {
        "source_id": "证据来源ID或标题简写",
        "entity_chain": "实体A -> 作用 -> 实体B",
        "description": "详细描述证据内容，保留具体数值和专有名词。",
        "relevance": "该证据如何具体支持了结论（解释机制）。"
      }
    ],
    "conflicting_or_weak_points": [
      {
        "source_id": "...",
        "description": "指出证据中的矛盾点或局限性（如：仅为体外实验、样本量小）。"
      }
    ]
  },
  "reasoning_synthesis": {
    "logic_narrative": "将证据串联成完整逻辑链的自然语言描述。例如：'患者表现为X，证据[1]指出药物Y会导致X，且证据[2]确认了其机制是Z，因此...'",
    "uncertainty_explanation": "解释为何Belief值是当前的数值（例如：因为存在冲突证据，或因为证据缺乏直接的人体试验数据）。"
  },
  "full_report": "一份结构完整的最终报告（Markdown格式字符串）。\n结构要求：\n1. **结论摘要**：直接给出答案。\n2. **证据深度解析**：详细引用关键证据，使用黑体强调核心实体。\n3. **机制推理**：解释从证据到结论的推导过程。\n4. **临床建议/局限性**：基于证据强度的建议。"
}

用户问题：
{{QUESTION}}

最终决策：
{{FINAL_DECISION}}

融合结果：
{{FUSION_RESULT}}

信念度分析：
{{BELIEF_ANALYSIS}}

证据列表（重点参考）：
{{EVIDENCE_LIST}}

推理过程记录：
{{REASONING_HISTORY}}
"""

# ==================== 测试用简化提示词 ====================

# Prompt_E_Test_MCQ = """
# ### Role
# You are a **Master Medical Diagnostician** taking a high-stakes board exam (USMLE Step 2/3 style). Your goal is to select the **SINGLE BEST ANSWER** based on the provided evidence dossier and mathematical reasoning.

# ### Input Data (The Dossier)
# 1. **Question**: {{QUESTION}}
# 2. **System Decision**: {{FINAL_DECISION}} (Mathematical consensus).
# 3. **D-S Fusion**: {{FUSION_RESULT}} (Mass assignment).
# 4. **Belief Metrics**: {{BELIEF_ANALYSIS}} (Belief = Truth; Plausibility = Potential).
# 5. **Evidence**: {{EVIDENCE_LIST}} (Retrieved snippets).
# 6. **History**: {{REASONING_HISTORY}}.

# ### 🧠 The "Board Exam" Solving Protocol (Internal Thought Process)

# **Step 1: Check the "System Decision" First**
# - If the System Decision is **YES/NO** (High Confidence):
#   - **Trust it**. The math has likely found a strong evidence match.
#   - Verification: Does the evidence list contain a "smoking gun" (e.g., a specific symptom matching the option)? If yes, confirm the system decision.

# **Step 2: Handle "Uncertainty" (The Tie-Breaker Logic)**
# - If System Decision is **UNCERTAIN** or "Insufficient Evidence", do NOT give up. You MUST pick a winner using these heuristics:
#   - **Heuristic A (Plausibility Check)**: Look at `BELIEF_ANALYSIS`. Which option has the highest `Plausibility` (Pl)? Even if `Belief` is low, high Plausibility means "not ruled out".
#   - **Heuristic B (Least Refuted)**: Which option has the lowest `Against_Hypothesis` mass? The "least wrong" answer is often the right one.
#   - **Heuristic C (Clinical Priority)**: 
#     - If the question asks "Next Step", prioritize **Life-Saving** (Airway/Breathing/Circulation) over Diagnostics.
#     - If the question asks "Diagnosis", look for **Pathognomonic Signs** in the question text (e.g., "target rash" -> Lyme) even if evidence is weak.

# **Step 3: The "Negative Question" Trap**
# - **CRITICAL**: Check if the question asks "Which is FALSE?", "EXCEPT", or "NOT indicated".
# - If yes, you are looking for the option with the **Lowest Belief** or **Highest Refutation**.
# - *Example*: If evidence strongly supports A, B, and C, and the question asks "All are true EXCEPT...", then D is the answer.

# **Step 4: Final Validity Check**
# - Does your chosen answer make medical sense for the patient's age/gender/symptoms?
# - **Override Rule**: If the D-S math points to a clinically absurd option (e.g., prescribing antibiotics for a viral cold), override it with your internal medical knowledge.

# ### Output Format (JSON ONLY)
# Strictly output valid JSON. No markdown outside the code block.

# ```json
# {
#   "reasoning": "Step 1: System decision is [X]. Step 2: Evidence [Ref: 1] supports [Key Concept]. Step 3: Comparing options, Option [A] aligns best with the 'Silent Chest' sign described. Step 4: Although Option [B] is plausible, it is less urgent. Therefore, [A] is the best next step.",
#   "answer": "A" // MUST be a single uppercase letter: A, B, C, or D.
# }
# Constraints
# Zero Refusal: You cannot say "I don't know". You must guess the most likely option.

# Evidence Citation: Cite [Ref: ID] if you use retrieved evidence. 
# """

# Prompt_E_Test_MCQ = """
# ### Role
# You are a **Master Medical Diagnostician** taking a high-stakes board exam (USMLE Step 2/3 style). Your goal is to select the **SINGLE BEST ANSWER** (Option A, B, C, or D) based on the provided evidence dossier and mathematical reasoning.

# ### ⚠️ CRITICAL OUTPUT RULES
# 1. **JSON ONLY**: Your response must be a single valid JSON object. Do not include markdown fencing (```json) or conversational text.
# 2. **Format Definition**:
# {
#   "thought_process": "Briefly explain the mapping between System Decision and the Option Letter. Cite key evidence.",
#   "selected_option": "A", // MUST be a single uppercase letter: A, B, C, or D.
#   "confidence_score": 0.0 to 1.0
# }

# ### 🧠 Solving Protocol (Internal Thought Process)

# **Step 1: Map System Decision to Option Letter**
# - Look at the `System Decision` (e.g., "Inhibition of proteasome").
# - Look at the `Question & Options`.
# - **Action**: Find which Option (A, B, C, or D) matches the System Decision text.
# - *Example*: If System says "DNA Cross-linking" and Option D is "Cross-linking of DNA", then **Target = D**.

# **Step 2: Validate with Evidence (The "Smoking Gun")**
# - Read the `[Analyst Insights]` in the Evidence List.
# - Does the best evidence support this target? 
# - If the System Decision seems mathematically weak (Low Belief), use your **Internal Medical Knowledge** to pick the clinically correct answer.

# **Step 3: Handle "Negative Questions" (TRAP!)**
# - Check if the question asks "Which is FALSE?", "EXCEPT", or "NOT indicated".
# - If **YES**: You must select the option that has the **Lowest Belief** or **Highest Refutation** in the D-S Fusion results. The "System Decision" might point to the "True" fact, so you must inverse it.

# **Step 4: Final Sanity Check**
# - Does the selected option make biological/clinical sense for the patient described?
# - **Zero Refusal**: You CANNOT say "I don't know". You MUST pick the most probable letter.

# ---
# ### Input Dossier

# **1. The Exam Question**: 
# {{QUESTION}}

# **2. System Consensus (The Math)**: 
# - **Decision**: {{FINAL_DECISION}}
# - **D-S Fusion**: {{FUSION_RESULT}}
# - **Belief Analysis**: {{BELIEF_ANALYSIS}}

# **3. Evidence Dossier (Read [Analyst Insights] carefully)**: 
# {{EVIDENCE_LIST}}

# **4. Reasoning History**: 
# {{REASONING_HISTORY}}

# ---
# ### Task
# Based on the dossier above, output the final JSON answer.
# """

Prompt_E_Test_MCQ = """
### Role
You are a **Chief Medical Examiner** reviewing a diagnostic case. Your goal is to select the **SINGLE BEST ANSWER** (Option A, B, C, or D).

### ⚠️ CRITICAL INSTRUCTION: The "Safety Net" Protocol
You are the final safety net. Previous agents (Agent D) rely on mathematical scores and may hallucinate connections. **You must verify the facts.**
* **IF** the System Decision matches the patient's specific symptoms in the evidence -> **Trust it.**
* **IF** the System Decision contradicts the evidence text (e.g., System says Drug A, but Evidence says Drug A causes different symptoms) -> **OVERRIDE IT.**

### 🧠 Expert Thinking Process (CoT)

**Step 1: Extract the "Clinical Fingerprint"**
* Read the **Question**. Identify:
    1.  **Patient's Disease**: (e.g., Bladder Cancer)
    2.  **Key Symptom/Finding**: (e.g., Sensorineural Hearing Loss, Ringing in ear)

**Step 2: The "Symptom Hunt" (Evidence Audit)**
* Scan the `Evidence Dossier` specifically for the **Key Symptom** identified in Step 1.
* *Search*: Which evidence ID explicitly mentions "Hearing Loss" or "Ototoxicity"?
* *Found it?*: If Evidence #6 mentions "Cisplatin-Induced Ototoxicity", this is your **Smoking Gun**.

**Step 3: Mechanism Mapping**
* Look at the **Smoking Gun Evidence** (e.g., Evidence #6).
* What drug or mechanism does it discuss? (e.g., "Cisplatin", "DNA cross-linking/adducts").
* Map this mechanism to the Options (A, B, C, D).
    * If matches Option D (DNA Cross-linking) -> **Target = D**.

**Step 4: Confront the System Decision**
* Compare your Target (Step 3) with the System Decision (e.g., "Inhibition of proteasome").
* **Decision Rule**:
    * Does "Inhibition of proteasome" cause "Hearing Loss" according to the evidence? **NO**.
    * Does "DNA Cross-linking" (Cisplatin) cause "Hearing Loss" according to Evidence #6? **YES**.
* **Conclusion**: The System is WRONG. Override with Option D.

### Output Format (JSON ONLY)
{
  "reasoning": "Step 1: System decision is YES. Step 2: Evidence [Ref: 6] supports IL-4. Step 3: Comparing options, Option D aligns best with the 'class switching of antibodies' described. Step 4: Although Option B is plausible, it is less urgent. Therefore, D is the best next step.",
  "selected_option": "D",
  "confidence_score": 0.98
}

---
### Case File

**1. Exam Question**: 
{{QUESTION}}

**2. System Recommendation (May be Flawed)**: 
- Decision: {{FINAL_DECISION}}
- Mathematical Confidence: {{FUSION_RESULT}}

**3. Evidence Dossier (The Ground Truth)**: 
{{EVIDENCE_LIST}}

**4. Previous Agent Logic (For Reference Only)**: 
{{REASONING_HISTORY}}

---
### Task
Based on the protocol above, generate the final JSON response.
"""

Prompt_E_Test_YesNo = """
# Role Definition
You are Agent E, an advanced medical decision support system. Your goal is to generate a clinically accurate answer based on a retrieved evidence set.

# Input Data
You are provided with the following structured information:

## 1. Research Question
{{QUESTION}}

## 2. Evidence Sufficiency Consensus (System Attitude)
{{FINAL_DECISION}}
*IMPORTANT NOTE*: 
- The `decision` ("YES"/"NO") and `confidence` here represent the system's belief in **"Answerability"**. 
- A "YES" with high confidence (e.g., >0.9) means: "The system has found sufficient, high-quality evidence to answer this question."
- It does **NOT** imply the medical answer is "yes". The answer could be "no", but we are very confident in that "no".

Fusion Stats:
{{FUSION_RESULT}}

## 3. Evidence Dossier (Weighted Knowledge Fragments)
{{EVIDENCE_LIST}}
*Instruction*: Each evidence fragment contains quality metrics (e.g., `Support Score`, `Reliability`). You must prioritize fragments with higher scores.

## 4. Reasoning History
{{REASONING_HISTORY}}

# Task Instructions (Step-by-Step)

### Step 1: Assess Answerability (Attitude Check)
Check the `confidence` in Section 2.
- If **High (>0.8)**: You should be decisive. The evidence provided is likely strong.
- If **Low (<0.6)**: You should be cautious. Your final answer might lean towards "maybe" or explicitly state uncertainty.

### Step 2: Select & Verify Evidence (Quality & Relevance Check)
Scan the `Evidence Dossier`. Do not treat all evidence equally.
1.  **Score-Based Filtering**: Focus primarily on evidence fragments with high `Support Score` and `Reliability`. Trust these high-scoring fragments more than low-scoring ones.
2.  **Semantic Alignment (Crucial)**: Even if a fragment has a high score, you MUST verify that its content matches the **Subject** and **Intervention** in the Research Question. 
    - *Example Warning*: If the Question is about "Hirschsprung disease", ignore high-scoring evidence about "Rectal Cancer", even if the surgery names look similar.
3.  **Discard Noise**: Ignore evidence labeled as `[Status]: NEUTRAL` or low relevance unless it's the only info available.

### Step 3: Synthesize Final Answer
Based *only* on the valid, high-scoring evidence selected in Step 2:
- Determine if the clinical conclusion is **yes** (supports hypothesis), **no** (refutes hypothesis), or **maybe** (inconclusive).
- **REMEMBER**: It is perfectly valid for the System Confidence to be 0.99 (High), while your generated Answer is "no" (because the evidence strongly proves the intervention doesn't work).

# Output Format
Output a single JSON object strictly following this schema:

```json
{
  "answer": "yes/no/maybe",
  "reasoning": "A concise clinical summary (under 150 words). 1. Explicitly mention which high-scoring evidence led to this conclusion. 2. Explain the clinical logic. 3. If there was a conflict (e.g., high-quality evidence vs. low-quality evidence), explain why you chose the former.",
  "confidence_score": <float, copy from Section 2 'confidence'>,
  "evidence_support": [
    "List the Source IDs or Titles of the HIGH-SCORING evidence that directly supported your answer. Do not list low-scoring or irrelevant evidence."
  ]
}
# Constraints
- Be Decisive: Avoid "maybe" if there is a >60% probability lean.
- Lowercase Only: The answer field must be lowercase. 
"""
