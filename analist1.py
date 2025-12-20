from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import set_argument
from retriever import retrieve_process

args = set_argument()
llm = ChatOpenAI(
    model = "gpt-4o-mini", 
    temperature = 0,
    api_key = args.api_key_gpt,
    base_url = args.api_url_gpt
    )

prompt = """
# Role
你是一位严谨的生物医学证据评估专家。你的任务是针对给定的【待解决问题】，评估【检索到的信息片段】作为证据的价值。

# Input Data
1. **待解决问题 (Question)**{question}: 一个复杂的生物医学问题。
2. **检索到的信息片段 (Evidence Snippet)**{context}: 一段来自论文、网页或数据库的文本。
3. **元数据 (Metadata)**{metadata}: (可选) 来源年份、期刊名称、作者等。

# Evaluation Dimensions (必须严格遵循的分析维度)

请从以下 5 个维度对证据进行深度解析：

1.  **相关性与粒度 (Relevance & Granularity)**:
    * 该信息是否包含问题中的关键实体（基因、药物、疾病等）？
    * 粒度是否匹配？（例如：问题询问具体分子通路，证据只谈论宏观疗效，则粒度不匹配）。
    * 分析该片段的核心意图是否真正解决了问题中的疑问？
    * 评分：0-10分（10为极度相关且粒度完美）。

2.  **逻辑立场 (Stance/Polarity)**:
    * 相对于问题的假设或陈述，该证据是：
        * `Support`: 明确支持。
        * `Refute`: 明确反驳/阴性结果。
        * `Neutral`: 提及相关概念但无明确方向性结论。
        * `Conditional`: 仅在特定条件下（如特定剂量、特定基因型）支持。

3.  **生物医学情境匹配 (Contextual Fit) [关键]**:
    * **物种检查**: 证据是基于人类 (Human)、动物模型 (Mouse/Primate)、细胞 (Cell line) 还是 计算机模拟 (In silico)？
    * **人群/样本**: 年龄、疾病分期、合并症是否与问题隐含的背景一致？
    * **适用性缺口 (Applicability Gap)**: 证据是否明确指出该证据在应用到当前问题时存在的“逻辑跳跃”或“推断风险”。
    * 如果情境严重不匹配（如用体外实验直接回答临床预后问题），必须大幅降低总评分。

4.  **证据质量 (Quality)**:
    * 基于文本内容的科学严谨性判断。
    * 是否包含具体的统计数据（P值、CI）、样本量 (N) 或实验设计描述？
    * (如果有元数据) 来源是否权威？
    * 该证据是否涵盖回答问题所需要的关键要素或因果链条？
    * 是否提供充分细节（如人群、机制、统计数据等）

5.  **时效性 (Timeliness)**:
    * 该证据是否可能过时？（特别是对于药物指南、临床试验结果）。

6.  **可解释性 (Explainability)**:
    * 该证据是否易于理解和解释？是否包含复杂的术语或模糊的表述？

7.  **语义理解（Semantic Understanding）**:
    * 该证据是否存在歧义？是否有多种可能的解释？
    * 考虑片段中的语义情景，是否存在乐观等情形，从而导致非字面性意思。


# Output Format (JSON)

请仅输出一个标准的 JSON 对象，不要包含任何额外的 Markdown 格式或解释文本。
注意：所有字段均为必填项，不能省略，并输出为中文。
{{
  "relevance_score": <0-10, 整数>,
  "stance": "<Support | Refute | Neutral | Conditional>",
  "context_analysis": {{
    "primary_findings": "<核心结果，包含数据>",
    "semantic_understanding":"<语义理解，如：是否存在歧义、乐观等情形>",
    "model_organism": "<Human | Mouse | Cell | Unknown | ...>",
    "is_context_mismatch": <true | false>,
    "explicit_limitations": "<提取文中提到的局限性>"
  }},
  "key_fact": "<提炼出的核心事实，不超过30字>",
  "reasoning": "<简要说明评分理由，特别是指出任何逻辑漏洞或情境不匹配>",
  "utility_verdict": "<High | Medium | Low | Discard>"
}}
"""
metadata = ""
# ========== 测试案例1 ==========   正确答案：B  原来流程答案：A  该流程答案：A
# question = "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?"
# option = {
#       "A": "Disclose the error to the patient and put it in the operative report",
#       "B": "Tell the attending that he cannot fail to disclose this mistake",
#       "C": "Report the physician to the ethics committee",
#       "D": "Refuse to dictate the operative report"
#     }
# ========== 测试案例2 ==========   正确答案：D  原来流程答案：C  该流程答案：D
# question = "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?"
# option = {
#       "A": "Inhibition of proteasome",
#       "B": "Hyperstabilization of microtubules",
#       "C": "Generation of free radicals",
#       "D": "Cross-linking of DNA"
# }
# ========== 测试案例3 ==========   正确答案：D  原来流程答案：B  该流程答案：A/B
# question = "A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and blood pressure is 80/50 mm Hg. There is blood oozing around the site of a peripheral intravenous line. Pelvic examination shows mucopurulent discharge from the cervical os and left adnexal tenderness. Laboratory studies show:\nPlatelet count 14,200/mm3\nFibrinogen 83 mg/mL (N = 200–430 mg/dL)\nD-dimer 965 ng/mL (N < 500 ng/mL)\nWhen phenol is applied to a sample of the patient's blood at 90°C, a phosphorylated N-acetylglucosamine dimer with 6 fatty acids attached to a polysaccharide side chain is identified. A blood culture is most likely to show which of the following?"
# option = {
#       "A": "Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar",
#       "B": "Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal blood agar",
#       "C": "Spore-forming, gram-positive bacilli forming yellow colonies on casein agar",
#       "D": "Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar"
# }
# ========== 测试案例4 ==========   正确答案：D  原来流程答案：B  该流程答案：D
# question = "A 39-year-old man presents to the emergency department because of progressively worsening chest pain and nausea that started at a local bar 30 minutes prior. The pain radiates to the epigastric area. He has a 5-year history of untreated hypertension. He has smoked 1 pack of cigarettes daily for the past 5 years and started abusing cocaine 2 weeks before his emergency room visit. The patient is diaphoretic and in marked distress. What should be the first step in management?"
# option = {
#         "A": "Diltiazem",
#         "B": "Labetalol",
#         "C": "Propranolol",
#         "D": "Reassurance and continuous monitoring"
# }
# ========== 测试案例5 ==========   正确答案：D  原来流程答案：A  该流程答案：A
# question = "A 62-year-old patient has been hospitalized for a week due to a stroke. One week into the hospitalization, he develops a fever and purulent cough. His vitals include: heart rate 88/min, respiratory rate 20/min, temperature 38.4°C (101.1°F), and blood pressure 110/85 mm Hg. On physical examination, he has basal crackles on the right side of the chest. Chest radiography shows a new consolidation on the same side. Complete blood count is as follows:\nHemoglobin 16 mg/dL\nHematocrit 50%\nLeukocyte count 8,900/mm3\nNeutrophils 72%\nBands 4%\nEosinophils 2%\nBasophils 0%\nLymphocytes 17%\nMonocytes 5%\nPlatelet count 280,000/mm3\nWhat is the most likely causal microorganism?"
# option = {
#           "A": "Streptococcus pneumoniae",
#           "B": "Mycobacterium tuberculosis",
#           "C": "Haemophilus influenzae",
#           "D": "Staphylococcus aureus"
# }
# ========== 测试案例6 ==========   正确答案：D  原来流程答案：C  该流程答案：D
# question = "A 22-year-old woman is brought to the emergency department because of a 2-day history of fever, intermittent rigors, and night sweats. She also has a 1-month history of progressive fatigue. Five weeks ago, she was hospitalized and received intravenous antibiotics for treatment of bacterial meningitis while visiting relatives in Guatemala. Her temperature is 39.4°C (102.9°F), pulse is 130/min, and blood pressure is 105/70 mm Hg. Examination shows pallor and scattered petechiae and ecchymoses. Laboratory studies show a hemoglobin concentration of 9.0 g/dL, a leukocyte count of 1,100/mm3 with 30% segmented neutrophils, and a platelet count of 20,000/mm3 . Blood cultures grow coagulase-negative staphylococci. The patient was most likely treated with which of the following antibiotics?"
# option = {
#           "A": "Doxycycline",
#           "B": "Trimethoprim/sulfamethoxazole",
#           "C": "Linezolid",
#           "D": "Chloramphenicol"
# }
# ========== 测试案例7 ==========   正确答案：B  原来流程答案：D  该流程答案：C/D
# question = "An otherwise healthy 50-year-old man comes to the physician because of a 6-month history of increasingly frequent episodes of upper abdominal pain, nausea, vomiting, and diarrhea. He has had a 3.2-kg (7-lb) weight loss during this time. Physical examination shows bilateral pitting pedal edema. An endoscopy shows prominent rugae in the gastric fundus. Biopsy shows parietal cell atrophy. Which of the following is the most likely underlying cause?"
# option = {
#           "A": "Serotonin-secreting gastric tumor",
#           "B": "Proliferation of gastric mucus-producing cells",
#           "C": "Excessive somatostatin secretion",
#           "D": "Ectopic secretion of gastrin"
# }
# ========== 测试案例8 ==========   正确答案：A  原来流程答案：C  该流程答案：C
question = "A 17-year-old girl is referred by her dentist for a suspected eating disorder. She has been visiting the same dentist since childhood and for the past 2 years has had at least 2 visits for dental caries. She eventually admitted to him that she regularly induces vomiting by putting her fingers down her throat. She says she has been doing this for the last few years and purging at least once a week. More recently, she has been inducing emesis more often and even looked into diuretics as she feels that she is gaining more and more weight compared to her ‘skinny friends’. Her BMI is at the 50th percentile for her age and sex. Which of the following features is most consistent with this patient’s condition?"
option = {
          "A": "Patients with this disorder are not further sub-typed",
          "B": "Patients do not usually initiate treatment",
          "C": "Patients can have a history of both anorexia and bulimia",
          "D": "Patients will typically have a BMI between 17–18.5 kg/m2"
}
external_knowledge = retrieve_process(question)
context = f"""
    "options": {option},
    "External knowledge": {external_knowledge}
"""
parser = StrOutputParser()
template = PromptTemplate(
    input_variables = ["question", "context", "metadata"],
    template = prompt,
)

chain = template | llm | parser
res = chain.invoke({
    "question": question,
    "context": context,
    "metadata": metadata
})
print(res)
full_context = f"expert analysis: {res}"

prompt1 = """
    You are a biomedical expert answering MedQA.
    This task is to select the option from 'A', 'B', 'C', 'D' that can answer this question.

    Question:
    {question}

    Options:
    {option}

    Expert Analysis Process:
    {full_context} 

    Instructions:
    1. Analyze the evidence carefully.
    2. Select the single best answer (A, B, or C).
    3. Provide a brief explanation.
    4. You MUST output a valid JSON object strictly in the following format:
    {{
        "explanation": "Your brief explanation here.",
        "final_answer": "A" 
    }}
    (Note: final_answer must be one of "A", "B", or "C")
    """

template1 = PromptTemplate(
    input_variables=["question","option","full_context"],
    template=prompt1
)

chain = template1 | llm | parser
res = chain.invoke({
    "question": question,
    "option": option,
    "full_context": full_context
})
print(f"最终QA答案{res}")