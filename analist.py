from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import set_argument

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
# ===========测试案例1============   专家答案：A  原本流程答案：B  该流程答案：B
# question = "Risk stratification in emergency surgical patients: is the APACHE II score a reliable marker of physiological impairment?"
# context= """
#             "The APACHE II (Acute Physiology and Chronic Health Evaluation II) score used as an intensive care unit (ICU) admission score in emergency surgical patients is not independent of the effects of treatment and might lead to considerable bias in the comparability of defined groups of patients and in the evaluation of treatment policies. Postoperative monitoring with the APACHE II score is clinically irrelevant.",
#             "Inception cohort study.",
#             "Secondary referral center.",
#             "Eighty-five consecutive emergency surgical patients admitted to the surgical ICU in 1999. The APACHE II score was calculated before surgery; after admission to the ICU; and on postoperative days 3, 7, and 10.",
#             "APACHE II scores and predicted and observed mortality rates.",
#             "The mean +/- SD APACHE II score of 24.2 +/- 8.3 at admission to the ICU was approximately 36% greater than the initial APACHE II score of 17.8 +/- 7.7, a difference that was highly statistically significant (P<.001). The overall mortality of 32% favorably corresponds with the predicted mortality of 34% according to the initial APACHE II score. However, the predicted mortality of 50% according to the APACHE II score at admission to the ICU was significantly different from the observed mortality rate (P =.02). In 40 long-term patients (>/=10 days in the ICU), the difference between the APACHE II scores of survivors and patients who died was statistically significant on day 10 (P =.04)."
#         """
# ===========测试案例2============   专家答案：A  原本流程答案：B  该流程答案：A
# question = "\"Occult\" posttraumatic lesions of the knee: can magnetic resonance substitute for diagnostic arthroscopy?"
# context = """
#             We investigated the actual role of MRI versus arthroscopy in the detection and characterization of occult bone and/or cartilage injuries in patients with previous musculoskeletal trauma of the knee, pain and severe functional impairment. Occult post-traumatic osteochondral injuries of the knee are trauma-related bone and/or cartilage damage missed at plain radiography.
#             We retrospectively selected 70 patients (men:women = 7:3; age range: 35 +/- 7 years) with a history of acute musculoskeletal trauma, negative conventional radiographs, pain and limited joint movements. All patients were submitted to conventional radiography, arthroscopy and MRI, the latter with 0.5 T units and T1-weighted SE. T2-weighted GE and FIR sequences with fat suppression.
#             We identified three types of occult post-traumatic injuries by morpho-topographic and signal intensity patterns: bone bruises (no. 25), subchondral (no. 33) and osteochondral (no. 35) injuries. Arthroscopy depicted 45 osteochondral and 19 chondral injuries. A bone bruise was defined as a typical subcortical area of signal loss, with various shapes, on T1-weighted images and of increased signal intensity on T2-weighted and FIR images. The cortical bone and articular cartilage were normal in all cases, while osteochondral injuries exhibited associated bone and cartilage damage with the same abnormal MR signal intensity. Sprain was the mechanism of injury in 52 cases, bruise in 12 and stress in 6. In 52 sprains (30 in valgus), the injury site was the lateral compartment in 92.3% of cases (100% in valgus), associated with meniscal damage in 73% of cases (90% in valgus) and with ligament injury in 90.4% (100% in valgus). In 12 bruises, the injury site was the lateral compartment in 58.3% of cases, the knee cap in 25% and the medial compartment in 16.7%; meniscal damage was associated in 25% of cases and ligament damage in 8.3%. In 6 stress injuries, the injury site was localized in the medial tibial condyle in 80% of cases, while meniscal and ligament tears were absent.
#         """
# ===========测试案例3============    专家答案：A  原本流程答案：B  该流程答案：C
# question = "Is muscle power related to running speed with changes of direction?"
# context = """
#         The purpose of this study was to identify the relationships between leg muscle power and sprinting speed with changes of direction.
#         the study was designed to describe relationships between physical qualities and a component of sports performance.
#         testing was conducted in an indoor sports hall and a biomechanics laboratory.
#         15 male participants were required to be free of injury and have recent experience competing in sports involving sprints with changes of direction.
#         subjects were timed in 8 m sprints in a straight line and with various changes of direction. They were also tested for bilateral and unilateral leg extensor muscle concentric power output by an isokinetic squat and reactive strength by a drop jump.
#         The correlations between concentric power and straight sprinting speed were non-significant whereas the relationships between reactive strength and straight speed were statistically significant. Correlations between muscle power and speed while changing direction were generally low and non-significant for concentric leg power with some moderate and significant (p<0.05) coefficients found for reactive strength. The participants who turned faster to one side tended to have a reactive strength dominance in the leg responsible for the push-off action.
#         """
# ===========测试案例4============    专家答案：A  原本流程答案：C  该流程答案：A
# question= "Is portable ultrasonography accurate in the evaluation of Schanz pin placement during extremity fracture fixation in austere environments?"
# context = """
#         The purpose of this study was to investigate the efficacy of ultrasonography to confirm Schanz pin placement in a cadaveric model, and the interobserver repeatability of the ultrasound methodology.
#         This investigation is a repeated measures cadaveric study with multiple examiners.
#         Cadaveric preparation and observations were done by an orthopaedic traumatologist and resident, and two general surgery traumatologists.
#         A total of 16 Schanz pins were equally placed in bilateral femora and tibiae. Four examiners took measurements of pin protrusion beyond the distal cortices using first ultrasonography and then by direct measurement after gross dissection.MAIN OUTCOME MEASURE(S): Distal Schanz pin protrusion length measurements from both ultrasonography and direct measurement post dissection.
#         Schanz pin protrusion measurements are underestimated by ultrasonography (p<0.01) by an average of 10 percent over the range of 5 to 18 mm, and they display a proportional bias that increases the under reporting as the magnitude of pin protrusion increases. Ultrasound data demonstrate good linear correlation and closely represent actual protrusion values in the 5 to 12 mm range. Interobserver repeatability analysis demonstrated that all examiners were not statistically different in their measurements despite minimal familiarity with the ultrasound methodology (p>0.8).
#     """
# ===========测试案例5============    专家答案：A  原本流程答案：C  该流程答案：A
# question = "Exploratory study in patients with mild to moderate sleep apnoea, limited treatment duration; concomitant hypnotic treatment (35%); lack of correction for multiplicity of testing.\nProof of concept study: does fenofibrate have a role in sleep apnoea syndrome?"
# context = """
#           To investigate the effect of fenofibrate on sleep apnoea indices.
#           Proof-of-concept study comprising a placebo run-in period (1 week, 5 weeks if fibrate washout was required) and a 4-week randomized, double-blind treatment period. Thirty-four subjects (mean age 55 years, body mass index 34 kg/m 2 , fasting triglycerides 3.5 mmol/L) with diagnosed sleep apnoea syndrome not treated with continuous positive airways pressure were enrolled and randomized to once daily treatment with fenofibrate (145 mg NanoCrystal(R) tablet) or placebo. 
#           Overnight polysomnography, computerized attention/vigilance tests and blood sampling for measurement of lipids, insulin, fasting plasma glucose and fibrinogen were performed at the end of each study period.\nNCT00816829.\nAs this was an exploratory study, a range of sleep variables were evaluated. 
#           The apnoea/hypopnoea index (AHI) and percentage of time spent with arterial oxygen saturation (SpO(2))<90% were relevant as they have been evaluated in other clinical trials. Other variables included total apnoeas, hypopnoeas and oxygen desaturations, and non-cortical micro-awakenings related to respiratory events per hour.\nFenofibrate treatment significantly reduced the percentage of time with SpO(2)<90% (from 9.0% to 3.5% vs. 10.0% to 11.5% with placebo, p = 0.007), although there was no significant change in the AHI (reduction vs. control 14% (95%CI -47 to 40%, p = 0.533). Treatment reduced obstructive apnoeas (by 44%, from 18.5 at baseline to 15.0 at end of treatment vs. 29.0 to 30.5 on placebo, p = 0.048), and non-cortical micro-awakenings per hour (from 23.5 to 18.0 vs. 24.0 to 25.0 with placebo, p = 0.004). 
#           Other sleep variables were not significantly influenced by fenofibrate.
#     """
# ===========测试案例6============    专家答案：A  原本流程答案：C  该流程答案：C
# question = "Implementation of epidural analgesia for labor: is the standard of effective analgesia reachable in all women?"
# context = """
#             Social and cultural factors combined with little information may prevent the diffusion of epidural analgesia for pain relief during childbirth. The present study was launched contemporarily to the implementation of analgesia for labor in our Department in order to perform a 2 years audit on its use. The goal is to evaluate the epidural acceptance and penetration into hospital practice by women and care givers and safety and efficacy during childbirth.
#             This audit cycle measured epidural analgesia performance against 4 standards: (1) Implementation of epidural analgesia for labor to all patients; (2) Acceptance and good satisfaction level reported by patients and caregivers. (3) Effectiveness of labor analgesia; (4) No maternal or fetal side effects.
#             During the audit period epidural analgesia increased from "15.5%" of all labors in the first trimester of the study to "51%" in the last trimester (p<0.005). Satisfaction levels reported by patients and care givers were good. A hierarchical clustering analysis identified two clusters based on VAS (Visual Analogue Scale) time course: in 226 patients (cluster 1) VAS decreased from 8.5±1.4 before to 4.1±1.3 after epidural analgesia; in 1002 patients (cluster 2) VAS decreased from 8.12±1.7 before (NS vs cluster 1), to 0.76±0.79 after (p<0.001 vs before and vs cluster 2 after). 
#             No other differences between clusters were observed.
#     """
# ===========测试案例7============    专家答案：A  原本流程答案：B  该流程答案：C
# question = "Longer term quality of life and outcome in stroke patients: is the Barthel index alone an adequate measure of outcome?"
# context = """
#         To consider whether the Barthel Index alone provides sufficient information about the long term outcome of stroke.
#         Cross sectional follow up study with a structured interview questionnaire and measures of impairment, disability, handicap, and general health. The scales used were the hospital anxiety and depression scale, mini mental state examination, Barthel index, modified Rankin scale, London handicap scale, Frenchay activities index, SF36, Nottingham health profile, life satisfaction index, and the caregiver strain index.
#         South east London.
#         People, and their identified carers, resident in south east London in 1989-90 when they had their first in a life-time stroke aged under 75 years.\nObservational study.
#         Comparison and correlation of the individual Barthel index scores with the scores on other outcome measures.
#         One hundred and twenty three (42%) people were known to be alive, of whom 106 (86%) were interviewed. The median age was 71 years (range 34-79). The mean interval between the stroke and follow up was 4.9 years. The rank correlation coefficients between the Barthel and the different dimensions of the SF36 ranged from r = 0.217 (with the role emotional dimension) to r = 0.810 (with the physical functioning dimension); with the Nottingham health profile the range was r = -0.189 (with the sleep dimension, NS) to r = -0.840 (with the physical mobility dimension); with the hospital and anxiety scale depression component the coefficient was r = -0.563, with the life satisfaction index r = 0.361, with the London handicap scale r = 0.726 and with the Frenchay activities index r = 0.826.
# """
# ===========测试案例8============    专家答案：A  原本流程答案：B  该流程答案：B
# question = "Does the SCL 90-R obsessive-compulsive dimension identify cognitive impairments?"
# context = """
#           To investigate the relevance of the Symptom Checklist 90-R Obsessive-Compulsive subscale to cognition in individuals with brain tumor.
#           A prospective study of patients assessed with a neuropsychological test battery.
#           A university medical center.\nNineteen adults with biopsy-confirmed diagnoses of malignant brain tumors were assessed prior to aggressive chemotherapy.
#           Included in the assessment were the Mattis Dementia Rating Scale, California Verbal Learning Test, Trail Making Test B, Symptom Checklist 90-R, Mood Assessment Scale, Beck Anxiety Inventory, and Chronic Illness Problem Inventory.
#           The SCL 90-R Obsessive-Compulsive subscale was not related to objective measures of attention, verbal memory, or age. It was related significantly to symptoms of depression (r = .81, P<.005), anxiety (r = .66, P<.005), and subjective complaints of memory problems (r = .75, P<.005). Multivariate analyses indicated that reported symptoms of depression contributed 66% of the variance in predicting SCL 90-R Obsessive-Compulsive Scores, whereas symptoms of anxiety contributed an additional 6% (P<.0001).
# """
# ===========测试案例9============    专家答案：A  原本流程答案：B  该流程答案：A
# question = "Predicting admission at triage: are nurses better than a simple objective score?"
# context = """
#             In this single-centre prospective study, triage nurses estimated the probability of admission using a 100 mm visual analogue scale (VAS), and GAPS was generated automatically from triage data. We compared calibration using rank sum tests, discrimination using area under receiver operating characteristic curves (AUC) and accuracy with McNemar's test.
#             Of 1829 attendances, 745 (40.7%) were admitted, not significantly different from GAPS' prediction of 750 (41.0%, p=0.678). In contrast, the nurses' mean VAS predicted 865 admissions (47.3%), overestimating by 6.6% (p<0.0001). GAPS discriminated between admission and discharge as well as nurses, its AUC 0.876 compared with 0.875 for VAS (p=0.93). As a binary predictor, its accuracy was 80.6%, again comparable with VAS (79.0%), p=0.18. 
#             In the minority of attendances, when nurses felt at least 95% certain of the outcome, VAS' accuracy was excellent, at 92.4%. However, in the remaining majority, GAPS significantly outperformed VAS on calibration (+1.2% vs +9.2%, p<0.0001), discrimination (AUC 0.810 vs 0.759, p=0.001) and accuracy (75.1% vs 68.9%, p=0.0009). 
#             When we used GAPS, but 'over-ruled' it when clinical certainty was ≥95%, this significantly outperformed either method, with AUC 0.891 (0.877-0.907) and accuracy 82.5% (80.7%-84.2%).
#         """
# ===========测试案例10===========    专家答案：A  原本流程答案：B  该流程答案：C
question = "Telemedicine and type 1 diabetes: is technology per se sufficient to improve glycaemic control?"
context = """
             Each patient received a smartphone with an insulin dose advisor (IDA) and with (G3 group) or without (G2 group) the telemonitoring/teleconsultation function. Patients were classified as \"high users\" if the proportion of \"informed\" meals using the IDA exceeded 67% (median) and as \"low users\" if not. 
             Also analyzed was the respective impact of the IDA function and teleconsultations on the final HbA1c levels.\nAmong the high users, the proportion of informed meals remained stable from baseline to the end of the study 6months later (from 78.1±21.5% to 73.8±25.1%; P=0.107), but decreased in the low users (from 36.6±29.4% to 26.7±28.4%; P=0.005). 
             As expected, HbA1c improved in high users from 8.7% [range: 8.3-9.2%] to 8.2% [range: 7.8-8.7%]in patients with (n=26) vs without (n=30) the benefit of telemonitoring/teleconsultation (-0.49±0.60% vs -0.52±0.73%, respectively; P=0.879). 
             However, although HbA1c also improved in low users from 9.0% [8.5-10.1] to 8.5% [7.9-9.6], those receiving support via teleconsultation tended to show greater improvement than the others (-0.93±0.97 vs -0.46±1.05, respectively; P=0.084).
        """
template = PromptTemplate(
    input_variables=["question", "context", "metadata"],
    template=prompt
)
parser = StrOutputParser()

chain = template | llm | parser
res = chain.invoke({
    "question": question,
    "context": context,
    "metadata": metadata
})
print(f"专家分析结论{res}")
full_context = f"expert analysis: {res}"

prompt1 = """
    You are a biomedical expert answering PubMedQA.
    The task is to answer the question with "yes", "no", or "maybe".

    Question:
    {question}

    Options:
    A: yes
    B: no
    C: maybe

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
    input_variables=["question", "full_context"],
    template=prompt1
)

chain = template1 | llm | parser
res = chain.invoke({
    "question": question,
    "full_context": full_context
})
print(f"最终QA答案{res}")