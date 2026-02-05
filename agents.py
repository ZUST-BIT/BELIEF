"""
智能体模块
包含用于不同任务的智能体实现
"""

import json
import re
import requests
from typing import Dict, Any, List
from datetime import datetime
from prompt import Prompt_A, Prompt_B, Prompt_C, Prompt_D, Prompt_E, Prompt_E_Test_MCQ, Prompt_E_Test_YesNo
from config import set_argument
from llm_client import call_llm


def extract_json_from_response(response: str) -> dict:
    """
    从LLM响应中提取JSON对象
    支持处理模型先输出推理过程再输出JSON的情况
    
    Args:
        response: LLM的原始响应文本
        
    Returns:
        解析后的JSON字典，失败则返回None
    """
    if not response:
        return None
    
    # 方法1: 尝试提取 ```json ... ``` 代码块
    if "```json" in response:
        try:
            json_str = response.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    # 方法2: 尝试提取 ``` ... ``` 代码块
    if "```" in response:
        try:
            json_str = response.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    # 方法3: 使用正则表达式查找最后一个完整的JSON对象 {...}
    # 这处理模型先输出推理文本，最后输出JSON的情况
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    if matches:
        # 从后往前尝试解析（通常最后一个是完整的输出JSON）
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # 方法4: 查找从第一个 { 到最后一个 } 的内容
    first_brace = response.find('{')
    last_brace = response.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            json_str = response[first_brace:last_brace + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 方法5: 直接尝试解析整个响应
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    return None

class AgentA:
    """
    智能体A - 生物医学问题分析智能体
    功能：根据用户的生物医学问题，调用外部大模型API进行结构化分析
    """
    
    def __init__(self):
        """初始化智能体A，加载配置"""
        self.args = set_argument()
        
    def _call_llm_api(self, prompt: str, temperature: float = 0.7) -> str:
        """
        调用LLM（统一接口）
        
        Args:
            prompt: 输入提示词
            temperature: 温度参数
            
        Returns:
            模型返回的文本响应
        """
        return call_llm(prompt, temperature=temperature, max_tokens=2000)
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        分析生物医学问题
        
        Args:
            question: 用户输入的生物医学问题
            
        Returns:
            结构化的分析结果（JSON格式）
        """
        # 将用户问题插入到 Prompt_A 模板中
        prompt = Prompt_A.replace("{{QUESTION}}", question)
        
        # 调用大模型API
        response = self._call_llm_api(prompt)
        # 解析JSON响应
        result = extract_json_from_response(response)
        if result is not None:
            return result
        else:
            print(f"JSON解析失败，原始响应: {response[:500]}...")
            return {
                "error": "JSON解析失败",
                "raw_response": response
            }
    
    def run(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        运行智能体A
        
        Args:
            question: 用户输入的生物医学问题
            verbose: 是否打印详细信息
            
        Returns:
            分析结果
        """
        if verbose:
            print(f"[智能体A] 正在分析问题: {question}")
        
        result = self.analyze_question(question)
        
        if verbose:
            print(f"[智能体A] 分析完成")
            print(f"结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return result


class AgentB:
    """
    智能体B - 生物医学证据分析智能体
    功能：检索知识片段，并将其转换为结构化的PICO和研究设计信息
    """
    
    def __init__(self):
        """初始化智能体B，加载配置"""
        self.args = set_argument()
        self.agent_a = AgentA()  # 初始化智能体A用于问题分析
        
    def _call_llm_api(self, prompt: str, temperature: float = 0.3) -> str:
        """
        调用LLM（统一接口）
        
        Args:
            prompt: 输入提示词
            temperature: 温度参数
            
        Returns:
            模型返回的文本响应
        """
        return call_llm(prompt, temperature=temperature, max_tokens=2000)
    
    def analyze_evidence(self, evidence_text: str) -> Dict[str, Any]:
        """
        分析单个证据片段
        
        Args:
            evidence_text: 证据文本
            
        Returns:
            结构化的PICO和研究设计信息
        """
        # 将证据文本插入到 Prompt_B 模板中
        prompt = Prompt_B.replace("{{EVIDENCE_TEXT}}", evidence_text)
        
        # 调用大模型API
        response = self._call_llm_api(prompt)
        
        # 解析JSON响应（使用健壮版本）
        result = extract_json_from_response(response)
        if result is not None:
            return result
        else:
            print(f"JSON解析失败，原始响应: {response[:500]}...")
            return {
                "error": "JSON解析失败",
                "raw_response": response
            }
    
    def analyze_evidence_list(self, question: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析给定的证据列表
        
        Args:
            question: 用户输入的生物医学问题
            evidence_list: 外部提供的证据列表
            
        Returns:
            包含证据分析结果
        """
        # 分析每个证据片段
        analyzed_evidences = []
        
        for idx, evidence in enumerate(evidence_list, 1):
            evidence_content = evidence.get('content', '')
            source_type = evidence.get('source_type', evidence.get('type', 'user_input'))
            
            
            # 对所有包含足够内容的证据进行分析
            if evidence_content and len(evidence_content.strip()) > 50:
                analysis = self.analyze_evidence(evidence_content)
                
                analyzed_evidences.append({
                    "evidence_id": idx,
                    "source_type": source_type,
                    "metadata": evidence.get('metadata', {}),
                    "analysis": analysis,
                    "original_content": evidence_content[:80000] + "..." if len(evidence_content) > 80000 else evidence_content
                })
            else:
                # 内容太短，跳过分析
                analyzed_evidences.append({
                    "evidence_id": idx,
                    "source_type": source_type,
                    "metadata": evidence.get('metadata', {}),
                    "original_content": evidence_content[:300] + "..." if len(evidence_content) > 300 else evidence_content,
                    "analysis": {"note": "Content too short, PICO extraction skipped"}
                })
        
        # 返回完整结果
        return {
            # "question": question,
            "evidence_count": len(evidence_list),
            "analyzed_evidences": analyzed_evidences
        }
    
    def run(self, question: str, evidence_list: List[Dict[str, Any]] = None, verbose: bool = False) -> Dict[str, Any]:
        """
        运行智能体B
        
        Args:
            question: 用户输入的生物医学问题
            evidence_list: 可选，外部提供的证据列表。如果为None，则使用内部检索
            verbose: 是否打印详细信息
            
        Returns:
            完整的分析结果
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[智能体B] 开始处理问题: {question}")
            print(f"{'='*60}\n")
        
        # 如果提供了外部证据列表，直接分析；否则进行检索
        if evidence_list is not None:
            result = self.analyze_evidence_list(question, evidence_list)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[智能体B] 处理完成")
            print(f"{'='*60}")
            if 'question_analysis' in result:
                print(f"\n问题类型: {result['question_analysis'].get('question_type', 'Unknown')}")
                print(f"分析模式: {result['question_analysis'].get('analysis_mode', 'Unknown')}")
            print(f"证据数量: {result['evidence_count']}")
            print(f"分析的证据数量: {len(result['analyzed_evidences'])}")
            
            # 打印前3个证据的分析结果
            print(f"\n证据分析摘要（前3个）：")
            for i, ev in enumerate(result['analyzed_evidences'][:3], 1):
                print(f"\n--- 证据 {i} ---")
                print(f"来源: {ev['source_type']}")
                if 'analysis' in ev and 'pico' in ev['analysis']:
                    pico = ev['analysis']['pico']
                    print(f"P: {pico.get('P', 'N/A')}")
                    print(f"I: {pico.get('I', 'N/A')}")
                    print(f"C: {pico.get('C', 'N/A')}")
                    print(f"O: {pico.get('O', 'N/A')}")
                    print(f"研究类型: {ev['analysis'].get('study_type', 'Unknown')}")
        
        return result


class AgentC:
    """
    智能体C - 基于D-S理论的证据评估智能体
    功能：使用NLI判断证据与假设的关系，结合可靠性计算BPA（基本概率分配）
    """
    
    def __init__(self):
        """初始化智能体C，加载配置"""
        self.args = set_argument()
        
    def _call_llm_api(self, prompt: str, temperature: float = 0.1) -> str:
        """
        调用LLM（统一接口）
        
        Args:
            prompt: 输入提示词
            temperature: 温度参数
            
        Returns:
            模型返回的文本响应
        """
        return call_llm(prompt, temperature=temperature, max_tokens=1500)

    def _format_evidence_for_prompt(self, content: str, analysis: Dict[str, Any], source_type: str = "Unknown") -> str:
        """
        [核心优化] 将原始文本和Agent B的分析结果整合成大模型易读的格式
        """
        formatted_text = f"[Source Type]: {source_type}\n"

        formatted_text += f"### Evidence Content:\n{content}\n"
        
        if analysis:
            formatted_text += "\n### Pre-Analysis (Reference Info):\n"
            if analysis.get('clinical_summary'):
                formatted_text += f"- Summary: {analysis['clinical_summary']}\n"
            
            pico = analysis.get('pico', {})
            if pico:
                formatted_text += "- Structured PICO:\n"
                if pico.get('population'): formatted_text += f"  * Population: {pico['population']}\n"
                if pico.get('intervention'): formatted_text += f"  * Intervention: {pico['intervention']}\n"
                if pico.get('outcome'): formatted_text += f"  * Outcome: {pico['outcome']}\n"
            
            if analysis.get('study_design'):
                formatted_text += f"- Study Design: {analysis['study_design']}\n"
                
        return formatted_text

    def _format_result_for_generator(self, evidence_content: str, analysis_result: Dict[str, Any]) -> str:
        """
        [核心新增 Helper 2] 
        将 '原始证据' + 'Agent C的分析结果' 组装成一段给生成模型看的文本块。
        这样下游模型直接读这个字段就能写文章，不用解析复杂的JSON。
        """
        # 1. 提取关键信息 (使用 .get 避免报错)
        reasoning = analysis_result.get('step_by_step_reasoning', {})
        nli = analysis_result.get('nli_analysis', {})
        bpa = analysis_result.get('bpa_components', {})
        rel = analysis_result.get('reliability_assessment', {})
        
        # 2. 组装文本 (使用清晰的分隔符和标签)
        formatted_block = f"""
<<<< EVIDENCE ANALYSIS REPORT >>>>
[Status]: {nli.get('dominant_relation', 'NEUTRAL')}
[Quality Metrics]: 
  - Reliability: {rel.get('adjusted_reliability', 0):.2f} ({rel.get('evidence_type', 'Unknown')})
  - Support Score: {bpa.get('support_hypothesis', 0):.4f}
  - Contradiction Score: {bpa.get('against_hypothesis', 0):.4f}

[Analyst Insights (Agent C)]:
  - Logical Inference: {reasoning.get('logical_inference', 'N/A')}
  - Trap Check: {reasoning.get('trap_check', 'None triggered')}
  - Normalized Hypothesis: {reasoning.get('normalized_hypothesis', 'N/A')}

[Original Evidence Content]:
{evidence_content}
<<<< END REPORT >>>>
"""
        return formatted_block
    
    def calculate_bpa(
        self, 
        entailment: float, 
        contradiction: float, 
        neutral: float,
        reliability: float
    ) -> Dict[str, float]:
        """
        根据NLI分数和可靠性计算BPA
        
        BPA公式：
        - m(H) = r × P(entailment)  # 支持假设
        - m(¬H) = r × P(contradiction)  # 反对假设
        - m(Θ) = 1 - r + r × P(neutral)  # 不确定性
        
        Args:
            entailment: 蕴含分数
            contradiction: 矛盾分数
            neutral: 中立分数
            reliability: 证据可靠性 (0-1)
            
        Returns:
            BPA字典
        """
        # 计算BPA
        m_support = reliability * entailment
        m_against = reliability * contradiction
        m_uncertainty = (1 - reliability) + reliability * neutral
        
        # 归一化确保总和为1
        total = m_support + m_against + m_uncertainty
        if total > 0:
            m_support /= total
            m_against /= total
            m_uncertainty /= total
        
        return {
            "support_hypothesis": round(m_support, 4),
            "against_hypothesis": round(m_against, 4),
            "uncertainty": round(m_uncertainty, 4)
        }

    # --- [新增辅助方法] ---
    def _format_question_pico(self, pico_data: Dict[str, str]) -> str:
        """
        将Agent A提取的PICO结构化数据格式化为字符串
        """
        if not pico_data:
            return "N/A"
        return (
            f" - Population (P): {pico_data.get('P', 'N/A')}\n"
            f" - Intervention (I): {pico_data.get('I', 'N/A')}\n"
            f" - Comparator (C): {pico_data.get('C', 'N/A')}\n"
            f" - Outcome (O): {pico_data.get('O', 'N/A')}"
        )
    
    def evaluate_evidence(
            self, 
            hypothesis: str, 
            evidence_content: str, 
            evidence_analysis: Dict[str, Any],
            evidence_type: str = "Unknown",
            question_pico: Dict[str, str] = None
            ) -> Dict[str, Any]:
        """
        评估单个证据片段
        """
        # 1. 构造增强型证据描述 (原始文本 + 分析摘要)
        rich_evidence_text = self._format_evidence_for_prompt(evidence_content, evidence_analysis, evidence_type)
        # print(f"构建好的证据描述文本为:\n{rich_evidence_text}\n")
        # 2. [准备数据] 格式化问题的 PICO
        question_pico_str = self._format_question_pico(question_pico)
        # 2. 替换Prompt
        # 注意：这里假设 Prompt_C 中有一个 {{EVIDENCE_TEXT}} 占位符
        # 建议 Prompt_C 的相关部分写成： "Here is the evidence (including content and pre-analysis): \n {{EVIDENCE_TEXT}}"
        prompt = Prompt_C.replace("{{HYPOTHESIS}}", hypothesis)
        prompt = prompt.replace("{{EVIDENCE_TEXT}}", rich_evidence_text)
        prompt = prompt.replace("{{QUESTION_PICO}}", question_pico_str)
        
        # 3. 调用 LLM
        response = self._call_llm_api(prompt)
        
        try:
            result = extract_json_from_response(response)
            
            if result is None:
                raise ValueError("JSON Extraction failed")

            # 4. 验证和计算BPA
            nli = result.get('nli_analysis', {})
            
            # --- [核心修改开始] ---
            # 优化可靠性获取逻辑：移除基于关键词的激进兜底，改为保守兜底
            llm_reliability = result.get('reliability_assessment', {}).get('adjusted_reliability')
            
            if llm_reliability is not None:
                # 情况A: 模型成功返回了可靠性 -> 使用模型的值
                final_reliability = float(llm_reliability)
            else:
                # 情况B: 模型未返回可靠性 -> 使用保守的默认值
                # 不再去判断 'meta' 或 'review'，因为我们不知道它是否相关。
                # 如果模型没给分，说明模型可能困惑或解析失败，此时不能给高分。
                # 给一个中性的低分，避免产生高分噪声。
                print(f"[Warning] Evidence {evidence_type} missing reliability score. Using default 0.4.")
                final_reliability = 0.4 
            # --- [核心修改结束] ---
            
            calculated_bpa = self.calculate_bpa(
                nli.get('entailment_score', 0),
                nli.get('contradiction_score', 0),
                nli.get('neutral_score', 0),
                final_reliability
            )
            
            result['bpa_components'] = calculated_bpa
            
            # 生成给生成模型的文本 (保持不变)
            generator_text = self._format_result_for_generator(evidence_content, result)
            result['content_for_generator'] = generator_text
            
            # 保留调试信息 (保持不变)
            result['processed_input_snippet'] = rich_evidence_text[:200] + "..." 
            
            return result
            
        except Exception as e:
            print(f"评估失败: {e}")
            return {"error": str(e)}
    
    def evaluate_evidence_batch(
        self,
        hypothesis: str,
        agent_b_result: Dict[str, Any],
        question_pico: Dict[str, str] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量评估多个证据 - 适配新的JSON结构
        """
        results = []
        
        # 1. 解析 Agent B 的结果结构
        # 兼容处理：既支持直接传 list，也支持传包含 analyzed_evidences 的 dict
        if isinstance(agent_b_result, dict):
            evidence_list = agent_b_result.get('analyzed_evidences', [])
            # total_count = agent_b_result.get('evidence_count', 0)
        elif isinstance(agent_b_result, list):
            evidence_list = agent_b_result
        else:
            print("Error: agent_b_result 格式错误")
            return []

        # print(f"[智能体C] 收到 {len(evidence_list)} 条证据待评估 (总计: {total_count})")

        for item in evidence_list:
            ev_id = item.get('evidence_id')
            source_type = item.get('source_type', 'Unknown')
            original_content = item.get('original_content', '')
            analysis_data = item.get('analysis', {}) # 获取 Agent B 的分析数据
            
            # print(f"正在评估证据 #{ev_id} ({source_type})...")
            
            # 只有当内容有效时才评估
            if original_content or analysis_data:
                evaluation = self.evaluate_evidence(
                    hypothesis=hypothesis, 
                    evidence_content=original_content[:2000000], # 依然限制长度
                    evidence_analysis=analysis_data,          # 传入分析数据
                    evidence_type=source_type,
                    question_pico=question_pico
                )
                
                results.append({
                    "evidence_id": ev_id,
                    "source_type": source_type,
                    "metadata": item.get('metadata', {}),
                    "evaluation": evaluation
                })
            else:
                results.append({
                    "evidence_id": ev_id,
                    "error": "Empty content"
                })
        
        return results
    
    def run(
        self, 
        hypothesis: str, 
        agent_b_result: Dict[str, Any], 
        question_pico: Dict[str, str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        运行智能体C
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[智能体C] 开始D-S证据评估")
            print(f"{'='*60}")
            print(f"假设命题: {hypothesis}")
        
        # 传入完整的 agent_b_result
        evaluations = self.evaluate_evidence_batch(hypothesis, agent_b_result, question_pico)
        
        # 收集所有有效的BPA
        valid_bpas = []
        for ev in evaluations:
            if 'evaluation' in ev and 'bpa_components' in ev['evaluation']:
                bpa = ev['evaluation']['bpa_components']
                if not ev['evaluation'].get('error'):
                    valid_bpas.append(bpa)
        
        if verbose:
            print(f"\n[智能体C] 评估完成，有效BPA: {len(valid_bpas)}")
            if valid_bpas:
                avg_support = sum(b['support_hypothesis'] for b in valid_bpas) / len(valid_bpas)
                print(f"BPA均值 -> 支持: {avg_support:.4f}")
        
        return {
            "hypothesis": hypothesis,
            "evaluations": evaluations,
            "bpa_list": valid_bpas
        }


class AgentD:
    """
    智能体D - 多证据融合与推理智能体
    功能：使用D-S理论融合多个BPA，处理冲突，执行链式推理
    """
    
    def __init__(self):
        """初始化智能体D，加载配置"""
        self.args = set_argument()
        
    def _call_llm_api(self, prompt: str, temperature: float = 0.3) -> str:
        """
        调用LLM（统一接口）
        """
        return call_llm(prompt, temperature=temperature, max_tokens=2000)
    
    def _format_evaluations_for_llm(self, evaluations: List[Dict[str, Any]]) -> str:
        """
        [Helper] 将 Agent C 的评估列表格式化为 LLM 易读的字符串
        """
        formatted_text = ""
        for ev in evaluations:
            ev_data = ev.get('evaluation', {})
            bpa = ev_data.get('bpa_components', {})
            # 提取 Agent C 生成的内容摘要
            content = ev_data.get('content_for_generator', 'No content summary available.')
            
            formatted_text += f"""
--- Evidence ID: {ev.get('evidence_id')} (Type: {ev.get('source_type')}) ---
[Agent C Score]: Support={bpa.get('support_hypothesis', 0):.2f}, Reliability={ev_data.get('reliability_assessment', {}).get('adjusted_reliability', 0):.2f}
[Content Analysis]:
{content}
------------------------------------------------------------
"""
        return formatted_text

    def analyze_competition_and_decide(
        self,
        question: str,
        fod: List[str],
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        [核心方法] 调用 LLM 进行竞争性假设评估和最终决策
        """
        # 1. 准备 Prompt 输入
        evals_text = self._format_evaluations_for_llm(evaluations)
        
        prompt = Prompt_D.replace("{{QUESTION}}", question)
        prompt = prompt.replace("{{FOD}}", str(fod))
        prompt = prompt.replace("{{EVALUATIONS_LIST}}", evals_text)
        
        # 2. 调用 LLM
        print("[智能体D] 正在进行竞争性假设评估 (LLM Reasoning)...")
        response = self._call_llm_api(prompt)
        
        # 3. 解析结果
        result = extract_json_from_response(response)
        if result is None:
            print(f"JSON解析失败，原始响应: {response[:500]}...")
            return {
                "error": "解析失败", 
                "raw_response": response,
                "final_decision": {"decision": "UNCERTAIN", "confidence": 0.0, "reason": "LLM解析失败"}
            }
            
        return result

    def calculate_conflict_coefficient(self, bpa1: Dict[str, float], bpa2: Dict[str, float]) -> float:
        """
        计算两个BPA之间的冲突系数K
        
        K = Σ m1(A) × m2(B) for all A∩B = ∅
        
        对于识别框架 Θ = {H, ¬H, Θ}:
        K = m1(H) × m2(¬H) + m1(¬H) × m2(H)
        """
        m1_h = bpa1.get('support_hypothesis', 0)
        m1_nh = bpa1.get('against_hypothesis', 0)
        m2_h = bpa2.get('support_hypothesis', 0)
        m2_nh = bpa2.get('against_hypothesis', 0)
        
        K = m1_h * m2_nh + m1_nh * m2_h
        return round(K, 4)
    
    def dempster_combine(self, bpa1: Dict[str, float], bpa2: Dict[str, float]) -> Dict[str, float]:
        """
        Dempster组合规则
        
        m(A) = [Σ m1(X) × m2(Y) for X∩Y=A] / (1 - K)
        """
        K = self.calculate_conflict_coefficient(bpa1, bpa2)
        
        if K >= 1.0:
            # 完全冲突，无法融合
            return {
                "support_hypothesis": 0.0,
                "against_hypothesis": 0.0,
                "uncertainty": 1.0
            }
        
        m1_h = bpa1.get('support_hypothesis', 0)
        m1_nh = bpa1.get('against_hypothesis', 0)
        m1_u = bpa1.get('uncertainty', 0)
        
        m2_h = bpa2.get('support_hypothesis', 0)
        m2_nh = bpa2.get('against_hypothesis', 0)
        m2_u = bpa2.get('uncertainty', 0)
        
        # 计算组合后的BPA
        norm_factor = 1.0 - K
        
        m_h = (m1_h * m2_h + m1_h * m2_u + m1_u * m2_h) / norm_factor
        m_nh = (m1_nh * m2_nh + m1_nh * m2_u + m1_u * m2_nh) / norm_factor
        m_u = (m1_u * m2_u) / norm_factor
        
        return {
            "support_hypothesis": round(m_h, 4),
            "against_hypothesis": round(m_nh, 4),
            "uncertainty": round(m_u, 4)
        }
    
    def murphy_average_combine(self, bpa_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Murphy平均组合规则（适用于高冲突场景）
        
        步骤：
        1. 对所有BPA求平均得到 m_avg
        2. 将 m_avg 自融合 n 次
        """
        if not bpa_list:
            return {"support_hypothesis": 0.0, "against_hypothesis": 0.0, "uncertainty": 1.0}
        
        n = len(bpa_list)
        
        # 步骤1: 计算平均BPA
        avg_h = sum(b.get('support_hypothesis', 0) for b in bpa_list) / n
        avg_nh = sum(b.get('against_hypothesis', 0) for b in bpa_list) / n
        avg_u = sum(b.get('uncertainty', 0) for b in bpa_list) / n
        
        m_avg = {
            "support_hypothesis": avg_h,
            "against_hypothesis": avg_nh,
            "uncertainty": avg_u
        }
        
        # 步骤2: 自融合 n 次
        result = m_avg
        for _ in range(n - 1):
            result = self.dempster_combine(result, m_avg)
        
        return result
    
    def analyze_reasoning_strategy(
        self,
        question: str,
        fod: List[str],
        bpa_list: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        使用LLM分析推理策略和证据关系
        """
        # 构建prompt
        prompt = Prompt_D.replace("{{FRAME_OF_DISCERNMENT}}", str(fod))
        prompt = prompt.replace("{{BPA_LIST}}", json.dumps(bpa_list, indent=2))
        prompt = prompt.replace("{{QUESTION_CONTEXT}}", question)
        
        # 调用LLM
        response = self._call_llm_api(prompt)
        
        # 解析JSON（使用健壮版本）
        result = extract_json_from_response(response)
        if result is not None:
            return result
        else:
            print(f"JSON解析失败，原始响应: {response[:500]}...")
            return {"error": "解析失败", "raw_response": response}
    
    def fuse_evidence(
        self,
        bpa_list: List[Dict[str, float]],
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        融合多个证据的BPA
        
        Args:
            bpa_list: BPA列表
            strategy: 融合策略 ('auto', 'dempster', 'murphy')
        
        Returns:
            融合结果
        """
        if not bpa_list:
            return {
                "fused_bpa": {"support_hypothesis": 0.0, "against_hypothesis": 0.0, "uncertainty": 1.0},
                "method": "none",
                "conflict_coefficient": 0.0
            }
        
        if len(bpa_list) == 1:
            return {
                "fused_bpa": bpa_list[0],
                "method": "single",
                "conflict_coefficient": 0.0
            }
        
        # 计算平均冲突系数
        conflicts = []
        for i in range(len(bpa_list) - 1):
            K = self.calculate_conflict_coefficient(bpa_list[i], bpa_list[i + 1])
            conflicts.append(K)
        
        avg_conflict = sum(conflicts) / len(conflicts) if conflicts else 0.0
        
        # 自动选择策略
        if strategy == "auto":
            if avg_conflict < 0.3:
                strategy = "dempster"
            else:
                strategy = "murphy"
        
        # 执行融合
        if strategy == "dempster":
            result = bpa_list[0]
            for bpa in bpa_list[1:]:
                result = self.dempster_combine(result, bpa)
            method = "Dempster组合规则"
        else:  # murphy
            result = self.murphy_average_combine(bpa_list)
            method = "Murphy平均规则"
        
        return {
            "fused_bpa": result,
            "method": method,
            "strategy": strategy,
            "conflict_coefficient": round(avg_conflict, 4),
            "evidence_count": len(bpa_list)
        }
    
    def calculate_belief_plausibility(
        self,
        fused_bpa: Dict[str, float],
        fod: List[str]
    ) -> Dict[str, Any]:
        """
        计算每个命题的信念度(Belief)和似真度(Plausibility)
        
        Bel(A) = m(A)
        Pl(A) = 1 - m(¬A)
        """
        m_h = fused_bpa.get('support_hypothesis', 0)
        m_nh = fused_bpa.get('against_hypothesis', 0)
        m_u = fused_bpa.get('uncertainty', 0)
        
        # 对于第一个假设（通常是肯定答案）
        bel_h = m_h
        pl_h = 1.0 - m_nh
        
        # 对于否定假设
        bel_nh = m_nh
        pl_nh = 1.0 - m_h
        
        return {
            "hypothesis_positive": {
                "belief": round(bel_h, 4),
                "plausibility": round(pl_h, 4),
                "uncertainty_interval": round(pl_h - bel_h, 4)
            },
            "hypothesis_negative": {
                "belief": round(bel_nh, 4),
                "plausibility": round(pl_nh, 4),
                "uncertainty_interval": round(pl_nh - bel_nh, 4)
            }
        }
    
    def make_decision(
        self,
        belief_pl: Dict[str, Any],
        threshold: float = 0.6  # 保持默认绝对阈值
    ) -> Dict[str, Any]:
        """
        基于信念度做出决策 (引入相对优势逻辑)
        
        策略：
        1. 绝对优势：任意一方 Belief >= threshold (默认0.6) -> 直接决策
        2. 相对优势：
           如果 Max(Bel) > 0.5 且 (Max(Bel) - Min(Bel)) > 0.3 -> 决策
           (解释：虽然没到0.6，但一方明显压倒另一方，且自身过半)
        """
        bel_pos = belief_pl['hypothesis_positive']['belief']
        bel_neg = belief_pl['hypothesis_negative']['belief']
        
        decision = "UNCERTAIN"
        confidence = 0.0
        reason = ""

        # 1. 绝对优势判定
        if bel_pos >= threshold:
            decision = "YES"
            confidence = bel_pos
            reason = f"正向信念度 {bel_pos:.4f} 超过绝对阈值 {threshold}"
        elif bel_neg >= threshold:
            decision = "NO"
            confidence = bel_neg
            reason = f"负向信念度 {bel_neg:.4f} 超过绝对阈值 {threshold}"
            
        # 2. 相对优势判定 (补救措施)
        elif decision == "UNCERTAIN":
            diff = abs(bel_pos - bel_neg)
            relative_threshold = 0.3 # 相对差距阈值
            min_absolute_support = 0.45 # 最低绝对支持度（稍微降低要求）

            if bel_pos > bel_neg and bel_pos > min_absolute_support and diff > relative_threshold:
                decision = "YES"
                confidence = bel_pos
                reason = f"正向信念度虽未达绝对阈值，但具有显著相对优势 (差值 {diff:.4f} > {relative_threshold})"
            elif bel_neg > bel_pos and bel_neg > min_absolute_support and diff > relative_threshold:
                decision = "NO"
                confidence = bel_neg
                reason = f"负向信念度虽未达绝对阈值，但具有显著相对优势 (差值 {diff:.4f} > {relative_threshold})"
            else:
                # 确实是不确定
                confidence = max(bel_pos, bel_neg)
                reason = f"信念度不足且无显著相对优势（正向:{bel_pos:.4f}, 负向:{bel_neg:.4f}）"
        
        return {
            "decision": decision,
            "confidence": round(confidence, 4),
            "reason": reason
        }
    
    def run(
        self,
        question: str,
        fod: List[str],
        bpa_list: List[Dict[str, float]],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        运行智能体D - 完整的多证据融合与推理
        
        Args:
            question: 原始问题
            fod: 识别框架
            bpa_list: BPA列表
            verbose: 是否打印详细信息
        
        Returns:
            融合结果和决策
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[智能体D] 开始多证据融合与推理")
            print(f"{'='*60}")
            print(f"问题: {question}")
            print(f"识别框架: {fod}")
            print(f"证据BPA数量: {len(bpa_list)}")
        
        # 步骤1: 使用LLM分析推理策略
        print("\n[步骤1] 分析证据关系和推理策略...")
        reasoning_analysis = self.analyze_reasoning_strategy(question, fod, bpa_list)
        
        recommended_strategy = reasoning_analysis.get('fusion_strategy', {}).get('primary_method', 'auto')
        if verbose:
            print(f"✅ 推荐融合策略: {recommended_strategy}")
            conflict_level = reasoning_analysis.get('conflict_analysis', {}).get('conflict_level', 'unknown')
            print(f"✅ 冲突程度: {conflict_level}")
        
        # 步骤2: 执行BPA融合
        print("\n[步骤2] 执行BPA融合...")
        fusion_result = self.fuse_evidence(bpa_list, strategy=recommended_strategy)
        if verbose:
            print(f"✅ 使用方法: {fusion_result['method']}")
            print(f"✅ 冲突系数: {fusion_result['conflict_coefficient']}")
            fused = fusion_result['fused_bpa']
            print(f"\n融合后的BPA:")
            print(f"  支持假设: {fused['support_hypothesis']:.4f}")
            print(f"  反对假设: {fused['against_hypothesis']:.4f}")
            print(f"  不确定性: {fused['uncertainty']:.4f}")
        
        # # 步骤3: 计算信念度和似真度
        print("\n[步骤3] 计算信念度和似真度...")
        belief_pl = self.calculate_belief_plausibility(fusion_result['fused_bpa'], fod)
        if verbose:
            pos = belief_pl['hypothesis_positive']
            neg = belief_pl['hypothesis_negative']
            print(f"\n正向假设:")
            print(f"  信念度(Belief): {pos['belief']:.4f}")
            print(f"  似真度(Plausibility): {pos['plausibility']:.4f}")
            print(f"  不确定区间: [{pos['belief']:.4f}, {pos['plausibility']:.4f}]")
            print(f"\n负向假设:")
            print(f"  信念度(Belief): {neg['belief']:.4f}")
            print(f"  似真度(Plausibility): {neg['plausibility']:.4f}")
        
        # # 步骤4: 做出决策
        print("\n[步骤4] 做出最终决策...")
        decision = self.make_decision(belief_pl)
        if verbose:
            print(f"\n{'='*60}")
            print(f"最终决策: {decision['decision']}")
            print(f"置信度: {decision['confidence']:.4f}")
            print(f"理由: {decision['reason']}")
            print(f"{'='*60}")
        
        return {
            "question": question,
            "frame_of_discernment": fod,
            "reasoning_analysis": reasoning_analysis,
            "fusion_result": fusion_result,
            "belief_plausibility": belief_pl,
            "final_decision": decision
        }


    # def run(
    #     self,
    #     question: str,
    #     fod: List[str],
    #     evaluations: List[Dict[str, Any]], # 注意：这里类型变了，不再是 bpa_list
    #     verbose: bool = True
    # ) -> Dict[str, Any]:
    #     """
    #     运行智能体D - 完整流程
    #     """
    #     if verbose:
    #         print(f"\n{'='*60}")
    #         print(f"[智能体D] 开始多证据融合与裁决")
    #         print(f"{'='*60}")
    #         print(f"识别框架 (FoD): {fod}")
    #         print(f"输入证据数量: {len(evaluations)}")
        
    #     # ------------------------------------------------------------------
    #     # 核心逻辑：直接让 LLM 裁判进行“竞争性评估”
    #     # 不再在 Python 里跑复杂的 Dempster 公式，因为分组逻辑太依赖语义理解
    #     # ------------------------------------------------------------------
        
    #     llm_decision_result = self.analyze_competition_and_decide(question, fod, evaluations)
        
    #     # 提取关键信息用于后续流程
    #     final_decision = llm_decision_result.get('final_decision', {})
    #     mapping = llm_decision_result.get('mapping_analysis', {})
    #     conflict = llm_decision_result.get('conflict_analysis', {})
        
    #     if verbose:
    #         print(f"\n[智能体D] 裁决完成")
    #         print(f"✅ 最终决策: {final_decision.get('decision')}")
    #         print(f"✅ 置信度: {final_decision.get('confidence')}")
    #         print(f"✅ 理由: {final_decision.get('reason')}")
    #         print(f"✅ 冲突状态: {conflict.get('conflict_level')}")
            
    #         # 打印映射关系（调试用）
    #         print(f"\n[证据分组情况]:")
    #         dist = mapping.get('evidence_distribution', {})
    #         for opt, ids in dist.items():
    #             print(f"  - {opt}: {ids}")

    #     # 构造返回结构，保持与下游 Agent E 的兼容性
    #     # 注意：这里我们构造一个虚拟的 'fusion_result' 和 'belief_plausibility'
    #     # 因为 LLM 已经直接给出了最终的 confidence，反推这些数值即可
        
    #     confidence = final_decision.get('confidence', 0.0)
        
    #     # 构造虚拟的融合结果 (适配 main.py 接口)
    #     fusion_result_mock = {
    #         "fused_bpa": {
    #             "support_hypothesis": confidence,
    #             "against_hypothesis": 0.0,
    #             "uncertainty": 1.0 - confidence
    #         },
    #         "method": "LLM_Competitive_Reasoning", # 标记这是 LLM 裁决的
    #         "conflict_coefficient": 0.0, # 暂时置0，或从 LLM 读取
    #         "evidence_count": len(evaluations)
    #     }
        
    #     belief_pl_mock = {
    #         "hypothesis_positive": {
    #             "belief": confidence,
    #             "plausibility": 1.0, 
    #             "uncertainty_interval": 1.0 - confidence
    #         },
    #         "hypothesis_negative": {
    #             "belief": 0.0,
    #             "plausibility": 1.0 - confidence,
    #             "uncertainty_interval": 1.0 - confidence
    #         }
    #     }

    #     return {
    #         "question": question,
    #         "frame_of_discernment": fod,
    #         # 直接透传 LLM 的分析结果
    #         "reasoning_analysis": {
    #             "conflict_analysis": conflict,
    #             "reasoning_chains": llm_decision_result.get('reasoning_chains', []),
    #             "reasoning_explanation": {
    #                  "overall_reasoning_path": final_decision.get('reason'),
    #                  "key_supporting_evidence": mapping.get('evidence_distribution', {}).get(mapping.get('dominant_option'), [])
    #             }
    #         },
    #         "fusion_result": fusion_result_mock,
    #         "belief_plausibility": belief_pl_mock,
    #         "final_decision": final_decision,
            
    #         # 保留原始的 LLM 完整输出供调试
    #         "raw_llm_output": llm_decision_result
    #     }

class CompletenessController:
    """
    完备性控制器 - 基于信度熵决定是否需要继续检索
    """
    
    def __init__(self):
        """初始化控制器"""
        pass
    
    def calculate_deng_entropy(self, bpa: Dict[str, float]) -> float:
        """
        计算Deng熵（修正信度熵）
        
        H(m) = -Σ m(A) × log2[m(A) / (2^|A| - 1)]
        
        对于三元素FoD {H, ¬H, Θ}:
        - |{H}| = 1, |{¬H}| = 1, |{Θ}| = 2
        """
        import math
        
        m_h = bpa.get('support_hypothesis', 0)
        m_nh = bpa.get('against_hypothesis', 0)
        m_u = bpa.get('uncertainty', 0)
        
        entropy = 0.0
        
        # 对于单元素集合 {H} 和 {¬H}，|A| = 1
        if m_h > 0:
            entropy -= m_h * math.log2(m_h / (2**1 - 1)) if m_h < 1 else 0
        if m_nh > 0:
            entropy -= m_nh * math.log2(m_nh / (2**1 - 1)) if m_nh < 1 else 0
        
        # 对于全集 Θ，|A| = 2（包含H和¬H）
        if m_u > 0:
            entropy -= m_u * math.log2(m_u / (2**2 - 1)) if m_u < 1 else 0
        
        return round(entropy, 4)
    
    def analyze_completeness(
        self,
        fused_bpa: Dict[str, float],
        belief_pl: Dict[str, Any],
        conflict_coef: float
    ) -> Dict[str, Any]:
        """
        分析证据完备性，决定下一步动作
        
        Returns:
            state: A/B/C
            action: stop/expand_retrieval/targeted_retrieval
            reason: 决策理由
        """
        m_h = fused_bpa.get('support_hypothesis', 0)
        m_nh = fused_bpa.get('against_hypothesis', 0)
        m_u = fused_bpa.get('uncertainty', 0)
        
        bel_pos = belief_pl['hypothesis_positive']['belief']
        bel_neg = belief_pl['hypothesis_negative']['belief']
        uncertainty_interval = belief_pl['hypothesis_positive']['uncertainty_interval']
        
        entropy = self.calculate_deng_entropy(fused_bpa)
        
        # 状态A：高信度，低不确定性
        if (bel_pos > 0.6 or bel_neg > 0.6) and uncertainty_interval < 0.3:
            return {
                "state": "A",
                "action": "stop",
                "reason": "证据充分，信念度高且不确定性低，可以得出结论",
                "entropy": entropy,
                "should_continue": False,
                "confidence": "high"
            }
        
        # 状态B：高无知
        if m_u > 0.6:
            return {
                "state": "B",
                "action": "expand_retrieval",
                "reason": "证据质量不足或相关性低，需要扩展检索范围",
                "entropy": entropy,
                "should_continue": True,
                "confidence": "low",
                "suggestion": "扩展关键词，增加数据源（如Google Scholar、教科书）"
            }
        
        # 状态C：高冲突
        if conflict_coef > 0.3 or abs(m_h - m_nh) < 0.2:
            return {
                "state": "C",
                "action": "targeted_retrieval",
                "reason": "证据存在冲突，需要针对性检索高质量证据解决争议",
                "entropy": entropy,
                "should_continue": True,
                "confidence": "moderate",
                "suggestion": "检索Meta分析、系统评价或争议性综述"
            }
        
        # 中等状态：可以停止但建议增强
        if entropy > 1.5:
            return {
                "state": "B-",
                "action": "optional_expand",
                "reason": "证据熵较高，建议继续检索以降低不确定性（可选）",
                "entropy": entropy,
                "should_continue": False,
                "confidence": "moderate",
                "suggestion": "可选：增加1-2轮检索以提高置信度"
            }
        
        return {
            "state": "A-",
            "action": "stop",
            "reason": "证据基本充分，可以得出结论",
            "entropy": entropy,
            "should_continue": False,
            "confidence": "moderate"
        }


class AgentE:
    """
    智能体E - 报告生成与上下文维护
    功能：生成最终医学问答报告，维护推理过程记录
    """
    
    def __init__(self):
        """初始化智能体E"""
        self.args = set_argument()
        self.reasoning_history = []  # 维护推理历史
        
    def _call_llm_api(self, prompt: str, temperature: float = 0) -> str:
        """
        调用LLM（统一接口）
        """
        return call_llm(prompt, temperature=temperature, max_tokens=3000)
    
    def add_reasoning_round(
        self,
        round_num: int,
        evidence_count: int,
        bpa_summary: Dict[str, Any],
        note: str
    ):
        """
        添加一轮推理记录
        
        Args:
            round_num: 轮次
            evidence_count: 证据数量
            bpa_summary: BPA摘要
            note: 备注说明
        """
        self.reasoning_history.append({
            "round": round_num,
            "evidence_count": evidence_count,
            "bpa_summary": bpa_summary,
            "note": note,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def generate_report(
        self,
        question: str,
        final_decision: Dict[str, Any],
        fusion_result: Dict[str, Any],
        # belief_analysis: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        reasoning_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成最终报告
        
        Args:
            question: 用户问题
            final_decision: 最终决策
            fusion_result: 融合结果
            belief_analysis: 信念度分析
            evidence_list: 证据列表
            reasoning_history: 推理历史（可选）
        
        Returns:
            结构化报告
        """
        # 准备数据
        if reasoning_history is None:
            reasoning_history = self.reasoning_history
        
        # 构建prompt
        prompt = Prompt_E_Test_YesNo.replace("{{QUESTION}}", question)
        prompt = prompt.replace("{{FINAL_DECISION}}", json.dumps(final_decision, indent=2))
        prompt = prompt.replace("{{FUSION_RESULT}}", json.dumps(fusion_result, indent=2))
        # prompt = prompt.replace("{{BELIEF_ANALYSIS}}", json.dumps(belief_analysis, indent=2))
        
        # 简化证据列表（只保留关键信息）
        simplified_evidence = []
        for ev in evidence_list[:10]:  # 限制最多10条
            simplified_evidence.append({
                "source_type": ev.get('source_type', 'Unknown'),
                "metadata": ev.get('metadata', {}),
                "content_snippet": ev.get('content', '')[:500000] + "..."
            })
        
        prompt = prompt.replace("{{EVIDENCE_LIST}}", json.dumps(simplified_evidence, indent=2, ensure_ascii=False))
        prompt = prompt.replace("{{REASONING_HISTORY}}", json.dumps(reasoning_history, indent=2, ensure_ascii=False))
        print(f"输入给生成模型的完整内容是：\n{prompt}\n")
        # 调用LLM生成报告
        response = self._call_llm_api(prompt)
        # print(f"生成模型的原始响应内容是：\n{response}\n")
        # 解析JSON（使用健壮版本）
        report = extract_json_from_response(response)
        if report is not None:
            return report
        else:
            print(f"JSON解析失败，原始响应: {response[:500]}...")
            # 返回基础报告
            return {
                "direct_answer": f"基于当前证据，{final_decision['decision']}",
                "decision": final_decision['decision'],
                "confidence_level": "moderate",
                "full_report": response,
                "error": "JSON解析失败，返回原始文本"
            }
    
    def run(
        self,
        question: str,
        final_decision: Dict[str, Any],
        fusion_result: Dict[str, Any],
        # belief_analysis: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        运行智能体E - 生成完整报告
        
        Args:
            question: 用户问题
            final_decision: 最终决策
            fusion_result: 融合结果
            belief_analysis: 信念度分析
            evidence_list: 证据列表
            verbose: 是否打印详细信息
        
        Returns:
            完整报告
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[智能体E] 开始生成最终报告")
            print(f"{'='*60}")
        
        report = self.generate_report(
            question,
            final_decision,
            fusion_result,
            # belief_analysis,
            evidence_list,
            self.reasoning_history
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[智能体E] 报告生成完成")
            print(f"{'='*60}")
            print(f"\n直接答案: {report.get('direct_answer', 'N/A')}")
            print(f"决策: {report.get('decision', 'N/A')}")
            print(f"置信水平: {report.get('confidence_level', 'N/A')}")
            
            if 'full_report' in report:
                print(f"\n完整报告:")
                print("-"*60)
                print(report['full_report'])
                print("-"*60)
        
        return report

