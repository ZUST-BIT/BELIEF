"""
智能体模块
包含用于不同任务的智能体实现
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from prompt import (Prompt_A, Prompt_B, Prompt_E, Prompt_E_Test_MCQ, Prompt_E_Test_YesNo,
                    Prompt_C_Optimized, Prompt_DirectLLM, Prompt_DirectLLM_YesNo,
                    Prompt_FinalAggregator, Prompt_FinalAggregator_YesNo)
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

    # 步骤0: 预处理——移除 <think>...</think> 块（兼容 Qwen3 等 think 模式模型）
    if '</think>' in response:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    elif '<think>' in response:
        before = response.split('<think>')[0].strip()
        if before:
            response = before
        else:
            inner = response.split('<think>', 1)[1]
            fb, lb = inner.find('{'), inner.rfind('}')
            if fb != -1 and lb > fb:
                try:
                    return json.loads(inner[fb:lb + 1])
                except json.JSONDecodeError:
                    pass

    if not response:
        return None

    # ── 内部工具：状态机扫描出所有完整 JSON 对象的字节范围 ──────────────────────
    # 正确处理字符串内的 { } （如 "STRONGLY_SUPPORTS_{H}"），不依赖正则，无嵌套层级限制
    def _scan_json_spans(text: str):
        spans = []
        n = len(text)
        i = 0
        while i < n:
            if text[i] != '{':
                i += 1
                continue
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                ch = text[j]
                if esc:
                    esc = False
                elif ch == '\\' and in_str:
                    esc = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            spans.append((i, j + 1))
                            break
                j += 1
            i += 1
        return spans

    # 方法1: 提取最后一个 ```json ... ``` 代码块（LLM 有时输出多个，取最后）
    if "```json" in response:
        parts = response.split("```json")
        for chunk in reversed(parts[1:]):
            json_str = chunk.split("```")[0].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    # 方法2: 提取最后一个 ``` ... ``` 代码块
    if "```" in response:
        parts = response.split("```")
        # 奇数索引段（1, 3, 5…）是代码块内容，取最后一个
        code_blocks = [parts[k].strip() for k in range(1, len(parts), 2)]
        for block in reversed(code_blocks):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # 方法3: 状态机扫描——找出所有合法 JSON 对象，按长度从大到小尝试解析
    spans = _scan_json_spans(response)
    if spans:
        spans_sorted = sorted(spans, key=lambda x: x[1] - x[0], reverse=True)
        for start, end in spans_sorted:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                continue

    # 方法4: 直接尝试解析整个响应（响应本身就是纯 JSON）
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
        
    def _call_llm_api(self, prompt: str, temperature: float = 0) -> str:
        """
        调用LLM（统一接口）
        
        Args:
            prompt: 输入提示词
            temperature: 温度参数
            
        Returns:
            模型返回的文本响应
        """
        return call_llm(prompt, temperature=temperature, max_tokens=4096)
    
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
        
    def _call_llm_api(self, prompt: str, temperature: float = 0) -> str:
        """
        调用LLM（统一接口）
        
        Args:
            prompt: 输入提示词
            temperature: 温度参数
            
        Returns:
            模型返回的文本响应
        """
        return call_llm(prompt, temperature=temperature, max_tokens=4096)
    
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
    功能：利用LLM进行证据多维标签分类，并由Python规则引擎计算BPA（基本概率分配）
    """
    
    def __init__(self):
        """初始化智能体C，加载配置"""
        self.args = set_argument()
        
    def _call_llm_api(self, prompt: str, temperature: float = 0,
                       max_retries: int = 3, retry_delay: float = 5.0) -> str:
        """
        调用LLM（统一接口），带重试逻辑，批量场景下应对短暂网络抖动。

        Args:
            prompt: 输入提示词
            temperature: 温度参数
            max_retries: 最大重试次数
            retry_delay: 初次重试等待秒数（指数退避）

        Returns:
            模型返回的文本响应（失败时返回空字符串）
        """
        import time
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = call_llm(prompt, temperature=temperature, max_tokens=4096)
                if result:          # 非空即成功
                    return result
                # 空响应视为软失败，继续重试
                raise ValueError("LLM returned empty response")
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = retry_delay * (2 ** (attempt - 1))   # 指数退避: 5s, 10s, 20s
                    print(f"[Agent C] API调用第{attempt}次失败，{wait:.0f}s后重试... ({e})")
                    time.sleep(wait)
        print(f"[Agent C] API调用连续{max_retries}次失败，跳过本条评估: {last_error}")
        return ""

    def _format_evidence_for_prompt(self, content: str, analysis: Dict[str, Any], source_type: str = "Unknown") -> str:
        """
        [核心优化] 将原始文本和Agent B的分析结果整合成大模型易读的格式
        """
        formatted_text = f"[Source Type]: {source_type}\n"

        formatted_text += f"### Evidence Content:\n{content}\n"
        
        if analysis:
            formatted_text += (
                "\n### Pre-Analysis (Reference Info Only — trust original evidence text if conflict exists):\n"
            )
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
            [重构版] 适配标签分类体系，组装给生成模型看的文本块
            """
            # 兼容新的 reasoning_trace 键名（重构后）和旧的 step_by_step_reasoning 键名
            reasoning = analysis_result.get('reasoning_trace',
                        analysis_result.get('step_by_step_reasoning', {}))
            metrics = analysis_result.get('metrics', {})
            labels = analysis_result.get('labels', {})
            bpa = analysis_result.get('bpa_components', {})

            # 确定主要的结论倾向
            status = "NEUTRAL"
            if bpa.get('support_hypothesis', 0) > bpa.get('against_hypothesis', 0):
                status = "SUPPORT"
            elif bpa.get('against_hypothesis', 0) > bpa.get('support_hypothesis', 0):
                status = "REFUTE"

            formatted_block = f"""
        <<<< EVIDENCE ANALYSIS REPORT (Label-Based) >>>>
        [Status]: {status}
        [Classification Labels]:
        - Source Privilege   : {labels.get('source_privilege', 'N/A')}
        - Relevance          : {labels.get('relevance', 'N/A')}
        - Source Quality     : {labels.get('source_quality', 'N/A')}
        - Quality Trap       : {labels.get('quality_trap', 'N/A')}
        - Direction Polarity : {labels.get('direction_polarity', 'N/A')}
        - Direction Strength : {labels.get('direction_strength', 'N/A')}
        - Mapped FoD Option  : {labels.get('mapped_fod_option', 'N/A')}

        [Computed BPA (Rule Engine)]:
        - Reliability (W)  : {metrics.get('adjusted_reliability_W', 0):.3f}
        - Degree of Support: {metrics.get('degree_of_support_D', 0):.3f}
        - Support          : {bpa.get('support_hypothesis', 0):.4f}
        - Refute           : {bpa.get('against_hypothesis', 0):.4f}
        - Uncertainty      : {bpa.get('uncertainty', 0):.4f}

        [Analyst Reasoning (Agent C)]:
        - Relevance   : {reasoning.get('relevance_reasoning', reasoning.get('relevance_analysis', 'N/A'))}
        - Quality     : {reasoning.get('source_quality_reasoning', reasoning.get('reliability_rationale', 'N/A'))}
        - Direction   : {reasoning.get('direction_reasoning', reasoning.get('alignment_analysis', 'N/A'))}
        - Mapping     : {reasoning.get('mapping_reasoning', 'N/A')}

        [Original Evidence Content]:
        {evidence_content}
        <<<< END REPORT >>>>
        """
            return formatted_block

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
    
    _RELIABILITY_MAP: Dict[str, float] = {
        'GOLD_STANDARD':       0.90,   # 上限0.90，保留至少10%不确定性空间
        'SYSTEMATIC_REVIEW':   0.82,
        'RCT':                 0.75,
        'COHORT_CASE_CONTROL': 0.60,
        'CASE_SERIES':         0.38,
        'UNCLEAR_BASIC':       0.28,
    }
    _TRAP_PENALTY_MAP: Dict[str, float] = {
        'NO_TRAP':                0.00,
        'WEAK_SUBGROUP':          0.15,
        'ANIMAL_MODEL_ONLY':      0.20,
        'CONTRADICTORY_INTERNAL': 0.20,
    }
    _DIRECTION_STRENGTH_MAP: Dict[str, float] = {
        'STRONGLY': 0.95,   # 与可靠性乘积最高约0.855，避免独裁性1.0
        'WEAKLY':   0.50,
        'NONE':     0.00,
    }
    _RELEVANCE_SCALE_MAP: Dict[str, float] = {
        'HIGHLY_RELEVANT':    1.0,
        'PARTIALLY_RELEVANT': 0.5,
        'IRRELEVANT':         0.0,
    }
    def _normalize_option_text(self, text: str) -> str:
        """
        归一化选项文本，降低大小写、标点、空白差异带来的匹配问题
        """
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[\[\]\(\)\{\},.;:!?\'"“”‘’/_\-]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _fuzzy_match_option(self, target: str, frame_of_discernment: List[str]) -> Optional[str]:
        """
        将LLM输出的选项名称匹配到FoD已知选项
        匹配顺序：
      1. 归一化后精确匹配
      2. 归一化后包含匹配
      3. 相似度匹配（阈值控制）
        """
        from difflib import SequenceMatcher
        if not target or not frame_of_discernment:
            return None

        target_norm = self._normalize_option_text(target)

        # 1) 精确匹配
        for opt in frame_of_discernment:
            if self._normalize_option_text(opt) == target_norm:
                return opt

    # 2) 包含匹配（双向）
        for opt in frame_of_discernment:
            opt_norm = self._normalize_option_text(opt)
            if target_norm in opt_norm or opt_norm in target_norm:
                return opt

    # 3) 相似度匹配
        best_opt = None
        best_score = 0.0
        for opt in frame_of_discernment:
            opt_norm = self._normalize_option_text(opt)
            score = SequenceMatcher(None, target_norm, opt_norm).ratio()
            if score > best_score:
                best_score = score
                best_opt = opt

        # 阈值不要太低，宁可不匹配也不要误匹配
        if best_score >= 0.78:
            return best_opt

        return None

    def compute_bpa_from_tags(
        self,
        labels: Dict[str, str],
        frame_of_discernment: List[str]
    ) -> Tuple[Dict[str, float], float, float]:
        """
        [核心规则引擎] 根据LLM分类标签，通过确定性规则计算BPA值。

        新版标签格式：
        - source_privilege
        - relevance
        - source_quality
        - quality_trap
        - direction_polarity   : SUPPORTS / REFUTES / NEUTRAL
        - direction_strength   : STRONGLY / WEAKLY / NONE
        - mapped_fod_option    : FoD中的具体选项 or NONE

        设计原则：
        - LLM负责理解、分类、映射到FoD
        - Python只负责量化与BPA计算
        """
        bpa: Dict[str, float] = {opt: 0.0 for opt in frame_of_discernment}
        bpa['uncertainty_theta'] = 0.0

        # ── Step 1: 相关性检查 ──────────────────────────────────
        relevance = labels.get('relevance', 'IRRELEVANT')
        relevance_scale = self._RELEVANCE_SCALE_MAP.get(relevance, 0.0)
        if relevance_scale == 0.0:
            bpa['uncertainty_theta'] = 1.0
            return bpa, 0.0, 0.0

        # ── Step 2: 计算调整后的可靠性 ──────────────────────────
        source_privilege = labels.get('source_privilege', 'EXTERNAL_LITERATURE')
        if source_privilege == 'GOLD_STANDARD':
            base_reliability = self._RELIABILITY_MAP.get('GOLD_STANDARD', 0.90)
        else:
            source_quality = labels.get('source_quality', 'UNCLEAR_BASIC')
            base_reliability = self._RELIABILITY_MAP.get(source_quality, 0.30)

        trap_key = labels.get('quality_trap', 'NO_TRAP')
        penalty = self._TRAP_PENALTY_MAP.get(trap_key, 0.0)

        adjusted_reliability = max(0.0, (base_reliability - penalty) * relevance_scale)

        # ── Step 3: 读取新版方向标签 ────────────────────────────
        polarity = str(labels.get('direction_polarity', 'NEUTRAL')).upper().strip()
        strength = str(labels.get('direction_strength', 'NONE')).upper().strip()
        mapped_option_raw = labels.get('mapped_fod_option', 'NONE')

        if polarity == 'NEUTRAL' or strength == 'NONE':
            bpa['uncertainty_theta'] = 1.0
            return bpa, adjusted_reliability, 0.0

        degree_of_support = self._DIRECTION_STRENGTH_MAP.get(strength, 0.0)
        if degree_of_support <= 0.0:
            bpa['uncertainty_theta'] = 1.0
            return bpa, adjusted_reliability, 0.0

        mass = adjusted_reliability * degree_of_support
        mass = min(mass, 0.90)

        matched_opt = None
        if mapped_option_raw and str(mapped_option_raw).upper().strip() != 'NONE':
            matched_opt = self._fuzzy_match_option(str(mapped_option_raw), frame_of_discernment)

        # ── Step 4: mass分配 ───────────────────────────────────
        if polarity == 'SUPPORTS':
            # 支持某个具体FoD选项
            if matched_opt is not None:
                bpa[matched_opt] = mass
            else:
                # LLM未能可靠映射到FoD，保守归入不确定性
                bpa['uncertainty_theta'] = mass

        elif polarity == 'REFUTES':
            # 多分类下，反对某个选项 != 自动支持其余所有选项
            # 因此：
            # - 二分类：可以转给对立项
            # - 多分类：保守归入不确定性
            if matched_opt is not None:
                opposite_opts = [opt for opt in frame_of_discernment if opt != matched_opt]
                if len(frame_of_discernment) == 2 and len(opposite_opts) == 1:
                    bpa[opposite_opts[0]] = mass
                else:
                    bpa['uncertainty_theta'] = mass
            else:
                bpa['uncertainty_theta'] = mass

        else:
            # 异常标签，保守处理
            bpa['uncertainty_theta'] = 1.0
            return bpa, adjusted_reliability, 0.0

        # ── Step 5: 计算剩余不确定性 ───────────────────────────
        assigned = sum(v for k, v in bpa.items() if k != 'uncertainty_theta')
        bpa['uncertainty_theta'] = max(0.0, round(1.0 - assigned, 6))

        total_mass = sum(bpa.values())
        if total_mass > 1.001:
            bpa = {k: v / total_mass for k, v in bpa.items()}

        return bpa, adjusted_reliability, degree_of_support

    def evaluate_evidence(
            self,
            hypothesis: str,
            evidence_content: str,
            evidence_analysis: Dict[str, Any],
            evidence_type: str = "Unknown",
            question_pico: Dict[str, str] = None,
            frame_of_discernment: List[str] = None
            ) -> Dict[str, Any]:
        """
        [重构版] 评估单个证据片段

        流程：
          1. 调用LLM → 获取5维分类标签（不再让LLM计算数字）
          2. 调用 compute_bpa_from_tags → 规则引擎计算BPA（确定性、可溯源）
          3. 组装下游期望的标准字段
        """
        # 1. 构造增强型证据描述 (原始文本 + Agent B结构化分析)
        rich_evidence_text = self._format_evidence_for_prompt(evidence_content, evidence_analysis, evidence_type)
        question_pico_str = self._format_question_pico(question_pico)

        # 2. 默认FoD
        if not frame_of_discernment:
            frame_of_discernment = ['SUPPORT', 'REFUTE']

        fod_text = "N/A"
        if frame_of_discernment:
                fod_text = "\n".join([f"- {opt}" for opt in frame_of_discernment])

        prompt = Prompt_C_Optimized.replace("{{HYPOTHESIS}}", hypothesis)
        prompt = prompt.replace("{{EVIDENCE_TEXT}}", rich_evidence_text)
        prompt = prompt.replace("{{QUESTION_PICO}}", question_pico_str)
        prompt = prompt.replace("{{FRAME_OF_DISCERNMENT}}", fod_text)

        response = self._call_llm_api(prompt)

        try:
            result = extract_json_from_response(response)
            if result is None:
                raise ValueError("JSON Extraction failed")

            labels = result.get('labels', {})

            # ── 容错：兼容LLM扁平化输出（labels字段直接平铺在顶层）──────────
            _EXPECTED_LABEL_KEYS = {
                'source_privilege',
                'relevance',
                'source_quality',
                'quality_trap',
                'direction_polarity',
                'direction_strength',
                'mapped_fod_option'
            }

            if not labels or not (_EXPECTED_LABEL_KEYS & set(labels.keys())):
                flat_labels = {k: result[k] for k in _EXPECTED_LABEL_KEYS if k in result}
                if flat_labels:
                    labels = flat_labels

            # 新版字段默认值
            labels.setdefault('direction_polarity', 'NEUTRAL')
            labels.setdefault('direction_strength', 'NONE')
            labels.setdefault('mapped_fod_option', 'NONE')

            # 合法值清洗
            valid_polarities = {'SUPPORTS', 'REFUTES', 'NEUTRAL'}
            valid_strengths = {'STRONGLY', 'WEAKLY', 'NONE'}

            if str(labels.get('direction_polarity', '')).upper() not in valid_polarities:
                labels['direction_polarity'] = 'NEUTRAL'

            if str(labels.get('direction_strength', '')).upper() not in valid_strengths:
                labels['direction_strength'] = 'NONE'

            if not labels.get('mapped_fod_option'):
                labels['mapped_fod_option'] = 'NONE'

            result['labels'] = labels

            # 规则引擎计算BPA
            bpa_dist, adjusted_reliability, degree_of_support = self.compute_bpa_from_tags(
                labels, frame_of_discernment
            )

            # 折叠为下游兼容的三元组件
            fod_masses = {k: v for k, v in bpa_dist.items() if k != 'uncertainty_theta'}
            m_uncertainty = bpa_dist.get('uncertainty_theta', 1.0)
            total_fod_mass = sum(fod_masses.values())

            polarity = str(labels.get('direction_polarity', 'NEUTRAL')).upper()
            if polarity == 'REFUTES':
                m_support = 0.0
                m_refute = total_fod_mass
            elif polarity == 'SUPPORTS':
                m_support = total_fod_mass
                m_refute = 0.0
            else:
                m_support = 0.0
                m_refute = 0.0

            result['metrics'] = {
                'adjusted_reliability_W': round(adjusted_reliability, 4),
                'degree_of_support_D': round(degree_of_support, 4),
            }
            result['bpa_distribution'] = {k: round(v, 4) for k, v in bpa_dist.items()}
            result['bpa_components'] = {
                "support_hypothesis": round(m_support, 4),
                "against_hypothesis": round(m_refute, 4),
                "uncertainty": round(m_uncertainty, 4),
            }

            result['content_for_generator'] = self._format_result_for_generator(evidence_content, result)
            result['processed_input_snippet'] = rich_evidence_text[:200] + "..."

            return result

        except Exception as e:
            print(f"[Agent C] 评估失败: {e}")
            return {"error": str(e)}
        
    def _compress_evidence_text(self, content: str, max_len: int = 12000) -> str:
        """
        轻量证据压缩：
        优先保留 Title / Summary / Results / Conclusion 等高价值片段。
        若无法识别结构，则退化为前 max_len 字符。
        """
        if not content:
            return ""

        content = content.strip()
        if len(content) <= max_len:
            return content

        important_markers = ["Title:", "Summary:", "Abstract:", "RESULTS:", "Results:", "CONCLUSION:", "Conclusion:"]
        selected_parts = []

        for marker in important_markers:
            idx = content.find(marker)
            if idx != -1:
                snippet = content[idx: idx + 2000]
                selected_parts.append(snippet)

        compressed = "\n".join(selected_parts).strip()
        if compressed and len(compressed) >= 500:
            return compressed[:max_len]

        return content[:max_len]
    

    def evaluate_evidence_batch(
        self,
        hypothesis: str,
        agent_b_result: Dict[str, Any],
        question_pico: Dict[str, str] = None,
        frame_of_discernment: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量评估多个证据 - 适配重构后的标签分类体系

        Args:
            hypothesis: 假设命题字符串（含问题和FoD描述）
            agent_b_result: Agent B 的输出结构（analyzed_evidences列表）
            question_pico: Agent A 提取的PICO字典
            frame_of_discernment: FoD选项列表，传给规则引擎BPA计算
        """
        results = []

        # 兼容处理：既支持直接传 list，也支持传包含 analyzed_evidences 的 dict
        if isinstance(agent_b_result, dict):
            evidence_list = agent_b_result.get('analyzed_evidences', [])
        elif isinstance(agent_b_result, list):
            evidence_list = agent_b_result
        else:
            print("Error: agent_b_result 格式错误")
            return []

        for item in evidence_list:
            ev_id = item.get('evidence_id')
            # 优先读取顶层 source_type（原始来源类型，如 user_context），
            # 而非 analysis.study_design（Agent B 推断的研究类型），两者含义不同。
            # 旧代码错误使用 item.get('study_design') 该字段不在顶层，永远 fallback 到 Unknown
            source_type = item.get('source_type', 'Unknown')
            original_content = item.get('original_content', '')
            analysis_data = item.get('analysis', {})
            compressed_content = self._compress_evidence_text(original_content, max_len=12000)
            if original_content or analysis_data:
                evaluation = self.evaluate_evidence(
                    hypothesis=hypothesis,
                    evidence_content=compressed_content,
                    evidence_analysis=analysis_data,
                    evidence_type=source_type,
                    question_pico=question_pico,
                    frame_of_discernment=frame_of_discernment,
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
        frame_of_discernment: List[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        运行智能体C（重构版）

        Args:
            hypothesis: 问题+FoD描述字符串
            agent_b_result: Agent B 输出
            question_pico: Agent A 提取的PICO
            frame_of_discernment: FoD选项列表，供规则引擎精确分配BPA
            verbose: 是否打印调试信息
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[智能体C] 开始D-S证据评估（标签驱动模式）")
            print(f"{'='*60}")
            print(f"假设命题: {hypothesis}")
            if frame_of_discernment:
                print(f"识别框架: {frame_of_discernment}")

        evaluations = self.evaluate_evidence_batch(
            hypothesis, agent_b_result, question_pico, frame_of_discernment
        )

        # 收集所有有效的BPA（使用 bpa_distribution，保留完整 FoD 键分布）
        valid_bpas = []
        for ev in evaluations:
            if 'evaluation' in ev and 'bpa_distribution' in ev['evaluation']:
                if not ev['evaluation'].get('error'):
                    valid_bpas.append(ev['evaluation']['bpa_distribution'])

        if verbose:
            print(f"\n[智能体C] 评估完成，有效BPA: {len(valid_bpas)}")
            if valid_bpas:
                avg_uncertainty = sum(b.get('uncertainty_theta', 0.0) for b in valid_bpas) / len(valid_bpas)
                avg_assigned = sum(
                    sum(v for k, v in b.items() if k != 'uncertainty_theta')
                    for b in valid_bpas
                ) / len(valid_bpas)

                print(f"BPA均值 -> 已分配质量: {avg_assigned:.4f}")
                print(f"BPA均值 -> 不确定性: {avg_uncertainty:.4f}")
                
        return {
            "hypothesis": hypothesis,
            "evaluations": evaluations,
            "bpa_list": valid_bpas
        }


class AgentD:
    """
    智能体D - 纯数学多证据融合引擎
    功能：严格按照 Dempster-Shafer 理论执行 BPA 融合、冲突处理及置信度计算。
    无需调用大语言模型，完全消除推理幻觉。
    """
    
    def __init__(self):
        """初始化智能体D"""
        pass # 如果有本地配置可以保留 self.args = set_argument()

    # 核心数学计算部分 (保持原样，纯 Python 逻辑)
    def calculate_conflict_coefficient(self, bpa1: Dict[str, float], bpa2: Dict[str, float]) -> float:
        """
        计算两个BPA之间的冲突系数K（支持任意FoD键）。

        对于单元素焦集（每个选项独占一个焦集），不同选项之间的集合交集为空，
        因此 K = Σ_{i≠j} m1(opt_i) × m2(opt_j)
        """
        option_keys = [k for k in set(list(bpa1.keys()) + list(bpa2.keys()))
                       if k != 'uncertainty_theta']
        K = 0.0
        for k1 in option_keys:
            for k2 in option_keys:
                if k1 != k2:
                    K += bpa1.get(k1, 0.0) * bpa2.get(k2, 0.0)
        return round(min(K, 1.0), 4)
    
    def dempster_combine(self, bpa1: Dict[str, float], bpa2: Dict[str, float]) -> Dict[str, float]:
        """
        Dempster 组合规则（支持任意 FoD 键）。

        组合公式（归一化后）：
          m(opt_i) = [m1(opt_i)·m2(opt_i) + m1(opt_i)·m2(Θ) + m1(Θ)·m2(opt_i)] / (1-K)
          m(Θ)     = [m1(Θ)·m2(Θ)] / (1-K)
        """
        option_keys = sorted(set(
            [k for k in list(bpa1.keys()) + list(bpa2.keys()) if k != 'uncertainty_theta']
        ))
        K = self.calculate_conflict_coefficient(bpa1, bpa2)
        if K >= 1.0:
            res = {k: 0.0 for k in option_keys}
            res['uncertainty_theta'] = 1.0
            return res

        norm = 1.0 - K
        u1 = bpa1.get('uncertainty_theta', 0.0)
        u2 = bpa2.get('uncertainty_theta', 0.0)

        result = {}
        for k in option_keys:
            m1k = bpa1.get(k, 0.0)
            m2k = bpa2.get(k, 0.0)
            result[k] = round((m1k * m2k + m1k * u2 + u1 * m2k) / norm, 6)
        result['uncertainty_theta'] = round((u1 * u2) / norm, 6)
        return result
    
    def murphy_average_combine(self, bpa_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Murphy 平均组合规则（支持任意 FoD 键，适用于高冲突场景）"""
        if not bpa_list:
            return {'uncertainty_theta': 1.0}

        # 收集所有 FoD 选项键
        all_option_keys = set()
        for bpa in bpa_list:
            all_option_keys.update(k for k in bpa.keys() if k != 'uncertainty_theta')

        n = len(bpa_list)
        # 计算各 FoD 选项的均值 BPA
        m_avg: Dict[str, float] = {
            k: sum(b.get(k, 0.0) for b in bpa_list) / n
            for k in all_option_keys
        }
        m_avg['uncertainty_theta'] = sum(
            b.get('uncertainty_theta', 0.0) for b in bpa_list
        ) / n

        # 对均值 BPA 应用 n-1 次 Dempster 组合
        result = m_avg
        for _ in range(n - 1):
            result = self.dempster_combine(result, m_avg)
        return result

    def calculate_belief_plausibility(self, fused_bpa: Dict[str, float]) -> Dict[str, Any]:
        """
        计算每个 FoD 选项的信念度(Belief)和似真度(Plausibility)。

        对单元素焦集：
          Bel(opt_i) = m(opt_i)
          Pl(opt_i)  = m(opt_i) + m(Θ)   （Θ 可能包含 opt_i）
        """
        option_keys = [k for k in fused_bpa.keys() if k != 'uncertainty_theta']
        u = fused_bpa.get('uncertainty_theta', 0.0)

        result = {}
        for opt in option_keys:
            bel = fused_bpa.get(opt, 0.0)
            pl  = bel + u
            result[opt] = {
                "belief":               round(bel, 4),
                "plausibility":         round(pl, 4),
                "uncertainty_interval": round(pl - bel, 4),
            }
        return result

    def make_decision(self, fused_bpa: Dict[str, float], threshold: float = 0.4) -> Dict[str, Any]:
        """
        基于融合 BPA 做出决策：选出质量最高的 FoD 选项。

        规则：
          1. 若最高选项质量 ≥ threshold → 直接输出该选项
          2. 若最高选项质量 > uncertainty 且 > 次高选项 × 1.5 → 有优势，输出该选项
          3. 否则 → UNCERTAIN
        """
        option_masses = {k: v for k, v in fused_bpa.items() if k != 'uncertainty_theta'}
        u = fused_bpa.get('uncertainty_theta', 1.0)

        if not option_masses or max(option_masses.values()) == 0:
            return {"decision": "UNCERTAIN", "confidence": 0.0,
                    "reason": "所有选项质量为零，证据不足"}

        best_opt    = max(option_masses, key=option_masses.get)
        best_mass   = option_masses[best_opt]
        other_vals  = sorted([v for k, v in option_masses.items() if k != best_opt], reverse=True)
        second_mass = other_vals[0] if other_vals else 0.0

        if best_mass >= threshold:
            return {
                "decision":   best_opt,
                "confidence": round(best_mass, 4),
                "reason":     f"选项 '{best_opt}' 获得最高质量 {best_mass:.4f}，超过阈值 {threshold}"
            }
        elif best_mass > u and best_mass > second_mass * 1.5:
            return {
                "decision":   best_opt,
                "confidence": round(best_mass, 4),
                "reason":     f"选项 '{best_opt}' 获得最高质量 {best_mass:.4f}，显著优于其他选项（次高: {second_mass:.4f}）"
            }
        else:
            return {
                "decision":   "UNCERTAIN",
                "confidence": round(best_mass, 4),
                "reason":     f"证据不足，最高质量选项 '{best_opt}' 仅 {best_mass:.4f}，与不确定性({u:.4f})相当"
            }

    # 流程控制与生成指引 (替代 LLM)
    def fuse_evidence(self, bpa_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """智能融合路由：计算平均冲突，自动选择策略"""
        if not bpa_list:
            return {"fused_bpa": {"support_hypothesis": 0.0, "against_hypothesis": 0.0, "uncertainty": 1.0}, "method": "none", "conflict_coefficient": 0.0}
        if len(bpa_list) == 1:
            return {"fused_bpa": bpa_list[0], "method": "single", "conflict_coefficient": 0.0}
        
        # 计算全局平均冲突系数 K
        conflicts = [self.calculate_conflict_coefficient(bpa_list[i], bpa_list[i + 1]) for i in range(len(bpa_list) - 1)]
        avg_conflict = sum(conflicts) / len(conflicts) if conflicts else 0.0
        
        # 纯规则路由：K<0.3 用 Dempster，否则用 Murphy
        strategy = "dempster" if avg_conflict < 0.3 else "murphy"
        
        if strategy == "dempster":
            result = bpa_list[0]
            for bpa in bpa_list[1:]:
                result = self.dempster_combine(result, bpa)
            method_desc = "Dempster组合规则 (由于证据表现出低冲突、高一致性)"
        else:
            result = self.murphy_average_combine(bpa_list)
            method_desc = "Murphy平均规则 (由于检测到证据间存在高冲突)"
            
        return {"fused_bpa": result, "method": method_desc, "strategy": strategy, "conflict_coefficient": round(avg_conflict, 4)}

    def generate_generation_guidance(self, k: float, bel_pl: Dict[str, Any], decision: str) -> str:
        """
        将数学指标转化为给 Agent E (生成模型) 的强制执行指令。
        bel_pl: {opt_name: {belief, plausibility, uncertainty_interval}, ...}
        decision: 最优 FoD 选项名，或 "UNCERTAIN"
        """
        guidance_lines = []

        # 1. 冲突态势
        if k >= 0.4:
            guidance_lines.append(
                f"- 【高冲突警示】冲突系数高达 {k:.2f}。**必须**在回答中明确指出学术界存在争议，"
                f"并展示多个候选选项的证据对比。"
            )
        elif k <= 0.2:
            guidance_lines.append(
                f"- 【高一致性】冲突系数极低 ({k:.2f})。证据方向统一，请直接综合陈述结论。"
            )

        # 2. 置信度语气
        if decision != "UNCERTAIN" and decision in bel_pl:
            confidence = bel_pl[decision]["belief"]
            if confidence >= 0.5:
                guidance_lines.append(
                    f"- 【语气定调】证据支持力度较高 (Belief={confidence:.2f})。"
                    f"请以较为笃定的语气指出 '{decision}' 为最佳答案。"
                )
            else:
                guidance_lines.append(
                    f"- 【语气定调】证据支持有限 (Belief={confidence:.2f})。"
                    f"请使用倾向性但保守的语气（例如：当前证据更倾向于 '{decision}'...）。"
                )
        else:
            guidance_lines.append(
                "- 【语气定调】无法得出确定结论。请以客观中立的语气解释为何目前无法确定最佳选项。"
            )

        return "\n".join(guidance_lines)

    def run(self, question: str, fod: List[str], bpa_list: List[Dict[str, float]], verbose: bool = False) -> Dict[str, Any]:
        """运行纯数学引擎的完整流程"""
        if verbose:
            print(f"\n{'='*40}\n[智能体D] 纯数学融合引擎启动\n{'='*40}")
        
        # 步骤1: 纯数学融合
        fusion_result = self.fuse_evidence(bpa_list)
        
        # 步骤2: 计算每个FoD选项的信念度和似真度
        belief_pl = self.calculate_belief_plausibility(fusion_result['fused_bpa'])
        
        # 步骤3: 基于融合BPA直接决策（选出质量最高选项）
        decision = self.make_decision(fusion_result['fused_bpa'])
        
        # 步骤4: 生成供下游 Agent E 使用的提示词指令
        k_value = fusion_result['conflict_coefficient']
        generation_guidance = self.generate_generation_guidance(k_value, belief_pl, decision['decision'])
        
        if verbose:
            print(f"✅ 最终决策: {decision['decision']} (置信度: {decision['confidence']:.4f})")
            print(f"✅ K 值: {k_value} | 策略: {fusion_result.get('strategy', 'N/A')}")
            print(f"✅ 已生成下游生成指令。")

        return {
            "question": question,
            "frame_of_discernment": fod,
            "fusion_result": fusion_result,
            "belief_plausibility": belief_pl,
            "final_decision": decision,
            "generation_guidance_for_LLM": generation_guidance
        }

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


class AgentDirectLLM:
    """
    直接LLM推理智能体
    功能：接收智能体B输出的半结构化证据，直接调用LLM生成答案（绕过D-S融合流程）
    该智能体作为可选分支存在，由主流程中的开关 ENABLE_DIRECT_LLM_BRANCH 控制。
    """

    def __init__(self):
        self.args = set_argument()

    def _call_llm_api(self, prompt: str, temperature: float = 0) -> str:
        return call_llm(prompt, temperature=temperature, max_tokens=2000)

    def run(
        self,
        question: str,
        agent_b_result: Dict[str, Any],
        task_mode: str = "SELECTION",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        运行直接LLM推理分支。

        Args:
            question:       原始医学问题
            agent_b_result: 智能体B输出的半结构化证据分析结果
            task_mode:      来自Agent A的任务模式，"SELECTION"→MCQ，其他→Yes/No
            verbose:        是否打印调试信息

        Returns:
            MCQ模式：包含 selected_option / reasoning / confidence_score / key_evidence_used
            YesNo模式：包含 answer / reasoning / confidence_score / key_evidence_used
        """
        is_mcq = (task_mode == "SELECTION")

        if verbose:
            print(f"\n{'='*60}")
            mode_label = "MCQ（选择题）" if is_mcq else "Yes/No（是非题）"
            print(f"[直接LLM分支] 开始直接推理 | 题型: {mode_label}")
            print(f"{'='*60}")

        # 将 agent_b 中的 analyzed_evidences 格式化为精简可读摘要
        analyzed_evidences = agent_b_result.get('analyzed_evidences', [])
        evidence_summaries = []
        for ev in analyzed_evidences[:15]:  # 最多取15条，避免超Token
            summary: Dict[str, Any] = {
                "evidence_id": ev.get('evidence_id'),
                "source_type": ev.get('source_type', 'Unknown'),
                "metadata": ev.get('metadata', {}),
            }
            analysis = ev.get('analysis', {})
            if analysis:
                summary["clinical_summary"] = analysis.get('clinical_summary', '')
                summary["study_design"] = analysis.get(
                    'study_design', analysis.get('study_type', '')
                )
                pico = analysis.get('pico', {})
                if pico:
                    summary["pico"] = pico
            # 兜底：节选原始内容
            if not summary.get("clinical_summary"):
                summary["content_snippet"] = ev.get('original_content', '')[:600]
            evidence_summaries.append(summary)

        # 根据题型选择对应 Prompt
        prompt_template = Prompt_DirectLLM if is_mcq else Prompt_DirectLLM_YesNo
        prompt = prompt_template.replace("{{QUESTION}}", question)
        prompt = prompt.replace(
            "{{ANALYZED_EVIDENCE}}",
            json.dumps(evidence_summaries, ensure_ascii=False, indent=2)
        )

        response = self._call_llm_api(prompt)
        result = extract_json_from_response(response)

        if result is None:
            print(f"[直接LLM分支] JSON解析失败，原始响应: {response[:300]}...")
            result = {
                "selected_option" if is_mcq else "answer": "UNKNOWN",
                "reasoning": response[:500],
                "confidence_score": 0.0,
                "key_evidence_used": [],
                "error": "JSON解析失败"
            }

        if verbose:
            ans_key = "selected_option" if is_mcq else "answer"
            print(f"[直接LLM分支] 答案: {result.get(ans_key)}")
            print(f"[直接LLM分支] 置信度:   {result.get('confidence_score')}")

        return result


class AgentFinalAggregator:
    """
    最终聚合智能体
    功能：综合D-S推理分支（智能体E输出）与直接LLM分支的结果，生成最终答案。
    只在 ENABLE_DIRECT_LLM_BRANCH=True 时由主流程调用。
    """

    def __init__(self):
        self.args = set_argument()

    def _call_llm_api(self, prompt: str, temperature: float = 0) -> str:
        return call_llm(prompt, temperature=temperature, max_tokens=2000)

    def run(
        self,
        question: str,
        ds_result: Dict[str, Any],
        direct_llm_result: Dict[str, Any],
        task_mode: str = "SELECTION",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        综合两条分支的结果，输出最终答案。

        Args:
            question:          原始医学问题
            ds_result:         智能体E生成的DS推理最终报告
            direct_llm_result: 直接LLM分支的推理结果
            task_mode:         来自Agent A的任务模式，"SELECTION"→MCQ，其他→Yes/No
            verbose:           是否打印调试信息

        Returns:
            包含 final_answer / agreement / reasoning / confidence_score / integration_note 的字典
        """
        is_mcq = (task_mode == "SELECTION")

        if verbose:
            print(f"\n{'='*60}")
            mode_label = "MCQ（选择题）" if is_mcq else "Yes/No（是非题）"
            print(f"[最终聚合] 综合两条分支结果 | 题型: {mode_label}")
            print(f"{'='*60}")

        # 根据题型选择对应 Prompt
        prompt_template = Prompt_FinalAggregator if is_mcq else Prompt_FinalAggregator_YesNo
        prompt = prompt_template.replace("{{QUESTION}}", question)
        prompt = prompt.replace(
            "{{DS_RESULT}}",
            json.dumps(ds_result, ensure_ascii=False, indent=2)
        )
        prompt = prompt.replace(
            "{{DIRECT_LLM_RESULT}}",
            json.dumps(direct_llm_result, ensure_ascii=False, indent=2)
        )

        response = self._call_llm_api(prompt)
        result = extract_json_from_response(response)

        if result is None:
            print(f"[最终聚合] JSON解析失败，原始响应: {response[:300]}...")
            # 兜底：选择置信度更高的一方
            ds_conf = ds_result.get('confidence_score', 0.0)
            llm_conf = direct_llm_result.get('confidence_score', 0.0)
            if ds_conf >= llm_conf:
                # DS分支答案字段：MCQ用answer(AgentE输出)，YesNo同
                chosen = ds_result.get('answer',
                            ds_result.get('selected_option',
                            ds_result.get('decision', 'UNKNOWN')))
                note = f"JSON解析失败，兜底选择DS分支（置信度 {ds_conf:.2f} >= LLM分支 {llm_conf:.2f}）"
            else:
                # 直接LLM分支：MCQ用selected_option，YesNo用answer
                chosen = direct_llm_result.get(
                    'selected_option' if is_mcq else 'answer', 'UNKNOWN'
                )
                note = f"JSON解析失败，兜底选择LLM分支（置信度 {llm_conf:.2f} > DS分支 {ds_conf:.2f}）"
            result = {
                "final_answer": chosen,
                "agreement": "unknown",
                "reasoning": response[:500],
                "confidence_score": max(ds_conf, llm_conf),
                "integration_note": note,
                "error": "JSON解析失败"
            }

        if verbose:
            print(f"[最终聚合] 最终答案:     {result.get('final_answer')}")
            print(f"[最终聚合] 两分支一致性: {result.get('agreement')}")
            print(f"[最终聚合] 置信度:       {result.get('confidence_score')}")

        return result

