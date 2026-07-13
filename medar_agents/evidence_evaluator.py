"""Evidence evaluator agent with DS labeling and BPA rules."""

import re
from typing import Dict, Any, List, Optional, Tuple

from prompt import Prompt_C_Optimized
from llm_client import call_llm
from .json_utils import extract_json_from_response
from .llm_chain import build_llm_chain


class EvidenceEvaluator:
    """
    Use LLM for label classification, then compute BPA via rules.
    """

    def __init__(self):
        self._chain = build_llm_chain(
            lambda prompt: prompt,
            temperature=0,
            max_tokens=4096,
            caller=self._call_llm_api,
        )

    def _call_llm_api(
        self,
        prompt: str,
        temperature: float = 0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> str:
        import time

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = call_llm(prompt, temperature=temperature, max_tokens=4096)
                if result:
                    return result
                raise ValueError("LLM returned empty response")
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = retry_delay * (2 ** (attempt - 1))
                    print(f"[EvidenceEvaluator] API调用第{attempt}次失败，{wait:.0f}s后重试... ({e})")
                    time.sleep(wait)
        print(f"[EvidenceEvaluator] API调用连续{max_retries}次失败，跳过本条评估: {last_error}")
        return ""

    def _format_evidence_for_prompt(self, content: str, analysis: Dict[str, Any], source_type: str = "Unknown") -> str:
        formatted_text = f"[Source Type]: {source_type}\n"
        formatted_text += f"### Evidence Content:\n{content}\n"

        if analysis:
            formatted_text += "\n### Pre-Analysis (Reference Info Only — trust original evidence text if conflict exists):\n"
            if analysis.get("clinical_summary"):
                formatted_text += f"- Summary: {analysis['clinical_summary']}\n"

            pico = analysis.get("pico", {})
            if pico:
                formatted_text += "- Structured PICO:\n"
                if pico.get("population"):
                    formatted_text += f"  * Population: {pico['population']}\n"
                if pico.get("intervention"):
                    formatted_text += f"  * Intervention: {pico['intervention']}\n"
                if pico.get("outcome"):
                    formatted_text += f"  * Outcome: {pico['outcome']}\n"

            if analysis.get("study_design"):
                formatted_text += f"- Study Design: {analysis['study_design']}\n"

        return formatted_text

    def _format_result_for_generator(self, evidence_content: str, analysis_result: Dict[str, Any]) -> str:
        reasoning = analysis_result.get(
            "reasoning_trace",
            analysis_result.get("step_by_step_reasoning", {}),
        )
        metrics = analysis_result.get("metrics", {})
        labels = analysis_result.get("labels", {})
        bpa = analysis_result.get("bpa_components", {})

        status = "NEUTRAL"
        if bpa.get("support_hypothesis", 0) > bpa.get("against_hypothesis", 0):
            status = "SUPPORT"
        elif bpa.get("against_hypothesis", 0) > bpa.get("support_hypothesis", 0):
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

[Analyst Reasoning (Evidence Evaluator)]:
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
        if not pico_data:
            return "N/A"
        return (
            f" - Population (P): {pico_data.get('P', 'N/A')}\n"
            f" - Intervention (I): {pico_data.get('I', 'N/A')}\n"
            f" - Comparator (C): {pico_data.get('C', 'N/A')}\n"
            f" - Outcome (O): {pico_data.get('O', 'N/A')}"
        )

    _RELIABILITY_MAP: Dict[str, float] = {
        "GOLD_STANDARD": 0.90,
        "SYSTEMATIC_REVIEW": 0.82,
        "RCT": 0.75,
        "COHORT_CASE_CONTROL": 0.60,
        "CASE_SERIES": 0.38,
        "UNCLEAR_BASIC": 0.28,
    }
    _TRAP_PENALTY_MAP: Dict[str, float] = {
        "NO_TRAP": 0.00,
        "WEAK_SUBGROUP": 0.15,
        "ANIMAL_MODEL_ONLY": 0.20,
        "CONTRADICTORY_INTERNAL": 0.20,
    }
    _DIRECTION_STRENGTH_MAP: Dict[str, float] = {
        "STRONGLY": 0.95,
        "WEAKLY": 0.50,
        "NONE": 0.00,
    }
    _RELEVANCE_SCALE_MAP: Dict[str, float] = {
        "HIGHLY_RELEVANT": 1.0,
        "PARTIALLY_RELEVANT": 0.5,
        "IRRELEVANT": 0.0,
    }

    def _normalize_option_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[\[\]\(\)\{\},.;:!?\'\"“”‘’/_\-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _fuzzy_match_option(self, target: str, frame_of_discernment: List[str]) -> Optional[str]:
        from difflib import SequenceMatcher

        if not target or not frame_of_discernment:
            return None

        target_norm = self._normalize_option_text(target)

        for opt in frame_of_discernment:
            if self._normalize_option_text(opt) == target_norm:
                return opt

        for opt in frame_of_discernment:
            opt_norm = self._normalize_option_text(opt)
            if target_norm in opt_norm or opt_norm in target_norm:
                return opt

        best_opt = None
        best_score = 0.0
        for opt in frame_of_discernment:
            opt_norm = self._normalize_option_text(opt)
            score = SequenceMatcher(None, target_norm, opt_norm).ratio()
            if score > best_score:
                best_score = score
                best_opt = opt

        if best_score >= 0.78:
            return best_opt

        return None

    def compute_bpa_from_tags(
        self,
        labels: Dict[str, str],
        frame_of_discernment: List[str],
    ) -> Tuple[Dict[str, float], float, float]:
        bpa: Dict[str, float] = {opt: 0.0 for opt in frame_of_discernment}
        bpa["uncertainty_theta"] = 0.0

        relevance = labels.get("relevance", "IRRELEVANT")
        relevance_scale = self._RELEVANCE_SCALE_MAP.get(relevance, 0.0)
        if relevance_scale == 0.0:
            bpa["uncertainty_theta"] = 1.0
            return bpa, 0.0, 0.0

        source_privilege = labels.get("source_privilege", "EXTERNAL_LITERATURE")
        if source_privilege == "GOLD_STANDARD":
            base_reliability = self._RELIABILITY_MAP.get("GOLD_STANDARD", 0.90)
        else:
            source_quality = labels.get("source_quality", "UNCLEAR_BASIC")
            base_reliability = self._RELIABILITY_MAP.get(source_quality, 0.30)

        trap_key = labels.get("quality_trap", "NO_TRAP")
        penalty = self._TRAP_PENALTY_MAP.get(trap_key, 0.0)

        adjusted_reliability = max(0.0, (base_reliability - penalty) * relevance_scale)

        polarity = str(labels.get("direction_polarity", "NEUTRAL")).upper().strip()
        strength = str(labels.get("direction_strength", "NONE")).upper().strip()
        mapped_option_raw = labels.get("mapped_fod_option", "NONE")

        if polarity == "NEUTRAL" or strength == "NONE":
            bpa["uncertainty_theta"] = 1.0
            return bpa, adjusted_reliability, 0.0

        degree_of_support = self._DIRECTION_STRENGTH_MAP.get(strength, 0.0)
        if degree_of_support <= 0.0:
            bpa["uncertainty_theta"] = 1.0
            return bpa, adjusted_reliability, 0.0

        mass = adjusted_reliability * degree_of_support
        mass = min(mass, 0.90)

        matched_opt = None
        if mapped_option_raw and str(mapped_option_raw).upper().strip() != "NONE":
            matched_opt = self._fuzzy_match_option(str(mapped_option_raw), frame_of_discernment)

        if matched_opt is not None:
            bpa[matched_opt] = mass
        else:
            bpa["uncertainty_theta"] = mass

        assigned = sum(v for k, v in bpa.items() if k != "uncertainty_theta")
        bpa["uncertainty_theta"] = max(0.0, round(1.0 - assigned, 6))

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
        question_pico: Optional[Dict[str, str]] = None,
        frame_of_discernment: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        rich_evidence_text = self._format_evidence_for_prompt(evidence_content, evidence_analysis, evidence_type)
        question_pico_str = self._format_question_pico(question_pico)

        if not frame_of_discernment:
            frame_of_discernment = ["SUPPORT", "REFUTE"]

        fod_text = "N/A"
        if frame_of_discernment:
            fod_text = "\n".join([f"- {opt}" for opt in frame_of_discernment])

        prompt = Prompt_C_Optimized.replace("{{HYPOTHESIS}}", hypothesis)
        prompt = prompt.replace("{{EVIDENCE_TEXT}}", rich_evidence_text)
        prompt = prompt.replace("{{QUESTION_PICO}}", question_pico_str)
        prompt = prompt.replace("{{FRAME_OF_DISCERNMENT}}", fod_text)

        response = self._chain.invoke(prompt)

        try:
            result = extract_json_from_response(response)
            if result is None:
                raise ValueError("JSON Extraction failed")

            labels = result.get("labels", {})

            expected_label_keys = {
                "source_privilege",
                "relevance",
                "source_quality",
                "quality_trap",
                "direction_polarity",
                "direction_strength",
                "mapped_fod_option",
            }

            if not labels or not (expected_label_keys & set(labels.keys())):
                flat_labels = {k: result[k] for k in expected_label_keys if k in result}
                if flat_labels:
                    labels = flat_labels

            labels.setdefault("direction_polarity", "NEUTRAL")
            labels.setdefault("direction_strength", "NONE")
            labels.setdefault("mapped_fod_option", "NONE")

            valid_polarities = {"SUPPORTS", "REFUTES", "NEUTRAL"}
            valid_strengths = {"STRONGLY", "WEAKLY", "NONE"}

            if str(labels.get("direction_polarity", "")).upper() not in valid_polarities:
                labels["direction_polarity"] = "NEUTRAL"

            if str(labels.get("direction_strength", "")).upper() not in valid_strengths:
                labels["direction_strength"] = "NONE"

            if not labels.get("mapped_fod_option"):
                labels["mapped_fod_option"] = "NONE"

            result["labels"] = labels

            bpa_dist, adjusted_reliability, degree_of_support = self.compute_bpa_from_tags(
                labels, frame_of_discernment
            )

            fod_masses = {k: v for k, v in bpa_dist.items() if k != "uncertainty_theta"}
            m_uncertainty = bpa_dist.get("uncertainty_theta", 1.0)
            total_fod_mass = sum(fod_masses.values())

            polarity = str(labels.get("direction_polarity", "NEUTRAL")).upper()
            if polarity == "REFUTES":
                m_support = 0.0
                m_refute = total_fod_mass
            elif polarity == "SUPPORTS":
                m_support = total_fod_mass
                m_refute = 0.0
            else:
                m_support = 0.0
                m_refute = 0.0

            result["metrics"] = {
                "adjusted_reliability_W": round(adjusted_reliability, 4),
                "degree_of_support_D": round(degree_of_support, 4),
            }
            result["bpa_distribution"] = {k: round(v, 4) for k, v in bpa_dist.items()}
            result["bpa_components"] = {
                "support_hypothesis": round(m_support, 4),
                "against_hypothesis": round(m_refute, 4),
                "uncertainty": round(m_uncertainty, 4),
            }

            result["content_for_generator"] = self._format_result_for_generator(evidence_content, result)
            result["processed_input_snippet"] = rich_evidence_text[:200] + "..."

            return result

        except Exception as e:
            print(f"[EvidenceEvaluator] 评估失败: {e}")
            return {"error": str(e)}

    def _compress_evidence_text(self, content: str, max_len: int = 12000) -> str:
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
        question_pico: Optional[Dict[str, str]] = None,
        frame_of_discernment: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        results = []

        if isinstance(agent_b_result, dict):
            evidence_list = agent_b_result.get("analyzed_evidences", [])
        elif isinstance(agent_b_result, list):
            evidence_list = agent_b_result
        else:
            print("Error: agent_b_result 格式错误")
            return []

        for item in evidence_list:
            ev_id = item.get("evidence_id")
            source_type = item.get("source_type", "Unknown")
            original_content = item.get("original_content", "")
            analysis_data = item.get("analysis", {})
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
                    "metadata": item.get("metadata", {}),
                    "evaluation": evaluation,
                })
            else:
                results.append({
                    "evidence_id": ev_id,
                    "error": "Empty content",
                })
        return results

    def run(
        self,
        hypothesis: str,
        agent_b_result: Dict[str, Any],
        question_pico: Optional[Dict[str, str]] = None,
        frame_of_discernment: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if verbose:
            print(f"\n{'='*60}")
            print("[EvidenceEvaluator] 开始D-S证据评估（标签驱动模式）")
            print(f"{'='*60}")
            print(f"假设命题: {hypothesis}")
            if frame_of_discernment:
                print(f"识别框架: {frame_of_discernment}")

        evaluations = self.evaluate_evidence_batch(
            hypothesis, agent_b_result, question_pico, frame_of_discernment
        )

        valid_bpas = []
        for ev in evaluations:
            if "evaluation" in ev and "bpa_distribution" in ev["evaluation"]:
                if not ev["evaluation"].get("error"):
                    valid_bpas.append(ev["evaluation"]["bpa_distribution"])

        if verbose:
            print(f"\n[EvidenceEvaluator] 评估完成，有效BPA: {len(valid_bpas)}")
            if valid_bpas:
                avg_uncertainty = sum(b.get("uncertainty_theta", 0.0) for b in valid_bpas) / len(valid_bpas)
                avg_assigned = sum(
                    sum(v for k, v in b.items() if k != "uncertainty_theta")
                    for b in valid_bpas
                ) / len(valid_bpas)

                print(f"BPA均值 -> 已分配质量: {avg_assigned:.4f}")
                print(f"BPA均值 -> 不确定性: {avg_uncertainty:.4f}")

        return {
            "hypothesis": hypothesis,
            "evaluations": evaluations,
            "bpa_list": valid_bpas,
        }
