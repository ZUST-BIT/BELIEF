"""Answer arbiter agent for DS and direct LLM arbitration."""

import json
from typing import Dict, Any, Optional

from prompt import Prompt_FinalAggregator, Prompt_FinalAggregator_YesNo
from .json_utils import extract_json_from_response
from .llm_chain import build_llm_chain
from .numeric_utils import clamp_confidence, safe_float
from .task_modes import SELECTION, normalize_task_mode


class AnswerArbiter:
    """
    Arbitration layer between DS results and direct LLM results.
    """

    def __init__(self):
        self.weight_exponent = 1.5
        self.override_margin = 0.12
        self.high_confidence = 0.75
        self.low_conflict_threshold = 0.15
        self.high_conflict_threshold = 0.40
        self._chain = build_llm_chain(
            lambda prompt: prompt,
            temperature=0.2,
            max_tokens=2500,
        )

    def _safe_float(self, x, default=0.0) -> float:
        return safe_float(x, default=default)

    def _safe_confidence(self, value: Any) -> float:
        return clamp_confidence(value)

    def _normalize_mcq_option(self, x: str) -> str:
        if not x:
            return "UNKNOWN"
        x = str(x).strip().upper()
        return x if x in {"A", "B", "C", "D"} else "UNKNOWN"

    def _normalize_yesno_answer(self, x: str) -> str:
        if not x:
            return "maybe"
        x = str(x).strip().lower().replace("-", "_").replace(" ", "_")
        if x in {
            "yes",
            "true",
            "support",
            "supported",
            "strong_yes",
            "support_association",
            "fact_confirmed",
            "favor_intervention",
            "h",
        }:
            return "yes"
        if x in {
            "no",
            "false",
            "refute",
            "refuted",
            "strong_no",
            "refute_association",
            "fact_contradicted",
            "favor_comparator",
            "no_significant_difference",
            "¬h",
        }:
            return "no"
        return "maybe"

    def _map_answer_to_score(self, answer: str, tendency: Optional[str] = None) -> float:
        if not answer:
            return 0.0
        ans = str(answer).lower().strip()

        if ans == "yes":
            return 1.0
        if ans == "no":
            return -1.0
        if ans == "maybe":
            if tendency:
                tend = str(tendency).lower()
                if "strong_yes" in tend:
                    return 0.6
                if "strong_no" in tend:
                    return -0.6
                if "lean_yes" in tend:
                    return 0.3
                if "lean_no" in tend:
                    return -0.3
            return 0.0
        return 0.0

    def _extract_ds_info(self, ds_result: Dict[str, Any], is_mcq: bool) -> Dict[str, Any]:
        if is_mcq:
            option = self._normalize_mcq_option(
                ds_result.get("selected_option", ds_result.get("answer", "UNKNOWN"))
            )
            conf = self._safe_confidence(
                ds_result.get("confidence_score", ds_result.get("decision_confidence", 0.0)),
            )
            conflict = self._safe_confidence(
                ds_result.get("fusion_result", {}).get("conflict_coefficient", 0.0),
            )
            return {
                "answer": option,
                "confidence": conf,
                "reason": ds_result.get("reason", ds_result.get("decision_reason", "")),
                "conflict": conflict,
                "raw": ds_result,
            }

        answer = self._normalize_yesno_answer(ds_result.get("answer", "maybe"))
        conf = self._safe_confidence(ds_result.get("confidence_score", 0.0))
        return {
            "answer": answer,
            "confidence": conf,
            "reason": ds_result.get("reasoning", ""),
            "conflict": self._safe_confidence(
                ds_result.get("fusion_result", {}).get("conflict_coefficient", 0.0)
            ),
            "raw": ds_result,
        }

    def _extract_llm_info(self, llm_result: Dict[str, Any], is_mcq: bool) -> Dict[str, Any]:
        if is_mcq:
            option = self._normalize_mcq_option(llm_result.get("selected_option", "UNKNOWN"))
            conf = self._safe_confidence(llm_result.get("confidence_score", 0.0))
            return {
                "answer": option,
                "confidence": conf,
                "reason": llm_result.get("reasoning", ""),
                "tendency": llm_result.get("directional_tendency", ""),
                "raw": llm_result,
            }

        answer = self._normalize_yesno_answer(llm_result.get("answer", "maybe"))
        conf = self._safe_confidence(llm_result.get("confidence_score", 0.0))
        return {
            "answer": answer,
            "confidence": conf,
            "reason": llm_result.get("reasoning", ""),
            "tendency": llm_result.get("directional_tendency", ""),
            "raw": llm_result,
        }

    def _build_arbitration_context_mcq(self, ds_info: Dict[str, Any], llm_info: Dict[str, Any]) -> Dict[str, Any]:
        ds_ans = ds_info["answer"]
        llm_ans = llm_info["answer"]
        ds_conf = ds_info["confidence"]
        llm_conf = llm_info["confidence"]
        conflict = ds_info.get("conflict", 0.0)

        agreement = (ds_ans != "UNKNOWN" and ds_ans == llm_ans)

        w_ds = ds_conf ** self.weight_exponent
        w_llm = llm_conf ** self.weight_exponent
        total_w = w_ds + w_llm
        ds_weight_norm = (w_ds / total_w) if total_w > 0 else 0.5
        llm_weight_norm = (w_llm / total_w) if total_w > 0 else 0.5

        if agreement:
            recommended_source = "BOTH"
            recommended_answer = ds_ans
            rationale = "Both branches agree on the same option."
        else:
            if ds_conf >= self.high_confidence and (ds_conf - llm_conf) >= self.override_margin:
                recommended_source = "DS"
                recommended_answer = ds_ans
                rationale = "DS branch has a clearly stronger confidence advantage."
            elif llm_conf >= self.high_confidence and (llm_conf - ds_conf) >= self.override_margin:
                recommended_source = "LLM"
                recommended_answer = llm_ans
                rationale = "Direct LLM branch has a clearly stronger confidence advantage."
            else:
                if conflict <= self.low_conflict_threshold and ds_ans != "UNKNOWN":
                    recommended_source = "DS"
                    recommended_answer = ds_ans
                    rationale = "DS conflict is low, so fused evidence is relatively stable."
                elif conflict >= self.high_conflict_threshold and llm_conf >= ds_conf and llm_ans != "UNKNOWN":
                    recommended_source = "LLM"
                    recommended_answer = llm_ans
                    rationale = "DS conflict is high and LLM confidence is not lower."
                else:
                    if ds_weight_norm >= llm_weight_norm and ds_ans != "UNKNOWN":
                        recommended_source = "DS"
                        recommended_answer = ds_ans
                        rationale = "Weighted arbitration slightly favors DS."
                    else:
                        recommended_source = "LLM"
                        recommended_answer = llm_ans
                        rationale = "Weighted arbitration slightly favors Direct LLM."

        advice = []
        if agreement:
            advice.append("The two branches agree; strong prior preference should be given to that answer unless there is an obvious logical flaw.")
        else:
            advice.append("The two branches disagree; examine whether DS retrieved evidence is actually mechanism-relevant or merely topic-relevant.")
            advice.append("If the DS evidence is only indirectly related but the LLM uses strong domain mechanism knowledge, the LLM answer may override DS.")
            advice.append("If the LLM reasoning is vague but DS provides directly aligned evidence with low conflict, prefer DS.")

        if conflict >= self.high_conflict_threshold:
            advice.append("DS conflict is high; be cautious about blindly trusting DS fusion.")
        elif conflict <= self.low_conflict_threshold:
            advice.append("DS conflict is low; DS fused conclusion is relatively internally stable.")

        if abs(ds_conf - llm_conf) <= 0.08:
            advice.append("Confidence gap is small; the final answer may come from either branch or a carefully justified third option within the legal answer set.")
        else:
            advice.append("Confidence gap is meaningful; use that as a strong prior but not as an absolute rule.")

        return {
            "mode": "MCQ",
            "ds_answer": ds_ans,
            "llm_answer": llm_ans,
            "ds_confidence": round(ds_conf, 4),
            "llm_confidence": round(llm_conf, 4),
            "ds_weight_norm": round(ds_weight_norm, 4),
            "llm_weight_norm": round(llm_weight_norm, 4),
            "agreement": agreement,
            "conflict_coefficient": round(conflict, 4),
            "recommended_source": recommended_source,
            "recommended_answer": recommended_answer,
            "recommendation_rationale": rationale,
            "advice": advice,
        }

    def _build_arbitration_context_yesno(self, ds_info: Dict[str, Any], llm_info: Dict[str, Any]) -> Dict[str, Any]:
        ds_ans = ds_info["answer"]
        llm_ans = llm_info["answer"]
        ds_conf = ds_info["confidence"]
        llm_conf = llm_info["confidence"]

        ds_score = self._map_answer_to_score(ds_ans)
        llm_score = self._map_answer_to_score(llm_ans, llm_info.get("tendency"))

        w_ds = ds_conf ** self.weight_exponent
        w_llm = llm_conf ** self.weight_exponent
        total_w = w_ds + w_llm
        final_score = 0.0 if total_w == 0 else (ds_score * w_ds + llm_score * w_llm) / total_w

        if final_score > 0.35:
            recommended_answer = "yes"
        elif final_score < -0.35:
            recommended_answer = "no"
        else:
            recommended_answer = "maybe"

        agreement = (ds_ans == llm_ans)
        ds_contribution = ds_score * w_ds
        llm_contribution = llm_score * w_llm
        if agreement and recommended_answer == ds_ans:
            recommended_source = "BOTH"
        elif abs(ds_contribution) > abs(llm_contribution):
            recommended_source = "DS"
        elif abs(llm_contribution) > abs(ds_contribution):
            recommended_source = "LLM"
        else:
            recommended_source = "BALANCED"

        advice = [
            "Use weighted score as a prior, not as an absolute rule.",
            "If one branch clearly misunderstands the question type or uses irrelevant evidence, override it.",
            "Keep the final reasoning aligned with the final answer.",
        ]

        return {
            "mode": "YESNO",
            "ds_answer": ds_ans,
            "llm_answer": llm_ans,
            "ds_confidence": round(ds_conf, 4),
            "llm_confidence": round(llm_conf, 4),
            "agreement": agreement,
            "weighted_score": round(final_score, 4),
            "recommended_source": recommended_source,
            "recommended_answer": recommended_answer,
            "advice": advice,
        }

    def _build_prompt(
        self,
        question: str,
        ds_result: Dict[str, Any],
        direct_llm_result: Dict[str, Any],
        arbitration_context: Dict[str, Any],
        task_mode: str = "SELECTION",
    ) -> str:
        task_mode = normalize_task_mode(task_mode)
        is_mcq = task_mode == SELECTION
        prompt_template = Prompt_FinalAggregator if is_mcq else Prompt_FinalAggregator_YesNo
        prompt = prompt_template.replace("{{QUESTION}}", question)
        prompt = prompt.replace("{{DS_RESULT}}", json.dumps(ds_result, ensure_ascii=False, indent=2))
        prompt = prompt.replace("{{DIRECT_LLM_RESULT}}", json.dumps(direct_llm_result, ensure_ascii=False, indent=2))
        prompt = prompt.replace("{{ARBITRATION_CONTEXT}}", json.dumps(arbitration_context, ensure_ascii=False, indent=2))
        return prompt

    def _validate_and_repair_result(
        self,
        result: Dict[str, Any],
        arbitration_context: Dict[str, Any],
        is_mcq: bool,
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            result = {}

        legal_mcq = {"A", "B", "C", "D"}
        legal_yesno = {"yes", "no", "maybe"}

        recommended_answer = arbitration_context.get("recommended_answer", "UNKNOWN")
        ds_answer = arbitration_context.get("ds_answer", "UNKNOWN")
        llm_answer = arbitration_context.get("llm_answer", "UNKNOWN")

        raw_answer = result.get("final_answer", recommended_answer if recommended_answer else "UNKNOWN")
        if is_mcq:
            final_answer = self._normalize_mcq_option(raw_answer)
            if final_answer not in legal_mcq:
                if self._normalize_mcq_option(recommended_answer) in legal_mcq:
                    final_answer = self._normalize_mcq_option(recommended_answer)
                elif self._normalize_mcq_option(ds_answer) in legal_mcq:
                    final_answer = self._normalize_mcq_option(ds_answer)
                elif self._normalize_mcq_option(llm_answer) in legal_mcq:
                    final_answer = self._normalize_mcq_option(llm_answer)
                else:
                    final_answer = "UNKNOWN"
        else:
            final_answer = self._normalize_yesno_answer(raw_answer)
            if final_answer not in legal_yesno:
                final_answer = recommended_answer if recommended_answer in legal_yesno else "maybe"

        result["final_answer"] = final_answer

        if ds_answer != "UNKNOWN" and llm_answer != "UNKNOWN" and ds_answer == llm_answer:
            result["agreement"] = "agree"
        else:
            result["agreement"] = "disagree"

        conf = self._safe_confidence(result.get("confidence_score", 0.0))

        if final_answer == recommended_answer:
            conf = max(conf, min(0.95, max(
                self._safe_float(arbitration_context.get("ds_confidence", 0.0), 0.0),
                self._safe_float(arbitration_context.get("llm_confidence", 0.0), 0.0),
            )))
        else:
            conf = min(conf if conf > 0 else 0.65, 0.85)

        result["confidence_score"] = round(conf, 3)

        if not result.get("reasoning"):
            result["reasoning"] = (
                "Final answer was selected by integrating DS evidence fusion and direct clinical reasoning under external arbitration guidance."
            )

        if not result.get("integration_note"):
            result["integration_note"] = f"Recommended source: {arbitration_context.get('recommended_source', 'N/A')}"

        result["arbitration_recommendation"] = arbitration_context.get("recommended_answer")
        result["recommended_source"] = arbitration_context.get("recommended_source")
        if "weighted_score" in arbitration_context:
            result["weighted_score"] = arbitration_context.get("weighted_score")
        result["ds_answer"] = ds_answer
        result["llm_answer"] = llm_answer
        result["ds_confidence"] = arbitration_context.get("ds_confidence")
        result["llm_confidence"] = arbitration_context.get("llm_confidence")

        return result

    def run(
        self,
        question: str,
        ds_result: Dict[str, Any],
        direct_llm_result: Dict[str, Any],
        task_mode: str = "SELECTION",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        task_mode = normalize_task_mode(task_mode)
        is_mcq = task_mode == SELECTION

        ds_info = self._extract_ds_info(ds_result, is_mcq=is_mcq)
        llm_info = self._extract_llm_info(direct_llm_result, is_mcq=is_mcq)

        if is_mcq:
            arbitration_context = self._build_arbitration_context_mcq(ds_info, llm_info)
        else:
            arbitration_context = self._build_arbitration_context_yesno(ds_info, llm_info)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[AnswerArbiter] 综合两条分支结果 | 题型: {'MCQ（选择题）' if is_mcq else 'Yes/No（是非题）'}")
            print(f"[AnswerArbiter] DS答案:  {arbitration_context.get('ds_answer')} (conf={arbitration_context.get('ds_confidence')})")
            print(f"[AnswerArbiter] LLM答案: {arbitration_context.get('llm_answer')} (conf={arbitration_context.get('llm_confidence')})")
            print(f"[AnswerArbiter] 推荐来源: {arbitration_context.get('recommended_source')}")
            print(f"[AnswerArbiter] 推荐答案: {arbitration_context.get('recommended_answer')}")
            if "conflict_coefficient" in arbitration_context:
                print(f"[AnswerArbiter] DS冲突系数: {arbitration_context.get('conflict_coefficient')}")
            if "weighted_score" in arbitration_context:
                print(f"[AnswerArbiter] 加权分数: {arbitration_context.get('weighted_score')}")
            print(f"{'='*60}")

        prompt = self._build_prompt(
            question=question,
            ds_result=ds_result,
            direct_llm_result=direct_llm_result,
            arbitration_context=arbitration_context,
            task_mode=task_mode,
        )

        response = self._chain.invoke(prompt)
        result = extract_json_from_response(response)

        if result is None:
            result = {
                "final_answer": arbitration_context.get("recommended_answer", "UNKNOWN" if is_mcq else "maybe"),
                "agreement": "agree" if arbitration_context.get("agreement") else "disagree",
                "reasoning": "LLM aggregation parsing failed, so the system fell back to the externally recommended arbitration result.",
                "confidence_score": max(
                    self._safe_float(arbitration_context.get("ds_confidence", 0.0), 0.0),
                    self._safe_float(arbitration_context.get("llm_confidence", 0.0), 0.0),
                ),
                "integration_note": "Fallback to externally computed arbitration recommendation.",
            }

        result = self._validate_and_repair_result(
            result=result,
            arbitration_context=arbitration_context,
            is_mcq=is_mcq,
        )

        if verbose:
            print(f"[AnswerArbiter] ✅ 最终答案:     {result.get('final_answer')}")
            print(f"[AnswerArbiter] 🤝 两分支一致性: {result.get('agreement')}")
            print(f"[AnswerArbiter] 📊 置信度:       {result.get('confidence_score')}")
            print(f"[AnswerArbiter] 📝 说明:         {result.get('integration_note')}")

        return result
