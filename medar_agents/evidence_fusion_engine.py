"""Evidence fusion engine (Dempster-Shafer)."""

from typing import Dict, Any, List


class EvidenceFusionEngine:
    """
    Pure DS fusion and decision making without any LLM calls.
    """

    def calculate_conflict_coefficient(self, bpa1: Dict[str, float], bpa2: Dict[str, float]) -> float:
        option_keys = [k for k in set(list(bpa1.keys()) + list(bpa2.keys())) if k != "uncertainty_theta"]
        k_value = 0.0
        for k1 in option_keys:
            for k2 in option_keys:
                if k1 != k2:
                    k_value += bpa1.get(k1, 0.0) * bpa2.get(k2, 0.0)
        return round(min(k_value, 1.0), 4)

    def dempster_combine(self, bpa1: Dict[str, float], bpa2: Dict[str, float]) -> Dict[str, float]:
        option_keys = sorted(set(
            [k for k in list(bpa1.keys()) + list(bpa2.keys()) if k != "uncertainty_theta"]
        ))
        k_value = self.calculate_conflict_coefficient(bpa1, bpa2)
        if k_value >= 1.0:
            res = {k: 0.0 for k in option_keys}
            res["uncertainty_theta"] = 1.0
            return res

        norm = 1.0 - k_value
        u1 = bpa1.get("uncertainty_theta", 0.0)
        u2 = bpa2.get("uncertainty_theta", 0.0)

        result = {}
        for k in option_keys:
            m1k = bpa1.get(k, 0.0)
            m2k = bpa2.get(k, 0.0)
            result[k] = round((m1k * m2k + m1k * u2 + u1 * m2k) / norm, 6)
        result["uncertainty_theta"] = round((u1 * u2) / norm, 6)
        return result

    def murphy_average_combine(self, bpa_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not bpa_list:
            return {"uncertainty_theta": 1.0}

        all_option_keys = set()
        for bpa in bpa_list:
            all_option_keys.update(k for k in bpa.keys() if k != "uncertainty_theta")

        n = len(bpa_list)
        m_avg: Dict[str, float] = {
            k: sum(b.get(k, 0.0) for b in bpa_list) / n
            for k in all_option_keys
        }
        m_avg["uncertainty_theta"] = sum(
            b.get("uncertainty_theta", 0.0) for b in bpa_list
        ) / n

        result = m_avg
        for _ in range(n - 1):
            result = self.dempster_combine(result, m_avg)
        return result

    def calculate_belief_plausibility(self, fused_bpa: Dict[str, float]) -> Dict[str, Any]:
        option_keys = [k for k in fused_bpa.keys() if k != "uncertainty_theta"]
        u = fused_bpa.get("uncertainty_theta", 0.0)

        result = {}
        for opt in option_keys:
            bel = fused_bpa.get(opt, 0.0)
            pl = bel + u
            result[opt] = {
                "belief": round(bel, 4),
                "plausibility": round(pl, 4),
                "uncertainty_interval": round(pl - bel, 4),
            }
        return result

    def make_decision(self, fused_bpa: Dict[str, float], threshold: float = 0.8) -> Dict[str, Any]:
        option_masses = {k: v for k, v in fused_bpa.items() if k != "uncertainty_theta"}
        u = fused_bpa.get("uncertainty_theta", 1.0)

        if not option_masses or max(option_masses.values()) == 0:
            return {
                "decision": "UNCERTAIN",
                "confidence": 0.0,
                "reason": "所有选项质量为零，证据不足",
            }

        best_opt = max(option_masses, key=option_masses.get)
        best_mass = option_masses[best_opt]
        other_vals = sorted([v for k, v in option_masses.items() if k != best_opt], reverse=True)
        second_mass = other_vals[0] if other_vals else 0.0

        if best_mass >= threshold:
            return {
                "decision": best_opt,
                "confidence": round(best_mass, 4),
                "reason": f"选项 '{best_opt}' 获得最高质量 {best_mass:.4f}，超过阈值 {threshold}",
            }
        if best_mass > u and best_mass > second_mass * 1.5:
            return {
                "decision": best_opt,
                "confidence": round(best_mass, 4),
                "reason": f"选项 '{best_opt}' 获得最高质量 {best_mass:.4f}，显著优于其他选项（次高: {second_mass:.4f}）",
            }
        return {
            "decision": "UNCERTAIN",
            "confidence": round(best_mass, 4),
            "reason": f"证据不足，最高质量选项 '{best_opt}' 仅 {best_mass:.4f}，与不确定性({u:.4f})相当",
        }

    def fuse_evidence(self, bpa_list: List[Dict[str, float]]) -> Dict[str, Any]:
        if not bpa_list:
            return {
                "fused_bpa": {"uncertainty_theta": 1.0},
                "method": "none",
                "strategy": "none",
                "conflict_coefficient": 0.0,
            }
        if len(bpa_list) == 1:
            return {
                "fused_bpa": dict(bpa_list[0]),
                "method": "single",
                "strategy": "single",
                "conflict_coefficient": 0.0,
            }

        conflicts = [
            self.calculate_conflict_coefficient(bpa_list[i], bpa_list[i + 1])
            for i in range(len(bpa_list) - 1)
        ]
        avg_conflict = sum(conflicts) / len(conflicts) if conflicts else 0.0

        strategy = "dempster" if avg_conflict < 0.3 else "murphy"

        if strategy == "dempster":
            result = bpa_list[0]
            for bpa in bpa_list[1:]:
                result = self.dempster_combine(result, bpa)
            method_desc = "Dempster组合规则 (由于证据表现出低冲突、高一致性)"
        else:
            result = self.murphy_average_combine(bpa_list)
            method_desc = "Murphy平均规则 (由于检测到证据间存在高冲突)"

        return {
            "fused_bpa": result,
            "method": method_desc,
            "strategy": strategy,
            "conflict_coefficient": round(avg_conflict, 4),
        }

    def generate_generation_guidance(self, k_value: float, bel_pl: Dict[str, Any], decision: str) -> str:
        guidance_lines = []

        if k_value >= 0.4:
            guidance_lines.append(
                f"- 【高冲突警示】冲突系数高达 {k_value:.2f}。**必须**在回答中明确指出学术界存在争议，"
                f"并展示多个候选选项的证据对比。"
            )
        elif k_value <= 0.2:
            guidance_lines.append(
                f"- 【高一致性】冲突系数极低 ({k_value:.2f})。证据方向统一，请直接综合陈述结论。"
            )

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
        if verbose:
            print(f"\n{'='*40}\n[EvidenceFusionEngine] 纯数学融合引擎启动\n{'='*40}")

        fusion_result = self.fuse_evidence(bpa_list)
        belief_pl = self.calculate_belief_plausibility(fusion_result["fused_bpa"])
        decision = self.make_decision(fusion_result["fused_bpa"])
        k_value = fusion_result["conflict_coefficient"]
        generation_guidance = self.generate_generation_guidance(k_value, belief_pl, decision["decision"])

        if verbose:
            print(f"✅ 最终决策: {decision['decision']} (置信度: {decision['confidence']:.4f})")
            print(f"✅ K 值: {k_value} | 策略: {fusion_result.get('strategy', 'N/A')}")
            print("✅ 已生成下游生成指令。")

        return {
            "question": question,
            "frame_of_discernment": fod,
            "fusion_result": fusion_result,
            "belief_plausibility": belief_pl,
            "final_decision": decision,
            "generation_guidance_for_LLM": generation_guidance,
        }
