"""Evidence completeness controller."""

from typing import Dict, Any


class EvidenceCompletenessController:
    """
    Decide whether more retrieval is needed based on entropy and conflict.
    """

    def __init__(self):
        pass

    def calculate_deng_entropy(self, bpa: Dict[str, float]) -> float:
        import math

        m_h = bpa.get("support_hypothesis", 0)
        m_nh = bpa.get("against_hypothesis", 0)
        m_u = bpa.get("uncertainty", 0)

        entropy = 0.0
        if m_h > 0:
            entropy -= m_h * math.log2(m_h / (2**1 - 1)) if m_h < 1 else 0
        if m_nh > 0:
            entropy -= m_nh * math.log2(m_nh / (2**1 - 1)) if m_nh < 1 else 0
        if m_u > 0:
            entropy -= m_u * math.log2(m_u / (2**2 - 1)) if m_u < 1 else 0

        return round(entropy, 4)

    def analyze_completeness(
        self,
        fused_bpa: Dict[str, float],
        belief_pl: Dict[str, Any],
        conflict_coef: float,
    ) -> Dict[str, Any]:
        m_h = fused_bpa.get("support_hypothesis", 0)
        m_nh = fused_bpa.get("against_hypothesis", 0)
        m_u = fused_bpa.get("uncertainty", 0)

        bel_pos = belief_pl["hypothesis_positive"]["belief"]
        bel_neg = belief_pl["hypothesis_negative"]["belief"]
        uncertainty_interval = belief_pl["hypothesis_positive"]["uncertainty_interval"]

        entropy = self.calculate_deng_entropy(fused_bpa)

        if (bel_pos > 0.6 or bel_neg > 0.6) and uncertainty_interval < 0.3:
            return {
                "state": "A",
                "action": "stop",
                "reason": "证据充分，信念度高且不确定性低，可以得出结论",
                "entropy": entropy,
                "should_continue": False,
                "confidence": "high",
            }

        if m_u > 0.6:
            return {
                "state": "B",
                "action": "expand_retrieval",
                "reason": "证据质量不足或相关性低，需要扩展检索范围",
                "entropy": entropy,
                "should_continue": True,
                "confidence": "low",
                "suggestion": "扩展关键词，增加数据源（如Google Scholar、教科书）",
            }

        if conflict_coef > 0.3 or abs(m_h - m_nh) < 0.2:
            return {
                "state": "C",
                "action": "targeted_retrieval",
                "reason": "证据存在冲突，需要针对性检索高质量证据解决争议",
                "entropy": entropy,
                "should_continue": True,
                "confidence": "moderate",
                "suggestion": "检索Meta分析、系统评价或争议性综述",
            }

        if entropy > 1.5:
            return {
                "state": "B-",
                "action": "optional_expand",
                "reason": "证据熵较高，建议继续检索以降低不确定性（可选）",
                "entropy": entropy,
                "should_continue": False,
                "confidence": "moderate",
                "suggestion": "可选：增加1-2轮检索以提高置信度",
            }

        return {
            "state": "A-",
            "action": "stop",
            "reason": "证据基本充分，可以得出结论",
            "entropy": entropy,
            "should_continue": False,
            "confidence": "moderate",
        }
