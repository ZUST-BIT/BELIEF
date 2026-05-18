"""Shared helpers for the MEDAR-QA pipeline."""

from typing import Dict, Any, List


JsonDict = Dict[str, Any]


def print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_step(title: str) -> None:
    print(f"\n{title}")
    print("-" * 80)


def dedup_against_user_context(papers: List[JsonDict], user_context_text: str, fingerprint_len: int = 60) -> List[JsonDict]:
    """
    Filter papers that overlap heavily with user context to avoid double counting.
    """
    if not user_context_text:
        return papers
    ctx_lower = user_context_text.lower()
    unique = []
    for paper in papers:
        content = paper.get("content", "")
        is_dup = False
        for line in content.split("\n"):
            line_stripped = line.strip()
            if line_stripped.lower().startswith("summary:"):
                body = line_stripped[len("summary:"):].strip()
                if ":" in body[:20]:
                    body = body[body.index(":") + 1:].strip()
                fingerprint = body[:fingerprint_len].lower()
                if fingerprint and fingerprint in ctx_lower:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(paper)
    return unique


def build_user_context_evidence(context: str) -> List[JsonDict]:
    if not context or not context.strip():
        return []
    return [{
        "source": "user_provided_context",
        "content": context.strip(),
        "score": 1.0,
        "type": "user_context",
        "metadata": {
            "is_user_provided": True,
            "description": "用户提供的背景上下文信息",
        },
    }]


def build_bpa_summary(bpa_list: List[JsonDict]) -> JsonDict:
    if not bpa_list:
        return {"bpa_count": 0}

    all_opts = set()
    for bpa in bpa_list:
        all_opts.update(k for k in bpa.keys() if k != "uncertainty_theta")

    return {
        "option_avg_masses": {
            opt: round(sum(b.get(opt, 0) for b in bpa_list) / len(bpa_list), 4)
            for opt in sorted(all_opts)
        },
        "avg_uncertainty": round(
            sum(b.get("uncertainty_theta", 0) for b in bpa_list) / len(bpa_list), 4
        ),
        "bpa_count": len(bpa_list),
    }


def build_completeness_inputs(agent_d_result: JsonDict) -> JsonDict:
    raw_fused = agent_d_result.get("fusion_result", {}).get("fused_bpa", {})
    opt_masses = {k: v for k, v in raw_fused.items() if k != "uncertainty_theta"}
    top_mass = max(opt_masses.values()) if opt_masses else 0.0
    u_mass = raw_fused.get("uncertainty_theta", 1.0)

    fused_bpa = {
        "support_hypothesis": top_mass,
        "against_hypothesis": round(sum(opt_masses.values()) - top_mass, 4),
        "uncertainty": u_mass,
    }
    belief_pl = {
        "hypothesis_positive": {
            "belief": top_mass,
            "plausibility": round(top_mass + u_mass, 4),
            "uncertainty_interval": u_mass,
        },
        "hypothesis_negative": {
            "belief": round(max(0.0, 1.0 - top_mass - u_mass), 4),
            "plausibility": round(1.0 - top_mass, 4),
            "uncertainty_interval": u_mass,
        },
    }
    conflict_coef = agent_d_result.get("fusion_result", {}).get("conflict_coefficient", 0)

    return {
        "fused_bpa": fused_bpa,
        "belief_pl": belief_pl,
        "conflict_coef": conflict_coef,
    }


def build_enhanced_evidence_input(agent_c_result: JsonDict, fallback_evidence: List[JsonDict]) -> List[JsonDict]:
    enhanced = []
    c_evaluations = agent_c_result.get("evaluations", [])
    if not c_evaluations:
        print("⚠️ 警告：未检测到增强分析，回退到原始证据。")
        return fallback_evidence

    print(f"正在组装 {len(c_evaluations)} 条增强型证据报告...")
    for ev in c_evaluations:
        ev_data = ev.get("evaluation", {})
        rich_content = ev_data.get("content_for_generator")
        if not rich_content:
            rich_content = f"[Raw Snippet]: {ev_data.get('processed_input_snippet', 'N/A')}"

        enhanced.append({
            "source": ev.get("source_type", "Unknown"),
            "content": rich_content,
            "score": max(
                (v for k, v in ev_data.get("bpa_distribution", {}).items()
                 if k != "uncertainty_theta"),
                default=0.0,
            ),
            "type": "analyzed_report",
            "metadata": ev.get("metadata", {}),
        })

    return enhanced
