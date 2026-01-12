# 上下管理器 - 负责管理证据和历史过程
class MultimodalContext:
    def __init__(self):
        # 1. 核心证据池：专门喂给生成器 (Generator) 用的纯净干货
        self.evidence_pool = [] 
        
        # 2. 可解释性日志 (Log)
        self.workflow_history = []
        
        # 3. 当前轮次的分析结果缓存
        self._current_round_findings = []
        self._current_round_idx = 0
        self._current_query = ""

    def start_round(self, round_idx, query):
        """
        开始新一轮检索分析
        
        :param round_idx: 当前轮次
        :param query: 当前查询
        """
        self._current_round_idx = round_idx
        self._current_query = query
        self._current_round_findings = []

    def add_analyst_result(self, analyst_result, source_type="Unknown"):
        """
        添加单条分析结果到当前轮次
        
        :param analyst_result: 单条分析结果 (dict)
        :param source_type: 证据来源类型 (PubMed Literature / Knowledge Graph / User Provided Context 等)
        """
        if not analyst_result:
            return
        
        # 缓存到当前轮次
        self._current_round_findings.append({
            "source_type": source_type,
            "thought_process": analyst_result.get("thought_process", ""),
            "relevance_score": analyst_result.get("relevance_score", 0),
            "key_entity_hit": analyst_result.get("key_entity_hit", False),
            "refined_evidence": analyst_result.get("refined_evidence", ""),
        })
        
        # 同时更新证据池 (仅保留精炼后的证据)
        refined = analyst_result.get("refined_evidence", "")
        if refined:
            # 去重检查
            if not any(e.get("refined_evidence") == refined for e in self.evidence_pool):
                self.evidence_pool.append({
                    "source_type": source_type,
                    "thought_process": analyst_result.get("thought_process", ""),
                    "refined_evidence": refined,
                    "relevance_score": analyst_result.get("relevance_score", 0),
                })

    def end_round(self, evaluation_result):
        """
        结束当前轮次，记录评估结果
        
        :param evaluation_result: 评估结果 (dict)
        """
        round_log = {
            "round_id": self._current_round_idx,
            "search_query": self._current_query,
            "analyst_findings": self._current_round_findings.copy(),
            "evaluator_decision": {
                "status": evaluation_result.get("final_decision", "NO-GO"),
                "belief_score": evaluation_result.get("ds_analysis", {}).get("belief_score", 0),
                "uncertainty_gap": evaluation_result.get("ds_analysis", {}).get("uncertainty_gap", 1.0),
                "conflict_detected": evaluation_result.get("ds_analysis", {}).get("conflict_detected", False),
                "reasoning": evaluation_result.get("ds_analysis", {}).get("reasoning", ""),
            }
        }
        
        # 如果是 NO-GO，记录改进策略
        if evaluation_result.get("final_decision") == "NO-GO":
            refinement = evaluation_result.get("refinement_strategy", {})
            round_log["refinement_strategy"] = {
                "missing_info": refinement.get("missing_information", ""),
                "next_queries": refinement.get("next_search_queries", []),
                "feedback": refinement.get("feedback_to_analysis_agent", "")
            }
        
        self.workflow_history.append(round_log)
        # 清空当前轮次缓存
        self._current_round_findings = []

    def get_generator_input(self):
        """
        [关键] 只提取 Generator 需要的干货
        """
        if not self.evidence_pool:
            return "暂无相关证据。"
        
        formatted_str = ""
        for idx, item in enumerate(self.evidence_pool):
            thought_process = item.get("thought_process", "")
            content = item.get("refined_evidence", "")
            score = item.get("relevance_score", 0)
            source = item.get("source_type", "Unknown")
            if content:
                formatted_str += f"[Evidence {idx+1}] (Source: {source}, Score: {score}, Thought Process: {thought_process}): {content}\n\n"
        return formatted_str.strip()

    def get_evidence_payload(self):
        """
        获取证据池中所有精炼后的证据列表
        """
        return [e.get("refined_evidence", "") for e in self.evidence_pool if e.get("refined_evidence")]

    def get_explainability_report(self):
        """
        [关键] 输出完整的可解释性报告
        """
        report = []
        report.append("📝 === 智能体思维链报告 (Explainability Log) ===")
        
        for r in self.workflow_history:
            report.append(f"\n📍 [Round {r['round_id']}]")
            report.append(f"   ❓ Search Query: {r['search_query']}")
            report.append(f"   🔎 Analyst Findings: Found {len(r['analyst_findings'])} valid evidences.")
            
            # 打印部分分析细节
            for i, f in enumerate(r['analyst_findings']):
                score = f.get('relevance_score', 0)
                content = f.get('refined_evidence', '')[:50]
                report.append(f"      - Ev {i+1} (Score: {score}): {content}...")
            
            # 打印评估结果
            eval_data = r.get('evaluator_decision', {})
            status = eval_data.get('status', 'NO-GO')
            status_icon = "✅" if status == 'GO' else "❌"
            report.append(f"   ⚖️ Evaluator Verdict: {status_icon} {status}")
            report.append(f"      - Belief Score: {eval_data.get('belief_score', 0):.2f}")
            report.append(f"      - Uncertainty Gap: {eval_data.get('uncertainty_gap', 1.0):.2f}")
            
            # 如果是 NO-GO，打印改进策略
            if status == 'NO-GO' and 'refinement_strategy' in r:
                strategy = r['refinement_strategy']
                report.append(f"      - Missing: {strategy.get('missing_info', '')}")
                report.append(f"      - Next Queries: {strategy.get('next_queries', [])}")
        
        return "\n".join(report)

    def export_workflow_log(self):
        """
        导出完整的工作流日志 (JSON 格式)
        """
        return {
            "evidence_pool": self.evidence_pool,
            "workflow_history": self.workflow_history
        }