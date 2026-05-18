"""Compatibility layer for refactored agents."""

from medar_agents import (
    extract_json_from_response,
    QuestionAnalyzer,
    EvidenceAnalyzer,
    EvidenceEvaluator,
    EvidenceFusionEngine,
    ReportGenerator,
    DirectReasoningAgent,
    AnswerArbiter,
    EvidenceCompletenessController,
)

__all__ = [
    "extract_json_from_response",
    "QuestionAnalyzer",
    "EvidenceAnalyzer",
    "EvidenceEvaluator",
    "EvidenceFusionEngine",
    "ReportGenerator",
    "DirectReasoningAgent",
    "AnswerArbiter",
    "EvidenceCompletenessController",
]

