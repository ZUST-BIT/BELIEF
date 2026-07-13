"""Shared task-mode validation for answer-producing agents."""

SELECTION = "SELECTION"
YES_NO = "YES_NO"
SUPPORTED_TASK_MODES = frozenset({SELECTION, YES_NO})


def normalize_task_mode(task_mode: str) -> str:
    """Normalize documented aliases and reject unsupported task modes."""
    normalized = str(task_mode or "").strip().upper()
    normalized = normalized.replace("-", "_").replace("/", "_")
    if normalized == "YESNO":
        normalized = YES_NO

    if normalized not in SUPPORTED_TASK_MODES:
        supported = ", ".join(sorted(SUPPORTED_TASK_MODES))
        raise ValueError(
            f"Unsupported task_mode '{task_mode}'. Expected one of: {supported}"
        )
    return normalized
