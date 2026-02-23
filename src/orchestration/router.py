"""
Backward-compatible router exports.
"""
from src.orchestration.workflow import (
    _keyword_classify,
    build_metadata_filter,
    classify_query,
)

__all__ = ["classify_query", "build_metadata_filter", "_keyword_classify"]
