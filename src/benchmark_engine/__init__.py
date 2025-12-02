"""
Benchmark engine module for BenchRight LLM evaluation.

This module provides a generic evaluation loop for running benchmarks
on language models, including LLM-as-Judge evaluation capabilities.
"""

from .engine import run_benchmark, exact_match_metric, contains_metric
from .llm_judge import LLMJudge, JUDGE_SYSTEM_PROMPT

__all__ = [
    "run_benchmark",
    "exact_match_metric",
    "contains_metric",
    "LLMJudge",
    "JUDGE_SYSTEM_PROMPT",
]
