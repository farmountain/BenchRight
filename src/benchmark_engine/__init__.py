"""
Benchmark engine module for BenchRight LLM evaluation.

This module provides a generic evaluation loop for running benchmarks
on language models, including LLM-as-Judge evaluation capabilities
and safety/hallucination testing.
"""

from .engine import run_benchmark, exact_match_metric, contains_metric
from .llm_judge import LLMJudge, JUDGE_SYSTEM_PROMPT
from .safety_tests import run_truthfulqa_eval, run_toxigen_eval

__all__ = [
    "run_benchmark",
    "exact_match_metric",
    "contains_metric",
    "LLMJudge",
    "JUDGE_SYSTEM_PROMPT",
    "run_truthfulqa_eval",
    "run_toxigen_eval",
]
