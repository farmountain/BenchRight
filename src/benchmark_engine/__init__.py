"""
Benchmark engine module for BenchRight LLM evaluation.

This module provides a generic evaluation loop for running benchmarks
on language models, including LLM-as-Judge evaluation capabilities,
safety/hallucination testing, robustness evaluation, performance profiling,
and regression analysis for version comparison.
"""

from .engine import run_benchmark, exact_match_metric, contains_metric
from .llm_judge import LLMJudge, JUDGE_SYSTEM_PROMPT
from .safety_tests import run_truthfulqa_eval, run_toxigen_eval
from .robustness import perturb_prompt, robustness_sweep
from .performance_profiler import (
    PerformanceProfiler,
    profile_model,
    create_mock_profiler,
)
from .reporting import (
    compare_runs,
    summarize_regressions,
    generate_regression_report,
)

__all__ = [
    "run_benchmark",
    "exact_match_metric",
    "contains_metric",
    "LLMJudge",
    "JUDGE_SYSTEM_PROMPT",
    "run_truthfulqa_eval",
    "run_toxigen_eval",
    "perturb_prompt",
    "robustness_sweep",
    "PerformanceProfiler",
    "profile_model",
    "create_mock_profiler",
    "compare_runs",
    "summarize_regressions",
    "generate_regression_report",
]
