"""
Benchmark engine module for BenchRight LLM evaluation.

This module provides a comprehensive suite of evaluation tools for running
benchmarks on language models, including:

- Generic evaluation loops (engine)
- LLM-as-Judge scoring (llm_judge)
- Safety and hallucination tests (safety_tests)
- Robustness testing (robustness)
- Performance profiling (performance_profiler)
- Run comparison and regression reporting (reporting)
"""

from .engine import run_benchmark, exact_match_metric, contains_metric
from .llm_judge import LLMJudge, MockLLMClient
from .safety_tests import run_truthfulqa_eval, run_toxigen_eval
from .robustness import perturb_prompt, robustness_sweep
from .performance_profiler import profile_model, ONNXProfiler
from .reporting import compare_runs, summarize_regressions, generate_comparison_report

__all__ = [
    # Engine (Week 5)
    "run_benchmark",
    "exact_match_metric",
    "contains_metric",
    # LLM Judge (Week 6)
    "LLMJudge",
    "MockLLMClient",
    # Safety Tests (Week 7)
    "run_truthfulqa_eval",
    "run_toxigen_eval",
    # Robustness (Week 8)
    "perturb_prompt",
    "robustness_sweep",
    # Performance Profiler (Week 9)
    "profile_model",
    "ONNXProfiler",
    # Reporting (Week 10)
    "compare_runs",
    "summarize_regressions",
    "generate_comparison_report",
]
