"""
Reporting module for BenchRight evaluation.

This module provides tools for comparing evaluation runs and identifying
regressions between model versions.

Example usage:
    >>> from benchmark_engine.reporting import compare_runs, summarize_regressions
    >>> diff_df = compare_runs(run_a_df, run_b_df, on="prompt")
    >>> regressions = summarize_regressions(diff_df, metric="score")
    >>> print(regressions)
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np


def compare_runs(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    on: str = "prompt",
    metrics: Optional[List[str]] = None,
    suffixes: tuple = ("_a", "_b")
) -> pd.DataFrame:
    """
    Compare metrics between two evaluation runs.

    This function merges two DataFrames containing evaluation results
    and computes differences for specified metrics.

    Args:
        run_a_df: DataFrame from the first (baseline) run.
        run_b_df: DataFrame from the second (new) run.
        on: Column name to join on. Default is "prompt".
        metrics: List of metric columns to compare. If None, automatically
                detects numeric columns.
        suffixes: Suffixes for disambiguating columns. Default is ("_a", "_b").

    Returns:
        A merged DataFrame with:
            - The join key column
            - Original metric columns from both runs (with suffixes)
            - Difference columns (metric_diff = new - baseline)
            - Percent change columns (metric_pct_change)

    Raises:
        ValueError: If the join column is not present in both DataFrames.

    Example:
        >>> run_a = pd.DataFrame({'prompt': ['p1', 'p2'], 'score': [0.8, 0.9]})
        >>> run_b = pd.DataFrame({'prompt': ['p1', 'p2'], 'score': [0.85, 0.75]})
        >>> diff = compare_runs(run_a, run_b)
        >>> print(diff[['prompt', 'score_a', 'score_b', 'score_diff']])
    """
    # Validate join column
    if on not in run_a_df.columns:
        raise ValueError(f"Column '{on}' not found in run_a_df")
    if on not in run_b_df.columns:
        raise ValueError(f"Column '{on}' not found in run_b_df")

    # Merge DataFrames
    merged = pd.merge(
        run_a_df,
        run_b_df,
        on=on,
        how="outer",
        suffixes=suffixes
    )

    # Auto-detect metrics if not specified
    if metrics is None:
        # Find numeric columns that appear in both (before merge)
        a_numeric = set(run_a_df.select_dtypes(include=[np.number]).columns)
        b_numeric = set(run_b_df.select_dtypes(include=[np.number]).columns)
        metrics = list(a_numeric & b_numeric)

    # Compute differences for each metric
    for metric in metrics:
        col_a = f"{metric}{suffixes[0]}"
        col_b = f"{metric}{suffixes[1]}"

        if col_a in merged.columns and col_b in merged.columns:
            # Absolute difference (new - baseline)
            merged[f"{metric}_diff"] = merged[col_b] - merged[col_a]

            # Percent change
            merged[f"{metric}_pct_change"] = np.where(
                merged[col_a] != 0,
                ((merged[col_b] - merged[col_a]) / merged[col_a].abs()) * 100,
                np.where(merged[col_b] != 0, np.inf, 0.0)
            )

    return merged


def summarize_regressions(
    diff_df: pd.DataFrame,
    metric: str = "score",
    threshold: float = 0.0,
    show_improvements: bool = False
) -> pd.DataFrame:
    """
    Identify and summarize regressions in a comparison DataFrame.

    A regression is defined as a case where the new model performs worse
    than the baseline (negative diff for metrics where higher is better).

    Args:
        diff_df: DataFrame from compare_runs() with difference columns.
        metric: The metric to analyze. Default is "score".
        threshold: Minimum absolute difference to consider significant.
                  Default is 0.0 (any negative change is a regression).
        show_improvements: If True, also include improvements (positive changes).
                          Default is False.

    Returns:
        A DataFrame containing:
            - All rows where the metric regressed (or improved if show_improvements)
            - Sorted by the magnitude of regression
            - Summary statistics at the end

    Example:
        >>> regressions = summarize_regressions(diff_df, metric="score", threshold=0.05)
        >>> print(f"Found {len(regressions)} regressions")
    """
    diff_col = f"{metric}_diff"
    pct_col = f"{metric}_pct_change"

    if diff_col not in diff_df.columns:
        raise ValueError(
            f"Difference column '{diff_col}' not found. "
            f"Make sure to run compare_runs() first with metric '{metric}'."
        )

    # Filter regressions (negative diff = new model worse)
    if show_improvements:
        # Show all changes above threshold
        mask = diff_df[diff_col].abs() > threshold
    else:
        # Only regressions (negative changes)
        mask = diff_df[diff_col] < -threshold

    regressions = diff_df[mask].copy()

    # Sort by magnitude of regression (most severe first)
    regressions = regressions.sort_values(diff_col, ascending=True)

    return regressions


def generate_comparison_report(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    on: str = "prompt",
    metrics: Optional[List[str]] = None,
    run_a_name: str = "Baseline",
    run_b_name: str = "New Model"
) -> Dict[str, Any]:
    """
    Generate a comprehensive comparison report between two runs.

    Args:
        run_a_df: DataFrame from the baseline run.
        run_b_df: DataFrame from the new run.
        on: Column name to join on.
        metrics: List of metrics to analyze.
        run_a_name: Display name for the baseline run.
        run_b_name: Display name for the new run.

    Returns:
        A dictionary containing:
            - 'summary': High-level statistics
            - 'by_metric': Per-metric breakdown
            - 'regressions': List of regressions per metric
            - 'improvements': List of improvements per metric
            - 'comparison_df': The full comparison DataFrame
    """
    # Perform comparison
    diff_df = compare_runs(run_a_df, run_b_df, on=on, metrics=metrics)

    # Auto-detect metrics if not specified
    if metrics is None:
        metrics = [col.replace("_diff", "") for col in diff_df.columns if col.endswith("_diff")]

    report: Dict[str, Any] = {
        'summary': {
            'run_a_name': run_a_name,
            'run_b_name': run_b_name,
            'total_examples': len(diff_df),
            'matched_examples': diff_df.dropna(subset=[f"{metrics[0]}_diff"]).shape[0] if metrics else 0,
        },
        'by_metric': {},
        'regressions': {},
        'improvements': {},
        'comparison_df': diff_df
    }

    for metric in metrics:
        diff_col = f"{metric}_diff"
        if diff_col not in diff_df.columns:
            continue

        # Calculate statistics
        diffs = diff_df[diff_col].dropna()

        metric_stats = {
            'mean_diff': diffs.mean(),
            'std_diff': diffs.std(),
            'median_diff': diffs.median(),
            'min_diff': diffs.min(),
            'max_diff': diffs.max(),
            'n_regressions': (diffs < 0).sum(),
            'n_improvements': (diffs > 0).sum(),
            'n_unchanged': (diffs == 0).sum(),
        }

        report['by_metric'][metric] = metric_stats

        # Get regressions and improvements
        report['regressions'][metric] = summarize_regressions(
            diff_df, metric=metric, show_improvements=False
        )
        report['improvements'][metric] = summarize_regressions(
            diff_df, metric=metric, threshold=0.0, show_improvements=True
        )
        # Filter improvements to only positive changes
        impr_df = report['improvements'][metric]
        if len(impr_df) > 0:
            report['improvements'][metric] = impr_df[impr_df[diff_col] > 0]

    return report


def print_comparison_report(report: Dict[str, Any]) -> None:
    """
    Print a formatted comparison report.

    Args:
        report: Report dictionary from generate_comparison_report().
    """
    summary = report['summary']

    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON REPORT")
    print("=" * 70)

    print(f"\n🔄 Comparing: {summary['run_a_name']} vs {summary['run_b_name']}")
    print(f"   Total examples: {summary['total_examples']}")
    print(f"   Matched examples: {summary['matched_examples']}")

    print("\n" + "-" * 70)
    print("📈 METRIC SUMMARY")
    print("-" * 70)

    for metric, stats in report['by_metric'].items():
        print(f"\n📌 {metric.upper()}:")
        print(f"   Mean change:    {stats['mean_diff']:+.4f}")
        print(f"   Std deviation:  {stats['std_diff']:.4f}")
        print(f"   Range:          [{stats['min_diff']:+.4f}, {stats['max_diff']:+.4f}]")
        print(f"   Regressions:    {stats['n_regressions']}")
        print(f"   Improvements:   {stats['n_improvements']}")
        print(f"   Unchanged:      {stats['n_unchanged']}")

        # Overall assessment
        if stats['mean_diff'] > 0:
            print(f"   ✅ Overall: IMPROVED by {stats['mean_diff']:.4f}")
        elif stats['mean_diff'] < 0:
            print(f"   ⚠️ Overall: REGRESSED by {abs(stats['mean_diff']):.4f}")
        else:
            print(f"   ➖ Overall: NO CHANGE")

    # Show top regressions if any
    print("\n" + "-" * 70)
    print("🔻 TOP REGRESSIONS")
    print("-" * 70)

    for metric, regressions in report['regressions'].items():
        if len(regressions) > 0:
            print(f"\n{metric}:")
            top_regressions = regressions.head(5)
            for _, row in top_regressions.iterrows():
                diff = row.get(f"{metric}_diff", 0)
                print(f"   • {row.get('prompt', 'N/A')[:40]}... : {diff:+.4f}")
        else:
            print(f"\n{metric}: No regressions found ✅")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Demonstration with synthetic DataFrames
    print("=" * 70)
    print("BenchRight Reporting Module Demo")
    print("=" * 70)

    # Create synthetic evaluation results for two runs
    np.random.seed(42)

    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis.",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "Describe machine learning.",
        "What causes rain?",
        "Define artificial intelligence.",
        "What is the largest planet?",
        "Explain gravity.",
    ]

    # Baseline model results (Run A)
    run_a_df = pd.DataFrame({
        'prompt': prompts,
        'score': [0.85, 0.78, 1.0, 0.92, 0.88, 0.75, 0.82, 0.79, 0.95, 0.70],
        'latency_ms': [45, 62, 25, 38, 55, 72, 48, 58, 32, 68],
        'coherence': [0.9, 0.85, 0.95, 0.88, 0.82, 0.78, 0.85, 0.80, 0.92, 0.75],
    })

    # New model results (Run B) - some improvements, some regressions
    run_b_df = pd.DataFrame({
        'prompt': prompts,
        'score': [0.88, 0.72, 1.0, 0.95, 0.85, 0.80, 0.78, 0.82, 0.93, 0.75],
        'latency_ms': [42, 58, 22, 35, 52, 68, 45, 55, 30, 65],
        'coherence': [0.92, 0.80, 0.96, 0.90, 0.80, 0.82, 0.82, 0.83, 0.90, 0.78],
    })

    print("\n📝 Sample Data:")
    print("-" * 70)
    print("\nBaseline Model (Run A):")
    print(run_a_df.to_string(index=False))
    print("\nNew Model (Run B):")
    print(run_b_df.to_string(index=False))

    # Compare runs
    print("\n\n📊 Running Comparison...")
    diff_df = compare_runs(run_a_df, run_b_df, on="prompt")

    print("\n📋 Comparison Results:")
    print(diff_df[['prompt', 'score_a', 'score_b', 'score_diff', 'score_pct_change']].to_string(index=False))

    # Summarize regressions
    print("\n\n🔻 Score Regressions (new model worse):")
    regressions = summarize_regressions(diff_df, metric="score")
    if len(regressions) > 0:
        print(regressions[['prompt', 'score_a', 'score_b', 'score_diff']].to_string(index=False))
    else:
        print("No regressions found!")

    # Generate full report
    print("\n")
    report = generate_comparison_report(
        run_a_df, run_b_df,
        run_a_name="Baseline v1.0",
        run_b_name="Candidate v1.1"
    )
    print_comparison_report(report)

    print("\nDemo complete!")
