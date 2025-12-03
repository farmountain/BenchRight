"""
Reporting module for BenchRight.

This module provides functions for comparing benchmark runs and identifying
performance regressions between model versions. It enables systematic
version comparison and regression detection for LLM evaluation.

Example usage:
    >>> from benchmark_engine.reporting import compare_runs, summarize_regressions
    >>> import pandas as pd
    >>>
    >>> # Create DataFrames for two runs
    >>> run_a = pd.DataFrame({'prompt': ['p1', 'p2'], 'score': [0.9, 0.8]})
    >>> run_b = pd.DataFrame({'prompt': ['p1', 'p2'], 'score': [0.85, 0.9]})
    >>>
    >>> # Compare runs
    >>> diff = compare_runs(run_a, run_b, on='prompt')
    >>> print(diff)
    >>>
    >>> # Find regressions
    >>> regressions = summarize_regressions(diff, metric='score')
    >>> print(regressions)
"""

import logging
from typing import List, Optional

import pandas as pd


# Configure module logger
logger = logging.getLogger(__name__)


def compare_runs(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    on: str = "prompt",
    suffixes: tuple = ("_a", "_b"),
) -> pd.DataFrame:
    """Compare metrics between two benchmark runs.

    This function merges two DataFrames from different benchmark runs
    and calculates the difference for all numeric columns. It enables
    side-by-side comparison of model performance across shared prompts
    or test cases.

    Args:
        run_a_df: DataFrame containing results from the first run (baseline).
                  Must contain the column specified by `on` parameter.
        run_b_df: DataFrame containing results from the second run (comparison).
                  Must contain the column specified by `on` parameter.
        on: Column name to join on. This should be a unique identifier
            for each test case (e.g., 'prompt', 'test_id'). Defaults to 'prompt'.
        suffixes: Tuple of suffixes to apply to overlapping column names.
                  Defaults to ('_a', '_b') for run_a and run_b respectively.

    Returns:
        A merged DataFrame containing:
            - The join column (e.g., 'prompt')
            - All columns from run_a with suffix '_a'
            - All columns from run_b with suffix '_b'
            - Difference columns for each numeric metric ('{metric}_diff')
              calculated as run_b value minus run_a value

    Raises:
        ValueError: If the `on` column is not present in both DataFrames.
        ValueError: If either DataFrame is empty.

    Example:
        >>> import pandas as pd
        >>> run_a = pd.DataFrame({
        ...     'prompt': ['What is AI?', 'Define ML'],
        ...     'score': [0.9, 0.8],
        ...     'latency_ms': [100, 120]
        ... })
        >>> run_b = pd.DataFrame({
        ...     'prompt': ['What is AI?', 'Define ML'],
        ...     'score': [0.85, 0.9],
        ...     'latency_ms': [90, 110]
        ... })
        >>> diff = compare_runs(run_a, run_b, on='prompt')
        >>> print(diff.columns.tolist())
        ['prompt', 'score_a', 'latency_ms_a', 'score_b', 'latency_ms_b',
         'score_diff', 'latency_ms_diff']
        >>> print(diff['score_diff'].tolist())
        [-0.05, 0.1]  # run_b - run_a
    """
    # Validate inputs
    if run_a_df.empty:
        raise ValueError("run_a_df cannot be empty")
    if run_b_df.empty:
        raise ValueError("run_b_df cannot be empty")
    if on not in run_a_df.columns:
        raise ValueError(f"Column '{on}' not found in run_a_df. Available: {list(run_a_df.columns)}")
    if on not in run_b_df.columns:
        raise ValueError(f"Column '{on}' not found in run_b_df. Available: {list(run_b_df.columns)}")

    logger.info(f"Comparing runs on column '{on}'")
    logger.debug(f"run_a_df shape: {run_a_df.shape}, run_b_df shape: {run_b_df.shape}")

    # Perform inner join on the specified column
    merged = pd.merge(
        run_a_df,
        run_b_df,
        on=on,
        how="inner",
        suffixes=suffixes,
    )

    if merged.empty:
        logger.warning(f"No matching rows found when joining on '{on}'")
        return merged

    logger.info(f"Merged {len(merged)} rows")

    # Find numeric columns and calculate differences
    # Look for matching column pairs with the suffixes
    suffix_a, suffix_b = suffixes
    columns_a = [c for c in merged.columns if c.endswith(suffix_a)]

    for col_a in columns_a:
        # Extract base column name
        base_name = col_a[: -len(suffix_a)]
        col_b = base_name + suffix_b

        if col_b in merged.columns:
            # Check if both columns are numeric
            if pd.api.types.is_numeric_dtype(merged[col_a]) and pd.api.types.is_numeric_dtype(merged[col_b]):
                diff_col = f"{base_name}_diff"
                merged[diff_col] = merged[col_b] - merged[col_a]
                logger.debug(f"Calculated difference for '{base_name}': mean diff = {merged[diff_col].mean():.4f}")

    return merged


def summarize_regressions(
    diff_df: pd.DataFrame,
    metric: str = "score",
    threshold: float = 0.0,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Identify and summarize cases where the new model performs worse.

    This function analyzes the difference DataFrame from `compare_runs`
    and extracts rows where the specified metric shows regression (i.e.,
    the new model underperforms compared to the baseline).

    Args:
        diff_df: DataFrame from `compare_runs` containing difference columns.
                 Must contain a column named '{metric}_diff'.
        metric: The metric name to analyze for regressions. The function
                looks for a column named '{metric}_diff'. Defaults to 'score'.
        threshold: Minimum absolute difference to consider as a regression.
                   Defaults to 0.0 (any negative change is a regression).
                   Use positive values to filter out small fluctuations.
        higher_is_better: If True, a negative difference indicates regression
                          (e.g., for accuracy, score). If False, a positive
                          difference indicates regression (e.g., for latency,
                          error rate). Defaults to True.

    Returns:
        A DataFrame containing only rows with regressions, sorted by
        severity (most severe regressions first). Includes all original
        columns plus a 'regression_severity' column showing the absolute
        magnitude of the regression.

    Raises:
        ValueError: If diff_df is empty.
        ValueError: If the difference column for the specified metric
                    is not found in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> # Assume diff_df is output from compare_runs
        >>> diff_df = pd.DataFrame({
        ...     'prompt': ['p1', 'p2', 'p3'],
        ...     'score_a': [0.9, 0.8, 0.7],
        ...     'score_b': [0.85, 0.9, 0.6],
        ...     'score_diff': [-0.05, 0.1, -0.1]
        ... })
        >>> regressions = summarize_regressions(diff_df, metric='score')
        >>> print(regressions['prompt'].tolist())
        ['p3', 'p1']  # Sorted by severity, most severe first
        >>> print(regressions['regression_severity'].tolist())
        [0.1, 0.05]

        >>> # For latency where lower is better
        >>> latency_diff = pd.DataFrame({
        ...     'prompt': ['p1', 'p2'],
        ...     'latency_ms_a': [100, 120],
        ...     'latency_ms_b': [110, 100],
        ...     'latency_ms_diff': [10, -20]
        ... })
        >>> regressions = summarize_regressions(
        ...     latency_diff, metric='latency_ms', higher_is_better=False
        ... )
        >>> print(regressions['prompt'].tolist())
        ['p1']  # Only p1 regressed (latency increased)
    """
    # Validate inputs
    if diff_df.empty:
        raise ValueError("diff_df cannot be empty")

    diff_col = f"{metric}_diff"
    if diff_col not in diff_df.columns:
        available_diff_cols = [c for c in diff_df.columns if c.endswith("_diff")]
        raise ValueError(
            f"Column '{diff_col}' not found in diff_df. "
            f"Available difference columns: {available_diff_cols}"
        )

    logger.info(f"Analyzing regressions for metric '{metric}' (higher_is_better={higher_is_better})")

    # Make a copy to avoid modifying original
    result = diff_df.copy()

    # Determine regression condition based on metric direction
    if higher_is_better:
        # For metrics where higher is better (e.g., accuracy, score),
        # a negative difference means regression
        regression_mask = result[diff_col] < -threshold
        result["regression_severity"] = -result[diff_col]
    else:
        # For metrics where lower is better (e.g., latency, error rate),
        # a positive difference means regression
        regression_mask = result[diff_col] > threshold
        result["regression_severity"] = result[diff_col]

    # Filter to only regressions
    regressions = result[regression_mask].copy()

    if regressions.empty:
        logger.info(f"No regressions found for metric '{metric}'")
        return regressions

    # Sort by severity (most severe first)
    regressions = regressions.sort_values("regression_severity", ascending=False)

    logger.info(
        f"Found {len(regressions)} regressions for '{metric}' "
        f"(threshold={threshold}, higher_is_better={higher_is_better})"
    )

    # Log summary statistics
    logger.debug(
        f"Regression severity: mean={regressions['regression_severity'].mean():.4f}, "
        f"max={regressions['regression_severity'].max():.4f}"
    )

    return regressions


def generate_regression_report(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    on: str = "prompt",
    metrics: Optional[List[str]] = None,
    metric_directions: Optional[dict] = None,
    threshold: float = 0.0,
) -> dict:
    """Generate a comprehensive regression report comparing two runs.

    This function provides a high-level interface for regression analysis,
    combining `compare_runs` and `summarize_regressions` for multiple
    metrics and generating summary statistics.

    Args:
        run_a_df: DataFrame containing baseline run results.
        run_b_df: DataFrame containing comparison run results.
        on: Column name to join on. Defaults to 'prompt'.
        metrics: List of metric names to analyze. If None, automatically
                 detects numeric columns. Defaults to None.
        metric_directions: Dictionary mapping metric names to their
                           direction (True if higher is better, False if
                           lower is better). Defaults to assuming higher
                           is better for all metrics.
        threshold: Minimum absolute difference for regression detection.
                   Defaults to 0.0.

    Returns:
        A dictionary containing:
            - 'comparison_df': The full merged DataFrame from compare_runs
            - 'total_cases': Number of test cases compared
            - 'metrics_analyzed': List of metrics that were analyzed
            - 'regressions': Dict mapping metric names to regression DataFrames
            - 'summary': Dict with summary statistics per metric

    Example:
        >>> report = generate_regression_report(run_a, run_b, on='prompt')
        >>> print(f"Total cases: {report['total_cases']}")
        >>> print(f"Metrics: {report['metrics_analyzed']}")
        >>> for metric, regressions_df in report['regressions'].items():
        ...     print(f"{metric}: {len(regressions_df)} regressions")
    """
    # Perform comparison
    diff_df = compare_runs(run_a_df, run_b_df, on=on)

    if diff_df.empty:
        return {
            "comparison_df": diff_df,
            "total_cases": 0,
            "metrics_analyzed": [],
            "regressions": {},
            "summary": {},
        }

    # Auto-detect metrics if not provided
    if metrics is None:
        # Find all difference columns
        diff_cols = [c for c in diff_df.columns if c.endswith("_diff")]
        metrics = [c[:-5] for c in diff_cols]  # Remove '_diff' suffix

    if metric_directions is None:
        metric_directions = {}

    # Analyze each metric
    regressions_dict = {}
    summary_dict = {}

    for metric in metrics:
        diff_col = f"{metric}_diff"
        if diff_col not in diff_df.columns:
            logger.warning(f"Skipping metric '{metric}': no difference column found")
            continue

        higher_is_better = metric_directions.get(metric, True)

        try:
            regressions = summarize_regressions(
                diff_df,
                metric=metric,
                threshold=threshold,
                higher_is_better=higher_is_better,
            )
            regressions_dict[metric] = regressions

            # Calculate summary statistics
            summary_dict[metric] = {
                "total_regressions": len(regressions),
                "regression_rate": len(regressions) / len(diff_df) if len(diff_df) > 0 else 0.0,
                "mean_diff": diff_df[diff_col].mean(),
                "std_diff": diff_df[diff_col].std(),
                "min_diff": diff_df[diff_col].min(),
                "max_diff": diff_df[diff_col].max(),
                "higher_is_better": higher_is_better,
            }

            if len(regressions) > 0:
                summary_dict[metric]["max_regression_severity"] = regressions["regression_severity"].max()
                summary_dict[metric]["mean_regression_severity"] = regressions["regression_severity"].mean()

        except ValueError as e:
            logger.error(f"Error analyzing metric '{metric}': {e}")
            continue

    logger.info(
        f"Regression report generated: {len(diff_df)} cases, "
        f"{len(metrics)} metrics analyzed"
    )

    return {
        "comparison_df": diff_df,
        "total_cases": len(diff_df),
        "metrics_analyzed": list(regressions_dict.keys()),
        "regressions": regressions_dict,
        "summary": summary_dict,
    }


if __name__ == "__main__":
    # Demonstration with synthetic data
    print("=" * 60)
    print("BenchRight Reporting Module Demo")
    print("=" * 60)

    # Create synthetic DataFrames for two benchmark runs
    print("\n📝 Creating synthetic benchmark data...")

    run_a = pd.DataFrame({
        "prompt": [
            "What is the capital of France?",
            "Explain machine learning.",
            "What is 2+2?",
            "Define artificial intelligence.",
            "What is the speed of light?",
        ],
        "score": [0.95, 0.88, 1.00, 0.92, 0.85],
        "latency_ms": [45, 62, 38, 55, 70],
        "tokens_per_second": [120, 95, 140, 105, 85],
    })

    run_b = pd.DataFrame({
        "prompt": [
            "What is the capital of France?",
            "Explain machine learning.",
            "What is 2+2?",
            "Define artificial intelligence.",
            "What is the speed of light?",
        ],
        "score": [0.92, 0.90, 1.00, 0.85, 0.88],
        "latency_ms": [42, 58, 40, 65, 68],
        "tokens_per_second": [125, 100, 135, 90, 88],
    })

    print("\n📊 Run A (Baseline):")
    print(run_a.to_string(index=False))

    print("\n📊 Run B (New Model):")
    print(run_b.to_string(index=False))

    # Compare runs
    print("\n" + "-" * 60)
    print("📈 Comparing Runs...")
    print("-" * 60)

    diff_df = compare_runs(run_a, run_b, on="prompt")

    print("\n📊 Comparison DataFrame:")
    # Show selected columns for readability
    display_cols = ["prompt", "score_a", "score_b", "score_diff", "latency_ms_diff"]
    print(diff_df[display_cols].to_string(index=False))

    # Find regressions for score (higher is better)
    print("\n" + "-" * 60)
    print("🔍 Finding Score Regressions (higher is better)...")
    print("-" * 60)

    score_regressions = summarize_regressions(diff_df, metric="score", higher_is_better=True)

    if not score_regressions.empty:
        print(f"\n⚠️ Found {len(score_regressions)} score regressions:")
        for _, row in score_regressions.iterrows():
            print(f"   - '{row['prompt'][:40]}...': {row['score_a']:.2f} → {row['score_b']:.2f} (Δ={row['score_diff']:.2f})")
    else:
        print("\n✅ No score regressions found!")

    # Find regressions for latency (lower is better)
    print("\n" + "-" * 60)
    print("🔍 Finding Latency Regressions (lower is better)...")
    print("-" * 60)

    latency_regressions = summarize_regressions(
        diff_df, metric="latency_ms", higher_is_better=False
    )

    if not latency_regressions.empty:
        print(f"\n⚠️ Found {len(latency_regressions)} latency regressions:")
        for _, row in latency_regressions.iterrows():
            print(
                f"   - '{row['prompt'][:40]}...': "
                f"{row['latency_ms_a']:.0f}ms → {row['latency_ms_b']:.0f}ms "
                f"(+{row['latency_ms_diff']:.0f}ms)"
            )
    else:
        print("\n✅ No latency regressions found!")

    # Generate full report
    print("\n" + "-" * 60)
    print("📋 Generating Full Regression Report...")
    print("-" * 60)

    report = generate_regression_report(
        run_a,
        run_b,
        on="prompt",
        metrics=["score", "latency_ms", "tokens_per_second"],
        metric_directions={
            "score": True,  # Higher is better
            "latency_ms": False,  # Lower is better
            "tokens_per_second": True,  # Higher is better
        },
    )

    print(f"\n📊 Report Summary:")
    print(f"   Total test cases: {report['total_cases']}")
    print(f"   Metrics analyzed: {report['metrics_analyzed']}")

    print("\n📈 Per-Metric Summary:")
    for metric, stats in report["summary"].items():
        direction = "↑" if stats["higher_is_better"] else "↓"
        print(f"\n   {metric} ({direction} = better):")
        print(f"      Mean diff: {stats['mean_diff']:.4f}")
        print(f"      Regressions: {stats['total_regressions']} ({stats['regression_rate']:.1%})")
        if stats["total_regressions"] > 0:
            print(f"      Max severity: {stats['max_regression_severity']:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
