# Week 10 — Regression & Version Comparison

### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 10, you will:

1. Understand *what regression testing is* and why it matters for model versioning.
2. Learn how to compare benchmark results across different model runs or versions.
3. Implement and use the `compare_runs` and `summarize_regressions` functions from `src/benchmark_engine/reporting.py`.
4. Understand how to identify and prioritize regressions by severity.
5. Apply critical thinking to balance improvements against regressions in model updates.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is Regression Testing?

### Simple Explanation

Imagine you're upgrading your car's engine for better fuel efficiency:

- **Did the new engine break anything?** → This is *regression testing*
- **Is the car still as fast?** → Comparing performance metrics
- **Are there new problems?** → Identifying where the new version is worse

> **Regression testing compares the performance of a new model version against a baseline to ensure improvements don't come at the cost of degraded performance in other areas.**

A good regression analysis helps you:
- Catch performance degradations before deployment
- Make informed decisions about model updates
- Quantify the trade-offs between model versions
- Prioritize which regressions need fixing

### The Three Pillars of Regression Analysis

| Pillar | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Comparison** | Side-by-side metric differences | See exactly what changed |
| **Regression Detection** | Cases where new model is worse | Identify problems to address |
| **Severity Ranking** | Magnitude of performance drops | Prioritize fixes effectively |

---

# 🧠 Section 2 — Key Concepts in Version Comparison

### What is a Regression?

A **regression** occurs when a new model version performs worse than the baseline on a specific metric or test case.

```
Regression = New Model Score < Baseline Score (for metrics where higher is better)
Regression = New Model Score > Baseline Score (for metrics where lower is better)
```

### Types of Metrics

| Metric Type | Direction | Regression Condition | Example |
|-------------|-----------|---------------------|---------|
| **Quality** | Higher is better | new < baseline | Accuracy, F1, Score |
| **Performance** | Lower is better | new > baseline | Latency, Error rate |
| **Efficiency** | Higher is better | new < baseline | Tokens/second, Throughput |

### Why Each Matters

| Metric | Production Scenario | Business Impact |
|--------|---------------------|-----------------|
| **Score regressions** | Model gives worse answers | User dissatisfaction, errors |
| **Latency regressions** | Slower response times | User experience degradation |
| **Throughput regressions** | Reduced capacity | Higher infrastructure costs |

---

# 🧪 Section 3 — The Reporting Module

### Module Design

The `reporting.py` module provides three main functions:

```python
def compare_runs(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    on: str = "prompt",
) -> pd.DataFrame:
    """
    Compare metrics between two benchmark runs.
    
    Args:
        run_a_df: DataFrame from baseline run
        run_b_df: DataFrame from comparison run (new model)
        on: Column to join on (default: "prompt")
        
    Returns:
        Merged DataFrame with difference columns for each metric
    """

def summarize_regressions(
    diff_df: pd.DataFrame,
    metric: str = "score",
    threshold: float = 0.0,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Identify cases where the new model performs worse.
    
    Args:
        diff_df: DataFrame from compare_runs
        metric: Metric name to analyze
        threshold: Minimum difference to count as regression
        higher_is_better: True for quality metrics, False for latency
        
    Returns:
        DataFrame of regressions sorted by severity
    """

def generate_regression_report(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    on: str = "prompt",
    metrics: List[str] = None,
    metric_directions: dict = None,
) -> dict:
    """
    Generate a comprehensive regression report for multiple metrics.
    
    Returns dict with comparison_df, regressions, and summary statistics
    """
```

### Key Design Decisions

1. **DataFrame-based:** Uses pandas for familiar data manipulation
2. **Flexible joining:** Join on any column (prompt, test_id, etc.)
3. **Direction-aware:** Handles both "higher is better" and "lower is better" metrics
4. **Severity ranking:** Sort regressions by magnitude to prioritize fixes
5. **Threshold support:** Filter out noise with configurable thresholds

---

# 🧪 Section 4 — Hands-on Lab: Using Regression Analysis

### Lab Overview

In this lab, you will:
1. Import the reporting functions
2. Create synthetic benchmark data for two model versions
3. Compare the runs and calculate differences
4. Identify and rank regressions
5. Generate a comprehensive regression report

### Step 1: Import the Module

```python
from src.benchmark_engine.reporting import (
    compare_runs,
    summarize_regressions,
    generate_regression_report,
)
import pandas as pd
```

### Step 2: Create Synthetic Benchmark Data

```python
# Baseline model results
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
})

# New model results
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
})
```

### Step 3: Compare Runs

```python
# Compare the two runs
diff_df = compare_runs(run_a, run_b, on="prompt")

# View the comparison
print(diff_df[['prompt', 'score_a', 'score_b', 'score_diff']])
```

### Step 4: Find Regressions

```python
# Find score regressions (higher is better)
score_regressions = summarize_regressions(
    diff_df, 
    metric="score", 
    higher_is_better=True
)

print(f"Found {len(score_regressions)} score regressions")
for _, row in score_regressions.iterrows():
    print(f"  {row['prompt']}: {row['score_a']} → {row['score_b']}")

# Find latency regressions (lower is better)
latency_regressions = summarize_regressions(
    diff_df,
    metric="latency_ms",
    higher_is_better=False
)

print(f"\nFound {len(latency_regressions)} latency regressions")
```

### Step 5: Generate Full Report

```python
# Generate comprehensive report
report = generate_regression_report(
    run_a,
    run_b,
    on="prompt",
    metrics=["score", "latency_ms"],
    metric_directions={
        "score": True,        # Higher is better
        "latency_ms": False,  # Lower is better
    },
)

# Access report components
print(f"Total test cases: {report['total_cases']}")
print(f"Metrics analyzed: {report['metrics_analyzed']}")

for metric, summary in report['summary'].items():
    print(f"\n{metric}:")
    print(f"  Regressions: {summary['total_regressions']}")
    print(f"  Rate: {summary['regression_rate']:.1%}")
```

---

# 🤔 Section 5 — Paul-Elder Critical Thinking Questions

### Question 1: EVIDENCE
**If a new model shows 5% improvement on average but has 10% of test cases with regressions, should you deploy it?**

*Consider: Severity of regressions, importance of affected use cases, rollback capability, user impact.*

### Question 2: ASSUMPTIONS
**What assumptions are we making when we compare runs using exact prompt matching?**

*Consider: Prompt wording variations, test set representativeness, production prompt distribution, edge cases.*

### Question 3: IMPLICATIONS
**If we only track average metrics without regression analysis, what might we miss?**

*Consider: Subset degradation, important edge cases, user-facing failures, compliance issues.*

### Question 4: POINT OF VIEW
**How might different stakeholders interpret the same regression report?**

*Consider: Engineers (fixability), Product (user impact), QA (test coverage), Leadership (risk tolerance).*

---

# 🔄 Section 6 — Inversion Thinking: How Can Regression Testing Fail?

Instead of asking "How does regression testing help us?", let's invert:

> **"How can regression testing give us false confidence?"**

### Potential Failure Modes

1. **Incomplete Test Coverage**
   - Test set doesn't represent production
   - Missing critical use cases
   - Overfit to benchmark prompts

2. **Threshold Blindness**
   - Small regressions accumulate over time
   - "Acceptable" regressions compound
   - Death by a thousand paper cuts

3. **Metric Gaming**
   - Optimizing for tracked metrics only
   - Goodhart's Law: when a measure becomes a target, it ceases to be a good measure
   - Missing qualitative regressions

4. **Statistical Noise**
   - Random variation mistaken for regression
   - Not enough test samples
   - Run-to-run variance not accounted for

### Defensive Practices

- **Representative test sets:** Sample from production logs
- **Multiple metrics:** Track quality, performance, and efficiency together
- **Trend analysis:** Look at regressions over multiple releases
- **Human review:** Spot-check regressions, especially severe ones
- **Statistical significance:** Use confidence intervals, not just point estimates
- **Canary deployments:** Detect regressions in production before full rollout

---

# 📝 Section 7 — Mini-Project: Regression Audit

### Task

Conduct a regression analysis comparing two model versions and produce an audit report.

### Instructions

1. **Create benchmark data:**
   - Define 10-20 test prompts
   - Simulate results for baseline and new model
   - Include varied performance (some improvements, some regressions)

2. **Run comparison:**
   - Use `compare_runs` to merge the results
   - Calculate differences for all metrics

3. **Analyze regressions:**
   - Use `summarize_regressions` for each metric
   - Rank by severity
   - Identify patterns in regressions

4. **Generate report:**
   - Use `generate_regression_report` for comprehensive analysis
   - Document findings and recommendations

### Submission Format

Create a markdown file `/examples/week10_regression_audit.md`:

```markdown
# Week 10 Mini-Project: Regression Audit Report

## Executive Summary
[1-2 sentences on overall regression posture]

## Models Compared
- **Baseline:** [Description of baseline model]
- **New Model:** [Description of new model version]

## Test Configuration
- Test cases: [number]
- Metrics analyzed: [list]

## Regression Summary

| Metric | Direction | Regressions | Rate | Max Severity |
|--------|-----------|-------------|------|--------------|
| score | ↑ better | ? | ?% | ? |
| latency_ms | ↓ better | ? | ?% | ? |
| ... | ... | ... | ... | ... |

## Notable Regressions

### Severe Regressions (> 10% degradation)
[List cases with significant performance drops]

### Moderate Regressions (5-10% degradation)
[List cases with moderate performance drops]

## Improvements
[Note any cases where new model improved]

## Recommendations
[2-3 actionable recommendations]

## Decision
- [ ] Deploy new model
- [ ] Investigate regressions first
- [ ] Roll back / reject update

## Limitations
[What this analysis did NOT cover]
```

---

# 🔧 Section 8 — Advanced: Extending the Reporting Module

### Adding Statistical Significance

For production use, add confidence intervals:

```python
# TODO: Implement statistical significance testing
def compare_runs_with_significance(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    on: str = "prompt",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compare runs with statistical significance testing.
    
    Uses bootstrap resampling or paired t-tests to determine
    if differences are statistically significant.
    """
    # from scipy import stats
    # 
    # diff_df = compare_runs(run_a_df, run_b_df, on=on)
    # 
    # # For each metric, calculate p-value
    # for col in diff_df.columns:
    #     if col.endswith('_diff'):
    #         t_stat, p_value = stats.ttest_rel(
    #             run_a_df[base_col], run_b_df[base_col]
    #         )
    #         diff_df[f'{base}_p_value'] = p_value
    #         diff_df[f'{base}_significant'] = p_value < (1 - confidence)
    # 
    # return diff_df
    pass
```

### Adding Trend Analysis

Track regressions over multiple releases:

```python
# TODO: Implement multi-version trend analysis
def analyze_regression_trend(
    version_dfs: List[Tuple[str, pd.DataFrame]],
    baseline_version: str,
    metric: str,
) -> pd.DataFrame:
    """
    Analyze regression trends across multiple model versions.
    
    Args:
        version_dfs: List of (version_name, results_df) tuples
        baseline_version: Version to use as comparison baseline
        metric: Metric to track
        
    Returns:
        DataFrame showing metric trend across versions
    """
    pass
```

### Adding Visualization

Create regression visualization:

```python
# TODO: Implement regression visualization
def plot_regression_summary(
    report: dict,
    save_path: Optional[str] = None,
) -> None:
    """
    Create visualization of regression analysis.
    
    Generates:
    - Bar chart of regression counts per metric
    - Scatter plot of baseline vs new performance
    - Distribution of differences
    """
    # import matplotlib.pyplot as plt
    # 
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # 
    # # Plot regression counts
    # metrics = list(report['summary'].keys())
    # counts = [report['summary'][m]['total_regressions'] for m in metrics]
    # axes[0].bar(metrics, counts)
    # axes[0].set_title('Regressions by Metric')
    # 
    # # ... more visualization code
    # 
    # if save_path:
    #     plt.savefig(save_path)
    # plt.show()
    pass
```

---

# ✔ Knowledge Mastery Checklist
- [ ] I can explain what regression testing is and why it matters for model versioning
- [ ] I understand the difference between quality and performance metrics
- [ ] I can use `compare_runs` to compare two benchmark DataFrames
- [ ] I can use `summarize_regressions` to identify and rank regressions
- [ ] I know how to interpret regression severity and prioritize fixes
- [ ] I understand potential failure modes and defensive practices
- [ ] I completed the mini-project regression audit

---

Week 10 complete.
Next: *Week 11 — Banking & Finance Use Cases*.
