# Week 9 — Performance Profiling

### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 9, you will:

1. Understand *what performance profiling is* and why it matters for LLM deployment.
2. Learn the key performance metrics: **latency**, **tokens per second**, and **memory usage**.
3. Implement and use the `PerformanceProfiler` class and `profile_model` function from `src/benchmark_engine/performance_profiler.py`.
4. Understand how to interpret and analyze performance data using pandas DataFrames.
5. Apply critical thinking to identify performance bottlenecks and optimization opportunities.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is Performance Profiling?

### Simple Explanation

Imagine you're testing a car before a road trip:

- **How fast can it go?** → This is like measuring *throughput* (tokens per second)
- **How long does it take to start?** → This is like measuring *latency* (inference time)
- **How much fuel does it use?** → This is like measuring *memory usage*

> **Performance profiling measures how efficiently an LLM runs inference—how fast, how many tokens, and how much resources it consumes.**

A well-profiled model helps you:
- Predict costs and resource requirements
- Identify bottlenecks and optimization opportunities
- Compare models objectively
- Set realistic SLAs for production systems

### The Three Pillars of Performance Profiling

| Pillar | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Latency** | Wall-clock time for inference | User experience, response time SLAs |
| **Throughput** | Tokens processed per second | Cost efficiency, batch processing capacity |
| **Memory** | RAM/VRAM consumption | Hardware requirements, concurrent users |

---

# 🧠 Section 2 — Key Performance Metrics

### Wall-Clock Latency

Latency is the time from sending a prompt to receiving the response:

```
Latency = End Time - Start Time
```

Measured in milliseconds (ms). Lower is better.

**Factors affecting latency:**
- Model size and complexity
- Input prompt length
- Hardware (CPU vs GPU)
- Batch size
- Memory bandwidth

### Tokens Per Second

Throughput measures how many tokens the model processes per unit time:

```
Tokens/sec = (Input Tokens + Output Tokens) / Inference Time
```

Higher is better. This metric helps compare:
- Different models
- Different hardware configurations
- Optimization techniques

### Memory Usage

Memory profiling tracks RAM or VRAM consumption during inference:

```
Memory Usage = Peak Memory - Baseline Memory
```

Measured in megabytes (MB). Lower is better for:
- Running larger batches
- Supporting more concurrent users
- Reducing hardware costs

### Why Each Matters

| Metric | Production Scenario | Business Impact |
|--------|---------------------|-----------------|
| **Latency** | Chat applications, real-time responses | User satisfaction, engagement |
| **Throughput** | Batch processing, document analysis | Cost per token, processing time |
| **Memory** | Cloud deployment, concurrent users | Instance sizing, infrastructure cost |

---

# 🧪 Section 3 — The Performance Profiler Module

### Module Design

The `performance_profiler.py` module provides two main components:

```python
class PerformanceProfiler:
    """
    Wraps an ONNX Runtime session and measures performance metrics.
    
    Attributes:
        session: ONNX Runtime InferenceSession
        tokenizer: HuggingFace tokenizer for encoding/decoding
        model_path: Path to the ONNX model file
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "gpt2",
        providers: Optional[List[str]] = None,
    ) -> None:
        """Initialize with an ONNX model path."""
        
    def profile_prompt(
        self,
        prompt: str,
        warmup_runs: int = 0,
        num_runs: int = 1,
    ) -> Dict[str, Any]:
        """Profile a single prompt and return metrics."""


def profile_model(
    model_path: str,
    prompts: List[str],
    tokenizer_name: str = "gpt2",
    providers: Optional[List[str]] = None,
    warmup_runs: int = 1,
    num_runs: int = 1,
    print_summary: bool = False,
) -> pd.DataFrame:
    """
    Profile an ONNX model on a list of prompts.
    
    Returns a DataFrame with per-prompt metrics:
    - prompt, input_tokens, output_tokens
    - latency_ms, tokens_per_second
    - memory_usage_mb, inference_time_seconds
    """
```

### Key Design Decisions

1. **ONNX Runtime Integration:** Uses ONNX Runtime for cross-platform inference
2. **Warmup Runs:** Allows warming up the session before measuring
3. **Multiple Runs:** Supports averaging over multiple runs for stability
4. **Memory Tracking:** Uses psutil when available for memory measurements
5. **Summary Statistics:** Optional printing of mean, std, min, max

---

# 🧪 Section 4 — Hands-on Lab: Using Performance Profiler

### Lab Overview

In this lab, you will:
1. Import the performance profiling functions
2. Profile a model on multiple prompts
3. Analyze the results using pandas
4. Generate summary statistics
5. Identify performance patterns

### Step 1: Import the Module

```python
from src.benchmark_engine.performance_profiler import (
    PerformanceProfiler,
    profile_model,
    create_mock_profiler,
)
```

### Step 2: Profile a Model

```python
# Define test prompts
prompts = [
    "What is the capital of France?",
    "Explain machine learning in simple terms.",
    "Write a short poem about the ocean.",
    "Summarize the key principles of software engineering.",
    "What are the benefits of cloud computing?",
]

# Profile the model
df = profile_model(
    model_path="path/to/model.onnx",
    prompts=prompts,
    print_summary=True
)

print(df)
```

### Step 3: Analyze Results

```python
# Basic statistics
print(f"Average latency: {df['latency_ms'].mean():.2f} ms")
print(f"Average throughput: {df['tokens_per_second'].mean():.2f} tokens/sec")

# Find slowest prompt
slowest_idx = df['latency_ms'].idxmax()
print(f"Slowest prompt: {df.loc[slowest_idx, 'prompt']}")
print(f"Latency: {df.loc[slowest_idx, 'latency_ms']:.2f} ms")

# Correlation between input tokens and latency
correlation = df['input_tokens'].corr(df['latency_ms'])
print(f"Correlation (input_tokens, latency): {correlation:.3f}")
```

### Step 4: Using Mock Profiler for Testing

```python
# Create a mock profiler for testing without a real model
def mock_model(prompt: str) -> str:
    return f"Response to: {prompt}"

profiler_fn = create_mock_profiler(mock_model)

# Profile a single prompt
result = profiler_fn("Hello world")
print(f"Latency: {result['latency_ms']:.2f} ms")
print(f"Tokens/sec: {result['tokens_per_second']:.2f}")
```

---

# 🤔 Section 5 — Paul-Elder Critical Thinking Questions

### Question 1: EVIDENCE
**If a model shows 50ms latency on short prompts but 500ms on long prompts, what might explain this 10x difference?**

*Consider: Tokenization time, attention complexity, memory bandwidth, sequence length scaling.*

### Question 2: ASSUMPTIONS
**What assumptions are we making when we measure latency on a single machine?**

*Consider: Network latency in production, cold start effects, concurrent requests, hardware variability.*

### Question 3: IMPLICATIONS
**If we optimize for tokens per second, what might we sacrifice?**

*Consider: Model quality, accuracy, coherence, generation diversity, safety.*

### Question 4: POINT OF VIEW
**How might different stakeholders interpret the same performance metrics?**

*Consider: Engineers (optimization), Product (UX), Finance (cost), Operations (capacity planning).*

---

# 🔄 Section 6 — Inversion Thinking: How Can Performance Profiling Fail?

Instead of asking "How do performance metrics help us?", let's invert:

> **"How can performance profiling give us false confidence?"**

### Potential Failure Modes

1. **Unrealistic Test Conditions**
   - Profiling on powerful hardware not used in production
   - Testing with short prompts when production uses long ones
   - No concurrent request simulation

2. **Cold Start Ignorance**
   - First request after model load is always slow
   - Warmup runs may not reflect production patterns
   - JIT compilation effects not measured

3. **Memory Leaks**
   - Metrics look good initially but degrade over time
   - Long-running profiling sessions needed to detect
   - Garbage collection timing affects measurements

4. **Synthetic Benchmark Bias**
   - Test prompts don't match production distribution
   - Average case measured, not worst case
   - Edge cases and outliers ignored

### Defensive Practices

- **Profile on production-like hardware:** Match CPU, memory, GPU specs
- **Use realistic prompt distributions:** Sample from production logs
- **Test at scale:** Simulate concurrent requests and load
- **Monitor over time:** Track metrics in production, not just benchmarks
- **Include warmup analysis:** Report cold start vs. warm latency separately

---

# 📝 Section 7 — Mini-Project: Performance Audit

### Task

Conduct a performance profiling evaluation and produce an audit report.

### Instructions

1. **Choose prompts:**
   - Select 10-20 representative prompts
   - Include varied lengths (short, medium, long)
   - Cover different use cases

2. **Run profiling:**
   - Use `profile_model` with `print_summary=True`
   - Perform multiple runs for statistical significance
   - Record warmup effects

3. **Analyze results:**
   - Calculate mean, std, min, max for each metric
   - Identify outliers and explain them
   - Look for correlations (input length vs. latency)

4. **Document findings:**
   - Create a performance report

### Submission Format

Create a markdown file `/examples/week09_performance_audit.md`:

```markdown
# Week 9 Mini-Project: Performance Audit Report

## Executive Summary
[1-2 sentences on overall performance posture]

## Model Under Test
[Description of the model profiled]

## Test Configuration
- Hardware: [CPU/GPU specs]
- Warmup runs: [number]
- Measurement runs: [number]

## Performance Metrics

| Prompt | Input Tokens | Latency (ms) | Tokens/sec | Memory (MB) |
|--------|--------------|--------------|------------|-------------|
| ...    | ...          | ...          | ...        | ...         |

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Latency (ms) | ??? | ??? | ??? | ??? |
| Tokens/sec | ??? | ??? | ??? | ??? |
| Memory (MB) | ??? | ??? | ??? | ??? |

## Key Findings
[3-5 bullet points on notable observations]

## Recommendations
[2-3 actionable recommendations based on findings]

## Limitations
[What this evaluation did NOT test]
```

---

# 🔧 Section 8 — Advanced: Extending the Performance Profiler

### Adding GPU Profiling

You can extend the profiler to measure GPU utilization:

```python
# TODO: Implement GPU memory tracking
def get_gpu_memory_usage() -> Optional[float]:
    """Get current GPU memory usage in MB."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)  # Convert to MB
    except Exception:
        return None
```

### Adding Percentile Metrics

For production deployments, percentiles (P50, P95, P99) are more useful than mean:

```python
# Calculate percentiles
def calculate_percentiles(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate P50, P95, P99 for a column."""
    return {
        "p50": df[column].quantile(0.50),
        "p95": df[column].quantile(0.95),
        "p99": df[column].quantile(0.99),
    }

percentiles = calculate_percentiles(df, "latency_ms")
print(f"P50: {percentiles['p50']:.2f} ms")
print(f"P95: {percentiles['p95']:.2f} ms")
print(f"P99: {percentiles['p99']:.2f} ms")
```

### Adding Batch Profiling

For batch inference, measure throughput at different batch sizes:

```python
# TODO: Implement batch profiling
def profile_batch(
    profiler: PerformanceProfiler,
    prompts: List[str],
    batch_sizes: List[int],
) -> pd.DataFrame:
    """Profile inference at different batch sizes."""
    # Group prompts into batches
    # Measure latency and throughput for each batch size
    # Return comparison DataFrame
    pass
```

---

# ✔ Knowledge Mastery Checklist
- [ ] I can explain what performance profiling is and why it matters
- [ ] I understand the three key metrics: latency, tokens/sec, memory
- [ ] I can use `PerformanceProfiler` to profile an ONNX model
- [ ] I can use `profile_model` to generate a metrics DataFrame
- [ ] I know how to interpret summary statistics (mean, std, min, max)
- [ ] I understand potential failure modes and defensive practices
- [ ] I completed the mini-project performance audit

---

Week 9 complete.
Next: *Week 10 — Regression Tests*.
