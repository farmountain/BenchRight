# Week 5 — Building a Generic Benchmark Engine
### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 5, you will:

1. Understand the design principles of a generic benchmark engine
2. Create reusable evaluation loops that work with any model, dataset, and metric
3. Implement type-safe interfaces using Python type hints
4. Build modular components: model functions, dataset iterators, and metric functions
5. Measure and report performance metrics (accuracy, latency, throughput)

---

# 🧠 Section 1 — Why Build a Generic Benchmark Engine?

### The Problem with Ad-Hoc Evaluation

In Week 4, we wrote evaluation code specific to each benchmark. This approach has limitations:

| Problem | Impact |
|---------|--------|
| **Code duplication** | Same evaluation loop written multiple times |
| **Inconsistent metrics** | Different benchmarks measured differently |
| **Hard to extend** | Adding new benchmarks requires rewriting code |
| **Poor reusability** | Code tied to specific model implementations |

### The Solution: A Generic Engine

A **generic benchmark engine** provides a single, reusable evaluation loop that works with:
- **Any model** that follows a simple interface
- **Any dataset** that yields input/reference pairs
- **Any metric** that scores model outputs

This abstraction separates concerns:
- The **engine** handles iteration, timing, and aggregation
- The **model function** handles inference
- The **dataset** handles data loading
- The **metric function** handles scoring

---

# 🧠 Section 2 — Core Design Patterns

### The Function Interface Pattern

Instead of requiring specific model classes, we use **callable interfaces**:

```python
# Model function: takes prompt, returns text
model_fn: Callable[[str], str]

# Metric function: takes (output, reference), returns score
metric_fn: Callable[[str, str], float]
```

This allows any function or method that matches the signature to work:
- ONNX Runtime inference
- Hugging Face transformers
- API-based models (OpenAI, Claude)
- Mock models for testing

### The Iterator Pattern

Datasets are **iterators** that yield `(input, reference)` tuples:

```python
dataset: Iterator[Tuple[str, str]]
```

This supports:
- Lists and tuples (converted with `iter()`)
- Generators (for large datasets)
- Streaming data sources
- Custom data loaders

### Result Aggregation

The engine returns a comprehensive result dictionary:

```python
{
    'scores': List[float],           # Individual scores
    'mean_score': float,             # Average score
    'total_examples': int,           # Count of examples
    'total_time_seconds': float,     # Total wall time
    'examples_per_second': float,    # Throughput
    'results': List[Dict]            # Detailed per-example results
}
```

---

# 🧪 Section 3 — The `run_benchmark` Function

### Function Signature

```python
def run_benchmark(
    model_fn: Callable[[str], str],
    dataset: Iterator[Tuple[str, str]],
    metric_fn: Callable[[str, str], float],
    batch_size: int = 1
) -> Dict[str, Any]:
    """
    Run a generic evaluation loop on a dataset.

    Args:
        model_fn: A callable that takes a prompt string and returns generated text.
        dataset: An iterator that yields (input, reference) tuples.
        metric_fn: A callable that computes a score for a (model_output, reference) pair.
        batch_size: Reserved for future batch processing support (currently unused).

    Returns:
        A dictionary containing scores, timing, and detailed results.
    """
```

### Implementation Walkthrough

```python
def run_benchmark(model_fn, dataset, metric_fn, batch_size=1):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    scores = []
    detailed_results = []
    start_time = time.time()

    for input_text, reference in dataset:
        # Time each inference
        inference_start = time.time()
        model_output = model_fn(input_text)
        inference_time = time.time() - inference_start

        # Compute the metric score
        score = metric_fn(model_output, reference)
        scores.append(score)

        # Store detailed results
        detailed_results.append({
            'input': input_text,
            'reference': reference,
            'model_output': model_output,
            'score': score,
            'inference_time_seconds': inference_time
        })

    total_time = time.time() - start_time

    return {
        'scores': scores,
        'mean_score': sum(scores) / len(scores) if scores else 0.0,
        'total_examples': len(scores),
        'total_time_seconds': total_time,
        'examples_per_second': len(scores) / total_time if total_time > 0 else 0.0,
        'results': detailed_results
    }
```

---

# 🧪 Section 4 — Built-in Metric Functions

### Exact Match Metric

```python
def exact_match_metric(output: str, reference: str) -> float:
    """
    Compute exact match score between model output and reference.
    
    Returns 1.0 if output matches reference (case-insensitive, stripped), 
    0.0 otherwise.
    """
    return 1.0 if output.strip().lower() == reference.strip().lower() else 0.0
```

**Use case:** Question answering, classification, short-form generation

### Contains Metric

```python
def contains_metric(output: str, reference: str) -> float:
    """
    Check if reference is contained in model output.
    
    Returns 1.0 if reference is found in output (case-insensitive), 
    0.0 otherwise.
    """
    return 1.0 if reference.strip().lower() in output.strip().lower() else 0.0
```

**Use case:** Long-form generation where the answer is embedded in text

---

# 🧪 Section 5 — Hands-on Lab: Using the Benchmark Engine

### Lab Overview

In this lab, you will:
1. Import the benchmark engine from the `src/benchmark_engine` module
2. Create a mock model function
3. Define a synthetic QA dataset
4. Run the benchmark and analyze results

### Step 1: Import the Engine

```python
from src.benchmark_engine.engine import run_benchmark, exact_match_metric, contains_metric
```

### Step 2: Create a Model Function

```python
def mock_model(prompt: str) -> str:
    """A simple mock model that returns predefined answers."""
    answers = {
        "capital of france": "Paris",
        "2+2": "4",
        "largest planet": "Jupiter",
        "formula for water": "H2O",
        "speed of light": "299792458 m/s",
    }
    prompt_lower = prompt.lower()
    for key, answer in answers.items():
        if key in prompt_lower:
            return answer
    return "I don't know"
```

### Step 3: Define the Dataset

```python
synthetic_dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2+2?", "4"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the speed of light?", "299792458 m/s"),
]
```

### Step 4: Run the Benchmark

```python
results = run_benchmark(
    model_fn=mock_model,
    dataset=iter(synthetic_dataset),
    metric_fn=exact_match_metric,
    batch_size=1
)

print(f"Total examples: {results['total_examples']}")
print(f"Mean score (accuracy): {results['mean_score']:.2%}")
print(f"Total time: {results['total_time_seconds']:.4f} seconds")
print(f"Throughput: {results['examples_per_second']:.2f} examples/second")
```

---

# 🤔 Section 6 — Paul-Elder Critical Thinking Questions

### Question 1: POINT OF VIEW
**What are the trade-offs between a highly generic engine vs. a specialized one?**

*Consider: Flexibility vs. optimization, ease of use vs. features, abstraction overhead.*

### Question 2: IMPLICATIONS
**If the metric function is slow (e.g., LLM-as-judge), how does that affect the benchmark results?**

*Consider: Timing accuracy, throughput measurement, separating inference time from metric time.*

### Question 3: ASSUMPTIONS
**What assumptions does the `run_benchmark` function make about the model and dataset?**

*Consider: Determinism, error handling, memory management, dataset size.*

---

# 🔄 Section 7 — Inversion Thinking: What Could Go Wrong?

Instead of asking "How does the engine help us?", let's invert:

> **"What could go wrong when using a generic benchmark engine?"**

### Potential Failure Modes

1. **Model Errors**
   - Model function raises an exception
   - Model returns None or empty string
   - Model hangs or times out

2. **Dataset Issues**
   - Empty dataset yields 0 examples
   - Malformed (input, reference) tuples
   - Dataset too large to fit in memory

3. **Metric Problems**
   - Metric function raises exceptions
   - Metric returns non-numeric values
   - Metric doesn't match evaluation goals

4. **Performance Issues**
   - Sequential processing is slow for large datasets
   - No checkpointing for long runs
   - Memory accumulation in detailed_results

### Defensive Practices

- Add try/except around model calls
- Validate dataset format before running
- Add progress logging for long runs
- Implement streaming result output
- Add timeout support for model calls

---

# 📝 Section 8 — Mini-Project: Extend the Benchmark Engine

### Task

Create a custom metric function and use it with the benchmark engine.

### Option A: Partial Match Metric

Implement a metric that gives partial credit for partially correct answers:

```python
def partial_match_metric(output: str, reference: str) -> float:
    """
    Compute partial match score using word overlap.
    
    Returns the proportion of reference words found in output.
    """
    output_words = set(output.strip().lower().split())
    reference_words = set(reference.strip().lower().split())
    
    if not reference_words:
        return 1.0
    
    overlap = output_words & reference_words
    return len(overlap) / len(reference_words)
```

### Option B: Length-Penalized Metric

Implement a metric that penalizes overly verbose responses:

```python
def length_penalized_metric(output: str, reference: str, max_ratio: float = 3.0) -> float:
    """
    Exact match with penalty for length.
    
    Returns 1.0 for exact match, reduced if output is much longer than reference.
    """
    if output.strip().lower() != reference.strip().lower():
        return 0.0
    
    length_ratio = len(output) / max(len(reference), 1)
    if length_ratio > max_ratio:
        return 1.0 / length_ratio
    return 1.0
```

### Submission

1. Implement your custom metric
2. Run the benchmark engine with your metric
3. Compare results with `exact_match_metric`
4. Write a short analysis of when your metric is useful

---

# ✔ Knowledge Mastery Checklist
- [ ] I understand why a generic benchmark engine is valuable
- [ ] I can explain the function interface pattern for models and metrics
- [ ] I can use `run_benchmark` with custom model functions
- [ ] I can create synthetic datasets for testing
- [ ] I understand the result dictionary structure
- [ ] I can implement custom metric functions
- [ ] I can analyze benchmark results and identify issues

---

Week 5 complete.
Next: *Week 6 — Automated Evaluation Pipelines (LLM-as-Judge)*.
