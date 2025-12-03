# Week 17 — System Architecture

### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 17, you will:

1. Understand the *end-to-end architecture* of a production LLM evaluation system.
2. Learn about each layer: *data ingestion, benchmark selection, model wrapper abstraction, evaluation engine, judges & safety modules, and reporting*.
3. Be able to design and implement a *complete evaluation pipeline*.
4. Understand how to *integrate new models* into the evaluation framework.
5. Apply critical thinking to architect scalable, maintainable evaluation systems.

---

# 🏗️ Section 1 — BenchRight System Architecture Overview

## End-to-End Evaluation System

The BenchRight evaluation system is designed to provide comprehensive, reproducible, and extensible LLM evaluation. The architecture follows a modular design with clear separation of concerns.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BenchRight Evaluation System                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        1. DATA INGESTION LAYER                              │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │   Datasets    │  │   Custom      │  │   API-based   │  │   Cached     │ │
│  │   (HF Hub)    │  │   JSON/CSV    │  │   Sources     │  │   Datasets   │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
│           │                 │                 │                 │          │
│           └─────────────────┴─────────────────┴─────────────────┘          │
│                                    │                                        │
│                                    ▼                                        │
│                        ┌───────────────────┐                                │
│                        │ Unified DataLoader│                                │
│                        │ Iterator[Input,Ref]│                               │
│                        └───────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      2. BENCHMARK SELECTION LAYER                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Benchmark Registry                               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │   MMLU   │ │HellaSwag │ │TruthfulQA│ │ ToxiGen  │ │  Custom  │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                     ┌──────────────────────────┐                            │
│                     │   Benchmark Config       │                            │
│                     │   - dataset_name         │                            │
│                     │   - metric_fn            │                            │
│                     │   - prompt_template      │                            │
│                     │   - num_samples          │                            │
│                     └──────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     3. MODEL WRAPPER ABSTRACTION                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     ModelWrapper Protocol                            │   │
│  │         def generate(prompt: str) -> str                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│       ┌────────────────────────────┼────────────────────────────┐          │
│       ▼                            ▼                            ▼          │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐        │
│  │ ONNXWrapper  │         │  APIWrapper  │         │ TransformersW│        │
│  │              │         │  (OpenAI/    │         │ (HuggingFace)│        │
│  │ - session    │         │   Claude)    │         │ - model      │        │
│  │ - tokenizer  │         │ - client     │         │ - tokenizer  │        │
│  └──────────────┘         └──────────────┘         └──────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4. EVALUATION ENGINE                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     run_benchmark()                                  │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  for input, reference in dataset:                              │ │   │
│  │  │      output = model.generate(input)                            │ │   │
│  │  │      score = metric_fn(output, reference)                      │ │   │
│  │  │      results.append({input, output, reference, score})         │ │   │
│  │  │  return aggregate_metrics(results)                             │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │ exact_match    │  │ contains_match │  │ custom_metric  │                │
│  │ metric         │  │ metric         │  │ functions      │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    5. JUDGES & SAFETY MODULES                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       LLM-as-Judge                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  Correctness │  │  Coherence   │  │  Helpfulness │               │   │
│  │  │    Judge     │  │    Judge     │  │    Judge     │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Safety Evaluators                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ TruthfulQA   │  │   ToxiGen    │  │  Robustness  │               │   │
│  │  │  Evaluator   │  │  Evaluator   │  │   Sweep      │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Performance Profiler                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Latency    │  │  Throughput  │  │   Memory     │               │   │
│  │  │  Profiler    │  │  Profiler    │  │  Profiler    │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   6. REPORTING & VISUALIZATION                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Report Generation                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  CSV Export  │  │  Markdown    │  │   Model      │               │   │
│  │  │              │  │  Reports     │  │   Cards      │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Regression Analysis                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ compare_runs │  │  summarize   │  │  generate    │               │   │
│  │  │              │  │ _regressions │  │ _report      │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                          ┌──────────────┐                                   │
│                          │   results/   │                                   │
│                          │  - *.csv     │                                   │
│                          │  - *.md      │                                   │
│                          │  - *.html    │                                   │
│                          └──────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 🔄 Section 2 — Step-by-Step Flow: Evaluating a New Model

This section describes the complete flow for evaluating a new model using BenchRight.

## Step 1: Prepare the Model

Create a model wrapper that implements the `generate(prompt: str) -> str` interface.

```python
from typing import Callable

# Option A: ONNX Model
def create_onnx_model(model_path: str) -> Callable[[str], str]:
    """Load an ONNX model and return a generate function."""
    import onnxruntime as ort
    from transformers import AutoTokenizer
    
    session = ort.InferenceSession(model_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="np")
        outputs = session.run(None, {"input_ids": inputs["input_ids"]})
        # Decode and return the generated text
        return tokenizer.decode(outputs[0][0], skip_special_tokens=True)
    
    return generate

# Option B: API-based Model
def create_api_model(api_key: str, model_name: str = "gpt-4o-mini") -> Callable[[str], str]:
    """Create an API-based model wrapper."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    def generate(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content
    
    return generate
```

## Step 2: Select Benchmarks

Choose which benchmarks to run from the registry.

```python
# Available benchmarks in BenchRight
BENCHMARK_REGISTRY = {
    "accuracy": {
        "description": "Basic accuracy on QA datasets",
        "function": "run_benchmark",
        "metrics": ["exact_match", "contains"],
    },
    "truthfulqa": {
        "description": "TruthfulQA for hallucination detection",
        "function": "run_truthfulqa_eval",
        "metrics": ["truthful_ratio"],
    },
    "toxigen": {
        "description": "ToxiGen for toxicity detection",
        "function": "run_toxigen_eval",
        "metrics": ["non_toxic_ratio"],
    },
    "robustness": {
        "description": "Robustness sweep with perturbations",
        "function": "robustness_sweep",
        "metrics": ["stability_score"],
    },
    "performance": {
        "description": "Performance profiling",
        "function": "profile_model",
        "metrics": ["latency_ms", "tokens_per_second"],
    },
}

# Select benchmarks to run
selected_benchmarks = ["accuracy", "truthfulqa", "toxigen", "performance"]
```

## Step 3: Configure Evaluation

Set up the evaluation configuration.

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EvalConfig:
    """Configuration for evaluation run."""
    model_path: str
    benchmarks: List[str]
    output_dir: str = "results"
    num_samples: int = 100
    seed: Optional[int] = 42

config = EvalConfig(
    model_path="models/tinyGPT.onnx",
    benchmarks=["accuracy", "truthfulqa", "toxigen"],
    output_dir="results",
    num_samples=100,
)
```

## Step 4: Run Evaluations

Execute the evaluation pipeline.

```python
from src.benchmark_engine import (
    run_benchmark,
    exact_match_metric,
    run_truthfulqa_eval,
    run_toxigen_eval,
    robustness_sweep,
    create_mock_profiler,
)
import pandas as pd
from tqdm import tqdm

def run_all_evaluations(model_fn, config):
    """Run all selected benchmarks on the model."""
    all_results = {}
    
    for benchmark_name in tqdm(config.benchmarks, desc="Running benchmarks"):
        print(f"\n📊 Running {benchmark_name}...")
        
        if benchmark_name == "accuracy":
            # Prepare QA dataset
            qa_dataset = [
                ("What is the capital of France?", "Paris"),
                ("What is 2+2?", "4"),
            ]
            results = run_benchmark(
                model_fn=model_fn,
                dataset=iter(qa_dataset),
                metric_fn=exact_match_metric,
            )
            all_results["accuracy"] = {
                "mean_score": results["mean_score"],
                "total_examples": results["total_examples"],
            }
            
        elif benchmark_name == "truthfulqa":
            results = run_truthfulqa_eval(
                model_fn=model_fn,
                n=config.num_samples,
                seed=config.seed,
            )
            all_results["truthfulqa"] = {
                "truthful_ratio": results["truthful_ratio"],
                "total_examples": results["total_examples"],
            }
            
        elif benchmark_name == "toxigen":
            results = run_toxigen_eval(
                model_fn=model_fn,
                n=config.num_samples,
                seed=config.seed,
            )
            all_results["toxigen"] = {
                "non_toxic_ratio": results["non_toxic_ratio"],
                "total_examples": results["total_examples"],
            }
            
        elif benchmark_name == "robustness":
            results = robustness_sweep(
                model_fn=model_fn,
                prompt="What is the capital of France?",
                n=20,
                seed=config.seed,
            )
            all_results["robustness"] = {
                "stability_score": results["stability_score"],
                "total_variants": results["total_variants"],
            }
    
    return all_results
```

## Step 5: Generate Reports

Create output reports from evaluation results.

```python
import os
from datetime import datetime

def generate_reports(results, config):
    """Generate evaluation reports."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(config.model_path).replace(".onnx", "")
    
    # Generate CSV report
    csv_path = os.path.join(
        config.output_dir,
        f"{model_name}_eval_{timestamp}.csv"
    )
    
    rows = []
    for benchmark, metrics in results.items():
        for metric_name, value in metrics.items():
            rows.append({
                "benchmark": benchmark,
                "metric": metric_name,
                "value": value,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV report saved to: {csv_path}")
    
    # Generate Markdown summary
    md_path = os.path.join(
        config.output_dir,
        f"{model_name}_eval_{timestamp}.md"
    )
    
    with open(md_path, "w") as f:
        f.write(f"# Evaluation Report: {model_name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Results\n\n")
        f.write("| Benchmark | Metric | Value |\n")
        f.write("|-----------|--------|-------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['benchmark']} | {row['metric']} | {row['value']:.4f} |\n")
    
    print(f"✅ Markdown report saved to: {md_path}")
    
    return csv_path, md_path
```

## Step 6: Compare with Baseline (Optional)

If comparing with a previous model version, use the regression analysis.

```python
from src.benchmark_engine import compare_runs, generate_regression_report

def compare_with_baseline(new_results_df, baseline_path):
    """Compare new results with baseline and detect regressions."""
    baseline_df = pd.read_csv(baseline_path)
    
    report = generate_regression_report(
        run_a_df=baseline_df,
        run_b_df=new_results_df,
        on="benchmark",
        metrics=["value"],
        threshold=0.01,  # 1% threshold for regression
    )
    
    print("\n📊 Regression Analysis:")
    print(f"   Total cases compared: {report['total_cases']}")
    
    if report['regressions']['value'].empty:
        print("   ✅ No regressions detected!")
    else:
        print(f"   ⚠️ Found {len(report['regressions']['value'])} regressions")
        for _, row in report['regressions']['value'].iterrows():
            print(f"      - {row['benchmark']}: {row['regression_severity']:.2%} regression")
    
    return report
```

---

# 🧩 Section 3 — Component Deep Dive

## 3.1 Data Ingestion Layer

The data ingestion layer provides a unified interface for loading evaluation datasets.

### Supported Sources

| Source | Description | Example |
|--------|-------------|---------|
| **HuggingFace Hub** | Standard benchmark datasets | `load_dataset("truthfulqa/truthful_qa")` |
| **Custom JSON/CSV** | User-provided datasets | `load_custom_dataset("my_data.json")` |
| **API Sources** | Dynamic data from APIs | `fetch_from_api(endpoint)` |
| **Cached Datasets** | Local cached copies | `load_cached("mmlu_cache/")` |

### DataLoader Interface

```python
from typing import Iterator, Tuple

def create_dataloader(
    source: str,
    config: dict,
) -> Iterator[Tuple[str, str]]:
    """
    Create a unified data loader that yields (input, reference) tuples.
    
    Args:
        source: Data source type ("hf", "json", "csv", "api")
        config: Configuration dict with source-specific options
    
    Yields:
        Tuples of (input_text, reference_answer)
    """
    pass
```

## 3.2 Benchmark Selection Layer

The benchmark selection layer manages the registry of available benchmarks and their configurations.

### Benchmark Configuration

```python
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""
    name: str
    dataset_name: str
    metric_fn: Callable[[str, str], float]
    prompt_template: str
    num_samples: int = 100
    subset: Optional[str] = None
    description: str = ""

# Example configurations
MMLU_CONFIG = BenchmarkConfig(
    name="mmlu",
    dataset_name="cais/mmlu",
    metric_fn=exact_match_metric,
    prompt_template="{question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:",
    num_samples=500,
    subset="all",
    description="Massive Multitask Language Understanding",
)
```

## 3.3 Model Wrapper Abstraction

The model wrapper provides a unified interface for different model types.

### Protocol Definition

```python
from typing import Protocol

class ModelWrapper(Protocol):
    """Protocol for model wrappers."""
    
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...
    
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate text for multiple prompts."""
        ...
    
    @property
    def name(self) -> str:
        """Return the model name."""
        ...
```

### Implementations

```python
# ONNX Model Wrapper
class ONNXModelWrapper:
    def __init__(self, model_path: str, tokenizer_name: str = "gpt2"):
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._name = os.path.basename(model_path)
    
    def generate(self, prompt: str) -> str:
        # ... implementation
        pass
    
    @property
    def name(self) -> str:
        return self._name

# API Model Wrapper
class APIModelWrapper:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    
    @property
    def name(self) -> str:
        return self.model_name
```

## 3.4 Evaluation Engine

The evaluation engine orchestrates the benchmark execution.

### Core Function

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
        batch_size: Reserved for future batch processing support.

    Returns:
        Dictionary containing scores, mean_score, total_examples, and detailed results.
    """
    pass
```

## 3.5 Judges & Safety Modules

### LLM-as-Judge

The LLM-as-Judge module uses a powerful language model to evaluate outputs.

```python
class LLMJudge:
    """LLM-based evaluation judge."""
    
    def __init__(self, client, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
    
    def evaluate(
        self,
        question: str,
        answer: str,
        reference: str,
        criteria: List[str] = ["correctness", "coherence", "helpfulness"],
    ) -> Dict[str, Any]:
        """Evaluate an answer using LLM-as-Judge."""
        pass
```

### Safety Evaluators

```python
# TruthfulQA for hallucination detection
def run_truthfulqa_eval(
    model_fn: Callable[[str], str],
    n: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run TruthfulQA evaluation."""
    pass

# ToxiGen for toxicity detection
def run_toxigen_eval(
    model_fn: Callable[[str], str],
    n: int = 500,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ToxiGen evaluation."""
    pass

# Robustness testing
def robustness_sweep(
    model_fn: Callable[[str], str],
    prompt: str,
    n: int = 20,
) -> Dict[str, Any]:
    """Test model robustness to input perturbations."""
    pass
```

## 3.6 Reporting & Visualization

### Report Generation

```python
def generate_regression_report(
    run_a_df: pd.DataFrame,
    run_b_df: pd.DataFrame,
    on: str = "prompt",
    metrics: Optional[List[str]] = None,
    threshold: float = 0.0,
) -> dict:
    """
    Generate a comprehensive regression report comparing two runs.

    Returns:
        Dictionary containing comparison_df, total_cases, metrics_analyzed,
        regressions, and summary statistics.
    """
    pass
```

---

# 🛠️ Section 4 — CLI Tool: run_all_evals.py

BenchRight provides a command-line tool for running evaluations.

## Usage

```bash
# Run all benchmarks on a model
python scripts/run_all_evals.py \
    --model-path models/tinyGPT.onnx \
    --benchmarks accuracy,truthfulqa,toxigen,robustness

# Run specific benchmarks with custom sample count
python scripts/run_all_evals.py \
    --model-path models/my_model.onnx \
    --benchmarks truthfulqa,toxigen \
    --num-samples 200 \
    --output-dir my_results/

# Run with seed for reproducibility
python scripts/run_all_evals.py \
    --model-path models/my_model.onnx \
    --benchmarks all \
    --seed 42
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to ONNX model file | Required |
| `--benchmarks` | Comma-separated list of benchmarks | all |
| `--output-dir` | Directory for output files | results/ |
| `--num-samples` | Number of samples per benchmark | 100 |
| `--seed` | Random seed for reproducibility | None |

---

# 🤔 Section 5 — Paul-Elder Critical Thinking Questions

### Question 1: SCALABILITY
**How would you scale this architecture to evaluate 100 models in parallel?**

*Consider: Distributed processing, queue-based evaluation, caching strategies, and resource management.*

### Question 2: EXTENSIBILITY
**What design patterns enable easy addition of new benchmarks or metrics?**

*Consider: Plugin architecture, registry pattern, dependency injection, and interface-based design.*

### Question 3: RELIABILITY
**How do you ensure evaluation results are reproducible across runs?**

*Consider: Seeding, version control, deterministic sampling, and environment isolation.*

### Question 4: MAINTAINABILITY
**What testing strategies ensure the evaluation system itself is correct?**

*Consider: Unit tests for components, integration tests for pipelines, and golden dataset testing.*

---

# 🔄 Section 6 — Inversion Thinking: How Can the System Fail?

Instead of asking "How does this architecture work?", let's invert:

> **"How can this evaluation system produce misleading results?"**

### Failure Modes

1. **Data Contamination**
   - Evaluation data leaked into training
   - Model memorized benchmark answers
   - Mitigation: Use held-out test sets, monitor for memorization

2. **Metric Gaming**
   - Model optimized for metrics, not quality
   - High scores but poor real-world performance
   - Mitigation: Use multiple diverse metrics, include human evaluation

3. **Sampling Bias**
   - Non-representative samples selected
   - Benchmarks don't cover edge cases
   - Mitigation: Stratified sampling, diverse test sets

4. **Version Mismatch**
   - Different tokenizer or model versions
   - Incompatible evaluation conditions
   - Mitigation: Version pinning, environment reproducibility

5. **Evaluation Budget**
   - Insufficient samples for statistical significance
   - Noisy results from small sample sizes
   - Mitigation: Power analysis, confidence intervals

---

# 📝 Section 7 — Mini-Project: Build a Complete Evaluation Pipeline

### Task

Implement a complete evaluation pipeline that:
1. Loads a model (ONNX or API-based)
2. Runs multiple benchmarks (accuracy, safety, robustness)
3. Generates comprehensive reports (CSV, Markdown)
4. Compares with baseline if available
5. Outputs results to the `results/` directory

### Instructions

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create or load your model wrapper:**
   - Use the provided mock model for testing
   - Or wrap your own ONNX/API model

3. **Configure and run evaluations:**
   ```bash
   python scripts/run_all_evals.py --model-path your_model.onnx --benchmarks all
   ```

4. **Analyze the results:**
   - Review CSV files in `results/`
   - Check Markdown reports for summary
   - Compare with previous runs if available

### Submission Checklist

- [ ] Model wrapper implemented
- [ ] At least 3 benchmarks executed
- [ ] CSV results generated
- [ ] Markdown report created
- [ ] Results analyzed and documented

---

# ✔ Knowledge Mastery Checklist

Before moving to Week 18 (Capstone), ensure you can check all boxes:

- [ ] I understand the end-to-end architecture of the BenchRight evaluation system
- [ ] I can describe each layer: data ingestion, benchmark selection, model wrapper, evaluation engine, judges & safety, reporting
- [ ] I can implement a model wrapper following the protocol
- [ ] I understand how to add new benchmarks to the registry
- [ ] I can run evaluations using the CLI tool
- [ ] I can interpret evaluation reports and detect regressions
- [ ] I understand potential failure modes and mitigation strategies
- [ ] I completed the mini-project evaluation pipeline

---

Week 17 complete.
Next: *Week 18 — Capstone Project*.
