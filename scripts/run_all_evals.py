#!/usr/bin/env python3
"""
run_all_evals.py - Run all BenchRight evaluations on a model.

This script provides a command-line interface for running comprehensive
evaluations on an LLM model using the BenchRight benchmark engine.

Usage:
    python scripts/run_all_evals.py --model-path models/tinyGPT.onnx --benchmarks accuracy,truthfulqa,toxigen

    python scripts/run_all_evals.py --model-path models/my_model.onnx --benchmarks all --num-samples 200

Example:
    # Run all benchmarks
    python scripts/run_all_evals.py --model-path models/tinyGPT.onnx --benchmarks all

    # Run specific benchmarks
    python scripts/run_all_evals.py --model-path models/tinyGPT.onnx --benchmarks accuracy,truthfulqa

    # Run with custom output directory
    python scripts/run_all_evals.py --model-path models/tinyGPT.onnx --benchmarks all --output-dir my_results
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(f"Processing: {desc}")
        return iterable


# Try to import benchmark engine components
try:
    from src.benchmark_engine import (
        run_benchmark,
        exact_match_metric,
        contains_metric,
        run_truthfulqa_eval,
        run_toxigen_eval,
        robustness_sweep,
        create_mock_profiler,
    )
    BENCHRIGHT_AVAILABLE = True
except ImportError:
    BENCHRIGHT_AVAILABLE = False
    print("Warning: BenchRight components not fully available. Using mock implementations.")


# ==============================================================================
# Constants
# ==============================================================================

AVAILABLE_BENCHMARKS = [
    "accuracy",
    "truthfulqa",
    "toxigen",
    "robustness",
    "performance",
]

BENCHMARK_DESCRIPTIONS = {
    "accuracy": "Basic accuracy on QA datasets using exact match",
    "truthfulqa": "TruthfulQA evaluation for hallucination detection",
    "toxigen": "ToxiGen evaluation for toxicity detection",
    "robustness": "Robustness sweep with input perturbations",
    "performance": "Performance profiling (latency, throughput)",
}


# ==============================================================================
# Mock Model Wrapper (for testing without ONNX)
# ==============================================================================

class MockModelWrapper:
    """Mock model wrapper for testing."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.name = os.path.basename(model_path)

        # Predefined answers for demo
        self.answers = {
            "capital of france": "Paris",
            "2+2": "4",
            "largest planet": "Jupiter",
            "formula for water": "H2O",
            "speed of light": "299792458 m/s",
            "color of sky": "Blue",
            "10%": "No, humans use virtually all of their brain.",
            "brain": "No, humans use all parts of their brain.",
            "goldfish": "Goldfish have memories lasting months, not seconds.",
            "teach old dog": "Yes, you can teach an old dog new tricks.",
        }

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        prompt_lower = prompt.lower()

        for key, answer in self.answers.items():
            if key in prompt_lower:
                return answer

        # Default safe response
        return "I believe in treating everyone with respect and dignity."


# ==============================================================================
# QA Dataset
# ==============================================================================

QA_DATASET = [
    ("What is the capital of France?", "Paris"),
    ("What is 2+2?", "4"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the speed of light?", "299792458 m/s"),
    ("What color is the sky on a clear day?", "Blue"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is 3+3?", "6"),
    ("What is the smallest prime number?", "2"),
    ("What planet is known as the Red Planet?", "Mars"),
]


# ==============================================================================
# Benchmark Runners
# ==============================================================================

def run_accuracy_benchmark(
    model_fn: Callable[[str], str],
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run accuracy benchmark."""
    # Use built-in dataset or limit samples
    dataset = QA_DATASET[:min(num_samples, len(QA_DATASET))]

    if BENCHRIGHT_AVAILABLE:
        results = run_benchmark(
            model_fn=model_fn,
            dataset=iter(dataset),
            metric_fn=exact_match_metric,
        )
    else:
        # Mock implementation
        scores = []
        results_list = []
        for input_text, reference in tqdm(dataset, desc="Accuracy"):
            output = model_fn(input_text)
            score = 1.0 if output.strip().lower() == reference.strip().lower() else 0.0
            scores.append(score)
            results_list.append({
                "input": input_text,
                "reference": reference,
                "output": output,
                "score": score,
            })
        results = {
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "total_examples": len(scores),
            "scores": scores,
            "results": results_list,
        }

    return {
        "benchmark": "accuracy",
        "mean_score": results["mean_score"],
        "total_examples": results["total_examples"],
    }


def run_safety_truthfulqa(
    model_fn: Callable[[str], str],
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run TruthfulQA evaluation."""
    if BENCHRIGHT_AVAILABLE:
        results = run_truthfulqa_eval(
            model_fn=model_fn,
            n=num_samples,
            seed=seed,
        )
    else:
        # Mock implementation with placeholder data
        truthful_count = int(num_samples * 0.8)  # Mock 80% truthful
        results = {
            "truthful_ratio": truthful_count / num_samples,
            "truthful_count": truthful_count,
            "untruthful_count": num_samples - truthful_count,
            "total_examples": num_samples,
        }

    return {
        "benchmark": "truthfulqa",
        "truthful_ratio": results["truthful_ratio"],
        "truthful_count": results.get("truthful_count", 0),
        "untruthful_count": results.get("untruthful_count", 0),
        "total_examples": results["total_examples"],
    }


def run_safety_toxigen(
    model_fn: Callable[[str], str],
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ToxiGen evaluation."""
    if BENCHRIGHT_AVAILABLE:
        results = run_toxigen_eval(
            model_fn=model_fn,
            n=num_samples,
            seed=seed,
        )
    else:
        # Mock implementation
        non_toxic_count = int(num_samples * 0.95)  # Mock 95% non-toxic
        results = {
            "non_toxic_ratio": non_toxic_count / num_samples,
            "non_toxic_count": non_toxic_count,
            "toxic_count": num_samples - non_toxic_count,
            "total_examples": num_samples,
        }

    return {
        "benchmark": "toxigen",
        "non_toxic_ratio": results["non_toxic_ratio"],
        "toxic_count": results.get("toxic_count", 0),
        "non_toxic_count": results.get("non_toxic_count", 0),
        "total_examples": results["total_examples"],
    }


def run_robustness_benchmark(
    model_fn: Callable[[str], str],
    num_samples: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run robustness sweep."""
    prompt = "What is the capital of France?"

    if BENCHRIGHT_AVAILABLE:
        results = robustness_sweep(
            model_fn=model_fn,
            prompt=prompt,
            n=num_samples,
            seed=seed,
        )
    else:
        # Mock implementation
        matching = int(num_samples * 0.9)  # Mock 90% stability
        results = {
            "stability_score": matching / num_samples,
            "matching_outputs": matching,
            "total_variants": num_samples,
        }

    return {
        "benchmark": "robustness",
        "stability_score": results["stability_score"],
        "matching_outputs": results.get("matching_outputs", 0),
        "total_variants": results.get("total_variants", num_samples),
    }


def run_performance_benchmark(
    model_fn: Callable[[str], str],
    num_samples: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run performance profiling."""
    prompts = [
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "What is 2+2?",
        "Write a short poem about the ocean.",
        "What is the speed of light?",
    ][:min(num_samples, 5)]

    latencies = []
    for prompt in tqdm(prompts, desc="Performance"):
        start = time.time()
        _ = model_fn(prompt)
        latencies.append((time.time() - start) * 1000)  # Convert to ms

    return {
        "benchmark": "performance",
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "total_prompts": len(prompts),
    }


# ==============================================================================
# Main Evaluation Runner
# ==============================================================================

def run_all_evaluations(
    model_fn: Callable[[str], str],
    benchmarks: List[str],
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run all selected benchmarks."""
    all_results = []

    benchmark_runners = {
        "accuracy": run_accuracy_benchmark,
        "truthfulqa": run_safety_truthfulqa,
        "toxigen": run_safety_toxigen,
        "robustness": run_robustness_benchmark,
        "performance": run_performance_benchmark,
    }

    for benchmark_name in tqdm(benchmarks, desc="Running benchmarks"):
        print(f"\n📊 Running {benchmark_name}...")

        if benchmark_name not in benchmark_runners:
            print(f"   ⚠️ Unknown benchmark: {benchmark_name}")
            continue

        runner = benchmark_runners[benchmark_name]
        results = runner(
            model_fn=model_fn,
            num_samples=num_samples,
            seed=seed,
        )
        all_results.append(results)

        # Print summary
        for key, value in results.items():
            if key != "benchmark":
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")

    return all_results


def save_results_to_csv(
    results: List[Dict[str, Any]],
    model_name: str,
    output_dir: str,
) -> str:
    """Save results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{model_name}_eval_{timestamp}.csv")

    # Flatten results into rows
    rows = []
    for result in results:
        benchmark = result.get("benchmark", "unknown")
        for key, value in result.items():
            if key != "benchmark":
                rows.append({
                    "benchmark": benchmark,
                    "metric": key,
                    "value": value,
                })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return csv_path


def save_summary_to_markdown(
    results: List[Dict[str, Any]],
    model_name: str,
    output_dir: str,
) -> str:
    """Save summary to Markdown file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"{model_name}_eval_{timestamp}.md")

    with open(md_path, "w") as f:
        f.write(f"# Evaluation Report: {model_name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write("| Benchmark | Metric | Value |\n")
        f.write("|-----------|--------|-------|\n")

        for result in results:
            benchmark = result.get("benchmark", "unknown")
            for key, value in result.items():
                if key != "benchmark":
                    if isinstance(value, float):
                        f.write(f"| {benchmark} | {key} | {value:.4f} |\n")
                    else:
                        f.write(f"| {benchmark} | {key} | {value} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- **accuracy**: Measures exact match on QA dataset (higher = better)\n")
        f.write("- **truthfulqa**: Measures truthfulness ratio (higher = more truthful)\n")
        f.write("- **toxigen**: Measures non-toxicity ratio (higher = less toxic)\n")
        f.write("- **robustness**: Measures output stability under perturbations (higher = more stable)\n")
        f.write("- **performance**: Measures inference latency in milliseconds (lower = faster)\n")

    return md_path


# ==============================================================================
# CLI Interface
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BenchRight evaluations on an LLM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all benchmarks:
    python scripts/run_all_evals.py --model-path models/tinyGPT.onnx --benchmarks all

  Run specific benchmarks:
    python scripts/run_all_evals.py --model-path models/tinyGPT.onnx --benchmarks accuracy,truthfulqa

  Run with custom settings:
    python scripts/run_all_evals.py --model-path models/tinyGPT.onnx --benchmarks all --num-samples 200 --output-dir my_results

Model Wrapper Requirements:
  This script currently uses a mock model for demonstration. To use a real model,
  implement a model wrapper that provides a `generate(prompt: str) -> str` method.
  
  For ONNX models:
    1. Load the model with onnxruntime.InferenceSession
    2. Use a tokenizer (e.g., from transformers) for input/output processing
    3. Implement the generate() method to tokenize, run inference, and decode
    
  See week17.md for detailed model wrapper implementation examples.
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the ONNX model file or model identifier. "
             "Note: Currently uses a mock model for demonstration.",
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
        help=f"Comma-separated list of benchmarks to run, or 'all'. "
             f"Available: {', '.join(AVAILABLE_BENCHMARKS)}",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files (default: results)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per benchmark (default: 100)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("BenchRight Evaluation Runner")
    print("=" * 60)

    # Parse benchmarks
    if args.benchmarks.lower() == "all":
        benchmarks = AVAILABLE_BENCHMARKS.copy()
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    # Validate benchmarks
    invalid_benchmarks = [b for b in benchmarks if b not in AVAILABLE_BENCHMARKS]
    if invalid_benchmarks:
        print(f"⚠️ Warning: Unknown benchmarks will be skipped: {invalid_benchmarks}")
        benchmarks = [b for b in benchmarks if b in AVAILABLE_BENCHMARKS]

    if not benchmarks:
        print("❌ Error: No valid benchmarks specified.")
        sys.exit(1)

    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Model Path:   {args.model_path}")
    print(f"   Benchmarks:   {', '.join(benchmarks)}")
    print(f"   Output Dir:   {args.output_dir}")
    print(f"   Num Samples:  {args.num_samples}")
    print(f"   Seed:         {args.seed}")

    # Create model wrapper
    print(f"\n🤖 Loading model...")

    # NOTE: This currently uses a mock model wrapper for demonstration.
    # To use a real ONNX model, replace MockModelWrapper with a class that:
    #   1. Loads the ONNX model using onnxruntime.InferenceSession
    #   2. Implements generate(prompt: str) -> str using a tokenizer
    # See week17.md for implementation examples.
    model = MockModelWrapper(args.model_path)
    model_name = model.name.replace(".onnx", "").replace(".", "_")

    print(f"   Model loaded: {model.name} (using mock model for demonstration)")

    # Run evaluations
    print(f"\n🚀 Starting evaluations...")
    start_time = time.time()

    results = run_all_evaluations(
        model_fn=model.generate,
        benchmarks=benchmarks,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    total_time = time.time() - start_time

    # Save results
    print(f"\n📝 Saving results...")

    csv_path = save_results_to_csv(results, model_name, args.output_dir)
    print(f"   CSV:      {csv_path}")

    md_path = save_summary_to_markdown(results, model_name, args.output_dir)
    print(f"   Markdown: {md_path}")

    # Print summary
    print(f"\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)

    for result in results:
        benchmark = result.get("benchmark", "unknown")
        print(f"\n{benchmark.upper()}:")
        for key, value in result.items():
            if key != "benchmark":
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")

    print(f"\n⏱️ Total time: {total_time:.2f} seconds")
    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
