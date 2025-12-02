"""
Generic benchmark engine for LLM evaluation.

This module provides a reusable evaluation loop for running benchmarks
on any model that follows a simple interface.

Example usage:
    >>> from benchmark_engine.engine import run_benchmark
    >>> results = run_benchmark(
    ...     model_fn=my_model,
    ...     dataset=my_dataset,
    ...     metric_fn=accuracy_metric,
    ...     batch_size=1
    ... )
"""

from typing import Callable, Iterator, Tuple, Any, Dict, List
import time


def run_benchmark(
    model_fn: Callable[[str], str],
    dataset: Iterator[Tuple[str, str]],
    metric_fn: Callable[[str, str], float],
    batch_size: int = 1
) -> Dict[str, Any]:
    """
    Run a generic evaluation loop on a dataset.

    This function provides a standardized way to evaluate any model on any
    dataset using any metric. It handles the evaluation loop, timing, and
    result aggregation.

    Args:
        model_fn: A callable that takes a prompt string and returns generated text.
                  Signature: (str) -> str
        dataset: An iterator that yields (input, reference) tuples.
                 - input: The prompt or question to send to the model
                 - reference: The expected/correct answer for scoring
        metric_fn: A callable that computes a score for a (model_output, reference) pair.
                   Signature: (str, str) -> float
                   Higher scores should indicate better performance.
        batch_size: Number of examples to process before yielding results.
                    Currently processes one at a time (batch_size=1 is recommended).

    Returns:
        A dictionary containing:
            - 'scores': List of individual scores for each example
            - 'mean_score': Average score across all examples
            - 'total_examples': Number of examples evaluated
            - 'total_time_seconds': Total evaluation time
            - 'examples_per_second': Throughput metric
            - 'results': List of detailed results for each example

    Raises:
        ValueError: If batch_size is less than 1
        StopIteration: If dataset is empty

    Example:
        >>> def simple_model(prompt):
        ...     return "Paris"
        >>>
        >>> dataset = [
        ...     ("What is the capital of France?", "Paris"),
        ...     ("What is 2+2?", "4"),
        ... ]
        >>>
        >>> def exact_match(output, reference):
        ...     return 1.0 if output.strip().lower() == reference.strip().lower() else 0.0
        >>>
        >>> results = run_benchmark(simple_model, iter(dataset), exact_match)
        >>> print(f"Accuracy: {results['mean_score']:.2%}")
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    scores: List[float] = []
    detailed_results: List[Dict[str, Any]] = []
    start_time = time.time()

    example_count = 0
    for input_text, reference in dataset:
        # Run model inference
        inference_start = time.time()
        model_output = model_fn(input_text)
        inference_time = time.time() - inference_start

        # Compute metric
        score = metric_fn(model_output, reference)
        scores.append(score)

        # Store detailed result
        detailed_results.append({
            'input': input_text,
            'reference': reference,
            'model_output': model_output,
            'score': score,
            'inference_time_seconds': inference_time
        })

        example_count += 1

    total_time = time.time() - start_time

    # Handle empty dataset case
    if example_count == 0:
        return {
            'scores': [],
            'mean_score': 0.0,
            'total_examples': 0,
            'total_time_seconds': total_time,
            'examples_per_second': 0.0,
            'results': []
        }

    return {
        'scores': scores,
        'mean_score': sum(scores) / len(scores),
        'total_examples': example_count,
        'total_time_seconds': total_time,
        'examples_per_second': example_count / total_time if total_time > 0 else 0.0,
        'results': detailed_results
    }


def exact_match_metric(output: str, reference: str) -> float:
    """
    Compute exact match score between model output and reference.

    Args:
        output: Model generated text
        reference: Expected/correct answer

    Returns:
        1.0 if output matches reference (case-insensitive, stripped), 0.0 otherwise
    """
    return 1.0 if output.strip().lower() == reference.strip().lower() else 0.0


def contains_metric(output: str, reference: str) -> float:
    """
    Check if reference is contained in model output.

    Args:
        output: Model generated text
        reference: Expected/correct answer

    Returns:
        1.0 if reference is found in output (case-insensitive), 0.0 otherwise
    """
    return 1.0 if reference.strip().lower() in output.strip().lower() else 0.0


if __name__ == "__main__":
    # Example demonstration with synthetic QA pairs
    print("=" * 60)
    print("BenchRight Benchmark Engine Demo")
    print("=" * 60)

    # Define a simple mock model
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

    # Create synthetic QA pairs
    synthetic_dataset = [
        ("What is the capital of France?", "Paris"),
        ("What is 2+2?", "4"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("What is the chemical formula for water?", "H2O"),
        ("What is the speed of light?", "299792458 m/s"),
    ]

    print("\n📝 Running benchmark on 5 synthetic QA pairs...")
    print("-" * 60)

    # Run the benchmark
    results = run_benchmark(
        model_fn=mock_model,
        dataset=iter(synthetic_dataset),
        metric_fn=exact_match_metric,
        batch_size=1
    )

    # Display results
    print("\n📊 Results:")
    print(f"   Total examples: {results['total_examples']}")
    print(f"   Mean score (accuracy): {results['mean_score']:.2%}")
    print(f"   Total time: {results['total_time_seconds']:.4f} seconds")
    print(f"   Throughput: {results['examples_per_second']:.2f} examples/second")

    print("\n📋 Detailed Results:")
    print("-" * 60)
    for i, result in enumerate(results['results']):
        status = "✓" if result['score'] == 1.0 else "✗"
        print(f"[{status}] Q{i+1}: {result['input'][:40]}...")
        print(f"     Expected: {result['reference']}")
        print(f"     Got: {result['model_output']}")
        print()

    print("=" * 60)
    print("Demo complete!")
