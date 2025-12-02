"""
Performance profiling module for BenchRight evaluation.

This module provides tools for measuring ONNX model performance including
latency, throughput, and memory usage.

Example usage:
    >>> from benchmark_engine.performance_profiler import profile_model
    >>> df = profile_model("model.onnx", ["Hello", "World"])
    >>> print(df[['prompt', 'latency_ms', 'tokens_per_second']])
"""

from typing import List, Dict, Any, Optional, Callable
import time
import numpy as np
import pandas as pd


# Mock profiler constants for realistic synthetic metrics
BASE_LATENCY_MS = 10.0  # Base latency in milliseconds
LATENCY_PER_TOKEN_MS = 0.5  # Additional latency per token
LATENCY_NOISE_STD = 2.0  # Standard deviation of latency noise
BASE_MEMORY_MB = 50.0  # Base memory usage in megabytes
MEMORY_PER_TOKEN_MB = 0.1  # Additional memory per token
MEMORY_NOISE_STD = 5.0  # Standard deviation of memory noise


def profile_model(
    model_path: str,
    prompts: List[str],
    tokenizer_fn: Optional[Callable[[str], np.ndarray]] = None,
    warmup_runs: int = 2,
    print_summary: bool = True
) -> pd.DataFrame:
    """
    Profile an ONNX model's performance on a set of prompts.

    This function wraps an ONNX Runtime session and measures key
    performance metrics for each prompt.

    Args:
        model_path: Path to the ONNX model file.
        prompts: List of prompt strings to profile.
        tokenizer_fn: Optional tokenization function that takes a prompt
                     string and returns a numpy array of input IDs.
                     If not provided, uses a simple placeholder tokenizer.
        warmup_runs: Number of warmup inference runs before profiling.
                    Default is 2.
        print_summary: Whether to print summary statistics. Default is True.

    Returns:
        A pandas DataFrame with columns:
            - 'prompt': The original prompt text
            - 'prompt_length': Length of prompt in characters
            - 'input_tokens': Number of input tokens (estimated)
            - 'latency_ms': Wall-clock latency in milliseconds
            - 'tokens_per_second': Estimated throughput
            - 'memory_mb': Rough memory usage in MB (if available)

    Example:
        >>> df = profile_model("my_model.onnx", ["Hello", "How are you?"])
        >>> print(df.head())
    """
    try:
        import onnxruntime as ort
    except ImportError:
        # If ONNX Runtime is not available, use mock profiling
        print("Warning: onnxruntime not available. Using mock profiler.")
        return _mock_profile_model(prompts, print_summary)

    # Initialize ONNX Runtime session
    try:
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
    except Exception as e:
        print(f"Warning: Could not load ONNX model: {e}. Using mock profiler.")
        return _mock_profile_model(prompts, print_summary)

    # Use provided tokenizer or default
    tokenizer = tokenizer_fn or _default_tokenizer

    # Get model input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    # Perform warmup runs
    if len(prompts) > 0:
        warmup_input = tokenizer(prompts[0])
        input_dict = {input_names[0]: warmup_input}

        for _ in range(warmup_runs):
            try:
                session.run(output_names, input_dict)
            except Exception:
                pass  # Warmup may fail with placeholder tokenizer

    results: List[Dict[str, Any]] = []

    for prompt in prompts:
        # Tokenize
        input_ids = tokenizer(prompt)
        input_tokens = input_ids.shape[-1] if len(input_ids.shape) > 1 else len(input_ids)

        # Prepare input
        input_dict = {input_names[0]: input_ids}

        # Measure memory before (rough estimate)
        memory_before = _get_memory_usage()

        # Run inference with timing
        start_time = time.perf_counter()
        try:
            outputs = session.run(output_names, input_dict)
            success = True
        except Exception as e:
            success = False
            outputs = None

        end_time = time.perf_counter()

        # Measure memory after
        memory_after = _get_memory_usage()

        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        tokens_per_second = input_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        memory_mb = memory_after - memory_before if memory_before and memory_after else None

        results.append({
            'prompt': prompt,
            'prompt_length': len(prompt),
            'input_tokens': input_tokens,
            'latency_ms': latency_ms,
            'tokens_per_second': tokens_per_second,
            'memory_mb': memory_mb if memory_mb and memory_mb > 0 else np.nan,
            'success': success
        })

    df = pd.DataFrame(results)

    if print_summary:
        _print_summary_stats(df)

    return df


def _mock_profile_model(
    prompts: List[str],
    print_summary: bool = True
) -> pd.DataFrame:
    """
    Mock profiler for when ONNX Runtime is not available.

    Generates synthetic but realistic-looking performance metrics
    for demonstration purposes.

    Args:
        prompts: List of prompt strings.
        print_summary: Whether to print summary statistics.

    Returns:
        DataFrame with mock performance metrics.
    """
    results: List[Dict[str, Any]] = []

    for prompt in prompts:
        # Estimate tokens (rough: ~4 chars per token)
        input_tokens = max(1, len(prompt) // 4)

        # Generate realistic mock metrics using named constants
        # Latency scales with input length
        latency_ms = (
            BASE_LATENCY_MS
            + (input_tokens * LATENCY_PER_TOKEN_MS)
            + np.random.normal(0, LATENCY_NOISE_STD)
        )
        latency_ms = max(1.0, latency_ms)  # Ensure positive

        # Tokens per second (typical range: 50-500)
        tokens_per_second = input_tokens / (latency_ms / 1000)

        # Memory (rough estimate) using named constants
        memory_mb = (
            BASE_MEMORY_MB
            + (input_tokens * MEMORY_PER_TOKEN_MB)
            + np.random.normal(0, MEMORY_NOISE_STD)
        )
        memory_mb = max(10.0, memory_mb)

        results.append({
            'prompt': prompt,
            'prompt_length': len(prompt),
            'input_tokens': input_tokens,
            'latency_ms': round(latency_ms, 2),
            'tokens_per_second': round(tokens_per_second, 2),
            'memory_mb': round(memory_mb, 2),
            'success': True
        })

    df = pd.DataFrame(results)

    if print_summary:
        _print_summary_stats(df)

    return df


def _default_tokenizer(text: str) -> np.ndarray:
    """
    Simple placeholder tokenizer.

    In production, use a proper tokenizer (e.g., from transformers).

    Args:
        text: Input text to tokenize.

    Returns:
        Numpy array of token IDs (placeholder).
    """
    # Simple character-based tokenization (placeholder)
    # Assumes ~4 characters per token on average
    n_tokens = max(1, len(text) // 4)
    return np.array([[i for i in range(n_tokens)]], dtype=np.int64)


def _get_memory_usage() -> Optional[float]:
    """
    Get current process memory usage in MB.

    Returns:
        Memory usage in MB, or None if unavailable.
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None
    except Exception:
        return None


def _print_summary_stats(df: pd.DataFrame) -> None:
    """
    Print summary statistics for the profiling results.

    Args:
        df: DataFrame with profiling results.
    """
    print("\n" + "=" * 60)
    print("📊 Performance Profiling Summary")
    print("=" * 60)

    if len(df) == 0:
        print("No results to summarize.")
        return

    print(f"\n📈 Latency (ms):")
    print(f"   Mean:   {df['latency_ms'].mean():.2f}")
    print(f"   Std:    {df['latency_ms'].std():.2f}")
    print(f"   Min:    {df['latency_ms'].min():.2f}")
    print(f"   Max:    {df['latency_ms'].max():.2f}")
    print(f"   Median: {df['latency_ms'].median():.2f}")

    print(f"\n⚡ Throughput (tokens/second):")
    print(f"   Mean:   {df['tokens_per_second'].mean():.2f}")
    print(f"   Std:    {df['tokens_per_second'].std():.2f}")

    if df['memory_mb'].notna().any():
        print(f"\n💾 Memory (MB):")
        print(f"   Mean:   {df['memory_mb'].mean():.2f}")
        print(f"   Std:    {df['memory_mb'].std():.2f}")

    success_rate = df['success'].mean() * 100 if 'success' in df.columns else 100.0
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    print(f"   Total prompts: {len(df)}")


class ONNXProfiler:
    """
    A reusable profiler class for ONNX models.

    This class maintains a session and provides methods for repeated
    profiling without re-loading the model.

    Attributes:
        session: The ONNX Runtime InferenceSession.
        tokenizer_fn: Function to tokenize prompts.

    Example:
        >>> profiler = ONNXProfiler("model.onnx")
        >>> latency = profiler.measure_latency("Hello world")
        >>> print(f"Latency: {latency:.2f} ms")
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_fn: Optional[Callable[[str], np.ndarray]] = None
    ) -> None:
        """
        Initialize the profiler with an ONNX model.

        Args:
            model_path: Path to the ONNX model file.
            tokenizer_fn: Optional tokenization function.
        """
        self.model_path = model_path
        self.tokenizer_fn = tokenizer_fn or _default_tokenizer
        self.session = None
        self._mock_mode = False

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
        except (ImportError, Exception) as e:
            print(f"Warning: Using mock profiler ({e})")
            self._mock_mode = True

    def measure_latency(self, prompt: str, n_runs: int = 5) -> float:
        """
        Measure average latency for a single prompt.

        Args:
            prompt: The prompt to profile.
            n_runs: Number of runs to average.

        Returns:
            Average latency in milliseconds.
        """
        if self._mock_mode:
            # Mock latency based on prompt length
            base = 10.0 + len(prompt) * 0.1
            return base + np.random.normal(0, 1)

        input_ids = self.tokenizer_fn(prompt)
        input_dict = {self.input_names[0]: input_ids}

        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            try:
                self.session.run(self.output_names, input_dict)
            except Exception:
                pass
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        return np.mean(latencies)

    def profile(self, prompts: List[str], print_summary: bool = True) -> pd.DataFrame:
        """
        Profile multiple prompts.

        Args:
            prompts: List of prompts to profile.
            print_summary: Whether to print summary statistics.

        Returns:
            DataFrame with profiling results.
        """
        if self._mock_mode:
            return _mock_profile_model(prompts, print_summary)
        return profile_model(
            self.model_path,
            prompts,
            self.tokenizer_fn,
            print_summary=print_summary
        )


if __name__ == "__main__":
    # Demonstration of performance profiler
    print("=" * 60)
    print("BenchRight Performance Profiler Demo")
    print("=" * 60)

    # Sample prompts for profiling
    sample_prompts = [
        "Hello",
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "How does machine learning work?",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "What is the meaning of life?",
        "Calculate the area of a circle with radius 5.",
    ]

    print("\n📝 Profiling prompts (mock mode - no ONNX model):")
    print("-" * 60)

    # Profile using mock (no actual ONNX model)
    df = profile_model("dummy_model.onnx", sample_prompts, print_summary=True)

    print("\n📋 Detailed Results:")
    print(df[['prompt', 'input_tokens', 'latency_ms', 'tokens_per_second']].to_string())

    print("\n" + "=" * 60)

    # Also demonstrate the class-based profiler
    print("\n🔧 Class-based Profiler Demo:")
    print("-" * 60)

    profiler = ONNXProfiler("dummy_model.onnx")

    # Measure single prompt latency
    test_prompt = "What is 2 + 2?"
    latency = profiler.measure_latency(test_prompt, n_runs=5)
    print(f"   Single prompt latency: {latency:.2f} ms")

    print("\n" + "=" * 60)
    print("Demo complete!")
