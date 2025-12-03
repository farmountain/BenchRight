"""
Performance profiling module for BenchRight.

This module provides functions for profiling ONNX Runtime model performance,
measuring wall-clock latency, tokens per second, and memory usage. It is
designed to wrap an ONNX Runtime session and provide comprehensive performance
metrics for LLM evaluation.

Example usage:
    >>> from benchmark_engine.performance_profiler import profile_model
    >>>
    >>> prompts = ["Hello world", "What is AI?", "Explain machine learning"]
    >>> df = profile_model("model.onnx", prompts)
    >>> print(df)
    >>>
    >>> # With summary stats
    >>> df = profile_model("model.onnx", prompts, print_summary=True)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd

# Try to import psutil for memory measurements
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import onnxruntime
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

# Try to import transformers for tokenization
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Configure module logger
logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """A performance profiler for ONNX Runtime inference sessions.

    This class wraps an ONNX Runtime session and provides methods for
    measuring performance metrics including latency, throughput, and
    memory usage.

    Attributes:
        session: The ONNX Runtime InferenceSession.
        tokenizer: The tokenizer used for encoding/decoding text.
        model_path: Path to the ONNX model file.

    Example:
        >>> profiler = PerformanceProfiler("model.onnx")
        >>> result = profiler.profile_prompt("Hello world")
        >>> print(f"Latency: {result['latency_ms']:.2f} ms")
        >>> print(f"Tokens/sec: {result['tokens_per_second']:.2f}")
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "gpt2",
        providers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the PerformanceProfiler with an ONNX model.

        Args:
            model_path: Path to the ONNX model file.
            tokenizer_name: Name of the Hugging Face tokenizer to use.
                           Defaults to "gpt2".
            providers: List of ONNX Runtime execution providers.
                      Defaults to ["CPUExecutionProvider"].

        Raises:
            ImportError: If onnxruntime or transformers is not installed.
            FileNotFoundError: If the model file does not exist.
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for PerformanceProfiler. "
                "Install it with: pip install onnxruntime"
            )

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for PerformanceProfiler. "
                "Install it with: pip install transformers"
            )

        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]

        # Load the ONNX model
        logger.info(f"Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers
        )

        # Load the tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(f"Model loaded. Inputs: {self.input_names}, Outputs: {self.output_names}")

    def _get_memory_usage_mb(self) -> Optional[float]:
        """Get current process memory usage in MB.

        Returns:
            Memory usage in MB, or None if psutil is not available.
        """
        if not PSUTIL_AVAILABLE:
            return None

        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB

    def profile_prompt(
        self,
        prompt: str,
        warmup_runs: int = 0,
        num_runs: int = 1,
    ) -> Dict[str, Any]:
        """Profile a single prompt and return performance metrics.

        This method tokenizes the prompt, runs inference, and measures
        various performance metrics including latency, tokens per second,
        and memory usage.

        Args:
            prompt: The text prompt to profile.
            warmup_runs: Number of warmup runs to perform before measuring.
                        Warmup runs are not included in the measurements.
                        Defaults to 0.
            num_runs: Number of inference runs to perform for averaging.
                     Defaults to 1.

        Returns:
            A dictionary containing:
                - prompt: The input prompt string
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - latency_ms: Wall-clock latency in milliseconds
                - tokens_per_second: Throughput in tokens per second
                - memory_usage_mb: Memory usage in MB (None if unavailable)
                - inference_time_seconds: Raw inference time in seconds
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        input_tokens = input_ids.shape[1]

        # Prepare input feed
        input_feed = {"input_ids": input_ids}

        # Handle attention mask if model expects it
        if "attention_mask" in self.input_names:
            if "attention_mask" in inputs:
                input_feed["attention_mask"] = inputs["attention_mask"]
            else:
                input_feed["attention_mask"] = np.ones_like(input_ids)

        # Perform warmup runs
        for _ in range(warmup_runs):
            self.session.run(None, input_feed)

        # Measure memory before inference
        memory_before = self._get_memory_usage_mb()

        # Run inference and measure time
        latencies = []
        output_tokens_count = 0

        for _ in range(num_runs):
            start_time = time.perf_counter()
            outputs = self.session.run(None, input_feed)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

            # Get output token count (from the last run)
            if outputs and len(outputs) > 0:
                output_array = outputs[0]
                # Output shape is typically (batch, seq_len, vocab_size) for LM head
                if len(output_array.shape) == 3:
                    output_tokens_count = output_array.shape[1]
                elif len(output_array.shape) == 2:
                    output_tokens_count = output_array.shape[1]
                else:
                    output_tokens_count = input_tokens  # Fallback

        # Measure memory after inference
        memory_after = self._get_memory_usage_mb()

        # Calculate average latency
        avg_latency_seconds = sum(latencies) / len(latencies)
        avg_latency_ms = avg_latency_seconds * 1000

        # Calculate tokens per second (input + output tokens)
        total_tokens = input_tokens + output_tokens_count
        tokens_per_second = total_tokens / avg_latency_seconds if avg_latency_seconds > 0 else 0.0

        # Calculate memory usage
        memory_usage_mb = None
        if memory_before is not None and memory_after is not None:
            memory_usage_mb = max(0, memory_after - memory_before) + memory_before

        return {
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens_count,
            "latency_ms": avg_latency_ms,
            "tokens_per_second": tokens_per_second,
            "memory_usage_mb": memory_usage_mb,
            "inference_time_seconds": avg_latency_seconds,
        }

    def profile_prompts(
        self,
        prompts: List[str],
        warmup_runs: int = 1,
        num_runs: int = 1,
    ) -> List[Dict[str, Any]]:
        """Profile multiple prompts and return performance metrics for each.

        Args:
            prompts: List of text prompts to profile.
            warmup_runs: Number of warmup runs before the first prompt.
                        Defaults to 1.
            num_runs: Number of inference runs per prompt for averaging.
                     Defaults to 1.

        Returns:
            List of dictionaries, each containing per-prompt metrics.
        """
        results = []

        for i, prompt in enumerate(prompts):
            # Only warmup on the first prompt
            runs_warmup = warmup_runs if i == 0 else 0

            result = self.profile_prompt(
                prompt=prompt,
                warmup_runs=runs_warmup,
                num_runs=num_runs,
            )
            results.append(result)

            logger.debug(
                f"Profiled prompt {i+1}/{len(prompts)}: "
                f"latency={result['latency_ms']:.2f}ms, "
                f"tokens/s={result['tokens_per_second']:.2f}"
            )

        return results


def profile_model(
    model_path: str,
    prompts: List[str],
    tokenizer_name: str = "gpt2",
    providers: Optional[List[str]] = None,
    warmup_runs: int = 1,
    num_runs: int = 1,
    print_summary: bool = False,
) -> pd.DataFrame:
    """Profile an ONNX model on a list of prompts and return a DataFrame.

    This is the main entry point for performance profiling. It creates a
    PerformanceProfiler, runs inference on all prompts, and returns the
    results as a pandas DataFrame.

    Args:
        model_path: Path to the ONNX model file.
        prompts: List of text prompts to profile.
        tokenizer_name: Name of the Hugging Face tokenizer to use.
                       Defaults to "gpt2".
        providers: List of ONNX Runtime execution providers.
                  Defaults to ["CPUExecutionProvider"].
        warmup_runs: Number of warmup runs before measuring.
                    Defaults to 1.
        num_runs: Number of inference runs per prompt for averaging.
                 Defaults to 1.
        print_summary: If True, prints summary statistics (mean, std).
                      Defaults to False.

    Returns:
        A pandas DataFrame with columns:
            - prompt: The input prompt string
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens
            - latency_ms: Wall-clock latency in milliseconds
            - tokens_per_second: Throughput in tokens per second
            - memory_usage_mb: Memory usage in MB (NaN if unavailable)
            - inference_time_seconds: Raw inference time in seconds

    Example:
        >>> prompts = ["Hello world", "What is AI?"]
        >>> df = profile_model("model.onnx", prompts, print_summary=True)
        >>> print(df)

        📊 Performance Summary:
        ==================================================
        Latency (ms):
          Mean: 45.23
          Std:  5.67
          Min:  38.12
          Max:  52.45
        Tokens per second:
          Mean: 156.78
          Std:  12.34
        ==================================================
    """
    # Create profiler
    profiler = PerformanceProfiler(
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        providers=providers,
    )

    # Profile all prompts
    results = profiler.profile_prompts(
        prompts=prompts,
        warmup_runs=warmup_runs,
        num_runs=num_runs,
    )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Print summary statistics if requested
    if print_summary:
        print("\n📊 Performance Summary:")
        print("=" * 50)

        # Latency stats
        print("Latency (ms):")
        print(f"  Mean: {df['latency_ms'].mean():.2f}")
        print(f"  Std:  {df['latency_ms'].std():.2f}")
        print(f"  Min:  {df['latency_ms'].min():.2f}")
        print(f"  Max:  {df['latency_ms'].max():.2f}")

        # Tokens per second stats
        print("Tokens per second:")
        print(f"  Mean: {df['tokens_per_second'].mean():.2f}")
        print(f"  Std:  {df['tokens_per_second'].std():.2f}")

        # Memory stats (if available)
        if df['memory_usage_mb'].notna().any():
            print("Memory usage (MB):")
            print(f"  Mean: {df['memory_usage_mb'].mean():.2f}")
            print(f"  Max:  {df['memory_usage_mb'].max():.2f}")

        # Total tokens processed
        total_input_tokens = df['input_tokens'].sum()
        total_output_tokens = df['output_tokens'].sum()
        total_time = df['inference_time_seconds'].sum()

        print("Total:")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Input tokens: {total_input_tokens}")
        print(f"  Output tokens: {total_output_tokens}")
        print(f"  Total time: {total_time:.2f}s")

        print("=" * 50)

    return df


def create_mock_profiler(
    model_fn: Callable[[str], str],
    tokenizer_fn: Optional[Callable[[str], int]] = None,
) -> Callable[[str], Dict[str, Any]]:
    """Create a mock profiler function for testing without ONNX models.

    This is useful for testing and demonstration when an actual ONNX
    model is not available.

    Args:
        model_fn: A callable that takes a prompt and returns generated text.
        tokenizer_fn: Optional callable that takes text and returns token count.
                     If None, uses a simple word-based estimate.

    Returns:
        A callable that takes a prompt and returns profiling metrics.

    Example:
        >>> def mock_model(prompt):
        ...     return "This is a response."
        >>>
        >>> profiler = create_mock_profiler(mock_model)
        >>> result = profiler("Hello world")
        >>> print(result['latency_ms'])
    """
    def default_tokenizer(text: str) -> int:
        """Estimate token count based on words (rough approximation)."""
        return len(text.split()) * 4 // 3 + 1  # Rough estimate

    token_counter = tokenizer_fn or default_tokenizer

    def profile_prompt(prompt: str) -> Dict[str, Any]:
        """Profile a prompt using the mock model."""
        input_tokens = token_counter(prompt)

        # Measure latency
        memory_before = None
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.perf_counter()
        output = model_fn(prompt)
        end_time = time.perf_counter()

        memory_after = None
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_after = process.memory_info().rss / (1024 * 1024)

        latency_seconds = end_time - start_time
        latency_ms = latency_seconds * 1000
        output_tokens = token_counter(output)
        total_tokens = input_tokens + output_tokens
        tokens_per_second = total_tokens / latency_seconds if latency_seconds > 0 else 0.0

        memory_usage_mb = None
        if memory_before is not None and memory_after is not None:
            memory_usage_mb = max(memory_before, memory_after)

        return {
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "tokens_per_second": tokens_per_second,
            "memory_usage_mb": memory_usage_mb,
            "inference_time_seconds": latency_seconds,
        }

    return profile_prompt


if __name__ == "__main__":
    # Demonstration with mock model
    print("=" * 60)
    print("BenchRight Performance Profiler Demo")
    print("=" * 60)

    print("\n📝 Note: This demo uses a mock model.")
    print("   For real profiling, provide an ONNX model path.")

    # Create a simple mock model
    def mock_model(prompt: str) -> str:
        """A mock model that simulates inference delay."""
        # Simulate some processing time
        time.sleep(0.01)  # 10ms delay
        return f"Response to: {prompt[:20]}..."

    # Create mock profiler
    profiler_fn = create_mock_profiler(mock_model)

    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "Write a short poem about the ocean.",
        "Summarize the key principles of software engineering.",
        "What are the benefits of cloud computing?",
    ]

    print("\n📊 Profiling Results:")
    print("-" * 60)

    results = []
    for prompt in test_prompts:
        result = profiler_fn(prompt)
        results.append(result)
        print(f"\nPrompt: {prompt[:40]}...")
        print(f"  Input tokens:  {result['input_tokens']}")
        print(f"  Output tokens: {result['output_tokens']}")
        print(f"  Latency:       {result['latency_ms']:.2f} ms")
        print(f"  Tokens/sec:    {result['tokens_per_second']:.2f}")
        if result['memory_usage_mb'] is not None:
            print(f"  Memory:        {result['memory_usage_mb']:.2f} MB")

    # Create DataFrame
    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("📊 Summary Statistics:")
    print("=" * 60)
    print(f"\nLatency (ms):")
    print(f"  Mean: {df['latency_ms'].mean():.2f}")
    print(f"  Std:  {df['latency_ms'].std():.2f}")
    print(f"  Min:  {df['latency_ms'].min():.2f}")
    print(f"  Max:  {df['latency_ms'].max():.2f}")

    print(f"\nTokens per second:")
    print(f"  Mean: {df['tokens_per_second'].mean():.2f}")
    print(f"  Std:  {df['tokens_per_second'].std():.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
