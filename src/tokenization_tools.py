"""
Tokenization tools for BenchRight LLM evaluation.

This module provides utilities for analyzing text tokenization,
which is essential for understanding LLM performance metrics.
"""

import pandas as pd
from transformers import AutoTokenizer


# Default tokenizer model
DEFAULT_TOKENIZER = "gpt2"


def count_tokens(prompt: str, tokenizer_name: str = DEFAULT_TOKENIZER) -> int:
    """
    Count the number of tokens in a prompt.

    Args:
        prompt: The text prompt to tokenize.
        tokenizer_name: Name of the Hugging Face tokenizer to use.
                       Defaults to 'gpt2'.

    Returns:
        The number of tokens in the prompt.

    Example:
        >>> count_tokens("Hello world")
        2
        >>> count_tokens("GPT-4 is a transformer model.")
        8
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoding = tokenizer(prompt, return_tensors="np")
    return int(encoding["input_ids"].shape[1])


def analyze_prompts(
    prompts: list[str], tokenizer_name: str = DEFAULT_TOKENIZER
) -> pd.DataFrame:
    """
    Analyze a list of prompts and return tokenization statistics.

    Args:
        prompts: List of text prompts to analyze.
        tokenizer_name: Name of the Hugging Face tokenizer to use.
                       Defaults to 'gpt2'.

    Returns:
        DataFrame with columns:
            - prompt: The original prompt text
            - n_tokens: Number of tokens in the prompt

    Example:
        >>> prompts = ["Hello world", "AI is amazing"]
        >>> df = analyze_prompts(prompts)
        >>> print(df)
                  prompt  n_tokens
        0    Hello world         2
        1  AI is amazing         4
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    results = []
    for prompt in prompts:
        encoding = tokenizer(prompt, return_tensors="np")
        n_tokens = int(encoding["input_ids"].shape[1])
        results.append({"prompt": prompt, "n_tokens": n_tokens})

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Demonstration of tokenization tools
    print("=" * 60)
    print("BenchRight Tokenization Tools Demo")
    print("=" * 60)

    # Example 1: Count tokens for a single prompt
    sample_prompt = "Hello, how are you today?"
    token_count = count_tokens(sample_prompt)
    print(f"\n📝 Single prompt analysis:")
    print(f"   Prompt: \"{sample_prompt}\"")
    print(f"   Token count: {token_count}")

    # Example 2: Analyze multiple prompts
    sample_prompts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-4 uses RLHF for alignment.",
        "Supercalifragilisticexpialidocious",
        "The price is $1,234.56 USD.",
    ]

    print(f"\n📊 Multi-prompt analysis:")
    df = analyze_prompts(sample_prompts)
    print(df.to_string(index=False))

    # Summary statistics
    print(f"\n📈 Summary:")
    print(f"   Total prompts: {len(df)}")
    print(f"   Average tokens: {df['n_tokens'].mean():.1f}")
    print(f"   Min tokens: {df['n_tokens'].min()}")
    print(f"   Max tokens: {df['n_tokens'].max()}")
