"""
Robustness testing module for BenchRight evaluation.

This module provides tools for testing model robustness by generating
perturbed variants of prompts and measuring response stability.

The key insight is that a robust model should produce semantically
similar outputs for minor variations of the same prompt.

Example usage:
    >>> from benchmark_engine.robustness import perturb_prompt, robustness_sweep
    >>> perturbed = perturb_prompt("What is the capital of France?", mode="typo")
    >>> results = robustness_sweep(my_model, "What is the capital of France?", n=20)
    >>> print(f"Stability score: {results['stability_score']:.2%}")
"""

from typing import Callable, Dict, Any, List, Optional
import random
import string


# Configuration constants for perturbation parameters
TYPO_PROBABILITY = 0.1  # Probability of introducing a typo per character
SYNONYM_REPLACEMENT_PROB = 0.3  # Probability of replacing a word with a synonym
WORD_SWAP_PROBABILITY = 0.2  # Probability of swapping adjacent words
CASE_CHANGE_PROBABILITY = 0.2  # Probability of changing case per character
SIMILARITY_THRESHOLD = 0.5  # Jaccard similarity threshold for semantic similarity


def perturb_prompt(prompt: str, mode: str = "typo", seed: Optional[int] = None) -> str:
    """
    Generate a perturbed variant of a prompt.

    This function applies various perturbation strategies to test model
    robustness against minor input variations.

    Args:
        prompt: The original prompt string to perturb.
        mode: The perturbation mode. Options:
              - "typo": Introduce random character-level typos
              - "synonym": Replace words with synonyms
              - "reorder": Reorder words while preserving meaning
              - "case": Change letter casing randomly
              - "whitespace": Add or modify whitespace
              - "punctuation": Add, remove, or change punctuation
              - "all": Apply a random combination of perturbations
        seed: Optional random seed for reproducibility.

    Returns:
        A perturbed version of the input prompt.

    Raises:
        ValueError: If an unknown perturbation mode is specified.

    Example:
        >>> original = "What is the capital of France?"
        >>> perturb_prompt(original, mode="typo")
        'Wht is the capital of Frnce?'
        >>> perturb_prompt(original, mode="case")
        'what IS the CAPITAL of france?'
    """
    if seed is not None:
        random.seed(seed)

    perturbation_functions = {
        "typo": _add_typos,
        "synonym": _apply_synonyms,
        "reorder": _reorder_words,
        "case": _change_case,
        "whitespace": _modify_whitespace,
        "punctuation": _modify_punctuation,
    }

    if mode == "all":
        # Apply a random subset of perturbations
        modes = list(perturbation_functions.keys())
        selected_modes = random.sample(modes, k=random.randint(1, 3))
        result = prompt
        for m in selected_modes:
            result = perturbation_functions[m](result)
        return result

    if mode not in perturbation_functions:
        raise ValueError(
            f"Unknown perturbation mode: '{mode}'. "
            f"Valid options: {list(perturbation_functions.keys()) + ['all']}"
        )

    return perturbation_functions[mode](prompt)


def robustness_sweep(
    model_fn: Callable[[str], str],
    prompt: str,
    n: int = 20,
    modes: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate perturbed variants and collect model outputs to assess stability.

    This function evaluates how consistently a model responds to minor
    variations of the same prompt. A robust model should produce
    semantically similar outputs for similar inputs.

    Args:
        model_fn: A callable that takes a prompt string and returns generated text.
                  Signature: (str) -> str
        prompt: The original prompt to test.
        n: Number of perturbed variants to generate. Default is 20.
        modes: List of perturbation modes to use. If None, uses all modes.
        seed: Optional random seed for reproducibility.

    Returns:
        A dictionary containing:
            - 'original_prompt': str - The original prompt
            - 'original_output': str - Model output for original prompt
            - 'stability_score': float - Fraction of outputs semantically similar
                                        to the original (placeholder implementation)
            - 'variants': List[Dict] - Details for each variant including:
                - 'perturbed_prompt': The perturbed prompt text
                - 'mode': The perturbation mode used
                - 'output': Model output for the perturbed prompt
                - 'is_similar': Boolean indicating semantic similarity (placeholder)
            - 'unique_outputs': int - Number of unique outputs generated
            - 'output_diversity': float - Ratio of unique outputs to total

    Example:
        >>> results = robustness_sweep(my_model, "What is 2+2?", n=10)
        >>> print(f"Stability: {results['stability_score']:.2%}")
        >>> print(f"Unique outputs: {results['unique_outputs']}")
    """
    if seed is not None:
        random.seed(seed)

    if modes is None:
        modes = ["typo", "synonym", "reorder", "case", "whitespace", "punctuation"]

    # Get original output
    original_output = model_fn(prompt)

    variants: List[Dict[str, Any]] = []
    outputs: List[str] = []

    for i in range(n):
        # Select a random perturbation mode
        mode = modes[i % len(modes)]

        # Generate perturbed prompt
        perturbed = perturb_prompt(prompt, mode=mode)

        # Get model output
        output = model_fn(perturbed)
        outputs.append(output)

        # Check semantic similarity (placeholder - uses exact match for now)
        # TODO: Replace with proper semantic similarity measure
        # Options include:
        # - Cosine similarity of embeddings
        # - BERTScore
        # - LLM-as-judge similarity
        is_similar = _is_semantically_similar(output, original_output)

        variants.append({
            'perturbed_prompt': perturbed,
            'mode': mode,
            'output': output,
            'is_similar': is_similar
        })

    # Calculate stability score
    similar_count = sum(1 for v in variants if v['is_similar'])
    stability_score = similar_count / n if n > 0 else 0.0

    # Calculate output diversity
    unique_outputs = len(set(outputs))
    output_diversity = unique_outputs / n if n > 0 else 0.0

    return {
        'original_prompt': prompt,
        'original_output': original_output,
        'stability_score': stability_score,
        'variants': variants,
        'unique_outputs': unique_outputs,
        'output_diversity': output_diversity
    }


def _add_typos(text: str, probability: float = 0.1) -> str:
    """
    Add random typos to text.

    Typo types include character deletion, insertion, substitution,
    and adjacent character swapping.

    Args:
        text: Input text.
        probability: Probability of each character being affected.

    Returns:
        Text with random typos.
    """
    result = []
    chars = list(text)

    for i, char in enumerate(chars):
        if char.isalpha() and random.random() < probability:
            typo_type = random.choice(['delete', 'insert', 'substitute', 'swap'])

            if typo_type == 'delete':
                continue  # Skip this character
            elif typo_type == 'insert':
                # Insert a random character before this one
                result.append(random.choice(string.ascii_lowercase))
                result.append(char)
            elif typo_type == 'substitute':
                # Replace with a nearby character on keyboard
                result.append(random.choice(string.ascii_lowercase))
            elif typo_type == 'swap' and i < len(chars) - 1:
                # Swap with next character (handle in next iteration)
                result.append(chars[i + 1] if i + 1 < len(chars) else char)
                chars[i + 1] = char if i + 1 < len(chars) else chars[i + 1]
            else:
                result.append(char)
        else:
            result.append(char)

    return ''.join(result)


def _apply_synonyms(text: str) -> str:
    """
    Replace words with synonyms.

    This is a simple placeholder implementation with a limited synonym map.
    In production, use a proper thesaurus or word embedding approach.

    TODO: Integrate with WordNet or similar resource for comprehensive synonyms.

    Args:
        text: Input text.

    Returns:
        Text with some words replaced by synonyms.
    """
    # Simple synonym map (placeholder) - only actual synonyms, no self-references
    synonyms = {
        'what': ['which'],
        'how': ['in what way'],
        'why': ['for what reason'],
        'big': ['large', 'huge', 'massive'],
        'small': ['tiny', 'little', 'miniature'],
        'good': ['excellent', 'great', 'fine'],
        'bad': ['poor', 'terrible', 'awful'],
        'fast': ['quick', 'rapid', 'swift'],
        'slow': ['sluggish', 'unhurried', 'leisurely'],
        'capital': ['main city'],
        'country': ['nation', 'state'],
        'city': ['town', 'metropolis'],
    }

    words = text.split()
    result = []

    for word in words:
        word_lower = word.lower().strip('.,?!')
        if word_lower in synonyms and random.random() < SYNONYM_REPLACEMENT_PROB:
            replacement = random.choice(synonyms[word_lower])
            # Preserve original casing for first letter
            if word[0].isupper():
                replacement = replacement.capitalize()
            # Preserve punctuation
            for punct in '.,?!':
                if word.endswith(punct):
                    replacement += punct
            result.append(replacement)
        else:
            result.append(word)

    return ' '.join(result)


def _reorder_words(text: str) -> str:
    """
    Reorder words while trying to preserve meaning.

    This is a simple implementation that swaps adjacent words occasionally.
    In production, use syntax-aware reordering.

    Args:
        text: Input text.

    Returns:
        Text with some word order changes.
    """
    words = text.split()
    if len(words) < 3:
        return text

    # Don't reorder the first and last words (often question words and punctuation)
    middle = words[1:-1]

    # Randomly swap adjacent pairs
    for i in range(len(middle) - 1):
        if random.random() < WORD_SWAP_PROBABILITY:
            middle[i], middle[i + 1] = middle[i + 1], middle[i]

    return ' '.join([words[0]] + middle + [words[-1]])


def _change_case(text: str) -> str:
    """
    Randomly change letter casing.

    Args:
        text: Input text.

    Returns:
        Text with random case changes.
    """
    result = []
    for char in text:
        if char.isalpha() and random.random() < CASE_CHANGE_PROBABILITY:
            result.append(char.swapcase())
        else:
            result.append(char)
    return ''.join(result)


def _modify_whitespace(text: str) -> str:
    """
    Add or modify whitespace randomly.

    Args:
        text: Input text.

    Returns:
        Text with whitespace modifications.
    """
    # Randomly add extra spaces
    words = text.split()
    result = []

    for i, word in enumerate(words):
        result.append(word)
        if i < len(words) - 1:
            # Add 1-3 spaces between words
            spaces = ' ' * random.choice([1, 1, 1, 2, 2, 3])
            result.append(spaces)

    return ''.join(result).strip()


def _modify_punctuation(text: str) -> str:
    """
    Add, remove, or change punctuation randomly.

    Args:
        text: Input text.

    Returns:
        Text with punctuation modifications.
    """
    result = []

    for char in text:
        if char in '.,?!;:':
            action = random.choice(['keep', 'keep', 'keep', 'remove', 'double'])
            if action == 'keep':
                result.append(char)
            elif action == 'double':
                result.append(char * 2)
            # 'remove' does nothing
        else:
            result.append(char)

    return ''.join(result)


def _is_semantically_similar(output1: str, output2: str) -> bool:
    """
    Check if two outputs are semantically similar.

    This is a placeholder implementation using simple heuristics.
    In production, replace with proper semantic similarity measures.

    TODO: Implement proper semantic similarity using:
          - Sentence embeddings (e.g., sentence-transformers)
          - BERTScore
          - LLM-as-judge comparison

    Args:
        output1: First output string.
        output2: Second output string.

    Returns:
        True if outputs are considered semantically similar.
    """
    # Normalize outputs
    norm1 = output1.lower().strip()
    norm2 = output2.lower().strip()

    # Exact match
    if norm1 == norm2:
        return True

    # Check word overlap (Jaccard similarity)
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if not words1 or not words2:
        return False

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    jaccard = intersection / union if union > 0 else 0.0

    # Consider similar if Jaccard exceeds threshold
    return jaccard > SIMILARITY_THRESHOLD


if __name__ == "__main__":
    # Demonstration of robustness tests
    print("=" * 60)
    print("BenchRight Robustness Tests Demo")
    print("=" * 60)

    # Example prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "What is 2 + 2?",
    ]

    print("\n📝 Prompt Perturbation Examples:")
    print("-" * 60)

    for prompt in test_prompts[:1]:
        print(f"\n🔤 Original: \"{prompt}\"")
        print("   Perturbations:")
        for mode in ["typo", "synonym", "reorder", "case", "whitespace", "punctuation"]:
            perturbed = perturb_prompt(prompt, mode=mode, seed=42)
            print(f"      [{mode:12}]: \"{perturbed}\"")

    print("\n\n📊 Robustness Sweep Demo:")
    print("-" * 60)

    # Define a stable mock model
    def stable_model(prompt: str) -> str:
        """A model that gives consistent answers."""
        prompt_lower = prompt.lower()
        if "capital" in prompt_lower and "france" in prompt_lower:
            return "Paris"
        if "2" in prompt_lower and "+" in prompt_lower:
            return "4"
        if "photosynthesis" in prompt_lower:
            return "Photosynthesis is the process plants use to convert light to energy."
        return "I don't know."

    # Define an unstable mock model
    def unstable_model(prompt: str) -> str:
        """A model that gives inconsistent answers."""
        answers = ["Paris", "London", "Berlin", "Rome", "Madrid"]
        return random.choice(answers)

    # Test stable model
    print("\n🛡️ Stable Model:")
    prompt = "What is the capital of France?"
    results = robustness_sweep(stable_model, prompt, n=10, seed=42)
    print(f"   Prompt: \"{prompt}\"")
    print(f"   Original output: \"{results['original_output']}\"")
    print(f"   Stability score: {results['stability_score']:.2%}")
    print(f"   Unique outputs: {results['unique_outputs']}/{len(results['variants'])}")
    print(f"   Output diversity: {results['output_diversity']:.2%}")

    # Test unstable model
    print("\n⚠️ Unstable Model:")
    results = robustness_sweep(unstable_model, prompt, n=10, seed=42)
    print(f"   Prompt: \"{prompt}\"")
    print(f"   Original output: \"{results['original_output']}\"")
    print(f"   Stability score: {results['stability_score']:.2%}")
    print(f"   Unique outputs: {results['unique_outputs']}/{len(results['variants'])}")
    print(f"   Output diversity: {results['output_diversity']:.2%}")

    print("\n   Sample variants:")
    for i, variant in enumerate(results['variants'][:3]):
        status = "✓" if variant['is_similar'] else "✗"
        print(f"      [{status}] {variant['mode']:12}: \"{variant['output']}\"")

    print("\n" + "=" * 60)
    print("Demo complete!")
