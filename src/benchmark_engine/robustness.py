"""
Robustness testing module for BenchRight.

This module provides functions for testing model robustness by applying
various perturbations to input prompts and measuring output stability.
Robustness testing helps identify models that may produce inconsistent
or unreliable outputs when inputs vary slightly.

Example usage:
    >>> def my_model(prompt: str) -> str:
    ...     return "Some response"
    >>> 
    >>> # Perturb a single prompt
    >>> perturbed = perturb_prompt("Hello world", mode="typo")
    >>> print(perturbed)  # e.g., "Helo world"
    >>> 
    >>> # Run a robustness sweep
    >>> results = robustness_sweep(my_model, "What is the capital of France?", n=20)
    >>> print(f"Stability score: {results['stability_score']:.2%}")
"""

import logging
import random
import re
import time
from typing import Callable, Dict, Any, List, Optional


# Configure module logger
logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

# Threshold for Jaccard similarity to consider outputs semantically similar
# This is a placeholder threshold - should be tuned or replaced with embedding-based similarity
SEMANTIC_SIMILARITY_THRESHOLD = 0.7


# ==============================================================================
# Prompt Perturbation Functions
# ==============================================================================

# Common typo mappings: character -> possible typo replacements
# Based on keyboard proximity and common mistakes
TYPO_MAPPINGS: Dict[str, List[str]] = {
    "a": ["s", "q", "z"],
    "b": ["v", "n", "g", "h"],
    "c": ["x", "v", "d", "f"],
    "d": ["s", "f", "e", "r", "c", "x"],
    "e": ["w", "r", "d", "s"],
    "f": ["d", "g", "r", "t", "v", "c"],
    "g": ["f", "h", "t", "y", "b", "v"],
    "h": ["g", "j", "y", "u", "n", "b"],
    "i": ["u", "o", "k", "j"],
    "j": ["h", "k", "u", "i", "m", "n"],
    "k": ["j", "l", "i", "o", "m"],
    "l": ["k", "o", "p"],
    "m": ["n", "j", "k"],
    "n": ["b", "m", "h", "j"],
    "o": ["i", "p", "l", "k"],
    "p": ["o", "l"],
    "q": ["w", "a"],
    "r": ["e", "t", "d", "f"],
    "s": ["a", "d", "w", "e", "x", "z"],
    "t": ["r", "y", "f", "g"],
    "u": ["y", "i", "h", "j"],
    "v": ["c", "b", "f", "g"],
    "w": ["q", "e", "a", "s"],
    "x": ["z", "c", "s", "d"],
    "y": ["t", "u", "g", "h"],
    "z": ["a", "x", "s"],
}

# Simple synonym mappings for common words
# TODO: Use a proper thesaurus or embedding-based approach in production systems
SYNONYM_MAPPINGS: Dict[str, List[str]] = {
    "what": ["which", "how"],
    "is": ["are", "was"],
    "the": ["a", "this"],
    "capital": ["main city", "chief city"],
    "large": ["big", "huge", "enormous"],
    "small": ["tiny", "little", "minute"],
    "good": ["great", "excellent", "fine"],
    "bad": ["poor", "terrible", "awful"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "leisurely"],
    "happy": ["glad", "joyful", "pleased"],
    "sad": ["unhappy", "sorrowful", "dejected"],
    "help": ["assist", "aid", "support"],
    "make": ["create", "build", "produce"],
    "give": ["provide", "offer", "supply"],
    "take": ["grab", "seize", "obtain"],
    "find": ["discover", "locate", "uncover"],
    "show": ["display", "demonstrate", "reveal"],
    "tell": ["inform", "explain", "describe"],
    "ask": ["inquire", "question", "request"],
    "important": ["significant", "crucial", "vital"],
    "beautiful": ["lovely", "gorgeous", "stunning"],
    "difficult": ["hard", "challenging", "tough"],
    "easy": ["simple", "straightforward", "effortless"],
}


def _inject_typos(text: str, num_typos: int = 1, seed: Optional[int] = None) -> str:
    """Inject typos into text by replacing characters with keyboard-adjacent ones.
    
    Args:
        text: The original text.
        num_typos: Number of typos to inject.
        seed: Random seed for reproducibility.
    
    Returns:
        Text with typos injected.
    """
    if seed is not None:
        random.seed(seed)
    
    chars = list(text)
    
    # Find positions of alphabetic characters that can have typos
    valid_positions = [
        i for i, c in enumerate(chars)
        if c.lower() in TYPO_MAPPINGS
    ]
    
    if not valid_positions:
        return text
    
    # Select random positions for typos (up to available positions)
    num_typos = min(num_typos, len(valid_positions))
    typo_positions = random.sample(valid_positions, num_typos)
    
    for pos in typo_positions:
        original_char = chars[pos]
        is_upper = original_char.isupper()
        lower_char = original_char.lower()
        
        if lower_char in TYPO_MAPPINGS:
            replacement = random.choice(TYPO_MAPPINGS[lower_char])
            if is_upper:
                replacement = replacement.upper()
            chars[pos] = replacement
    
    return "".join(chars)


def _substitute_synonyms(text: str, num_substitutions: int = 1, seed: Optional[int] = None) -> str:
    """Substitute words with synonyms.
    
    Args:
        text: The original text.
        num_substitutions: Number of word substitutions to make.
        seed: Random seed for reproducibility.
    
    Returns:
        Text with synonym substitutions.
    """
    if seed is not None:
        random.seed(seed)
    
    words = text.split()
    
    # Find words that have synonyms
    substitutable = []
    for i, word in enumerate(words):
        # Strip punctuation for matching
        clean_word = re.sub(r"[^\w]", "", word).lower()
        if clean_word in SYNONYM_MAPPINGS:
            substitutable.append((i, word, clean_word))
    
    if not substitutable:
        return text
    
    # Select random words for substitution
    num_substitutions = min(num_substitutions, len(substitutable))
    to_substitute = random.sample(substitutable, num_substitutions)
    
    for idx, original_word, clean_word in to_substitute:
        synonym = random.choice(SYNONYM_MAPPINGS[clean_word])
        
        # Preserve capitalization and punctuation
        if original_word[0].isupper():
            synonym = synonym.capitalize()
        
        # Preserve trailing punctuation
        trailing_punct = ""
        if original_word and not original_word[-1].isalnum():
            trailing_punct = original_word[-1]
        
        words[idx] = synonym + trailing_punct
    
    return " ".join(words)


def _reorder_words(text: str, seed: Optional[int] = None) -> str:
    """Reorder words in the text while trying to preserve meaning.
    
    This applies simple reordering strategies:
    - Swap adjacent words
    - Move modifiers
    
    Args:
        text: The original text.
        seed: Random seed for reproducibility.
    
    Returns:
        Text with reordered words.
    """
    if seed is not None:
        random.seed(seed)
    
    words = text.split()
    
    if len(words) < 3:
        return text
    
    # Strategy: swap two adjacent non-boundary words
    # Avoid swapping first and last words to maintain question structure
    swap_positions = list(range(1, len(words) - 1))
    
    if not swap_positions:
        return text
    
    # Pick a random position to swap with its neighbor
    pos = random.choice(swap_positions)
    
    # Swap with next word if available, otherwise previous
    if pos + 1 < len(words):
        words[pos], words[pos + 1] = words[pos + 1], words[pos]
    elif pos > 0:
        words[pos], words[pos - 1] = words[pos - 1], words[pos]
    
    return " ".join(words)


def perturb_prompt(
    prompt: str,
    mode: str,
    seed: Optional[int] = None,
) -> str:
    """Apply perturbations to a prompt string.
    
    This function modifies the input prompt according to the specified mode,
    which can introduce typos, substitute synonyms, or reorder words. These
    perturbations are used to test model robustness to input variations.
    
    Args:
        prompt: The original prompt string to perturb.
        mode: The type of perturbation to apply. Must be one of:
            - "typo": Inject character-level typos based on keyboard proximity
            - "synonym": Replace words with their synonyms
            - "reorder": Reorder words within the prompt
        seed: Random seed for reproducibility. Defaults to None.
    
    Returns:
        The perturbed prompt string.
    
    Raises:
        ValueError: If mode is not one of "typo", "synonym", or "reorder".
    
    Example:
        >>> perturb_prompt("What is the capital of France?", mode="typo", seed=42)
        'What is the capiral of France?'
        
        >>> perturb_prompt("What is the capital of France?", mode="synonym", seed=42)
        'Which is the main city of France?'
        
        >>> perturb_prompt("What is the capital of France?", mode="reorder", seed=42)
        'What the is capital of France?'
    """
    valid_modes = ["typo", "synonym", "reorder"]
    
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid perturbation mode: '{mode}'. "
            f"Must be one of: {valid_modes}"
        )
    
    if mode == "typo":
        return _inject_typos(prompt, num_typos=1, seed=seed)
    elif mode == "synonym":
        return _substitute_synonyms(prompt, num_substitutions=1, seed=seed)
    elif mode == "reorder":
        return _reorder_words(prompt, seed=seed)
    
    # Should not reach here due to validation above
    return prompt


def robustness_sweep(
    model_fn: Callable[[str], str],
    prompt: str,
    n: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate perturbed variants of a prompt and collect model outputs.
    
    This function creates n perturbed versions of the input prompt using
    different perturbation modes (typos, synonyms, reorderings) and collects
    the model's responses. It then calculates a stability score based on
    how consistent the outputs are across perturbations.
    
    Args:
        model_fn: A callable that takes a prompt string and returns generated text.
                  Signature: (str) -> str
        prompt: The original prompt to perturb.
        n: Number of perturbed variants to generate. Defaults to 20.
        seed: Random seed for reproducibility. Defaults to None.
    
    Returns:
        A dictionary containing:
            - original_prompt: The input prompt
            - original_output: Model output for the original prompt
            - perturbed_prompts: List of perturbed prompt strings
            - perturbed_outputs: List of model outputs for perturbed prompts
            - stability_score: Fraction of outputs semantically similar to original
                              (currently uses exact match as placeholder)
            - total_variants: Total number of variants tested (n)
            - matching_outputs: Number of outputs matching the original
            - perturbation_breakdown: Dict with counts per perturbation mode
            - total_time_seconds: Total time for the sweep
            - results: List of detailed results for each variant
    
    Example:
        >>> def my_model(prompt: str) -> str:
        ...     return "Paris"
        >>> results = robustness_sweep(my_model, "What is the capital of France?", n=10)
        >>> print(f"Stability: {results['stability_score']:.2%}")
    """
    if seed is not None:
        random.seed(seed)
    
    modes = ["typo", "synonym", "reorder"]
    
    # Get original output
    start_time = time.time()
    original_output = model_fn(prompt)
    
    perturbed_prompts: List[str] = []
    perturbed_outputs: List[str] = []
    detailed_results: List[Dict[str, Any]] = []
    mode_counts: Dict[str, int] = {"typo": 0, "synonym": 0, "reorder": 0}
    matching_count = 0
    
    for i in range(n):
        # Select a random perturbation mode
        mode = random.choice(modes)
        mode_counts[mode] += 1
        
        # Generate perturbed prompt with a unique seed
        variant_seed = (seed + i + 1) if seed is not None else None
        perturbed = perturb_prompt(prompt, mode=mode, seed=variant_seed)
        perturbed_prompts.append(perturbed)
        
        # Get model output for perturbed prompt
        inference_start = time.time()
        output = model_fn(perturbed)
        inference_time = time.time() - inference_start
        perturbed_outputs.append(output)
        
        # Check semantic similarity
        # TODO: Replace word overlap heuristic with embedding-based semantic similarity
        is_similar = _check_semantic_similarity(original_output, output)
        
        if is_similar:
            matching_count += 1
        
        detailed_results.append({
            "variant_index": i,
            "perturbation_mode": mode,
            "original_prompt": prompt,
            "perturbed_prompt": perturbed,
            "original_output": original_output,
            "perturbed_output": output,
            "is_semantically_similar": is_similar,
            "inference_time_seconds": inference_time,
        })
        
        # Log the perturbation and output
        logger.debug(
            f"Variant {i+1}/{n} [{mode}]: "
            f"'{perturbed[:50]}...' -> '{output[:50]}...'"
        )
    
    total_time = time.time() - start_time
    
    # Calculate stability score
    stability_score = matching_count / n if n > 0 else 0.0
    
    logger.info(
        f"Robustness sweep complete: {matching_count}/{n} outputs "
        f"semantically similar (stability: {stability_score:.2%})"
    )
    
    return {
        "original_prompt": prompt,
        "original_output": original_output,
        "perturbed_prompts": perturbed_prompts,
        "perturbed_outputs": perturbed_outputs,
        "stability_score": stability_score,
        "total_variants": n,
        "matching_outputs": matching_count,
        "perturbation_breakdown": mode_counts,
        "total_time_seconds": total_time,
        "examples_per_second": n / total_time if total_time > 0 else 0.0,
        "results": detailed_results,
    }


def _check_semantic_similarity(output1: str, output2: str) -> bool:
    """Check if two outputs are semantically similar.
    
    TODO: Implement proper semantic similarity using:
    - Embedding similarity (e.g., sentence transformers)
    - LLM-as-judge comparison
    - Token overlap metrics (BLEU, ROUGE)
    
    Currently uses a simple heuristic based on normalized text matching
    and word overlap as a placeholder.
    
    Args:
        output1: First output string.
        output2: Second output string.
    
    Returns:
        True if outputs are considered semantically similar, False otherwise.
    """
    # Normalize outputs for comparison
    def normalize(text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()
    
    norm1 = normalize(output1)
    norm2 = normalize(output2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return True
    
    # Word overlap heuristic
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return False
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Consider similar if Jaccard similarity meets threshold
    return jaccard >= SEMANTIC_SIMILARITY_THRESHOLD


if __name__ == "__main__":
    # Demonstration with mock model
    print("=" * 60)
    print("BenchRight Robustness Tests Demo")
    print("=" * 60)

    # Create a simple mock model
    def mock_model(prompt: str) -> str:
        """A simple mock model that returns consistent answers."""
        prompt_lower = prompt.lower()
        
        # Handle capital of France with various perturbations
        if "capital" in prompt_lower and "france" in prompt_lower:
            return "Paris"
        elif "main city" in prompt_lower and "france" in prompt_lower:
            return "Paris"
        elif "chief city" in prompt_lower and "france" in prompt_lower:
            return "Paris"
        elif "2+2" in prompt_lower or "2 + 2" in prompt_lower:
            return "4"
        elif "color" in prompt_lower and "sky" in prompt_lower:
            return "The sky is blue."
        else:
            return "I'm not sure about that."

    # Demo perturb_prompt
    print("\n📝 Demonstrating perturb_prompt:")
    print("-" * 60)
    
    original = "What is the capital of France?"
    print(f"Original: {original}")
    
    for mode in ["typo", "synonym", "reorder"]:
        perturbed = perturb_prompt(original, mode=mode, seed=42)
        print(f"[{mode:8}] {perturbed}")

    # Demo robustness_sweep
    print("\n📊 Running Robustness Sweep:")
    print("-" * 60)
    
    results = robustness_sweep(
        model_fn=mock_model,
        prompt="What is the capital of France?",
        n=10,
        seed=42
    )
    
    print(f"\n📈 Results:")
    print(f"   Original prompt: {results['original_prompt']}")
    print(f"   Original output: {results['original_output']}")
    print(f"   Total variants: {results['total_variants']}")
    print(f"   Matching outputs: {results['matching_outputs']}")
    print(f"   Stability score: {results['stability_score']:.2%}")
    print(f"   Time: {results['total_time_seconds']:.4f} seconds")
    
    print("\n📋 Perturbation Breakdown:")
    for mode, count in results['perturbation_breakdown'].items():
        print(f"   {mode}: {count}")
    
    print("\n📋 Sample Results:")
    for result in results['results'][:5]:
        status = "✓ Similar" if result['is_semantically_similar'] else "✗ Different"
        print(f"\n   [{status}] Variant {result['variant_index']+1} ({result['perturbation_mode']}):")
        print(f"       Perturbed: {result['perturbed_prompt'][:50]}...")
        print(f"       Output: {result['perturbed_output']}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
