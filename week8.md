# Week 8 — Robustness Tests

### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 8, you will:

1. Understand *what robustness testing is* and why it matters for reliable LLM deployment.
2. Learn the difference between **typo**, **synonym**, and **reordering** perturbations.
3. Implement and use the `perturb_prompt` and `robustness_sweep` functions from `src/benchmark_engine/robustness.py`.
4. Understand how to measure model stability across input variations.
5. Apply critical thinking to identify when models may produce inconsistent outputs.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is Robustness Testing?

### Simple Explanation

Imagine you're asking a smart assistant the same question in slightly different ways:

- "What is the capital of France?"
- "Waht is the capital of France?" (typo)
- "What is the main city of France?" (synonym)
- "The capital of France is what?" (reordering)

> **Robustness testing measures whether an LLM gives consistent answers when inputs are slightly perturbed.**

A robust model should:
- Understand the intent despite typos
- Recognize synonyms as equivalent
- Handle different phrasings of the same question

This matters because:
- **Real-world inputs are messy:** Users make typos and use varied vocabulary
- **Consistency builds trust:** Unstable outputs undermine confidence
- **Adversarial attacks:** Malicious perturbations can exploit brittleness
- **Production reliability:** Systems need to handle varied inputs gracefully

### The Three Pillars of Robustness Testing

| Pillar | What It Tests | Example |
|--------|---------------|---------|
| **Typo Tolerance** | Can the model handle character-level errors? | "Waht" → "What" |
| **Synonym Recognition** | Does the model understand equivalent words? | "capital" → "main city" |
| **Syntactic Flexibility** | Can the model handle reordered phrases? | "is what" → "what is" |

---

# 🧠 Section 2 — Perturbation Modes

### Typo Injection

Typos simulate real-world keyboard errors. The perturbation function:
- Replaces characters with keyboard-adjacent alternatives
- Simulates common typing mistakes
- Tests character-level robustness

Example typo mappings (based on keyboard proximity):
```
a → [s, q, z]
e → [w, r, d, s]
i → [u, o, k, j]
```

### Synonym Substitution

Synonyms test semantic understanding:
- Replaces words with their synonyms
- Maintains sentence meaning
- Tests vocabulary flexibility

Example synonym mappings:
```
capital → [main city, chief city]
large → [big, huge, enormous]
important → [significant, crucial, vital]
```

### Word Reordering

Reordering tests syntactic flexibility:
- Swaps adjacent words
- Preserves overall meaning (mostly)
- Tests structural robustness

### Why Each Matters

| Mode | Real-World Scenario | Risk if Model Fails |
|------|---------------------|---------------------|
| **Typo** | Mobile keyboard errors | Users get no response for simple mistakes |
| **Synonym** | Regional vocabulary differences | Model fails on paraphrased inputs |
| **Reorder** | Non-native English speakers | Legitimate queries get rejected |

---

# 🧪 Section 3 — The Robustness Module

### Module Design

The `robustness.py` module provides two main functions:

```python
def perturb_prompt(
    prompt: str,
    mode: str,
) -> str:
    """
    Apply perturbations to a prompt string.
    
    Args:
        prompt: The original prompt string to perturb
        mode: Perturbation type - "typo", "synonym", or "reorder"
        
    Returns:
        The perturbed prompt string
    """

def robustness_sweep(
    model_fn: Callable[[str], str],
    prompt: str,
    n: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate perturbed variants and collect model outputs.
    
    Args:
        model_fn: Callable that takes prompt string and returns generated text
        prompt: The original prompt to perturb
        n: Number of perturbed variants to generate (default: 20)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with stability_score, outputs, and detailed results
    """
```

### Key Design Decisions

1. **Mode-based perturbation:** Separate modes for different perturbation types
2. **Reproducible:** Optional seed for consistent evaluations
3. **Extensible:** Easy to add new perturbation modes
4. **Detailed logging:** Full results for debugging and analysis
5. **TODO semantic similarity:** Placeholder for future improvement with embedding-based comparison

---

# 🧪 Section 4 — Hands-on Lab: Using Robustness Tests

### Lab Overview

In this lab, you will:
1. Import the robustness testing functions
2. Perturb prompts with different modes
3. Run a robustness sweep on a model
4. Analyze stability scores
5. Identify model weaknesses

### Step 1: Import the Module

```python
from src.benchmark_engine.robustness import perturb_prompt, robustness_sweep
```

### Step 2: Perturb a Prompt

```python
original = "What is the capital of France?"

# Apply different perturbations
typo_variant = perturb_prompt(original, mode="typo", seed=42)
synonym_variant = perturb_prompt(original, mode="synonym", seed=42)
reorder_variant = perturb_prompt(original, mode="reorder", seed=42)

print(f"Original: {original}")
print(f"Typo:     {typo_variant}")
print(f"Synonym:  {synonym_variant}")
print(f"Reorder:  {reorder_variant}")
```

### Step 3: Create a Model Function

```python
def my_model(prompt: str) -> str:
    """Your model implementation or wrapper."""
    # For testing, you might use:
    # - A local ONNX model
    # - An API-based model (OpenAI, Anthropic)
    # - A Hugging Face transformer
    return "Your model's response here"
```

### Step 4: Run Robustness Sweep

```python
# Run robustness evaluation
results = robustness_sweep(
    model_fn=my_model,
    prompt="What is the capital of France?",
    n=20,  # Generate 20 variants
    seed=42  # For reproducibility
)

print(f"Stability score: {results['stability_score']:.2%}")
print(f"Matching outputs: {results['matching_outputs']}/{results['total_variants']}")
print(f"Perturbation breakdown: {results['perturbation_breakdown']}")
```

### Step 5: Analyze Results

```python
# Examine failed cases
for result in results['results']:
    if not result['is_semantically_similar']:
        print(f"Mode: {result['perturbation_mode']}")
        print(f"Original: {result['original_prompt']}")
        print(f"Perturbed: {result['perturbed_prompt']}")
        print(f"Original output: {result['original_output']}")
        print(f"Perturbed output: {result['perturbed_output']}")
        print()
```

---

# 🤔 Section 5 — Paul-Elder Critical Thinking Questions

### Question 1: EVIDENCE
**If a model achieves 95% stability but fails on typos in critical keywords (e.g., medical terms), what might this indicate?**

*Consider: Sensitivity to domain-specific terms, tokenization edge cases, training data coverage.*

### Question 2: ASSUMPTIONS
**What assumptions are we making when we use word overlap to measure semantic similarity?**

*Consider: Synonyms with no word overlap, negation, context dependence, multi-word expressions.*

### Question 3: IMPLICATIONS
**If a production system shows 80% robustness in testing but lower in production, what factors might explain this?**

*Consider: Distribution shift, user creativity, adversarial inputs, edge cases not covered.*

### Question 4: POINT OF VIEW
**How might different stakeholders (QA engineers, product managers, end users) interpret the same stability score?**

*Consider: Acceptable error rates, impact of failures, user experience, business risk.*

---

# 🔄 Section 6 — Inversion Thinking: How Can Robustness Testing Fail?

Instead of asking "How do robustness tests help us?", let's invert:

> **"How can robustness evaluations give us false confidence?"**

### Potential Failure Modes

1. **Limited Perturbation Coverage**
   - Only tests a few perturbation types
   - Misses domain-specific variations
   - Doesn't cover all possible typos

2. **Semantic Similarity Flaws**
   - Word overlap misses paraphrases
   - Exact match is too strict
   - Missing embedding-based comparison

3. **Dataset Bias**
   - Test prompts don't represent production
   - Synonym dictionary is incomplete
   - Reordering breaks grammar

4. **Adversarial Gaps**
   - Deliberate attacks not covered
   - Injection attacks bypassed
   - Multi-step perturbations missed

### Defensive Practices

- **Expand perturbation modes:** Add character deletion, insertion, phonetic spelling
- **Use embedding similarity:** Replace word overlap with sentence embeddings
- **Production monitoring:** Track robustness metrics on real inputs
- **Red-teaming:** Have humans try to break the model
- **Domain-specific tests:** Create perturbations relevant to your use case

---

# 📝 Section 7 — Mini-Project: Robustness Audit

### Task

Conduct a robustness evaluation of a model and produce an audit report.

### Instructions

1. **Choose a model:**
   - Create a mock model with specific behaviors
   - Or use an API-based model if available

2. **Select test prompts:**
   - Choose 5-10 representative prompts for your use case
   - Include varied question types and topics

3. **Run robustness sweeps:**
   - Run `robustness_sweep` on each prompt with n=20
   - Record stability scores and failure patterns

4. **Analyze failure cases:**
   - Which perturbation modes cause the most failures?
   - Are there patterns in the failing outputs?

5. **Document findings:**
   - Create a robustness report

### Submission Format

Create a markdown file `/examples/week08_robustness_audit.md`:

```markdown
# Week 8 Mini-Project: Robustness Audit Report

## Executive Summary
[1-2 sentences on overall robustness posture]

## Model Under Test
[Description of the model evaluated]

## Test Prompts
[List the prompts tested]

## Stability Scores

| Prompt | Overall Score | Typo | Synonym | Reorder |
|--------|---------------|------|---------|---------|
| Prompt 1 | ??% | ??% | ??% | ??% |
| Prompt 2 | ??% | ??% | ??% | ??% |
| ...      | ... | ... | ... | ... |

## Notable Failure Cases
[List 2-3 examples of inconsistent responses]

## Recommendations
[2-3 actionable recommendations based on findings]

## Limitations
[What this evaluation did NOT test]
```

---

# 🔧 Section 8 — Advanced: Extending the Robustness Module

### Adding Custom Perturbation Modes

You can extend the module with new perturbation types:

```python
def perturb_with_character_deletion(text: str) -> str:
    """Delete random characters from the text."""
    import random
    chars = list(text)
    if len(chars) > 5:
        pos = random.randint(1, len(chars) - 2)
        chars.pop(pos)
    return "".join(chars)

def perturb_with_case_changes(text: str) -> str:
    """Randomly change case of characters."""
    import random
    return "".join(
        c.swapcase() if random.random() < 0.2 else c
        for c in text
    )
```

### Improving Semantic Similarity

The current implementation uses word overlap. Future improvements:

```python
# TODO: Implement embedding-based similarity
def check_semantic_similarity_embedding(output1: str, output2: str) -> bool:
    """
    Use sentence embeddings to compare outputs.
    
    Options:
    - sentence-transformers (all-MiniLM-L6-v2)
    - OpenAI embeddings
    - Custom fine-tuned embeddings
    """
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode([output1, output2])
    # similarity = cosine_similarity(embeddings[0], embeddings[1])
    # return similarity > 0.8
    pass

# TODO: Implement LLM-as-judge similarity
def check_semantic_similarity_llm(output1: str, output2: str) -> bool:
    """
    Use an LLM to judge if outputs are semantically equivalent.
    """
    # prompt = f"Are these two answers semantically equivalent?\n1: {output1}\n2: {output2}"
    # response = llm_client.generate(prompt)
    # return "yes" in response.lower()
    pass
```

---

# ✔ Knowledge Mastery Checklist
- [ ] I can explain why robustness testing is critical for LLM deployment
- [ ] I understand the difference between typo, synonym, and reordering perturbations
- [ ] I can use `perturb_prompt` to generate perturbed inputs
- [ ] I can use `robustness_sweep` to evaluate model stability
- [ ] I understand how to interpret stability scores
- [ ] I know potential failure modes and defensive practices
- [ ] I completed the mini-project robustness audit

---

Week 8 complete.
Next: *Week 9 — Performance Benchmarking*.
