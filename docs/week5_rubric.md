# Week 5 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 5 mini-project: **Extending the Benchmark Engine with a Custom Metric Function**.

---

## Grading Criteria

Each criterion is scored on a scale of **0–3**:

| Score | Description |
|-------|-------------|
| 0 | Not attempted or fundamentally incorrect |
| 1 | Partial attempt with significant errors |
| 2 | Mostly correct with minor issues |
| 3 | Excellent—complete, correct, and well-executed |

---

### 1. Correctness of Code (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Code runs without errors, correctly implements a custom metric function following the (output, reference) -> float signature, integrates with run_benchmark, and produces valid results |
| 2 | Code runs with minor issues but demonstrates understanding of metric function design |
| 1 | Code has significant errors or metric function doesn't work with the engine |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Custom metric follows correct signature: `Callable[[str, str], float]`
- [ ] Metric returns values in expected range (typically 0.0 to 1.0)
- [ ] Metric integrates correctly with `run_benchmark`
- [ ] Edge cases are handled (empty strings, None values)

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows comparison between exact_match and custom metric, includes diverse test cases, and demonstrates when the new metric differs |
| 2 | Results table is complete but comparison could be clearer |
| 1 | Results table is incomplete or doesn't show metric comparison |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| Input | Model Output | Reference | Exact Match | Custom Metric |
|-------|--------------|-----------|-------------|---------------|
| [Q1] | [Output1] | [Ref1] | 0.0/1.0 | X.XX |
| [Q2] | [Output2] | [Ref2] | 0.0/1.0 | X.XX |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides thoughtful analysis of when the custom metric differs from exact match, explains the use case for the custom metric, discusses trade-offs |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Clear explanation of what the custom metric measures
- Specific examples where custom metric differs from exact match
- Discussion of when to use partial_match vs exact_match
- Analysis of metric's sensitivity and edge cases

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file, includes metric design rationale, clear examples, proper formatting |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Clear title and date
- [ ] Custom metric description and rationale
- [ ] Code implementation
- [ ] Comparison results table
- [ ] Analysis of when the metric is useful

---

## Total Score

| Category | Score |
|----------|-------|
| Correctness of Code | /3 |
| Quality of Results | /3 |
| Interpretation | /3 |
| Documentation | /3 |
| **Total** | **/12** |

---

## Example of an Excellent Submission

```markdown
# Week 5 Mini-Project Results

**Author:** [Your Name]  
**Date:** [Date]  
**Custom Metric:** Partial Match (Word Overlap)

## Custom Metric Design

### partial_match_metric

```python
def partial_match_metric(output: str, reference: str) -> float:
    """
    Compute partial match score using word overlap.
    
    Returns the proportion of reference words found in output.
    Useful when the model provides correct information in a different format.
    """
    output_words = set(output.strip().lower().split())
    reference_words = set(reference.strip().lower().split())
    
    if not reference_words:
        return 1.0
    
    overlap = output_words & reference_words
    return len(overlap) / len(reference_words)
```

### Design Rationale

The partial_match_metric addresses a limitation of exact_match: it gives credit
for partially correct answers. This is useful when:
- Models provide correct content with different wording
- Answers include additional context but contain the key information
- Formatting differs but substance is correct

## Comparison Results

| Question | Model Output | Reference | Exact Match | Partial Match |
|----------|--------------|-----------|-------------|---------------|
| Capital of France? | The capital is Paris | Paris | 0.0 | 1.0 |
| What is 2+2? | The answer is 4 | 4 | 0.0 | 1.0 |
| Largest planet? | Jupiter is the largest planet | Jupiter | 0.0 | 1.0 |
| Color of sky? | Blue like the ocean | Blue | 0.0 | 1.0 |
| H2O formula? | H2O is water | H2O | 0.0 | 0.5 |

## Benchmark Results

| Metric | Mean Score | Examples Evaluated |
|--------|------------|-------------------|
| exact_match | 0.20 | 5 |
| partial_match | 0.90 | 5 |

## Interpretation

**Key findings:**

1. **Dramatic difference in scores:** exact_match shows 20% while partial_match shows 90%, revealing that the model often knows the answer but phrases it differently.

2. **Verbose model outputs:** The model tends to add context ("The capital is...", "The answer is...") which breaks exact matching.

3. **Edge case identified:** "H2O is water" scores 0.5 because only the reference word "H2O" appears, but "is" and "water" don't match the single-word reference.

**When to use partial_match:**
- Long-form generation where key information is embedded
- Evaluating model understanding rather than exact phrasing
- Early development where formatting isn't finalized

**Limitations:**
- May give false positives if output contains reference word by chance
- Doesn't consider word order or semantic meaning
- Case-insensitive may miss important distinctions
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Correctness and robustness of metric implementation
2. Thoughtfulness of metric design and use case
3. Quality of comparison between metrics
4. Understanding of trade-offs in evaluation design
