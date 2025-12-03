# Week 2 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 2 mini-project: **Computing Perplexity on Custom Text**.

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
| 3 | Code runs without errors, correctly implements perplexity calculation using log-probabilities, handles at least 3 sentences, and produces valid perplexity values |
| 2 | Code runs with minor issues but demonstrates understanding of perplexity computation |
| 1 | Code has significant errors or only partially implements the required functionality |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Log-probabilities are correctly extracted from model outputs
- [ ] Perplexity formula is correctly implemented: exp(-1/N × Σ log P(token_i))
- [ ] At least 3 different sentences are evaluated
- [ ] Results are numerically reasonable (perplexity values > 1)

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table is complete with all sentences, perplexity values are clearly presented, and includes summary statistics (mean, min, max) |
| 2 | Results table has all sentences but may have minor formatting issues |
| 1 | Results table is incomplete or perplexity values appear incorrect |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| Sentence | Token Count | Perplexity |
|----------|-------------|------------|
| [Sentence 1] | N | XX.XX |
| [Sentence 2] | N | XX.XX |
| [Sentence 3] | N | XX.XX |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides thoughtful analysis of perplexity differences between sentences, discusses factors affecting perplexity, and draws reasonable conclusions about model behavior |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Observations about which sentences have lower/higher perplexity
- Discussion of factors affecting perplexity (common vs. rare words, sentence structure)
- Understanding of what perplexity tells us about model confidence
- Comparison with expected behavior based on sentence complexity

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file, clear headings, proper table formatting, includes explanation of methodology |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Clear title and date
- [ ] Methodology explanation
- [ ] Properly formatted results table
- [ ] Interpretation section
- [ ] Summary statistics

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
# Week 2 Mini-Project Results

**Author:** [Your Name]  
**Date:** [Date]  
**Environment:** Google Colab, Python 3.10, ONNX Runtime 1.16.0

## Methodology

Perplexity was calculated using the formula:
PPL = exp(-1/N × Σ log P(token_i))

where N is the number of tokens and P(token_i) is the probability assigned to each token.

## Results Table

| Sentence | Token Count | Perplexity |
|----------|-------------|------------|
| The cat sat on the mat. | 7 | 12.45 |
| Quantum entanglement demonstrates non-locality. | 6 | 89.23 |
| I went to the store to buy groceries. | 9 | 15.67 |

## Summary Statistics

- **Mean Perplexity:** 39.12
- **Min Perplexity:** 12.45
- **Max Perplexity:** 89.23

## Interpretation

The perplexity values reveal interesting patterns about model confidence:

1. **Common phrases have lower perplexity:** "The cat sat on the mat" (12.45) shows the model is confident with common English patterns.

2. **Technical content increases perplexity:** The quantum physics sentence (89.23) has significantly higher perplexity, indicating the model is less certain about specialized vocabulary.

3. **Everyday language is predictable:** The grocery store sentence (15.67) falls in the low range, suggesting conversational text is well-modeled.

**Limitations:**
- Small sample size (3 sentences)
- Single model evaluation
- Token-level analysis would provide deeper insights
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Correctness of perplexity calculation implementation
2. Quality and diversity of test sentences
3. Depth of interpretation and understanding of perplexity
4. Suggestions for exploring perplexity in future analyses
