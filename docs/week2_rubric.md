# Week 2 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 2 mini-project: **Tokenization Analysis - Same Length, Different Tokens**.

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
| 3 | Code runs without errors, correctly tokenizes both prompts, measures token counts accurately, and runs inference with latency measurement |
| 2 | Code runs with minor issues but demonstrates understanding of tokenization and latency measurement |
| 1 | Code has significant errors or only partially implements the required functionality |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Two prompts with same character count but different token counts are created
- [ ] Tokenizer correctly counts tokens for each prompt
- [ ] Inference runs successfully on both prompts
- [ ] Latency is measured accurately for comparison

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows character count, token count, and latency for both prompts; includes clear comparison of the differences |
| 2 | Results table has both prompts but may have minor formatting issues |
| 1 | Results table is incomplete or missing key measurements |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| Prompt | Characters | Tokens | Latency (ms) |
|--------|------------|--------|--------------|
| Prompt A (common words) | ~100 | N | XX.XX |
| Prompt B (rare words) | ~100 | M | XX.XX |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides thoughtful analysis of token count differences, explains why different prompts tokenize differently, and discusses the relationship between token count and latency |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Explanation of why common words tokenize into fewer tokens
- Discussion of BPE/tokenization algorithm behavior
- Analysis of token count's impact on inference latency
- Understanding of implications for cost and performance

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file, clear headings, proper table formatting, includes explanation of prompt design choices |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Clear title and date
- [ ] Both prompts displayed clearly
- [ ] Properly formatted results table
- [ ] Interpretation section
- [ ] Explanation of prompt design rationale

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
# Week 2 Mini-Project: Tokenization Analysis

**Author:** [Your Name]  
**Date:** [Date]  
**Environment:** Google Colab, Python 3.10, ONNX Runtime 1.16.0

## Prompts

### Prompt A (Common English Words) - 100 characters
"The quick brown fox jumps over the lazy dog. This is a simple sentence with common everyday words."

### Prompt B (Technical/Rare Words) - 100 characters  
"Quantum chromodynamics describes gluon-mediated interactions between quarks in hadrons via SU(3)."

## Results

| Prompt | Characters | Tokens | Latency (ms) |
|--------|------------|--------|--------------|
| Prompt A (common) | 100 | 22 | 38.4 |
| Prompt B (technical) | 100 | 41 | 52.7 |

## Analysis

### Token Count Comparison

- **Prompt A:** 22 tokens (0.22 tokens per character)
- **Prompt B:** 41 tokens (0.41 tokens per character)
- **Difference:** Prompt B requires 86% more tokens

### Why the Difference?

1. **Common words are in vocabulary:** Words like "the," "quick," and "brown" are single tokens because they appear frequently in training data.

2. **Rare words are split:** Technical terms like "chromodynamics" and "gluon-mediated" are split into subword pieces:
   - "chromodynamics" → "chrom", "ody", "nam", "ics" (4 tokens)
   - "gluon-mediated" → "gl", "u", "on", "-", "med", "iated" (6 tokens)

3. **BPE tokenization:** The tokenizer uses Byte Pair Encoding, which merges frequently occurring character sequences. Common patterns merge more aggressively.

### Latency Impact

- **Prompt A latency:** 38.4 ms
- **Prompt B latency:** 52.7 ms
- **Latency increase:** 37%

The 86% increase in tokens resulted in a 37% increase in latency. This sublinear relationship suggests some fixed overhead, but token count is a strong predictor of inference time.

### Implications

1. **Cost prediction:** API costs often charge per token—technical content is more expensive
2. **Latency budgets:** For real-time applications, prefer concise, common vocabulary
3. **Context limits:** Same character count uses different amounts of context window

## Limitations

- Only tested 2 prompts; more samples would give statistical confidence
- Single run per prompt; multiple runs would reduce variance
- Did not test extreme cases (non-Latin scripts, code, etc.)
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Quality of prompt pair design (same characters, different tokens)
2. Accuracy of token count and latency measurement
3. Depth of understanding about tokenization algorithms
4. Analysis of relationship between tokens and latency
