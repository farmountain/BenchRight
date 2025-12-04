# Week 3 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 3 mini-project: **Compare Perplexity Across Text Domains**.

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
| 3 | Code runs without errors, correctly implements perplexity calculation, evaluates both formal and casual text domains with 5-10 sentences each |
| 2 | Code runs with minor issues but demonstrates understanding of perplexity computation |
| 1 | Code has significant errors or only implements one domain |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Perplexity formula is correctly implemented: exp(-1/N × Σ log P(token_i))
- [ ] 5-10 sentences collected for formal text domain
- [ ] 5-10 sentences collected for casual text domain
- [ ] Results are numerically reasonable (perplexity values > 1)

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows perplexity for each sentence, includes domain averages, and compares formal vs casual domains |
| 2 | Results table has both domains but may lack comparative statistics |
| 1 | Results table is incomplete or missing one domain |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| Domain | Sentence | Perplexity |
|--------|----------|------------|
| Formal | [News/Wikipedia sentence] | XX.XX |
| Casual | [Chat/social media sentence] | XX.XX |

### Summary
| Domain | Mean Perplexity | Min | Max |
|--------|-----------------|-----|-----|
| Formal | XX.XX | XX.XX | XX.XX |
| Casual | XX.XX | XX.XX | XX.XX |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides thoughtful comparison of perplexity between domains, explains why differences occur, and draws conclusions about model training data |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Comparison of average perplexity between formal and casual domains
- Explanation of why one domain might have lower perplexity
- Discussion of what perplexity reveals about training data distribution
- Understanding of perplexity limitations for evaluation

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file, clear domain descriptions, proper table formatting, includes sources of sample texts |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Clear title and date
- [ ] Description of each text domain and source
- [ ] Perplexity calculation methodology
- [ ] Results tables with domain comparison
- [ ] Interpretation section

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
# Week 3 Mini-Project: Perplexity Across Text Domains

**Author:** [Your Name]  
**Date:** [Date]  
**Model:** tinyGPT (ONNX)

## Text Domains

### Formal Text (News/Wikipedia)
Source: News articles and Wikipedia excerpts
Characteristics: Proper grammar, complete sentences, formal vocabulary

### Casual Text (Social Media/Chat)
Source: Simulated chat messages and social media posts
Characteristics: Informal language, abbreviations, incomplete sentences

## Methodology

Perplexity was calculated using:
PPL = exp(-1/N × Σ log P(token_i))

## Results

### Formal Text

| Sentence | Tokens | Perplexity |
|----------|--------|------------|
| The Federal Reserve announced interest rate changes. | 8 | 28.5 |
| Scientists discovered a new species in the Amazon. | 9 | 31.2 |
| The technology sector showed strong quarterly growth. | 8 | 25.8 |
| International diplomats gathered for climate talks. | 7 | 35.4 |
| Research indicates significant progress in medicine. | 7 | 29.1 |

**Formal Average: 30.0**

### Casual Text

| Sentence | Tokens | Perplexity |
|----------|--------|------------|
| lol did u see that? so funny 😂 | 8 | 89.3 |
| omg cant believe it happened again | 7 | 76.5 |
| gonna grab food brb | 5 | 125.4 |
| tbh idk what to do anymore | 7 | 98.2 |
| ngl that was pretty wild | 6 | 82.1 |

**Casual Average: 94.3**

## Summary Statistics

| Domain | Mean PPL | Min | Max | Std Dev |
|--------|----------|-----|-----|---------|
| Formal | 30.0 | 25.8 | 35.4 | 3.6 |
| Casual | 94.3 | 76.5 | 125.4 | 18.7 |

## Interpretation

### Key Findings

1. **Casual text has 3x higher perplexity:** Average 94.3 vs 30.0 for formal text, indicating the model is much less confident about casual language.

2. **Training data bias:** tinyGPT was likely trained on more formal text (books, Wikipedia, articles), making it better at predicting formal language patterns.

3. **Abbreviations increase perplexity:** Terms like "lol," "brb," and "tbh" are likely rare in training data, causing higher perplexity.

4. **Emojis are challenging:** The 😂 emoji contributed to the highest perplexity sentence.

### Why Does This Matter?

- **Domain mismatch:** A model with low formal perplexity but high casual perplexity may struggle with chatbot applications
- **Evaluation selection:** Perplexity benchmarks should match deployment domain
- **Fine-tuning signal:** High perplexity domains indicate where fine-tuning could help

### Limitations

- Small sample sizes (5 sentences per domain)
- Single model tested
- Synthetic casual text (not real social media data)
- Perplexity doesn't capture semantic correctness
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Quality and authenticity of text samples in each domain
2. Correct implementation of perplexity calculation
3. Depth of comparative analysis between domains
4. Understanding of what perplexity reveals about model training
