# Week 1 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 1 mini-project: **Running tinyGPT ONNX on 3 prompts and measuring latency**.

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
| 3 | Code runs without errors, correctly loads the ONNX model, tokenizes inputs, runs inference, and measures latency for all 3 prompts |
| 2 | Code runs with minor issues (e.g., missing imports, small bugs) but demonstrates understanding of the process |
| 1 | Code has significant errors or only partially implements the required functionality |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Model loads successfully with `ort.InferenceSession`
- [ ] Tokenizer is properly initialized
- [ ] Inference runs on all 3 prompts
- [ ] Latency is measured using `time.time()` or equivalent

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table is complete with all 3 prompts, outputs are readable, latency values are in milliseconds and appear reasonable |
| 2 | Results table has all 3 prompts but may have minor formatting issues or unclear outputs |
| 1 | Results table is incomplete or latency values appear incorrect |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| Prompt | Output | Latency (ms) |
|--------|--------|--------------|
| Explain machine learning | [Generated text] | XX.XX |
| Summarize the Singapore financial system | [Generated text] | XX.XX |
| Describe a robot to a child | [Generated text] | XX.XX |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides thoughtful analysis of results, discusses latency patterns, considers factors affecting performance, and draws reasonable conclusions |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Observations about latency differences between prompts
- Discussion of factors affecting latency (input length, tokenization)
- Acknowledgment of limitations (small sample size, single run)
- Suggestions for improvement

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file, clear headings, proper table formatting, includes environment details (Python version, hardware) |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Clear title and date
- [ ] Environment information (optional but encouraged)
- [ ] Properly formatted results table
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
# Week 1 Mini-Project Results

**Author:** [Your Name]  
**Date:** [Date]  
**Environment:** Google Colab, Python 3.10, ONNX Runtime 1.16.0

## Results Table

| Prompt | Output | Latency (ms) |
|--------|--------|--------------|
| Explain machine learning | Machine learning is a subset of artificial intelligence that enables computers to learn from data... | 45.32 |
| Summarize the Singapore financial system | Singapore's financial system is a hub for Asian banking and finance, regulated by MAS... | 52.18 |
| Describe a robot to a child | A robot is like a helpful friend made of metal and wires that can do tasks for people... | 38.91 |

## Summary Statistics

- **Mean Latency:** 45.47 ms
- **Min Latency:** 38.91 ms
- **Max Latency:** 52.18 ms

## Interpretation

The latency measurements show consistent performance across all three prompts, with 
inference times ranging from 38.91 ms to 52.18 ms. The "Summarize Singapore" prompt 
showed the highest latency (52.18 ms), which correlates with its longer input length 
requiring more tokenization.

**Key observations:**
1. Latency correlates loosely with prompt length
2. All prompts complete in under 100ms, suitable for interactive applications
3. Results are from a single run; multiple runs would provide statistical confidence

**Limitations:**
- Sample size of 3 prompts is too small for robust conclusions
- No warm-up runs were performed, which may affect initial measurements
- CPU-only execution; GPU would likely show different performance characteristics

**Next steps:**
- Run multiple iterations to calculate mean and standard deviation
- Test with longer prompts to understand scaling behavior
- Compare CPU vs GPU performance
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. What was done well
2. Specific improvements for each criterion
3. Suggestions for deeper exploration in future weeks
