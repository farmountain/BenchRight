# Week 3 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 3 mini-project: **Evaluating tinyGPT on MMLU and HellaSwag**.

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
| 3 | Code runs without errors, correctly implements both MMLU and HellaSwag evaluations, uses proper evaluation methodology (exact match or log-prob for MMLU, continuation scoring for HellaSwag) |
| 2 | Code runs with minor issues but demonstrates understanding of benchmark evaluation |
| 1 | Code has significant errors or only implements one benchmark |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Dataset loading is correct (using datasets library or manual loading)
- [ ] MMLU evaluation uses appropriate method (exact match or log-probability)
- [ ] HellaSwag evaluation uses completion scoring methodology
- [ ] Sample sizes are appropriate (n=100 minimum recommended)

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows scores for both benchmarks with sample sizes, includes subject breakdown for MMLU if applicable |
| 2 | Results table has both benchmarks but may lack detail or breakdown |
| 1 | Results table is incomplete or missing one benchmark |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| Benchmark | Accuracy | Sample Size | Notes |
|-----------|----------|-------------|-------|
| MMLU (overall) | XX.X% | N | [Subjects tested] |
| HellaSwag | XX.X% | N | [Evaluation method] |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides thoughtful comparison of performance across benchmarks, discusses what each benchmark measures, compares to published baselines, and acknowledges limitations |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Comparison of tinyGPT performance to published model scores
- Discussion of what MMLU vs HellaSwag measure (knowledge vs reasoning)
- Analysis of performance patterns across MMLU subjects
- Understanding of small model limitations on knowledge benchmarks

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file, clear methodology section, proper formatting, includes benchmark descriptions |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Clear title and date
- [ ] Benchmark descriptions (what MMLU and HellaSwag measure)
- [ ] Evaluation methodology
- [ ] Results tables
- [ ] Interpretation and comparison to baselines

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
# Week 3 Mini-Project Results

**Author:** [Your Name]  
**Date:** [Date]  
**Model:** tinyGPT (ONNX)

## Benchmark Descriptions

### MMLU (Massive Multitask Language Understanding)
Tests knowledge across 57 subjects including STEM, humanities, and social sciences.
Measures: Factual knowledge and reasoning ability.

### HellaSwag
Tests commonsense reasoning through sentence completion tasks.
Measures: Physical and social commonsense understanding.

## Results

### MMLU Results

| Subject Category | Accuracy | Sample Size |
|------------------|----------|-------------|
| STEM | 25.8% | 50 |
| Humanities | 28.4% | 50 |
| Social Sciences | 27.2% | 50 |
| **Overall** | **27.1%** | **150** |

### HellaSwag Results

| Metric | Value |
|--------|-------|
| Accuracy | 31.5% |
| Sample Size | 100 |
| Random Baseline | 25% |

## Interpretation

**Key findings:**

1. **Near-random performance on MMLU:** tinyGPT achieves ~27% accuracy (random = 25%), indicating limited factual knowledge storage due to model size.

2. **Slightly better on HellaSwag:** 31.5% accuracy suggests marginally better commonsense reasoning, though still far from larger models (GPT-4: ~95%).

3. **Subject variation:** Humanities slightly outperformed STEM, possibly due to more common training patterns in text.

**Comparison to baselines:**
- GPT-3 (175B): MMLU ~43%, HellaSwag ~79%
- tinyGPT shows expected performance for a small model

**Limitations:**
- Small sample sizes (100-150 per benchmark)
- Single evaluation run
- Subset of MMLU subjects tested
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Correct implementation of benchmark evaluation methodology
2. Understanding of what each benchmark measures
3. Quality of comparison to published baselines
4. Recognition of model size limitations on knowledge-intensive tasks
