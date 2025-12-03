# Week 4 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 4 mini-project: **Evaluating tinyGPT on ARC, WinoGrande, and GSM8K**.

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
| 3 | Code runs without errors, correctly implements all three benchmark evaluations (ARC Easy/Challenge, WinoGrande, GSM8K), uses appropriate methodology for each |
| 2 | Code runs with minor issues but demonstrates understanding of benchmark evaluation |
| 1 | Code has significant errors or only implements one or two benchmarks |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] ARC Easy and ARC Challenge are evaluated separately
- [ ] WinoGrande coreference resolution is correctly handled
- [ ] GSM8K math problems use appropriate evaluation (chain-of-thought or direct answer)
- [ ] Sample sizes are appropriate (n=100 minimum per benchmark)

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows scores for all benchmarks with sample sizes, distinguishes ARC Easy from Challenge, notes GSM8K evaluation approach |
| 2 | Results table has most benchmarks but may lack detail |
| 1 | Results table is incomplete or missing benchmarks |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| Benchmark | Accuracy | Sample Size | Notes |
|-----------|----------|-------------|-------|
| ARC-Easy | XX.X% | N | Science questions |
| ARC-Challenge | XX.X% | N | Harder science |
| WinoGrande | XX.X% | N | Coreference |
| GSM8K | XX.X% | N | Math problems |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides thoughtful comparison across benchmarks, discusses difficulty levels, analyzes reasoning vs knowledge performance, compares to published baselines |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Comparison of ARC Easy vs Challenge performance
- Discussion of reasoning capabilities shown by WinoGrande
- Analysis of mathematical reasoning limitations on GSM8K
- Comparison to larger model performance
- Understanding of benchmark difficulty hierarchy

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file, clear benchmark descriptions, proper formatting, methodology for each benchmark |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Clear title and date
- [ ] Description of each benchmark and what it measures
- [ ] Evaluation methodology per benchmark
- [ ] Results tables with clear formatting
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
# Week 4 Mini-Project Results

**Author:** [Your Name]  
**Date:** [Date]  
**Model:** tinyGPT (ONNX)

## Benchmark Descriptions

### ARC (AI2 Reasoning Challenge)
- **Easy:** Grade-school science questions
- **Challenge:** Harder science questions requiring multi-step reasoning

### WinoGrande
Tests commonsense reasoning through pronoun resolution tasks.

### GSM8K
Grade school math word problems requiring multi-step arithmetic reasoning.

## Results

### Overall Results

| Benchmark | Accuracy | Sample Size | Random Baseline |
|-----------|----------|-------------|-----------------|
| ARC-Easy | 32.5% | 100 | 25% |
| ARC-Challenge | 26.8% | 100 | 25% |
| WinoGrande | 52.3% | 100 | 50% |
| GSM8K | 2.1% | 100 | ~0% |

## Interpretation

**Key findings:**

1. **Science reasoning near random:** ARC scores (26-32%) show limited scientific knowledge, slightly better on Easy vs Challenge as expected.

2. **Coreference at chance level:** WinoGrande at 52.3% is barely above random (50%), indicating minimal commonsense pronoun resolution ability.

3. **Math is extremely difficult:** GSM8K at 2.1% shows tinyGPT cannot reliably perform multi-step math reasoning—this is expected for small models.

**Difficulty hierarchy observed:**
GSM8K (hardest) < ARC-Challenge < ARC-Easy < WinoGrande (easiest)

**Comparison to larger models:**
| Model | GSM8K | ARC-C | WinoGrande |
|-------|-------|-------|------------|
| tinyGPT | 2.1% | 26.8% | 52.3% |
| GPT-3.5 | ~57% | ~85% | ~87% |
| GPT-4 | ~92% | ~96% | ~94% |

**Limitations:**
- Small sample sizes (100 per benchmark)
- No chain-of-thought prompting for GSM8K
- Single evaluation run
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Correct implementation of each benchmark's unique evaluation approach
2. Understanding of difficulty progression across benchmarks
3. Quality of comparison to published results
4. Insights about small model limitations on reasoning tasks
