# Week 8 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 8 mini-project: **Robustness Audit Report**.

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
| 3 | Code runs without errors, correctly runs robustness_sweep on 5-10 prompts with n=20 variants each, captures stability scores and perturbation breakdown |
| 2 | Code runs with minor issues but demonstrates understanding of robustness testing |
| 1 | Code has significant errors or tests fewer than 5 prompts |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] 5-10 representative prompts are selected
- [ ] robustness_sweep is run with n=20 variants per prompt
- [ ] Stability scores are computed correctly
- [ ] Perturbation breakdown (typo, synonym, reorder) is captured
- [ ] Failure cases are recorded

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows all prompts with overall and per-perturbation stability scores, includes 2-3 notable inconsistent response examples |
| 2 | Results table is mostly complete but may lack perturbation breakdown |
| 1 | Results table is incomplete or missing stability analysis |
| 0 | No results provided or results are unusable |

**Expected results format:**

| Prompt | Overall Score | Typo | Synonym | Reorder |
|--------|---------------|------|---------|---------|
| Prompt 1 | ??% | ??% | ??% | ??% |
| Prompt 2 | ??% | ??% | ??% | ??% |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides executive summary, analyzes which perturbation modes cause the most failures, identifies patterns in inconsistent responses, provides actionable recommendations |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Analysis of which perturbation type (typo/synonym/reorder) is most problematic
- Identification of prompt characteristics that lead to instability
- Discussion of implications for production use
- Recommendations for improving robustness

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file following the specified format, includes all required sections, professional formatting |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Executive Summary
- [ ] Model Under Test description
- [ ] Test Prompts list
- [ ] Stability Scores table with per-perturbation breakdown
- [ ] Notable Failure Cases
- [ ] Recommendations
- [ ] Limitations section

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
# Week 8 Mini-Project: Robustness Audit Report

## Executive Summary

TinyGPT shows moderate robustness with an average stability score of 68% across all prompts and perturbation types. Typo perturbations are handled best (78%), while word reordering causes the most inconsistencies (55%).

## Model Under Test

**Model:** tinyGPT (ONNX)  
**Evaluation Date:** [Date]  
**Variants per prompt:** 20  
**Seed:** 42

## Test Prompts

1. What is the capital of France?
2. Explain machine learning in simple terms.
3. What are the benefits of exercise?
4. How do I make a cup of coffee?
5. What is the speed of light?
6. Describe the water cycle.
7. What causes earthquakes?

## Stability Scores

| Prompt | Overall Score | Typo | Synonym | Reorder |
|--------|---------------|------|---------|---------|
| Capital of France | 85% | 90% | 85% | 80% |
| Machine learning | 60% | 70% | 55% | 55% |
| Benefits of exercise | 72% | 80% | 75% | 60% |
| Make coffee | 75% | 85% | 70% | 70% |
| Speed of light | 80% | 85% | 80% | 75% |
| Water cycle | 55% | 65% | 50% | 50% |
| Earthquakes | 48% | 60% | 45% | 40% |
| **Average** | **68%** | **78%** | **66%** | **61%** |

## Notable Failure Cases

### Case 1: Typo Sensitivity

**Original:** "What is the capital of France?"  
**Perturbed:** "Waht is the capitsl of France?"  
**Original Output:** "The capital of France is Paris."  
**Perturbed Output:** "I'm not sure what you're asking about."

**Analysis:** Minor typos caused complete failure to answer the question.

### Case 2: Synonym Confusion

**Original:** "Explain machine learning in simple terms."  
**Perturbed:** "Describe machine learning in basic terms."  
**Original Output:** "Machine learning is a type of AI that learns from data."  
**Perturbed Output:** "Machine learning uses algorithms to find patterns in datasets and make predictions."

**Analysis:** Semantically similar prompts produced significantly different explanations—not wrong, but inconsistent.

### Case 3: Word Order Sensitivity

**Original:** "What causes earthquakes?"  
**Perturbed:** "Earthquakes what causes?"  
**Original Output:** "Earthquakes are caused by tectonic plate movement."  
**Perturbed Output:** "Earthquakes are natural disasters that happen suddenly."

**Analysis:** Reordering caused the model to miss the "causes" question entirely.

## Perturbation Analysis

### Typo Handling (78% stable)
The model handles minor character-level perturbations reasonably well. Most typos in common words are tolerated.

### Synonym Recognition (66% stable)  
Performance drops with synonym substitution, suggesting the model is sensitive to exact word choice rather than semantic meaning.

### Word Reordering (61% stable)
This is the weakest area. Non-standard sentence structures often confuse the model, even when meaning is preserved.

## Recommendations

1. **Typo Resilience:**
   - Current handling is acceptable for production
   - Consider input spell-checking as preprocessing

2. **Synonym Understanding:**
   - Train with paraphrased data augmentation
   - Evaluate on paraphrase detection tasks

3. **Syntax Flexibility:**
   - Expose model to varied sentence structures
   - Consider more robust tokenization

4. **Production Safeguards:**
   - Implement input normalization
   - Add confidence thresholds for unstable responses

## Limitations

This evaluation did NOT test:
- Adversarial perturbations designed to exploit model weaknesses
- Multi-sentence or paragraph-level perturbations
- Domain-specific terminology variations
- Perturbations in non-English languages
- Combined perturbation modes
- Character-level perturbations beyond keyboard proximity
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Diversity and representativeness of test prompts
2. Quality of per-perturbation analysis
3. Identification of patterns in failures
4. Actionable recommendations for improving robustness
