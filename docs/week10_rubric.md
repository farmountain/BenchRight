# Week 10 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 10 mini-project: **Regression Audit Report comparing two model versions**.

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
| 3 | Code runs without errors, correctly uses compare_runs and summarize_regressions on 10-20 test prompts, handles both quality and performance metrics properly |
| 2 | Code runs with minor issues but demonstrates understanding of regression analysis |
| 1 | Code has significant errors or uses fewer than 10 test cases |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] 10-20 test prompts are defined
- [ ] Baseline and new model results are simulated or captured
- [ ] compare_runs merges DataFrames correctly
- [ ] summarize_regressions identifies performance drops
- [ ] Both "higher is better" and "lower is better" metrics are handled

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results include regression summary table, notable regressions (>10% and 5-10%), identified improvements, and clear pass/fail decision |
| 2 | Results table is mostly complete but may lack severity breakdown |
| 1 | Results table is incomplete or missing key comparisons |
| 0 | No results provided or results are unusable |

**Expected results format:**

### Regression Summary
| Metric | Direction | Regressions | Rate | Max Severity |
|--------|-----------|-------------|------|--------------|
| score | ↑ better | ? | ?% | ? |
| latency_ms | ↓ better | ? | ?% | ? |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides executive summary, analyzes severe vs moderate regressions, notes improvements, makes clear deployment recommendation with rationale |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Clear executive summary of regression posture
- Analysis of severe regressions (>10%) with investigation notes
- Trade-off analysis: improvements vs regressions
- Deployment decision with clear rationale
- Risk assessment for identified regressions

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
- [ ] Models Compared (baseline and new)
- [ ] Test Configuration
- [ ] Regression Summary table
- [ ] Severe Regressions (>10%) section
- [ ] Moderate Regressions (5-10%) section
- [ ] Improvements section
- [ ] Deployment Decision checkbox
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
# Week 10 Mini-Project: Regression Audit Report

## Executive Summary

The new model version (v1.2) shows a net improvement with 15% average score increase but introduces 3 regressions in edge cases. Recommend investigation before deployment.

## Models Compared

| Property | Baseline | New Model |
|----------|----------|-----------|
| Version | v1.1 | v1.2 |
| Training Date | 2024-01-15 | 2024-02-01 |
| Parameters | 10M | 10M |
| Changes | - | Added instruction tuning |

## Test Configuration

- Test cases: 15
- Metrics analyzed: score, latency_ms
- Seed: 42

## Regression Summary

| Metric | Direction | Total Cases | Regressions | Rate | Max Severity |
|--------|-----------|-------------|-------------|------|--------------|
| score | ↑ better | 15 | 3 | 20% | -15.3% |
| latency_ms | ↓ better | 15 | 2 | 13% | +18.5% |

## Detailed Comparison

| Prompt | Score (v1.1) | Score (v1.2) | Δ Score | Latency v1.1 | Latency v1.2 | Δ Latency |
|--------|--------------|--------------|---------|--------------|--------------|-----------|
| Capital of France? | 0.95 | 0.98 | +3.2% | 45ms | 42ms | -6.7% |
| Explain quantum physics | 0.72 | 0.85 | +18.1% | 62ms | 58ms | -6.5% |
| Write a poem | 0.80 | 0.68 | -15.0% | 55ms | 65ms | +18.2% |
| Math: 2+2 | 1.00 | 1.00 | 0% | 38ms | 38ms | 0% |
| Summarize article | 0.65 | 0.78 | +20.0% | 70ms | 72ms | +2.9% |
| Code debugging | 0.70 | 0.58 | -17.1% | 68ms | 75ms | +10.3% |
| ...remaining 9 prompts... | ... | ... | ... | ... | ... | ... |

## Severe Regressions (> 10% degradation)

### 1. Code Debugging (-17.1% score, +10.3% latency)
**Prompt:** "Find the bug in this Python code..."  
**Baseline output:** Correctly identified off-by-one error  
**New output:** Missed the bug, suggested unrelated changes  
**Root cause hypothesis:** Instruction tuning may have diluted code understanding  
**Priority:** HIGH

### 2. Poetry Writing (-15.0% score, +18.2% latency)
**Prompt:** "Write a short poem about nature"  
**Baseline output:** Creative, well-structured poem  
**New output:** Repetitive, less creative  
**Root cause hypothesis:** Instruction tuning optimized for factual over creative tasks  
**Priority:** MEDIUM

## Moderate Regressions (5-10% degradation)

### 1. Translation Task (-8.2% score)
**Prompt:** "Translate to French..."  
**Note:** Minor accuracy drop in idiomatic expressions  
**Priority:** LOW

## Improvements

| Prompt Type | Improvement | Notes |
|-------------|-------------|-------|
| Factual QA | +12.5% avg | Instruction tuning helped clarity |
| Summarization | +20.0% | Much better structure |
| Math reasoning | +15.3% | More step-by-step explanations |

## Decision

- [ ] Deploy new model - Not yet
- [x] Investigate regressions first
- [ ] Roll back / reject update

**Rationale:** While average metrics improved, the severe regression in code debugging is concerning for our developer-focused use case. Recommend:
1. Root cause analysis on code debugging samples
2. Targeted fine-tuning to address creative writing
3. Re-evaluate after fixes

## Limitations

This analysis did NOT cover:
- Edge cases beyond 15 test prompts
- Long-form generation comparison
- Multi-turn conversation regression
- Performance under load
- A/B test with real users
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Quality of baseline vs new model comparison
2. Identification and prioritization of regressions
3. Balance of improvements vs regressions analysis
4. Clarity and justification of deployment decision
