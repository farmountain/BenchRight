# Week 6 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 6 mini-project: **Building an LLM-as-Judge Pipeline**.

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
| 3 | Code runs without errors, correctly uses LLMJudge to evaluate at least 10 QA pairs, compares LLM-as-Judge scores with exact match scores, and produces valid results |
| 2 | Code runs with minor issues but demonstrates understanding of LLM-as-Judge |
| 1 | Code has significant errors or evaluates fewer than 10 QA pairs |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] LLMJudge is correctly initialized (or mock client used)
- [ ] At least 10 QA pairs are evaluated
- [ ] Both exact_match and LLM-as-Judge scores are computed
- [ ] Results include score and rationale from judge

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows all QA pairs with both metric scores, includes cases where metrics disagree, rationales are captured |
| 2 | Results table is mostly complete but may lack some comparisons |
| 1 | Results table is incomplete or missing metric comparison |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| # | Question | Reference | Model Answer | Exact Match | LLM-Judge | Rationale |
|---|----------|-----------|--------------|-------------|-----------|-----------|
| 1 | ... | ... | ... | 0.0/1.0 | X.XX | "..." |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides detailed analysis of 2-3 disagreement cases, explains when each metric is preferred, discusses LLM-as-Judge limitations |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Specific examples where exact match = 0 but LLM-Judge score is high
- Specific examples where exact match = 1 but LLM-Judge score is lower
- Discussion of when to use each metric type
- Understanding of LLM-as-Judge biases and limitations

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file following the specified format in week6.md, clear sections, proper formatting |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Dataset section with question/reference table
- [ ] Model outputs and scores table
- [ ] Analysis of cases where metrics disagree
- [ ] Key insight paragraph on when to use each metric

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
# Week 6 Mini-Project: LLM-as-Judge Analysis

## Dataset

| # | Question | Reference |
|---|----------|-----------|
| 1 | What is the capital of France? | Paris |
| 2 | What is 2+2? | 4 |
| 3 | Who wrote Romeo and Juliet? | Shakespeare |
| 4 | What is the chemical symbol for water? | H2O |
| 5 | How many continents are there? | 7 |
| 6 | What is the largest planet? | Jupiter |
| 7 | Who painted the Mona Lisa? | Leonardo da Vinci |
| 8 | What is the speed of light? | 299,792,458 m/s |
| 9 | What year did WW2 end? | 1945 |
| 10 | What is the capital of Japan? | Tokyo |

## Model Outputs and Scores

| # | Model Answer | Exact Match | LLM-Judge | Rationale |
|---|--------------|-------------|-----------|-----------|
| 1 | The capital of France is Paris | 0.0 | 0.95 | "Correct answer with complete sentence" |
| 2 | Four | 0.0 | 0.90 | "Correct answer, different format" |
| 3 | William Shakespeare wrote it | 0.0 | 0.95 | "Correct with additional context" |
| 4 | H2O | 1.0 | 1.00 | "Perfect match" |
| 5 | There are 7 continents | 0.0 | 0.95 | "Correct with explanation" |
| 6 | Jupiter | 1.0 | 1.00 | "Perfect match" |
| 7 | Da Vinci | 0.0 | 0.85 | "Correct but abbreviated name" |
| 8 | About 300,000 km/s | 0.0 | 0.80 | "Approximately correct, different units" |
| 9 | 1945 | 1.0 | 1.00 | "Perfect match" |
| 10 | Tokyo is Japan's capital | 0.0 | 0.95 | "Correct with context" |

## Summary

| Metric | Mean Score |
|--------|------------|
| Exact Match | 0.30 |
| LLM-Judge | 0.935 |

## Analysis

### Cases Where Metrics Disagree

**Case 1: "The capital of France is Paris" (Exact: 0.0, LLM: 0.95)**
The model provided the correct answer but as a complete sentence rather than a single word. LLM-as-Judge correctly identifies the semantic equivalence, while exact match fails.

**Case 2: "About 300,000 km/s" (Exact: 0.0, LLM: 0.80)**
This is interesting—the answer is approximately correct but uses different units and precision. LLM-as-Judge gave 0.80, acknowledging partial correctness but penalizing the approximation.

**Case 3: "Da Vinci" (Exact: 0.0, LLM: 0.85)**
The abbreviated name is commonly used but doesn't match the reference. LLM-as-Judge recognizes it as the same person but slightly penalizes the informal form.

### Key Insight

**When to use Exact Match:**
- Short, unambiguous answers (numbers, single words)
- High-stakes applications requiring precise formatting
- Benchmarks with standardized answer formats

**When to use LLM-as-Judge:**
- Long-form generation where phrasing varies
- Semantic correctness matters more than format
- Partial credit is appropriate
- Evaluating reasoning or explanation quality

**Limitations of LLM-as-Judge:**
- Potential bias toward verbose or confident-sounding answers
- May be too lenient on approximately correct answers
- Depends on judge model quality
- More expensive and slower than exact match
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Quality and diversity of QA pairs used
2. Depth of analysis on disagreement cases
3. Understanding of when to use each metric type
4. Recognition of LLM-as-Judge limitations and biases
