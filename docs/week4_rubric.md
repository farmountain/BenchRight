# Week 4 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 4 mini-project: **Design Your Own 10-Question Benchmark**.

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
| 3 | Code runs without errors, correctly implements a custom benchmark with 10 questions, includes expected answers, and evaluates tinyGPT on all questions |
| 2 | Code runs with minor issues but demonstrates understanding of benchmark design |
| 1 | Code has significant errors or has fewer than 10 questions |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Capability to test is clearly defined
- [ ] 10 questions with expected answers are provided
- [ ] Model is evaluated on all questions
- [ ] Scoring method (exact match or appropriate metric) is implemented

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows all 10 questions with model outputs, expected answers, and pass/fail status; includes overall accuracy |
| 2 | Results table has all questions but may lack detail |
| 1 | Results table is incomplete or missing questions |
| 0 | No results table provided or results are unusable |

**Expected results format:**

| # | Question | Expected | Model Output | Pass? |
|---|----------|----------|--------------|-------|
| 1 | [Question] | [Answer] | [Output] | ✅/❌ |
| ... | ... | ... | ... | ... |
| 10 | [Question] | [Answer] | [Output] | ✅/❌ |

**Overall: X/10 (XX%)**

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides thoughtful analysis of which questions the model got right/wrong, identifies patterns in failures, and discusses what the benchmark reveals about model capabilities |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Analysis of which question types were easy vs difficult
- Patterns in failures (e.g., math, reasoning, specific knowledge)
- Comparison to expected performance
- Suggestions for improving the benchmark

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file, clear description of capability tested, proper formatting, includes benchmark design rationale |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Clear title and date
- [ ] Description of capability being tested
- [ ] Rationale for question selection
- [ ] Results table
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
# Week 4 Mini-Project: Custom Benchmark

**Author:** [Your Name]  
**Date:** [Date]  
**Model:** tinyGPT (ONNX)

## Capability Tested

**Basic Arithmetic:** Testing whether tinyGPT can perform simple addition, subtraction, multiplication, and division.

## Benchmark Design Rationale

Arithmetic is a fundamental capability that reveals:
- Whether the model memorized math facts
- Ability to follow simple computational rules
- Consistency across different number ranges

I included a mix of:
- Simple single-digit operations (easy)
- Double-digit operations (medium)
- Operations with zero and negative numbers (edge cases)

## Benchmark Questions

| # | Question | Expected Answer | Difficulty |
|---|----------|-----------------|------------|
| 1 | What is 2 + 3? | 5 | Easy |
| 2 | What is 7 - 4? | 3 | Easy |
| 3 | What is 6 × 3? | 18 | Easy |
| 4 | What is 15 ÷ 3? | 5 | Easy |
| 5 | What is 25 + 17? | 42 | Medium |
| 6 | What is 50 - 23? | 27 | Medium |
| 7 | What is 12 × 8? | 96 | Medium |
| 8 | What is 100 ÷ 4? | 25 | Medium |
| 9 | What is 0 + 5? | 5 | Edge |
| 10 | What is 10 - 10? | 0 | Edge |

## Results

| # | Question | Expected | Model Output | Pass? |
|---|----------|----------|--------------|-------|
| 1 | 2 + 3? | 5 | "5" | ✅ |
| 2 | 7 - 4? | 3 | "3" | ✅ |
| 3 | 6 × 3? | 18 | "18" | ✅ |
| 4 | 15 ÷ 3? | 5 | "5" | ✅ |
| 5 | 25 + 17? | 42 | "42" | ✅ |
| 6 | 50 - 23? | 27 | "23" | ❌ |
| 7 | 12 × 8? | 96 | "84" | ❌ |
| 8 | 100 ÷ 4? | 25 | "25" | ✅ |
| 9 | 0 + 5? | 5 | "5" | ✅ |
| 10 | 10 - 10? | 0 | "0" | ✅ |

**Overall: 8/10 (80%)**

## Interpretation

### Performance by Difficulty

| Difficulty | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| Easy | 4 | 4 | 100% |
| Medium | 2 | 4 | 50% |
| Edge | 2 | 2 | 100% |

### Key Findings

1. **Single-digit arithmetic is strong:** All easy questions passed, suggesting memorization of basic facts.

2. **Medium difficulty reveals limits:** 50-23 and 12×8 failed, showing multi-digit computation is unreliable.

3. **Edge cases handled well:** Zero operations were correct, indicating some rule understanding.

### Failure Analysis

- **50 - 23 = "23":** Model may have repeated part of the input instead of computing
- **12 × 8 = "84":** Close to correct (96) but wrong—possible partial computation

### Benchmark Improvements

1. Add more double-digit operations to test limits
2. Include larger numbers to find breaking points
3. Test multi-step arithmetic (e.g., "2 + 3 × 4")
4. Add word problems to test reasoning

### Comparison to Industry Benchmarks

My 80% accuracy on simple arithmetic contrasts with:
- GSM8K (multi-step math): ~2% for tinyGPT
- This confirms the model handles simple operations but not complex reasoning
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Quality and thoughtfulness of benchmark design
2. Diversity of questions (easy, medium, edge cases)
3. Clarity of results presentation
4. Depth of interpretation and failure analysis
