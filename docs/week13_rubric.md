# Week 13 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 13 mini-project: **Bug-Fix Evaluation Audit**.

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
| 3 | Code runs without errors, correctly uses CodeEvaluator on 5+ bug-fix examples, runs unit tests, computes pass rates by bug type |
| 2 | Code runs with minor issues but demonstrates understanding of code evaluation |
| 1 | Code has significant errors or evaluates fewer than 5 examples |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] 5+ bug-fix examples with different bug types
- [ ] Each example has buggy code, expected fix, and unit tests
- [ ] CodeEvaluator runs unit tests correctly
- [ ] Pass/fail is determined by test results
- [ ] Pass rate is computed overall and by bug type

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows all examples with bug type, test count, pass/fail status, and pass rate; includes analysis by bug type |
| 2 | Results table is mostly complete but may lack bug type breakdown |
| 1 | Results table is incomplete or missing key metrics |
| 0 | No results provided or results are unusable |

**Expected results format:**

| # | Bug Type | Description | Test Count | Passed | Pass Rate |
|---|----------|-------------|------------|--------|-----------|
| 1 | off-by-one | Sum to N | 3 | ✅ | 100% |
| 2 | wrong-operator | Is Even | 4 | ❌ | 50% |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides executive summary, analyzes which bug types are hardest to fix, identifies failure patterns, provides prompt engineering and test design recommendations |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Analysis of which bug types have highest/lowest pass rates
- Identification of patterns in failures
- Recommendations for improving prompt design
- Discussion of limitations (security, efficiency not tested)

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
- [ ] Bug-Fix Examples Evaluated table
- [ ] Evaluation Results table
- [ ] Analysis by Bug Type
- [ ] Failure Analysis section
- [ ] Recommendations for Prompt Engineering
- [ ] Recommendations for Test Design
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
# Week 13 Mini-Project: Bug-Fix Evaluation Audit

## Executive Summary

Bug-fix evaluation across 7 examples shows a 71% overall pass rate. Off-by-one errors are handled well (100%), while logic errors and missing returns prove more challenging (50% each).

## Bug-Fix Examples Evaluated

| # | Bug Type | Description | Test Count |
|---|----------|-------------|------------|
| 1 | off-by-one | Sum to N (range issue) | 3 |
| 2 | off-by-one | List index bounds | 2 |
| 3 | wrong-operator | Is Even (== 1 vs == 0) | 4 |
| 4 | wrong-comparison | Filter greater (< vs >) | 3 |
| 5 | missing-return | Find max (no return) | 3 |
| 6 | string-order | Reverse string (append vs prepend) | 3 |
| 7 | logic-error | Fibonacci sequence | 3 |

## Evaluation Results

| # | Bug Type | Tests Run | Tests Passed | Pass Rate | Overall |
|---|----------|-----------|--------------|-----------|---------|
| 1 | off-by-one | 3 | 3 | 100% | ✅ |
| 2 | off-by-one | 2 | 2 | 100% | ✅ |
| 3 | wrong-operator | 4 | 4 | 100% | ✅ |
| 4 | wrong-comparison | 3 | 3 | 100% | ✅ |
| 5 | missing-return | 3 | 1 | 33% | ❌ |
| 6 | string-order | 3 | 3 | 100% | ✅ |
| 7 | logic-error | 3 | 0 | 0% | ❌ |

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| Total Examples | 7 |
| Passed | 5 |
| Failed | 2 |
| Overall Pass Rate | 71% |

## Analysis by Bug Type

### Bug Type Performance

| Bug Type | Examples | Passed | Pass Rate |
|----------|----------|--------|-----------|
| off-by-one | 2 | 2 | 100% |
| wrong-operator | 1 | 1 | 100% |
| wrong-comparison | 1 | 1 | 100% |
| string-order | 1 | 1 | 100% |
| missing-return | 1 | 0 | 0% |
| logic-error | 1 | 0 | 0% |

### Patterns Observed

1. **Simple operator/value fixes: 100% success**
   - Changing `==` to `!=`, `<` to `>`, or adjusting constants is well-handled
   
2. **Structural changes: 0% success**
   - Adding missing return statements required understanding program flow
   - Model added return in wrong location

3. **Multi-step logic: 0% success**
   - Fibonacci fix required understanding recursive relationship
   - Model's fix was syntactically valid but semantically wrong

## Failure Analysis

### Case 1: Missing Return (0% pass rate)

**Buggy Code:**
```python
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    # Bug: missing return
```

**Model's Fix:**
```python
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            return item  # Wrong: returns first larger, not max
    return max_val
```

**Issue:** Model misunderstood the algorithm's intent, adding a return inside the loop instead of after.

### Case 2: Logic Error in Fibonacci (0% pass rate)

**Buggy Code:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-3)  # Bug: n-3 should be n-2
```

**Model's Fix:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n) + fibonacci(n-1)  # Wrong: causes infinite recursion
```

**Issue:** Model changed the wrong operand and created an infinite loop.

## Recommendations

### For Prompt Engineering

1. **Include expected behavior explicitly:**
   - Add "The function should return X when given input Y"
   - Provide input/output examples in the prompt

2. **Describe the bug location:**
   - "The bug is in the loop/return/condition" helps focus attention

3. **Ask for explanation first:**
   - "First explain the bug, then provide the fix" improves accuracy

### For Test Design

1. **Include edge cases:**
   - Empty inputs, single elements, negative numbers
   - Boundary conditions (0, 1, max values)

2. **Test intermediate values:**
   - Not just final result, but algorithmic correctness

3. **Add assertion messages:**
   - Clearer test failures help debugging

## Limitations

### What This Evaluation Cannot Assess

1. **Security vulnerabilities:** Generated code may introduce injection, overflow, etc.
2. **Efficiency:** Fixes may be O(n²) when O(n) is possible
3. **Code style:** Readability, naming conventions not evaluated
4. **Real-world complexity:** Synthetic bugs simpler than production issues

### Future Improvements

1. Add security scanning with bandit or similar
2. Include time/space complexity analysis
3. Test on larger, multi-file codebases
4. Evaluate fix explanations alongside code
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Diversity of bug types tested
2. Quality of unit test coverage
3. Depth of failure analysis
4. Practicality of prompt engineering recommendations
