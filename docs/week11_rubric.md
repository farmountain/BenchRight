# Week 11 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 11 mini-project: **Compliance Summarization Audit**.

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
| 3 | Code runs without errors, correctly uses ComplianceJudge on 5+ regulatory texts, captures correctness and completeness scores, identifies omitted details |
| 2 | Code runs with minor issues but demonstrates understanding of compliance evaluation |
| 1 | Code has significant errors or evaluates fewer than 5 texts |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] 5+ regulatory text examples are defined
- [ ] Both good and poor summaries are included for comparison
- [ ] ComplianceJudge evaluates correctness and completeness
- [ ] Omitted details are captured
- [ ] Combined scores are computed

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows all evaluations with correctness, completeness, combined scores, and omitted details, includes omission analysis |
| 2 | Results table is mostly complete but may lack omission analysis |
| 1 | Results table is incomplete or missing key metrics |
| 0 | No results provided or results are unusable |

**Expected results format:**

| # | Correctness | Completeness | Combined | Omitted Details |
|---|-------------|--------------|----------|-----------------|
| 1 | 0.95 | 0.80 | 0.88 | Retention period |
| 2 | 0.60 | 0.40 | 0.50 | Threshold, deadline |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides executive summary, analyzes most commonly omitted detail types, recommends acceptable thresholds, discusses when human review is required |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Pattern analysis of commonly omitted details
- Threshold recommendations with rationale
- Risk assessment for using summaries without review
- Discussion of when human review is essential

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
- [ ] Regulatory Texts Evaluated table
- [ ] Evaluation Results table
- [ ] Omission Analysis (most commonly omitted details)
- [ ] Recommended Thresholds
- [ ] Required Human Review guidance
- [ ] Risk Assessment

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
# Week 11 Mini-Project: Compliance Summarization Audit

## Executive Summary

Compliance summarization evaluation reveals that LLM summaries achieve 85% correctness on average but only 68% completeness. Monetary thresholds and deadlines are most commonly omitted, posing significant compliance risk.

## Regulatory Texts Evaluated

| # | Type | Critical Details Count | Source |
|---|------|------------------------|--------|
| 1 | KYC | 5 | Customer ID verification |
| 2 | AML/SAR | 4 | Suspicious activity reporting |
| 3 | Data Retention | 4 | BSA requirements |
| 4 | Consumer Protection | 3 | Disclosure requirements |
| 5 | Privacy | 5 | GDPR-style requirements |

## Evaluation Results

| # | Type | Correctness | Completeness | Combined | Omitted Details |
|---|------|-------------|--------------|----------|-----------------|
| 1 | KYC (Good) | 0.95 | 0.90 | 0.93 | None |
| 2 | KYC (Poor) | 0.70 | 0.40 | 0.55 | Retention period, penalty |
| 3 | SAR (Good) | 0.90 | 0.85 | 0.88 | Filing deadline accuracy |
| 4 | SAR (Poor) | 0.50 | 0.25 | 0.38 | Threshold ($5k), timeline (30 days), non-disclosure |
| 5 | Data Retention | 0.85 | 0.70 | 0.78 | 24-hour retrieval requirement |
| 6 | Consumer | 0.92 | 0.88 | 0.90 | Timing of disclosure |
| 7 | Privacy | 0.88 | 0.75 | 0.82 | Data subject rights deadline |

## Omission Analysis

### Most Commonly Omitted Details

| Detail Type | Frequency | Risk Level |
|-------------|-----------|------------|
| Monetary thresholds | 4/7 (57%) | HIGH |
| Time limits/deadlines | 4/7 (57%) | HIGH |
| Penalty amounts | 3/7 (43%) | MEDIUM |
| Specific requirements | 2/7 (29%) | MEDIUM |
| Procedural details | 2/7 (29%) | LOW |

### Patterns Observed

1. **Numerical values are frequently lost:** The LLM tends to summarize away specific numbers ($5,000, 30 days, 5 years), replacing them with vague language ("promptly," "significant amount").

2. **Consequences are deprioritized:** Penalty information is often omitted, reducing perceived urgency of compliance.

3. **"Must" becomes "should":** Mandatory requirements are sometimes softened in summaries, which could lead to non-compliance.

4. **Context dependencies omitted:** Conditions and exceptions are frequently left out.

## Recommended Thresholds

### Acceptable Score Thresholds

| Dimension | Minimum Score | Rationale |
|-----------|---------------|-----------|
| Correctness | 0.85 | Must not misrepresent requirements |
| Completeness | 0.80 | Must include critical details |
| Combined | 0.82 | Overall quality floor |

### Threshold Justification

- **Correctness at 0.85:** Any score below indicates potential misrepresentation that could lead to compliance violations
- **Completeness at 0.80:** Missing >20% of details likely means missing at least one critical element
- **Combined at 0.82:** Provides buffer for minor omissions while catching serious gaps

## Required Human Review

Human compliance review is **MANDATORY** when:

1. **Correctness < 0.90:** Any potential misrepresentation requires expert verification
2. **Completeness < 0.85:** Missing details must be identified and added
3. **Any monetary value omitted:** All thresholds must be explicit
4. **Any deadline omitted:** All time requirements must be stated
5. **New or updated regulations:** First summarization of any regulation

Human review is **RECOMMENDED** when:
- Combined score is 0.82-0.90
- Regulation involves enforcement actions
- Summary will be used for training purposes

## Risk Assessment

### Risk Level: MODERATE to HIGH

**If these summaries were used without review:**

1. **Regulatory penalties:** Missing thresholds could lead to unreported transactions
2. **Audit findings:** Incomplete procedures would fail compliance audits
3. **Legal liability:** Misrepresented requirements could expose institution
4. **Reputational damage:** Compliance failures become public record

### Mitigation Requirements

1. Never deploy compliance summaries without human review
2. Use summaries as starting point, not final output
3. Maintain links to source regulatory text
4. Require sign-off from compliance officer
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Diversity and realism of regulatory text examples
2. Quality of omission pattern analysis
3. Practicality of threshold recommendations
4. Clarity of human review requirements
