# Week 15 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 15 mini-project: **RAG Evaluation Audit**.

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
| 3 | Code runs without errors, implements FAQ corpus with 8+ entries, uses vector store for retrieval, evaluates both retrieval quality and groundedness |
| 2 | Code runs with minor issues but demonstrates understanding of RAG evaluation |
| 1 | Code has significant errors or has fewer than 8 FAQ entries |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] FAQ corpus with 8+ entries across 3+ categories
- [ ] Vector store implementation (SimpleVectorStore or FAISS)
- [ ] 5+ test questions with expected FAQs
- [ ] Retrieval metrics computed (precision, recall, hit rate)
- [ ] Groundedness evaluated (string match and/or LLM judge)

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results show retrieval metrics per question, groundedness scores, includes both retrieval and generation analysis |
| 2 | Results table is mostly complete but may lack groundedness detail |
| 1 | Results table is incomplete or missing key metrics |
| 0 | No results provided or results are unusable |

**Expected results format:**

### Retrieval Evaluation
| # | Question | Expected FAQ | Retrieved FAQ | Hit? |
|---|----------|--------------|---------------|------|
| 1 | Refund? | faq_001 | faq_001 | ✅ |

### Groundedness
| # | Expected Phrases | Matched | Score |
|---|------------------|---------|-------|
| 1 | [30 days, refund] | 2/2 | 100% |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides executive summary, analyzes retrieval vs generation failures, identifies patterns, provides recommendations for retrieval, generation, and knowledge base |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Analysis of retrieval hit rate vs groundedness correlation
- Identification of retrieval failure patterns
- Discussion of hallucination or groundedness issues
- Recommendations for improving each RAG component

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file following the specified format, includes FAQ corpus description, all sections complete, professional formatting |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Executive Summary
- [ ] FAQ Corpus (categories and coverage)
- [ ] Retrieval Evaluation table and metrics
- [ ] Generation Evaluation / Groundedness Analysis
- [ ] Failure Analysis (retrieval and groundedness)
- [ ] Recommendations for each component
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
# Week 15 Mini-Project: RAG Evaluation Audit

## Executive Summary

RAG system evaluation shows 85% retrieval hit rate and 78% average groundedness. Main issues are semantic mismatch in retrieval and incomplete key phrase inclusion in generation.

## FAQ Corpus

### Categories and Coverage

| Category | # FAQs | Example Question |
|----------|--------|------------------|
| Returns | 2 | What is your return policy? |
| Shipping | 3 | How long does shipping take? |
| Orders | 2 | How do I track my order? |
| Payment | 1 | What payment methods accepted? |

### Total FAQs: 8

### FAQ Details

| ID | Category | Question Summary |
|----|----------|------------------|
| faq_001 | returns | Return policy (30 days, full refund) |
| faq_002 | shipping | Standard/Express shipping times |
| faq_003 | orders | Track order (Order History, email) |
| faq_004 | payment | Accepted payments (Visa, PayPal, etc.) |
| faq_005 | orders | Cancel order (1 hour window) |
| faq_006 | shipping | International shipping (50 countries) |
| faq_007 | returns | Damaged item (48 hours, photos) |
| faq_008 | shipping | Change address (account settings) |

## Retrieval Evaluation

### Test Questions

| # | Question | Expected FAQ | Retrieved FAQ (Top-1) | Hit? |
|---|----------|--------------|----------------------|------|
| 1 | Can I get a refund? | faq_001 | faq_001 | ✅ |
| 2 | How fast is delivery? | faq_002 | faq_002 | ✅ |
| 3 | Where's my order status? | faq_003 | faq_003 | ✅ |
| 4 | Can I pay with PayPal? | faq_004 | faq_004 | ✅ |
| 5 | My package arrived broken | faq_007 | faq_001 | ❌ |
| 6 | Ship to Canada? | faq_006 | faq_002 | ❌ |
| 7 | Cancel my order | faq_005 | faq_005 | ✅ |

### Retrieval Metrics

| Metric | Value |
|--------|-------|
| Average Precision | 71% |
| Average Recall | 71% |
| Hit Rate (Top-1) | 71% (5/7) |
| Hit Rate (Top-3) | 86% (6/7) |

## Generation Evaluation

### Sample Outputs

| # | Question | Answer Excerpt | Grounded? |
|---|----------|----------------|-----------|
| 1 | Refund? | "...within 30 days for a full refund..." | ✅ |
| 2 | Delivery? | "...5-7 business days for standard..." | ✅ |
| 3 | Order status? | "...Order History section..." | ✅ |
| 4 | PayPal? | "...accept PayPal and major cards..." | ✅ |
| 5 | Broken package | "...contact for replacement..." | ⚠️ |
| 6 | Canada shipping? | "...we offer shipping options..." | ❌ |
| 7 | Cancel order | "...within 1 hour of placing..." | ✅ |

## Groundedness Analysis

### String Match Results

| # | Question | Expected Phrases | Matched | Score |
|---|----------|------------------|---------|-------|
| 1 | Refund? | [30 days, refund] | 2/2 | 100% |
| 2 | Delivery? | [5-7 business days, express] | 2/2 | 100% |
| 3 | Order status? | [Order History, account] | 2/2 | 100% |
| 4 | PayPal? | [PayPal, accept] | 2/2 | 100% |
| 5 | Broken | [48 hours, replacement, photos] | 1/3 | 33% |
| 6 | Canada? | [international, 50 countries] | 0/2 | 0% |
| 7 | Cancel | [1 hour, cancel] | 2/2 | 100% |

### Overall Groundedness Score: 76%

### LLM Judge Results (Optional)

| # | Grounded | Hallucination | Notes |
|---|----------|---------------|-------|
| 1 | ✅ | ❌ | Well-grounded |
| 2 | ✅ | ❌ | Well-grounded |
| 5 | ⚠️ | ⚠️ | Missing key details |
| 6 | ❌ | ✅ | Generic response, not from FAQ |

## Failure Analysis

### Retrieval Failures

**Case 1: "My package arrived broken"**
- **Expected:** faq_007 (Damaged item)
- **Retrieved:** faq_001 (Return policy)
- **Cause:** "broken" → "return" semantic association, but faq_007 uses "damaged" terminology
- **Fix:** Add synonym mapping or improve embeddings

**Case 2: "Ship to Canada?"**
- **Expected:** faq_006 (International shipping)
- **Retrieved:** faq_002 (Standard shipping)
- **Cause:** "ship" keyword matched standard shipping over international
- **Fix:** Include country names in FAQ keywords

### Groundedness Failures

**Case 1: Broken Package (33% grounded)**
- Retrieved wrong FAQ, so generated generic response
- Missed: 48 hours deadline, photos requirement
- Root cause: Retrieval failure cascaded to generation

**Case 2: Canada Shipping (0% grounded)**
- Retrieved domestic shipping FAQ
- Answer was technically correct but didn't address international
- Root cause: Retrieval failure

## Recommendations

### For Retrieval

1. **Improve synonym coverage:**
   - Add "broken" = "damaged" mapping
   - Include country names in international FAQ keywords

2. **Use better embeddings:**
   - Replace bag-of-words with sentence transformers
   - Fine-tune on customer service domain

3. **Increase top-k:**
   - Retrieve top-3 instead of top-1
   - Let generation model pick most relevant

### For Generation

1. **Enforce grounding:**
   - Prompt: "Only answer using provided context"
   - Add confidence check before responding

2. **Handle retrieval failures:**
   - Detect low retrieval scores
   - Respond "I need to check on that" instead of guessing

3. **Include citations:**
   - Show which FAQ the answer came from
   - Helps users verify information

### For Knowledge Base

1. **Add synonym lists:**
   - broken/damaged, refund/return, etc.

2. **Standardize terminology:**
   - Consistent use of key terms across FAQs

3. **Regular updates:**
   - Schedule FAQ review and expansion

## Limitations

### What This Evaluation Cannot Assess

1. **Real user satisfaction:** No actual user feedback
2. **Edge cases:** Only tested 7 questions
3. **Long-term consistency:** Single evaluation run
4. **Multi-turn conversations:** Only single-turn Q&A

### Future Improvements

1. Test with 50+ diverse questions
2. Add user satisfaction surveys
3. Implement conversation-level evaluation
4. Test with different embedding models
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Quality and diversity of FAQ corpus
2. Distinction between retrieval and generation failures
3. Depth of groundedness analysis
4. Practical recommendations for each RAG component
