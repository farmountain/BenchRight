# Week 16 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 16 mini-project: **Creative Content Evaluation Audit**.

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
| 3 | Code runs without errors, correctly uses CreativeContentJudge on 5+ content samples, evaluates all three dimensions (brand voice, clarity, CTA), computes aggregate metrics |
| 2 | Code runs with minor issues but demonstrates understanding of creative evaluation |
| 1 | Code has significant errors or evaluates fewer than 5 samples |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Detailed brand guidelines defined (voice, tone, values)
- [ ] 5+ content samples with varied quality
- [ ] CreativeContentJudge evaluates brand_voice_score, clarity_score, cta_score
- [ ] Rationales are captured
- [ ] Aggregate metrics computed

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows all content with three dimension scores, overall score, rationales for best/worst, and aggregate metrics |
| 2 | Results table is mostly complete but may lack rationales |
| 1 | Results table is incomplete or missing key metrics |
| 0 | No results provided or results are unusable |

**Expected results format:**

| # | Product | Brand Voice | Clarity | CTA | Overall |
|---|---------|-------------|---------|-----|---------|
| 1 | ... | 4/5 | 5/5 | 4/5 | 4.3/5 |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides executive summary, identifies patterns in high/low scoring content, analyzes strongest/weakest dimensions, provides actionable recommendations |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Pattern analysis of what makes content score high/low
- Identification of which dimension is weakest across samples
- Actionable recommendations for content creators
- Discussion of evaluation limitations

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file following the specified format, includes brand guidelines, all sections complete, professional formatting |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Executive Summary
- [ ] Brand Guidelines Summary
- [ ] Content Samples table
- [ ] Evaluation Results with rationales
- [ ] Aggregate Metrics
- [ ] Analysis (patterns, strongest/weakest dimensions)
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
# Week 16 Mini-Project: Creative Content Evaluation Audit

## Executive Summary

Creative content evaluation across 6 samples shows an average overall score of 3.5/5. Brand voice alignment is strongest (3.8/5) while CTA strength is weakest (3.0/5), indicating writers understand the brand but struggle with calls-to-action.

## Brand Guidelines Summary

### Brand Profile

| Element | Description |
|---------|-------------|
| Name | GreenLeaf |
| Industry | Organic Skincare |
| Voice | Nurturing, knowledgeable, genuine |
| Target Audience | Health-conscious women 25-45 |

### Key Voice Elements

1. **Tone:** Educational but accessible, like a trusted friend who's a skincare expert
2. **Values:** Natural ingredients, cruelty-free, sustainability
3. **Vocabulary:** Use "nourishing," "botanical," "pure"; Avoid "chemical," "anti-aging," "miracle"
4. **CTA Style:** Inviting rather than pushy, focus on self-care journey

## Content Samples

### Overview

| # | Product/Topic | Type | Length |
|---|---------------|------|--------|
| 1 | Rosehip Serum | Product description | 65 words |
| 2 | Vitamin C Cream | Product description | 45 words |
| 3 | Lavender Night Cream | Product description | 50 words |
| 4 | Green Tea Cleanser | Product description | 70 words |
| 5 | Newsletter Subject | Email subject | 10 words |
| 6 | Social Media Post | Social post | 30 words |

## Evaluation Results

### Individual Scores

| # | Product | Brand Voice | Clarity | CTA | Overall |
|---|---------|-------------|---------|-----|---------|
| 1 | Rosehip Serum | 5/5 | 5/5 | 4/5 | 4.7/5 |
| 2 | Vitamin C Cream | 1/5 | 2/5 | 2/5 | 1.7/5 |
| 3 | Lavender Night Cream | 2/5 | 1/5 | 1/5 | 1.3/5 |
| 4 | Green Tea Cleanser | 5/5 | 4/5 | 5/5 | 4.7/5 |
| 5 | Newsletter Subject | 4/5 | 5/5 | 3/5 | 4.0/5 |
| 6 | Social Media Post | 4/5 | 4/5 | 3/5 | 3.7/5 |

### Score Rationales (Selected)

#### Highest Scoring: Rosehip Serum (4.7/5)

- **Brand Voice (5/5):** Perfect use of brand vocabulary ("nourishing botanical blend," "pure, gentle, effective"). Tone is nurturing and knowledgeable. Feels authentic to GreenLeaf.

- **Clarity (5/5):** Immediately understandable benefits (hydrate, even skin tone). Well-structured with clear product information. No jargon.

- **CTA (4/5):** "Ready to begin your journey to healthier skin?" is inviting but could be more specific about next steps.

#### Lowest Scoring: Lavender Night Cream (1.3/5)

- **Brand Voice (2/5):** Uses scientific jargon ("topical emollient formulation," "epidermal application," "stratum corneum") completely opposite to accessible brand voice. Feels like a medical document.

- **Clarity (1/5):** Incomprehensible to target audience. Requires scientific background. "Transepidermal water loss reduction" is not consumer-friendly.

- **CTA (1/5):** No call-to-action at all. Doesn't invite any customer engagement.

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| Average Brand Voice | 3.5/5 |
| Average Clarity | 3.5/5 |
| Average CTA Strength | 3.0/5 |
| Average Overall | 3.3/5 |
| Total Evaluated | 6 |

## Analysis

### Strongest Dimension: Brand Voice (3.5/5)

When content is good, it's very on-brand. The best samples use vocabulary guidelines perfectly ("nourishing," "botanical," "pure"). Issues only arise when writers default to technical/scientific language.

### Weakest Dimension: CTA Strength (3.0/5)

Even good content often has weak CTAs. Common issues:
- Questions that don't specify action ("Ready for better skin?")
- Missing urgency or incentive
- Vague "learn more" instead of specific next step

### Patterns in High-Scoring Content

1. **Uses approved vocabulary list:** "Nourishing," "botanical," "gentle," "pure"
2. **Addresses customer directly:** "Your skin," "your journey"
3. **Focuses on benefits, not features:** What it does FOR you, not what it IS
4. **Specific and inviting CTA:** "Take our free skin quiz"

### Patterns in Low-Scoring Content

1. **Technical jargon:** Scientific terms inappropriate for consumer brand
2. **Feature-focused:** Lists ingredients without benefits
3. **No personality:** Generic text that could be any brand
4. **Missing or weak CTA:** No clear next step for customer

## Recommendations

### For Content Creators

1. **Reference vocabulary guide:** Before writing, review approved/avoided terms
2. **CTA formula:** Include WHAT to do + WHY now + HOW easy
   - Example: "Take our 2-minute skin quiz to get your personalized routine"
3. **Read aloud test:** Content should sound like friendly expert, not textbook
4. **Benefit-first writing:** Start with what customer gets, then explain how

### For Brand Guidelines

1. **Add CTA examples:** Current guidelines focus on tone but lack CTA guidance
2. **Include anti-patterns:** Show what NOT to do alongside good examples
3. **Create content templates:** Provide starting points for common formats

### For Evaluation Process

1. **Weight CTA higher:** Currently weakest area, needs more focus
2. **Add engagement dimension:** Measure if content is interesting/memorable
3. **Calibrate with human review:** Validate LLM-judge scores monthly

## Limitations

### What This Evaluation Cannot Assess

1. **Actual conversion rates:** High-scoring content may not convert better
2. **A/B test performance:** Would need real user data
3. **Cultural nuances:** Single LLM judge may miss regional preferences
4. **Long-term brand impact:** Only measures individual pieces

### Future Improvements

1. Correlate scores with actual marketing performance
2. Add human evaluation for calibration
3. Test with multiple LLM judges for consistency
4. Include competitive analysis dimension
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Quality and detail of brand guidelines
2. Diversity of content samples (good and bad examples)
3. Identification of patterns across high/low scores
4. Actionable recommendations for improvement
