# Week 11 — Banking & Compliance Evaluation

### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 11, you will:

1. Understand *why compliance summarization is critical* in banking and financial services.
2. Learn how to design evaluation scenarios for *regulatory text summarization*.
3. Implement an LLM-as-Judge evaluation that scores *correctness* and detects *omission of critical details*.
4. Understand the risks and business impact of mis-summarizing regulations.
5. Apply critical thinking to evaluate the suitability of LLMs for compliance use cases.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is Compliance Summarization?

### Simple Explanation

Imagine you're a bank employee who needs to quickly understand a new regulation:

- **The regulation** is a long, complex legal document
- **You need** a short summary of the key requirements
- **If you miss something** → potential fines, legal issues, or customer harm

> **Compliance summarization uses LLMs to create concise summaries of regulations, KYC rules, and compliance requirements. Accuracy is paramount—missing a critical detail can have serious legal and financial consequences.**

This is different from general summarization because:
- **Completeness matters:** Omitting a key requirement is a failure
- **Precision matters:** Misrepresenting a rule is potentially worse than omitting it
- **Stakes are high:** Errors can lead to regulatory penalties, lawsuits, or reputational damage

### The Banking & Compliance Domain

| Document Type | What It Contains | Why Summarization Helps |
|---------------|------------------|------------------------|
| **KYC Rules** | Customer due diligence requirements | Quick reference for onboarding teams |
| **AML Regulations** | Anti-money laundering procedures | Training and compliance checklists |
| **Basel Accords** | Capital adequacy requirements | Executive briefings |
| **GDPR/Privacy** | Data protection requirements | Developer guidelines |
| **Dodd-Frank** | Financial reform rules | Risk assessment summaries |

---

# 🧠 Section 2 — The Scenario: Regulatory Text Summarization

### Scenario Description

A bank's compliance team uses an LLM to:
1. **Input:** Short regulatory texts or KYC rule excerpts (1-3 paragraphs)
2. **Output:** Concise summaries highlighting key requirements
3. **Use case:** Quick reference for compliance officers, training materials, internal guidelines

### Example Regulatory Texts

#### Example 1: KYC Customer Identification Rule
```
Financial institutions must verify the identity of each customer opening an account.
Verification requires obtaining the customer's name, date of birth, address, and 
identification number. For individuals, acceptable identification includes a 
government-issued photo ID (passport, driver's license) or two forms of non-photo 
identification. The institution must maintain records of the verification process 
for at least five years after the account is closed. Failure to comply may result 
in penalties up to $500,000 per violation.
```

#### Example 2: Suspicious Activity Reporting
```
Banks must file a Suspicious Activity Report (SAR) within 30 calendar days of 
detecting suspicious activity. Suspicious activity includes transactions over 
$5,000 that appear to involve money laundering, tax evasion, or other criminal 
activity. The bank must not notify the customer that a SAR has been filed. 
SAR records must be retained for five years. Late filing or failure to file 
may result in civil penalties and criminal prosecution.
```

#### Example 3: Data Retention Requirements
```
Under the Bank Secrecy Act, financial institutions must retain records of 
transactions over $3,000 for a minimum of five years. Records must include 
the customer's name, account number, transaction amount, and date. Electronic 
records are acceptable if they can be produced within 24 hours upon regulatory 
request. Institutions must designate a compliance officer responsible for 
ensuring retention requirements are met.
```

### What Makes This Challenging

| Challenge | Why It's Hard | Risk if Failed |
|-----------|---------------|----------------|
| **Critical details** | Amounts, timeframes, thresholds | Missed deadline = violation |
| **Legal precision** | "Must" vs "should" vs "may" | Incorrect compliance posture |
| **Completeness** | Every requirement matters | Omission = potential violation |
| **Context** | Rules interact with each other | Incomplete understanding |

---

# 🧪 Section 3 — Designing the Evaluation

### Evaluation Design

For banking & compliance summarization, we need an evaluation that assesses:

1. **Correctness:** Does the summary accurately represent the regulation?
2. **Completeness:** Are all critical details included (amounts, deadlines, penalties)?
3. **Omission Detection:** What critical details were left out?

### The Compliance Judge Prompt

```python
COMPLIANCE_JUDGE_PROMPT = """You are an expert compliance officer evaluating regulatory summaries.

Your task is to evaluate a summary of a regulatory text based on two criteria:

1. **Correctness (0.0-1.0):** Does the summary accurately represent the regulatory requirements?
   - 1.0: Fully accurate, no misrepresentations
   - 0.7-0.9: Minor inaccuracies that don't affect compliance
   - 0.4-0.6: Significant inaccuracies that could mislead compliance decisions
   - 0.0-0.3: Major errors or misrepresentations

2. **Completeness (0.0-1.0):** Are all critical details included?
   - 1.0: All critical details present (amounts, deadlines, penalties, requirements)
   - 0.7-0.9: Most critical details present, minor omissions
   - 0.4-0.6: Some critical details missing
   - 0.0-0.3: Major critical details missing

CRITICAL DETAILS to check for:
- Monetary thresholds and amounts
- Time limits and deadlines
- Penalty amounts and consequences
- Specific requirements (documents, actions, retention periods)
- Key terms ("must", "shall", prohibited actions)

Respond ONLY with a valid JSON object:
{
    "correctness_score": <float 0.0-1.0>,
    "completeness_score": <float 0.0-1.0>,
    "omitted_details": ["<list of critical details missing from summary>"],
    "rationale": "<brief explanation of scores>"
}
"""
```

### Key Design Decisions

1. **Two-dimensional scoring:** Correctness and completeness are both critical but distinct
2. **Explicit omission tracking:** The judge must identify what's missing, not just score
3. **Critical detail focus:** Specific guidance on what constitutes a critical detail
4. **Clear rubric:** Numeric guidance helps consistency across evaluations

---

# 🧪 Section 4 — Implementing the Compliance Evaluator

### The ComplianceJudge Class

```python
from typing import Dict, List, Any
import json

COMPLIANCE_JUDGE_PROMPT = """You are an expert compliance officer evaluating regulatory summaries.

Your task is to evaluate a summary of a regulatory text based on two criteria:

1. **Correctness (0.0-1.0):** Does the summary accurately represent the regulatory requirements?
2. **Completeness (0.0-1.0):** Are all critical details included?

CRITICAL DETAILS to check for:
- Monetary thresholds and amounts
- Time limits and deadlines  
- Penalty amounts and consequences
- Specific requirements (documents, actions, retention periods)
- Key terms ("must", "shall", prohibited actions)

Respond ONLY with a valid JSON object:
{
    "correctness_score": <float 0.0-1.0>,
    "completeness_score": <float 0.0-1.0>,
    "omitted_details": ["<list of critical details missing from summary>"],
    "rationale": "<brief explanation of scores>"
}
"""


class ComplianceJudge:
    """
    LLM-as-Judge for evaluating regulatory and compliance summaries.
    
    This specialized judge evaluates:
    - Correctness: Accuracy of the summary
    - Completeness: Whether critical details are included
    - Omissions: Specific details that were left out
    """
    
    def __init__(self, client, model: str = "gpt-4o-mini"):
        """
        Initialize the ComplianceJudge.
        
        Args:
            client: An LLM client with chat.completions.create() method
            model: Model to use for judging (default: gpt-4o-mini)
        """
        self.client = client
        self.model = model
        self.system_prompt = COMPLIANCE_JUDGE_PROMPT
    
    def evaluate_summary(
        self,
        regulatory_text: str,
        summary: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a regulatory summary.
        
        Args:
            regulatory_text: The original regulatory text
            summary: The model-generated summary
            
        Returns:
            Dictionary with correctness_score, completeness_score, 
            omitted_details, and rationale
        """
        user_prompt = f"""Regulatory Text:
{regulatory_text}

Summary to Evaluate:
{summary}

Evaluate the summary for correctness and completeness."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            return {
                "correctness_score": float(result.get("correctness_score", 0.0)),
                "completeness_score": float(result.get("completeness_score", 0.0)),
                "omitted_details": result.get("omitted_details", []),
                "rationale": result.get("rationale", ""),
            }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {
                "correctness_score": 0.0,
                "completeness_score": 0.0,
                "omitted_details": ["Error parsing judge response"],
                "rationale": f"Error: {str(e)}",
            }
    
    def compute_combined_score(
        self,
        correctness_score: float,
        completeness_score: float,
        correctness_weight: float = 0.5,
    ) -> float:
        """
        Compute a weighted combined score.
        
        Args:
            correctness_score: Score for accuracy (0-1)
            completeness_score: Score for completeness (0-1)
            correctness_weight: Weight for correctness (completeness = 1 - this)
            
        Returns:
            Weighted combined score
        """
        completeness_weight = 1.0 - correctness_weight
        return (correctness_score * correctness_weight + 
                completeness_score * completeness_weight)
```

### Usage Example

```python
# Initialize the judge
judge = ComplianceJudge(client)

# Regulatory text
regulatory_text = """
Financial institutions must file a Suspicious Activity Report (SAR) within 
30 calendar days of detecting suspicious activity. Transactions over $5,000 
that appear to involve money laundering must be reported. The bank must not 
notify the customer. SAR records must be retained for five years.
"""

# Model-generated summary
summary = """
Banks must file SARs for suspicious transactions. Records should be kept 
for documentation purposes.
"""

# Evaluate
result = judge.evaluate_summary(regulatory_text, summary)

print(f"Correctness: {result['correctness_score']:.2f}")
print(f"Completeness: {result['completeness_score']:.2f}")
print(f"Omitted Details: {result['omitted_details']}")
print(f"Rationale: {result['rationale']}")
```

---

# 🧪 Section 5 — Hands-on Lab: Evaluating Regulatory Summaries

### Lab Overview

In this lab, you will:
1. Create sample regulatory texts
2. Generate summaries (simulated or from an LLM)
3. Evaluate summaries using the ComplianceJudge
4. Analyze which critical details are commonly omitted

### Step 1: Define Test Cases

```python
# Test cases with regulatory texts and summaries of varying quality
test_cases = [
    {
        "name": "KYC - Complete Summary",
        "regulatory_text": """
Financial institutions must verify the identity of each customer opening an account.
Verification requires obtaining the customer's name, date of birth, address, and 
identification number. For individuals, acceptable identification includes a 
government-issued photo ID (passport, driver's license) or two forms of non-photo 
identification. The institution must maintain records of the verification process 
for at least five years after the account is closed. Failure to comply may result 
in penalties up to $500,000 per violation.
        """,
        "summary": """
Banks must verify customer identity when opening accounts using name, DOB, address, 
and ID number. Acceptable ID: government photo ID or two non-photo IDs. Records 
must be kept 5 years after account closure. Penalties up to $500,000 per violation.
        """,
        "expected_completeness": "high",
    },
    {
        "name": "KYC - Missing Critical Details",
        "regulatory_text": """
Financial institutions must verify the identity of each customer opening an account.
Verification requires obtaining the customer's name, date of birth, address, and 
identification number. For individuals, acceptable identification includes a 
government-issued photo ID (passport, driver's license) or two forms of non-photo 
identification. The institution must maintain records of the verification process 
for at least five years after the account is closed. Failure to comply may result 
in penalties up to $500,000 per violation.
        """,
        "summary": """
Banks need to verify customer identity when opening accounts. They should keep 
records of this process.
        """,
        "expected_completeness": "low",
    },
    {
        "name": "SAR - Inaccurate Summary",
        "regulatory_text": """
Banks must file a Suspicious Activity Report (SAR) within 30 calendar days of 
detecting suspicious activity. Suspicious activity includes transactions over 
$5,000 that appear to involve money laundering. The bank must not notify the 
customer that a SAR has been filed.
        """,
        "summary": """
Banks should file SARs within 60 days for transactions over $10,000. Customers 
should be informed when a SAR is filed about their account.
        """,
        "expected_completeness": "inaccurate",
    },
]
```

### Step 2: Run Evaluations

```python
# Evaluate each test case
for tc in test_cases:
    result = judge.evaluate_summary(
        tc["regulatory_text"],
        tc["summary"]
    )
    
    print(f"\n{'='*60}")
    print(f"Test: {tc['name']}")
    print(f"Expected: {tc['expected_completeness']}")
    print(f"{'='*60}")
    print(f"Correctness Score: {result['correctness_score']:.2f}")
    print(f"Completeness Score: {result['completeness_score']:.2f}")
    print(f"Combined Score: {judge.compute_combined_score(result['correctness_score'], result['completeness_score']):.2f}")
    print(f"\nOmitted Details:")
    for detail in result['omitted_details']:
        print(f"  - {detail}")
    print(f"\nRationale: {result['rationale']}")
```

---

# 🤔 Section 6 — Paul-Elder Critical Thinking Questions

### Question 1: RISK ASSESSMENT
**What are the potential consequences if a compliance summary incorrectly states a reporting deadline as 60 days instead of 30 days?**

*Consider: Regulatory violations, audit findings, potential fines, reputational damage, and the chain of decisions that might rely on this summary. How might this error propagate through an organization?*

### Question 2: TRUST CALIBRATION
**Should a compliance officer trust an LLM-generated summary of a regulation they haven't read themselves? Under what conditions?**

*Consider: The role of human oversight, the complexity of the regulation, the stakes involved, available verification methods, and organizational liability. When might automation be appropriate vs. when is human review essential?*

### Question 3: OMISSION DETECTION
**How can we systematically detect what an LLM summary has omitted from a regulatory text, and why is this particularly challenging?**

*Consider: The difficulty of proving a negative, the need for domain expertise to identify critical vs. non-critical details, the risk of "unknown unknowns," and how omission detection differs from error detection. What evaluation strategies might help?*

---

# 🔄 Section 7 — Inversion Thinking: How Can Compliance Summarization Fail?

Instead of asking "How does LLM summarization help compliance?", let's invert:

> **"How can LLM-generated compliance summaries cause regulatory violations?"**

### Failure Modes

1. **Threshold Errors**
   - LLM rounds or approximates monetary thresholds
   - "$5,000" becomes "around $5,000" or "several thousand dollars"
   - Consequence: Transactions fall through the cracks

2. **Deadline Dilution**
   - "30 calendar days" becomes "about a month" or "promptly"
   - Hard deadlines become soft suggestions
   - Consequence: Late filings, regulatory penalties

3. **Requirement Softening**
   - "Must" becomes "should" or "typically"
   - Mandatory requirements sound optional
   - Consequence: Non-compliance treated as acceptable

4. **Penalty Omission**
   - Consequences and penalties are dropped for brevity
   - Risk appears lower than it actually is
   - Consequence: Inadequate compliance priority

5. **Context Loss**
   - Exceptions and conditions are omitted
   - Nuanced rules become absolute statements
   - Consequence: Incorrect application of rules

### Defensive Practices

- **Critical Detail Extraction:** Separately extract and verify all numbers, dates, and thresholds
- **Requirement Classification:** Tag each requirement as mandatory/recommended/optional
- **Human-in-the-Loop:** Require compliance officer review before using summaries
- **Audit Trail:** Maintain links from summaries back to source text
- **Confidence Scoring:** Flag low-confidence summaries for additional review
- **Regression Testing:** Test summarization quality on known regulatory texts

---

# 📝 Section 8 — Mini-Project: Build a Compliance Summarization Evaluator

### Task

Create a complete compliance summarization evaluation pipeline that:
1. Uses the ComplianceJudge to evaluate regulatory summaries
2. Processes at least 5 regulatory text examples
3. Tracks both correctness and completeness scores
4. Identifies patterns in omitted critical details

### Instructions

1. **Create your regulatory dataset:**
   - 5 short regulatory texts (real or realistic)
   - Include variety: KYC, AML, reporting requirements, data retention
   - Each should have clear critical details (amounts, deadlines, penalties)

2. **Generate summaries:**
   - Create both good and poor summaries
   - Or use an LLM to generate summaries

3. **Run evaluations:**
   - Use ComplianceJudge to evaluate each summary
   - Record correctness and completeness scores
   - Track all omitted details

4. **Analyze results:**
   - Which types of details are most commonly omitted?
   - What patterns do you observe in correctness errors?
   - What thresholds would you set for acceptable summaries?

### Submission Format

Create a markdown file `/examples/week11_compliance_audit.md`:

```markdown
# Week 11 Mini-Project: Compliance Summarization Audit

## Executive Summary
[2-3 sentences on overall findings]

## Regulatory Texts Evaluated

| # | Type | Critical Details Count | 
|---|------|------------------------|
| 1 | KYC | 5 |
| 2 | AML | 4 |
| ... | ... | ... |

## Evaluation Results

| # | Correctness | Completeness | Combined | Omitted Details |
|---|-------------|--------------|----------|-----------------|
| 1 | 0.95 | 0.80 | 0.88 | Retention period |
| 2 | 0.60 | 0.40 | 0.50 | Threshold, deadline, penalty |
| ... | ... | ... | ... | ... |

## Omission Analysis

### Most Commonly Omitted Details
1. [Detail type] - [frequency]
2. [Detail type] - [frequency]

### Patterns Observed
[Analysis of what types of information LLMs tend to omit]

## Recommendations

### Acceptable Score Thresholds
- Correctness: [threshold]
- Completeness: [threshold]
- Combined: [threshold]

### Required Human Review
[When should human compliance review be mandatory?]

## Risk Assessment
[Assessment of risks if these summaries were used without review]
```

---

# 🔧 Section 9 — Advanced: Extending the Compliance Evaluator

### Adding Critical Detail Extraction

For production use, extract and verify critical details explicitly:

```python
def extract_critical_details(regulatory_text: str) -> dict:
    """
    Extract critical details from regulatory text.
    
    Returns:
        Dictionary with categorized critical details:
        - monetary_thresholds: List of amounts
        - deadlines: List of time requirements
        - penalties: List of consequences
        - requirements: List of mandatory actions
    """
    # TODO: Implement with LLM or rule-based extraction
    # 
    # prompt = f"""Extract all critical details from this regulatory text:
    # 
    # {regulatory_text}
    # 
    # Return JSON with:
    # - monetary_thresholds: ["$5,000", ...]
    # - deadlines: ["30 calendar days", ...]
    # - penalties: ["$500,000 per violation", ...]
    # - requirements: ["must file SAR", ...]
    # """
    pass


def verify_critical_details(
    extracted_from_source: dict,
    extracted_from_summary: dict,
) -> dict:
    """
    Verify that critical details from source appear in summary.
    
    Returns:
        Dictionary with verification results and missing details
    """
    pass
```

### Adding Regulatory Domain Classification

```python
def classify_regulatory_domain(text: str) -> str:
    """
    Classify the regulatory domain of a text.
    
    Returns one of:
    - "kyc": Know Your Customer
    - "aml": Anti-Money Laundering
    - "privacy": Data Protection/Privacy
    - "reporting": Regulatory Reporting
    - "capital": Capital Requirements
    - "consumer": Consumer Protection
    - "other": Other regulatory domain
    """
    pass
```

### Adding Compliance Risk Scoring

```python
def assess_summary_risk(
    correctness_score: float,
    completeness_score: float,
    omitted_details: list,
    regulatory_domain: str,
) -> dict:
    """
    Assess the risk level of using a summary.
    
    Returns:
        Dictionary with:
        - risk_level: "low", "medium", "high", "critical"
        - risk_factors: List of specific concerns
        - recommendation: "approve", "review", "reject"
    """
    pass
```

---

# ✔ Knowledge Mastery Checklist
- [ ] I can explain why compliance summarization is high-stakes and different from general summarization
- [ ] I understand the difference between correctness and completeness in regulatory summaries
- [ ] I can use the ComplianceJudge to evaluate regulatory summaries
- [ ] I know what types of critical details are most important in compliance contexts
- [ ] I understand the risks of omission in regulatory summarization
- [ ] I can identify when human review is essential vs. when automation is acceptable
- [ ] I completed the mini-project compliance audit

---

Week 11 complete.
Next: *Week 12 — Healthcare Use Cases*.
