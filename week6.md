# Week 6 — LLM-as-Judge: Automated Evaluation with Language Models
### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 6, you will:

1. Understand *what LLM-as-Judge is* and why it's a powerful evaluation paradigm.
2. Learn how to design effective **system prompts** for evaluation.
3. Implement and use the `LLMJudge` class from `src/benchmark_engine/llm_judge.py`.
4. Understand the trade-offs between automated and human evaluation.
5. Apply critical thinking to identify when LLM-as-Judge is appropriate.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is LLM-as-Judge?

### Simple Explanation

Imagine you're grading essays. Instead of reading each one yourself, you hire an expert teacher to grade them for you. The teacher reads each essay, compares it to a rubric (the expected answer), and gives a score with feedback.

> **LLM-as-Judge uses a powerful language model (like GPT-4) as that expert teacher to evaluate outputs from other models.**

This approach is valuable because:
- **Scalability:** Humans can't manually review thousands of outputs
- **Consistency:** LLMs apply the same criteria every time
- **Speed:** Evaluation happens in seconds, not hours
- **Cost:** Often cheaper than human annotators at scale

### When to Use LLM-as-Judge

| Use Case | LLM-as-Judge? | Reason |
|----------|---------------|--------|
| Grading factual QA | ✓ Yes | Easy to compare against reference |
| Evaluating creative writing | ✓ Yes | Can assess coherence, style |
| Safety-critical applications | ⚠ Maybe | Supplement with human review |
| Highly specialized domains | ⚠ Maybe | May lack domain expertise |
| Legal/compliance decisions | ✗ No | Requires human accountability |

---

# 🧠 Section 2 — Designing Effective Evaluation Prompts

### The Anatomy of a Good Judge Prompt

A well-designed judge prompt should:

1. **Define the role:** "You are an expert evaluator..."
2. **Specify criteria:** What aspects to evaluate (correctness, coherence, helpfulness)
3. **Provide context:** The question, answer, and reference
4. **Request structured output:** JSON format for easy parsing
5. **Set scoring guidelines:** Clear definitions for each score level

### Our System Prompt Structure

```python
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI-generated responses.

Your task is to evaluate an answer based on three criteria:
1. **Correctness**: Does the answer accurately address the question?
2. **Coherence**: Is the answer well-structured and easy to understand?
3. **Helpfulness**: Does the answer provide useful information?

Respond ONLY with a valid JSON object:
{"score": <float between 0.0 and 1.0>, "rationale": "<brief explanation>"}
"""
```

### Why These Three Criteria?

| Criterion | What It Measures | Why It Matters |
|-----------|-----------------|----------------|
| **Correctness** | Factual accuracy | Users need reliable information |
| **Coherence** | Logical structure | Confusing answers frustrate users |
| **Helpfulness** | Practical utility | Answers must address the actual need |

---

# 🧪 Section 3 — The `LLMJudge` Class

### Class Design

The `LLMJudge` class follows a simple, flexible design:

```python
class LLMJudge:
    def __init__(self, client, model="gpt-4o-mini", system_prompt=JUDGE_SYSTEM_PROMPT):
        """
        Initialize with an LLM client (e.g., OpenAI client).
        
        Args:
            client: An object with chat.completions.create() method
            model: Which model to use for judging
            system_prompt: Instructions for the judge
        """
        
    def score_answer(self, question, answer, reference) -> dict:
        """
        Evaluate an answer against a reference.
        
        Returns:
            {"score": float, "rationale": str}
        """
```

### Key Design Decisions

1. **Client Injection:** The class accepts any client object that follows the OpenAI API pattern. This makes it flexible for:
   - OpenAI's official client
   - Azure OpenAI
   - Local models with compatible APIs
   - Mock clients for testing

2. **Configurable Model:** Default is `gpt-4o-mini` for cost-effectiveness, but can use stronger models like `gpt-4o` for critical evaluations.

3. **Customizable Prompt:** The system prompt can be overridden for specialized evaluation scenarios (Weeks 11-16 will explore this).

4. **Structured Output:** Returns a dictionary with `score` and `rationale` for programmatic use.

---

# 🧪 Section 4 — Hands-on Lab: Using LLMJudge

### Lab Overview

In this lab, you will:
1. Import the LLMJudge class
2. Set up an OpenAI client (or mock client for testing)
3. Evaluate sample QA pairs
4. Analyze the results

### Step 1: Import the Module

```python
from src.benchmark_engine.llm_judge import LLMJudge, JUDGE_SYSTEM_PROMPT
```

### Step 2: Create an LLM Client

**Option A: With OpenAI API (requires API key)**
```python
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY environment variable
judge = LLMJudge(client)
```

**Option B: With Mock Client (for testing)**
```python
# See the notebook for a mock client implementation
judge = LLMJudge(mock_client)
```

### Step 3: Evaluate Answers

```python
# Define test cases
test_cases = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
        "reference": "Paris"
    },
    {
        "question": "What is 2+2?",
        "answer": "The sum of 2 and 2 equals 4.",
        "reference": "4"
    },
    {
        "question": "What is the largest planet?",
        "answer": "Mars is the largest planet.",  # Wrong!
        "reference": "Jupiter"
    },
]

# Evaluate each case
for tc in test_cases:
    result = judge.score_answer(
        question=tc["question"],
        answer=tc["answer"],
        reference=tc["reference"]
    )
    print(f"Q: {tc['question']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Rationale: {result['rationale']}")
    print()
```

---

# 🧠 Section 5 — Integrating LLMJudge with the Benchmark Engine

### Creating an LLM-as-Judge Metric Function

The benchmark engine expects a metric function with signature `(output, reference) -> float`. We can adapt LLMJudge:

```python
from src.benchmark_engine import run_benchmark, LLMJudge

def create_llm_judge_metric(judge: LLMJudge, question: str):
    """
    Create a metric function that uses LLM-as-Judge.
    
    Note: This captures the question in closure for use in the metric.
    """
    def llm_metric(output: str, reference: str) -> float:
        result = judge.score_answer(question, output, reference)
        return result["score"]
    return llm_metric

# For more complex scenarios, see the notebook examples
```

### Example: Evaluating a Model with LLM-as-Judge

```python
# Create judge
judge = LLMJudge(client)

# Evaluate each QA pair
results = []
for question, reference in dataset:
    answer = model_fn(question)
    result = judge.score_answer(question, answer, reference)
    results.append({
        "question": question,
        "answer": answer,
        "reference": reference,
        "score": result["score"],
        "rationale": result["rationale"]
    })

# Compute average score
mean_score = sum(r["score"] for r in results) / len(results)
print(f"Average LLM-as-Judge Score: {mean_score:.2%}")
```

---

# 🤔 Section 6 — Paul-Elder Critical Thinking Questions

### Question 1: EVIDENCE
**If LLM-as-Judge gives a high score to an answer that a human expert would rate poorly, what might explain this discrepancy?**

*Consider: Domain expertise, prompt design, biases in the judge model, ambiguity in the evaluation criteria.*

### Question 2: ASSUMPTIONS
**What assumptions are we making when we use a model like GPT-4 to judge outputs from another model?**

*Consider: Model objectivity, capability differences, potential biases, alignment between judge model and human preferences.*

### Question 3: IMPLICATIONS
**If an organization relies solely on LLM-as-Judge for model evaluation, what could go wrong in production?**

*Consider: Edge cases, adversarial inputs, distributional shifts, blind spots shared by judge and evaluated model.*

### Question 4: POINT OF VIEW
**How might the choice of judge model affect evaluation results? What if we used GPT-3.5 vs GPT-4?**

*Consider: Model capabilities, cost trade-offs, consistency, potential for the judge to have similar weaknesses as the evaluated model.*

---

# 🔄 Section 7 — Inversion Thinking: How Can LLM-as-Judge Fail?

Instead of asking "How does LLM-as-Judge help us?", let's invert:

> **"How can LLM-as-Judge give us misleading results?"**

### Potential Failure Modes

1. **Sycophancy Bias**
   - Judge models may rate verbose, confident-sounding answers higher
   - Even if the content is wrong or unhelpful
   - Mitigation: Include explicit criteria for conciseness

2. **Self-Preference Bias**
   - Models may prefer outputs that match their own style
   - If evaluating GPT outputs with GPT judge, may be overly generous
   - Mitigation: Use diverse judge models, compare with human evaluation

3. **Position Bias**
   - Order of information in the prompt can affect scores
   - Mitigation: Randomize presentation order, test for sensitivity

4. **Domain Blindness**
   - Judge may lack expertise in specialized domains (medical, legal)
   - Confidently wrong evaluations possible
   - Mitigation: Use domain-specific prompts, human validation

5. **Adversarial Manipulation**
   - Answers can be crafted to fool the judge but mislead humans
   - Example: Including "This is correct" in the answer
   - Mitigation: Robust prompts, adversarial testing

### Defensive Practices

- **Human Calibration:** Periodically compare LLM-as-Judge with human annotations
- **Multiple Judges:** Use diverse models and aggregate scores
- **Transparency:** Always log rationales for debugging
- **Thresholds:** Set minimum confidence thresholds for automated decisions

---

# 📝 Section 8 — Mini-Project: Build Your LLM-as-Judge Pipeline

### Task

Create a complete LLM-as-Judge evaluation pipeline that:
1. Uses the `LLMJudge` class to evaluate a model
2. Processes at least 10 QA pairs
3. Compares LLM-as-Judge scores with exact match scores
4. Analyzes when the two metrics disagree

### Instructions

1. **Create your dataset:**
   - 10 QA pairs with known correct answers
   - Include cases where partial credit is reasonable

2. **Implement two evaluation approaches:**
   - Exact match metric from Week 5
   - LLM-as-Judge metric from this week

3. **Run both evaluations** on the same model outputs

4. **Analyze disagreements:**
   - Which answers scored high on LLM-judge but 0 on exact match?
   - Which scored 1.0 on exact match but < 1.0 on LLM-judge?
   - What patterns do you observe?

### Submission Format

Create a markdown file `/examples/week06_results.md`:

```markdown
# Week 6 Mini-Project: LLM-as-Judge Analysis

## Dataset
| # | Question | Reference |
|---|----------|-----------|
| 1 | ...      | ...       |

## Model Outputs and Scores
| # | Model Answer | Exact Match | LLM-Judge | Rationale |
|---|--------------|-------------|-----------|-----------|
| 1 | ...          | 0.0         | 0.85      | "..."     |

## Analysis

### Cases Where Metrics Disagree
[Describe 2-3 examples where exact match and LLM-as-Judge differ]

### Key Insight
[One paragraph on when to use each metric type]
```

---

# ✔ Knowledge Mastery Checklist
- [ ] I can explain LLM-as-Judge in simple terms
- [ ] I understand the three evaluation criteria (correctness, coherence, helpfulness)
- [ ] I can use the `LLMJudge` class to evaluate answers
- [ ] I understand the trade-offs of automated vs. human evaluation
- [ ] I know potential failure modes of LLM-as-Judge
- [ ] I completed the mini-project comparing metrics

---

Week 6 complete.
Next: *Week 7 — Semantic Similarity Metrics (Embeddings, Cosine Similarity)*.
