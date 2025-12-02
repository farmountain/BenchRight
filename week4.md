# Week 4 — Industry Benchmark Suites
### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 4, you will:

1. Understand the purpose and design of major industry benchmarks: **MMLU, HellaSwag, BBH, TruthfulQA, and ToxiGen**.
2. Know why these benchmarks matter and what aspects of LLM capability they measure.
3. Recognize the limitations of small models on these benchmarks while understanding evaluation mechanics.
4. Load and evaluate model outputs on benchmark subsets using Hugging Face datasets.
5. Build intuition for designing your own custom benchmarks.

---

# 🧠 Section 1 — Overview of Industry Benchmarks

### MMLU (Massive Multitask Language Understanding)

**Purpose:** Measures world knowledge and problem-solving ability across 57 subjects.

MMLU contains 15,908 multiple-choice questions spanning subjects from elementary mathematics to professional law and medicine. It tests whether a model has acquired factual knowledge during pretraining and can apply it to answer questions correctly.

**Why it matters:** A model that scores well on MMLU demonstrates broad knowledge—crucial for applications like question answering, tutoring, and research assistance.

---

### HellaSwag

**Purpose:** Tests commonsense reasoning and narrative understanding through story completion.

HellaSwag presents a scenario and asks the model to choose the most plausible continuation from several options. The options are adversarially generated to be grammatically correct but nonsensical to humans with common sense.

**Why it matters:** Commonsense reasoning is essential for models to behave naturally in conversation and avoid absurd responses.

---

### BigBench Hard (BBH)

**Purpose:** Evaluates multi-step reasoning through 23 challenging tasks from BIG-Bench.

BBH focuses on tasks where chain-of-thought prompting significantly improves performance. These include logical deduction, causal reasoning, and algorithmic tasks that require multiple reasoning steps.

**Why it matters:** Real-world tasks often require combining multiple facts and reasoning steps—BBH tests this crucial capability.

---

### TruthfulQA

**Purpose:** Measures truthfulness and resistance to common human misconceptions.

TruthfulQA contains 817 questions across 38 categories designed to test whether models repeat common falsehoods or provide accurate information. Questions are crafted to target misconceptions that humans often believe.

**Why it matters:** As LLMs are used for information retrieval, it's critical they don't confidently spread misinformation.

---

### ToxiGen

**Purpose:** Detects implicit toxicity and hate speech across 13 demographic groups.

ToxiGen contains approximately 274,000 toxic and benign statements that test a model's ability to recognize subtle, adversarial hate speech. Unlike obvious slurs, ToxiGen tests for implicit bias and coded language.

**Why it matters:** Safe deployment requires models that can identify and avoid generating harmful content, even when it's disguised.

---

# 🧠 Section 2 — Why Small Models Struggle (But We Learn Anyway)

### The Reality of tinyGPT on Industry Benchmarks

Small models like tinyGPT will likely perform poorly on these benchmarks—and that's okay! Here's why we still evaluate them:

| Reason | Explanation |
|--------|-------------|
| **Learn evaluation mechanics** | Understanding how to run benchmarks is valuable regardless of results |
| **Establish baselines** | Poor results help calibrate expectations for larger models |
| **Identify failure modes** | Seeing where small models fail teaches us what capabilities emerge with scale |
| **Build infrastructure** | The evaluation pipeline works the same for any model size |
| **Practice interpretation** | Learning to analyze results is a transferable skill |

### Expected Results

| Benchmark | tinyGPT Expected | State-of-Art |
|-----------|-----------------|--------------|
| MMLU | ~25% (random) | 85%+ |
| HellaSwag | ~30-40% | 95%+ |
| TruthfulQA | ~25-30% | 60%+ |

**Key insight:** Random guessing on 4-choice questions gives 25%. Results near this indicate the model lacks the capability—but we still learn from running the evaluation!

---

# 🧪 Section 3 — Hands-on Lab: Evaluating on Industry Benchmarks

### Lab Overview

In this lab, you will:
1. Load small subsets (50 examples) from 2-3 benchmarks
2. Run tinyGPT to generate answers
3. Compute simple accuracy metrics
4. Log results to a DataFrame for analysis

### Step 1: Load Benchmark Data

```python
from datasets import load_dataset

# Load HellaSwag (commonsense reasoning)
hellaswag = load_dataset("hellaswag", split="validation")
hellaswag_sample = hellaswag.select(range(50))

# Load TruthfulQA (truthfulness)
truthfulqa = load_dataset("truthful_qa", "multiple_choice", split="validation")
truthfulqa_sample = truthfulqa.select(range(50))

print(f"HellaSwag sample: {len(hellaswag_sample)} examples")
print(f"TruthfulQA sample: {len(truthfulqa_sample)} examples")
```

### Step 2: Format Questions for the Model

```python
def format_hellaswag_question(example):
    """Format a HellaSwag example as a multiple-choice question."""
    context = example["ctx"]
    endings = example["endings"]
    
    prompt = f"Context: {context}\n\n"
    prompt += "Which ending makes the most sense?\n"
    for i, ending in enumerate(endings):
        prompt += f"{chr(65+i)}. {ending}\n"
    prompt += "\nAnswer:"
    
    correct_answer = chr(65 + int(example["label"]))
    return prompt, correct_answer


def format_truthfulqa_question(example):
    """Format a TruthfulQA example as a multiple-choice question."""
    question = example["question"]
    choices = example["mc1_targets"]["choices"]
    
    prompt = f"Question: {question}\n\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "\nAnswer:"
    
    # First choice is correct in mc1 format
    correct_answer = "A"
    return prompt, correct_answer
```

### Step 3: Evaluate Model on Benchmark

```python
def evaluate_on_benchmark(examples, format_fn, session, tokenizer, benchmark_name):
    """
    Evaluate model on a benchmark dataset.
    
    Args:
        examples: Dataset examples
        format_fn: Function to format questions
        session: ONNX inference session
        tokenizer: Tokenizer instance
        benchmark_name: Name for logging
    
    Returns:
        DataFrame with results
    """
    results = []
    correct = 0
    
    for i, example in enumerate(examples):
        prompt, correct_answer = format_fn(example)
        
        # Run inference
        inputs = tokenizer(prompt, return_tensors="np")
        outputs = session.run(None, {"input_ids": inputs["input_ids"]})
        
        # Get predicted token
        next_token_logits = outputs[0][0, -1, :]
        predicted_token_id = np.argmax(next_token_logits)
        predicted_text = tokenizer.decode([predicted_token_id]).strip().upper()
        
        # Extract answer letter (heuristic)
        model_answer = ""
        for char in predicted_text:
            if char in "ABCD":
                model_answer = char
                break
        
        is_correct = model_answer == correct_answer
        if is_correct:
            correct += 1
        
        results.append({
            "benchmark": benchmark_name,
            "question_id": i,
            "correct_answer": correct_answer,
            "model_answer": model_answer,
            "is_correct": is_correct,
        })
    
    accuracy = correct / len(examples) * 100
    print(f"{benchmark_name}: {accuracy:.1f}% accuracy ({correct}/{len(examples)})")
    
    return pd.DataFrame(results)
```

### Step 4: Run Evaluation and Log Results

```python
# Run on HellaSwag
hellaswag_results = evaluate_on_benchmark(
    hellaswag_sample, 
    format_hellaswag_question, 
    session, 
    tokenizer, 
    "HellaSwag"
)

# Run on TruthfulQA
truthfulqa_results = evaluate_on_benchmark(
    truthfulqa_sample, 
    format_truthfulqa_question, 
    session, 
    tokenizer, 
    "TruthfulQA"
)

# Combine results
all_results = pd.concat([hellaswag_results, truthfulqa_results])
print("\n📊 Full Results Summary:")
print(all_results.groupby("benchmark")["is_correct"].mean() * 100)
```

---

# 🤔 Section 4 — Paul-Elder Critical Thinking Questions

### Question 1: EVIDENCE
**If a model scores 70% on MMLU but only 40% on TruthfulQA, what might this reveal about its training data or methodology?**

*Consider: Knowledge vs truthfulness, training data biases, optimization targets.*

### Question 2: ASSUMPTIONS
**What assumptions are we making when we use 50-example subsets instead of full benchmarks?**

*Consider: Statistical significance, sample bias, variance in results.*

### Question 3: IMPLICATIONS
**If an organization publishes benchmark results without revealing their evaluation methodology (prompt format, sampling, etc.), what are the implications?**

*Consider: Reproducibility, gaming benchmarks, comparison validity.*

---

# 🔄 Section 5 — Inversion Thinking: What Goes Wrong If We Trust Only Benchmarks?

Instead of asking "How do benchmarks help us?", let's invert:

> **"What goes wrong if we treat benchmark scores as the only truth about a model?"**

### Failure Scenarios

1. **Benchmark Overfitting**
   - Models can be optimized specifically for benchmark formats
   - High benchmark scores but poor real-world performance
   - Example: Model memorizes MMLU questions but can't generalize

2. **Distribution Mismatch**
   - Benchmarks may not match your actual use case
   - Model excels at academic questions but fails at customer support
   - Real users don't interact like benchmark prompts

3. **Missing Dimensions**
   - Benchmarks measure specific capabilities
   - A model might be unsafe, slow, or unreliable despite good accuracy
   - Example: High MMLU score but produces toxic content

4. **Prompt Sensitivity**
   - Small changes in prompt format can dramatically change scores
   - Published results may use optimized prompts
   - Your deployment won't have the same prompt engineering

5. **Cherry-Picking**
   - Organizations may report best results from many runs
   - Results may not be reproducible
   - Selection bias in which benchmarks to report

### Defensive Practice

**Always complement benchmark scores with:**
- Real user testing / human evaluation
- Safety audits (red teaming)
- Domain-specific evaluation
- Latency and cost metrics
- Failure case analysis

---

# 📝 Section 6 — Mini-Project: Design Your Own Benchmark

### Task

Create a small 10-question benchmark to evaluate tinyGPT on a specific capability of your choosing.

### Instructions

1. **Choose a capability to test:**
   - Basic arithmetic
   - Common sense facts
   - Simple logical reasoning
   - Sentiment understanding
   - Something domain-specific

2. **Design 10 multiple-choice questions:**
   - Each question should have 4 options (A, B, C, D)
   - One option should be clearly correct
   - Other options should be plausible distractors

3. **Format your benchmark:**
```python
my_benchmark = [
    {
        "question": "What is 2 + 2?",
        "options": ["A. 3", "B. 4", "C. 5", "D. 6"],
        "correct": "B"
    },
    # ... 9 more questions
]
```

4. **Run tinyGPT on your benchmark:**
   - Use the evaluation function from the lab
   - Record accuracy

5. **Analyze results:**
   - What does the model get right/wrong?
   - Is the benchmark too easy or too hard?
   - What would you change?

### Submission Format

Create a markdown file with:
- Your benchmark name and purpose
- All 10 questions with correct answers
- Model accuracy results
- Analysis of performance (1-2 paragraphs)
- Suggestions for improving the benchmark

---

# ✔ Knowledge Mastery Checklist
- [ ] I can explain the purpose of MMLU, HellaSwag, BBH, TruthfulQA, and ToxiGen
- [ ] I understand why small models may struggle but evaluation is still valuable
- [ ] I can load benchmark data from Hugging Face datasets
- [ ] I can format questions and parse model outputs for evaluation
- [ ] I understand the dangers of relying solely on benchmark scores
- [ ] I designed and ran my own 10-question benchmark

---

Week 4 complete.
Next: *Week 5 — Building a Generic Benchmark Engine*.
