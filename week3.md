# Week 3 — Perplexity and Basic Benchmarking
### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 3, you will:

1. Understand *what perplexity is* and why it matters for language model evaluation.
2. Learn how perplexity relates to the exponentiated average negative log-likelihood.
3. Recognize the *limitations* of perplexity as a standalone metric.
4. Compute approximate perplexity on a small text corpus using an ONNX model.
5. Apply critical thinking frameworks to understand when perplexity fails.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is Perplexity?

### Simple Explanation

Imagine you're playing a guessing game where you have to predict the next word in a sentence. If you're good at guessing, you feel less "surprised" when the answer is revealed. If you're bad at guessing, you feel very "surprised."

> **Perplexity measures how "surprised" a language model is when it sees actual text.**

Here's a concrete example:

**Sentence:** "The cat sat on the ___"

A good language model might predict:
- "mat" — 40% probability
- "floor" — 25% probability
- "chair" — 15% probability
- "table" — 10% probability
- other words — 10%

If the actual word is "mat," the model isn't very surprised (low perplexity). But if the actual word is "elephant," the model would be very surprised (high perplexity).

**Key insight:**
- **Low perplexity** = The model predicted the text well → model is confident and usually correct
- **High perplexity** = The model was surprised by the text → model struggles with this type of content

### Technical Definition (Still Simple)

Perplexity is the **exponentiated average negative log-likelihood** of the tokens in a sequence.

In plain English:
1. For each word/token, the model assigns a probability
2. We take the negative log of each probability (surprised = big number, confident = small number)
3. We average these across all tokens
4. We exponentiate the result to get perplexity

**Lower perplexity = better model** (for that specific text).

---

# 🧠 Section 2 — Why Perplexity is Not Everything

### The Correlation Problem

Perplexity measures how well a model predicts the next token. But in real-world applications, we care about:

- **Usefulness:** Does the model answer questions correctly?
- **Safety:** Does the model avoid harmful content?
- **Coherence:** Does the model produce logical, well-structured responses?
- **Factual accuracy:** Does the model state true information?

A model can have **excellent perplexity** but still:
- Hallucinate facts confidently
- Produce grammatically perfect but meaningless text
- Be great at predicting common patterns but terrible at reasoning
- Excel on Wikipedia text but fail on conversation

### Real-World Disconnect

| Scenario | Perplexity | Real-World Performance |
|----------|------------|------------------------|
| Model trained on Wikipedia | Low on Wikipedia test set | May fail at casual conversation |
| Model with memorized training data | Very low perplexity | Fails on novel inputs |
| Model predicting common phrases | Low perplexity | May lack creativity/diversity |
| Model on domain it hasn't seen | High perplexity | Actually might still be useful |

### The Key Insight

> **Perplexity tells you how well a model fits a specific text distribution, not how useful the model is for your task.**

This is why industry evaluations use multiple metrics: MMLU for knowledge, HellaSwag for reasoning, TruthfulQA for accuracy, and more.

---

# 🧪 Section 3 — Hands-on Lab: Computing Pseudo-Perplexity

### Overview

In this lab, we'll compute **approximate perplexity** using our tinyGPT ONNX model. Since ONNX models output logits, we'll:
1. Convert logits to probabilities using softmax
2. Compute the negative log probability of actual tokens
3. Average and exponentiate to get perplexity

### Lab Steps

#### Step 1: Setup and Load Model

```python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

model_path = "/tmp/tinygpt.onnx"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
```

#### Step 2: Softmax Function

```python
def softmax(logits):
    """Convert logits to probabilities using softmax."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
```

#### Step 3: Compute Pseudo-Perplexity

```python
def compute_pseudo_perplexity(text, session, tokenizer):
    """
    Compute approximate perplexity for a text sequence.
    
    Uses sliding window approach: for each position, predict the next token
    and compute the negative log probability of the actual next token.
    """
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"][0]
    
    if len(input_ids) < 2:
        return float('inf')  # Can't compute perplexity for single token
    
    total_nll = 0.0
    count = 0
    
    # For each position, predict next token
    for i in range(len(input_ids) - 1):
        # Use tokens up to position i as context
        context = input_ids[:i+1].reshape(1, -1)
        
        # Get model output (logits)
        outputs = session.run(None, {"input_ids": context})
        logits = outputs[0][0, -1, :]  # Last position's logits
        
        # Convert to probabilities
        probs = softmax(logits)
        
        # Get probability of actual next token
        next_token = input_ids[i+1]
        prob = probs[next_token]
        
        # Compute negative log probability
        nll = -np.log(prob + 1e-10)  # Add small epsilon to avoid log(0)
        total_nll += nll
        count += 1
    
    # Compute perplexity
    avg_nll = total_nll / count
    perplexity = np.exp(avg_nll)
    
    return perplexity
```

#### Step 4: Test on Sample Sentences

```python
test_sentences = [
    "The cat sat on the mat.",
    "Machine learning is transforming technology.",
    "Colorless green ideas sleep furiously.",
    "asdf qwerty zxcv bnm poiu",
]

for sentence in test_sentences:
    ppl = compute_pseudo_perplexity(sentence, session, tokenizer)
    print(f"Perplexity: {ppl:.2f} | Text: {sentence}")
```

**Expected observations:**
- Common, grammatical sentences should have lower perplexity
- Nonsense or rare word combinations should have higher perplexity
- The famous Chomsky sentence "Colorless green ideas sleep furiously" is grammatically correct but semantically odd — watch its perplexity!

---

# 🤔 Section 4 — Paul-Elder Critical Thinking Questions

### Question 1: EVIDENCE
**If a model achieves the lowest perplexity on a benchmark, what evidence would you need to conclude it's the "best" model?**

*Consider: What tasks will the model be used for? Does low perplexity on the benchmark translate to those tasks?*

### Question 2: ASSUMPTIONS
**What assumptions are we making when we use perplexity computed on a held-out test set to evaluate a model?**

*Consider: Is the test set representative of real usage? Could the model have seen similar data during training?*

### Question 3: IMPLICATIONS
**If a company optimizes solely for perplexity when training their LLM, what are the potential consequences?**

*Consider: Model behavior, safety, usefulness, diversity of outputs.*

---

# 🔄 Section 5 — Inversion Thinking: When Can Good Perplexity Fail in Production?

Instead of asking "What makes a good perplexity score?", let's invert:

> **"When can a model have good perplexity but fail in production?"**

### Failure Scenarios

1. **Distribution Shift**
   - Model trained on formal text, deployed for casual chat
   - Low perplexity on formal test set, but users write informally
   - Result: Model seems "confused" by real user inputs

2. **Memorization Without Understanding**
   - Model memorized training data verbatim
   - Achieves near-perfect perplexity on similar text
   - Fails when asked to reason, generalize, or apply knowledge

3. **Confident Hallucination**
   - Model assigns high probability to plausible-sounding but false statements
   - Low perplexity because the text "sounds right"
   - Users receive misinformation delivered confidently

4. **Narrow Domain Excellence**
   - Model excels at Wikipedia-style factual text
   - Fails at code, math, creative writing, or conversation
   - Perplexity benchmark doesn't cover deployment domain

5. **Safety Blindspot**
   - Model predicts harmful content with high confidence
   - Low perplexity on toxic text (if such text was in training)
   - Deployed model produces harmful outputs

### Defensive Practice

**Always pair perplexity with task-specific evaluation:**
- Accuracy on question-answering
- Human preference ratings
- Safety benchmarks (TruthfulQA, ToxiGen)
- Domain-specific tests

---

# 📝 Section 6 — Mini-Project: Compare Perplexity Across Text Domains

### Task

Compare perplexity of tinyGPT on two different text domains:
1. **Formal text** (news articles, Wikipedia)
2. **Casual text** (chat messages, social media style)

### Instructions

1. **Collect sample texts** (5-10 sentences per domain):
   - Domain A: Formal news-style sentences
   - Domain B: Casual conversational sentences

2. **Compute perplexity** for each sentence using the pseudo-perplexity function

3. **Analyze results:**
   - Which domain has lower average perplexity?
   - What does this tell you about the model's training data?
   - Are there any surprising outliers?

4. **Visualize:** Create a histogram or box plot comparing perplexities

### Sample Data to Get Started

**Formal/News Domain:**
```python
formal_texts = [
    "The Federal Reserve announced a quarter-point interest rate increase.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "The United Nations Security Council convened for an emergency session.",
    "Economic indicators suggest moderate growth in the coming quarter.",
    "The research team published their findings in a peer-reviewed journal.",
]
```

**Casual/Chat Domain:**
```python
casual_texts = [
    "hey whats up, u free tonight?",
    "lol that was so funny i cant even",
    "gonna grab some food, want anything?",
    "ngl this new song is fire",
    "omg did u see what happened yesterday??",
]
```

### Submission Format

Create a markdown file with:
- Your sample texts for each domain
- Perplexity scores for each sentence
- Summary statistics (mean, std) per domain
- A brief analysis (2-3 paragraphs) explaining your findings
- One key insight about what perplexity reveals about model training

---

# ✔ Knowledge Mastery Checklist
- [ ] I can explain perplexity using a simple guessing game analogy
- [ ] I understand perplexity as exponentiated average negative log-likelihood
- [ ] I know why perplexity alone doesn't guarantee real-world usefulness
- [ ] I can compute approximate perplexity using an ONNX model
- [ ] I understand when good perplexity can still lead to production failures
- [ ] I completed the mini-project comparing perplexity across domains

---

Week 3 complete.
Next: *Week 4 — Industry Benchmark Suites (MMLU, HellaSwag, BBH, TruthfulQA, ToxiGen)*.
