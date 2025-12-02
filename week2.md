# Week 2 — Tokenization & ONNX Runtime Internals
### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 2, you will:

1. Understand *what a token is* and why tokenization is fundamental to LLM evaluation.
2. Learn how different tokenization strategies (BPE, WordPiece, SentencePiece) affect input representation.
3. Analyze the relationship between token count, input length, and inference latency.
4. Understand ONNX Runtime *execution providers* and how they affect performance and cost.
5. Enable ONNX Runtime profiling and interpret operator-level timing breakdowns.
6. Recognize how tokenization choices can mislead evaluation metrics.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is Tokenization?

### Simple Explanation

Imagine you're translating a sentence into a different language. Before translating, you need to break the sentence into smaller pieces—words, phrases, or even syllables. **Tokenization** is exactly this: breaking text into smaller units called **tokens** that the model can understand.

> **A token is the smallest unit of text that a language model processes.**

Tokens are not always whole words:
- `"unhappiness"` might become `["un", "happiness"]` (2 tokens)
- `"AI"` might be 1 token
- `"artificial intelligence"` might be `["artificial", " intelligence"]` (2 tokens)

### Why Does Tokenization Matter for Evaluation?

1. **Latency scales with token count:** More tokens = more computation = higher latency.
2. **Token count ≠ word count:** A prompt with 50 words might have 70 tokens—or 40.
3. **Comparison fairness:** Comparing latency of prompts requires understanding their token counts.
4. **Cost implications:** API pricing is often per-token, so tokenization directly affects cost.

**Key insight:** Two prompts with the *same character length* can have *very different token counts*, leading to different latencies. This is why we must always consider tokenization in evaluation.

---

# 🧠 Section 2 — ONNX Runtime Execution Providers

### What is an Execution Provider?

An **execution provider (EP)** is the backend that ONNX Runtime uses to run model operations. Think of it as choosing which "engine" powers your model:

| Execution Provider | Hardware | Use Case |
|-------------------|----------|----------|
| **CPUExecutionProvider** | CPU | Default, works everywhere |
| **CUDAExecutionProvider** | NVIDIA GPU | High throughput, low latency |
| **TensorrtExecutionProvider** | NVIDIA GPU | Optimized for production inference |
| **CoreMLExecutionProvider** | Apple Silicon | macOS/iOS deployment |
| **DirectMLExecutionProvider** | Windows GPU | Cross-vendor GPU on Windows |

### Why Do Execution Providers Affect Latency and Cost?

1. **Hardware acceleration:** GPU providers can be 10-100x faster than CPU for large models.
2. **Memory management:** Different EPs have different memory allocation strategies.
3. **Operator coverage:** Not all EPs support all ONNX operators—some fall back to CPU.
4. **Optimization level:** Some EPs apply graph optimizations that reduce computation.
5. **Cost trade-off:** GPU instances cost more but may be cheaper per-token due to speed.

**Key insight:** Switching from CPU to GPU doesn't just change speed—it can change the *cost-performance trade-off* of your entire evaluation pipeline.

---

# 🧪 Section 3 — Hands-on Lab: Tokenization & ONNX Profiling

This lab guides you through inspecting tokenization, measuring latency vs. token count, and enabling ONNX profiling.

### Lab Outline

#### Part 1: Inspect Tokenization of Several Prompts

**Goal:** Understand how different prompts tokenize differently.

**Steps:**
1. Load a GPT-2 tokenizer using `transformers.AutoTokenizer`
2. Define 5+ prompts with varying content (technical terms, names, numbers, punctuation)
3. For each prompt, print:
   - Original text
   - Token IDs
   - Decoded tokens
   - Token count

**Example prompts to analyze:**
- `"Hello world"` — Simple, common words
- `"Supercalifragilisticexpialidocious"` — Single long word
- `"GPT-4, BERT, and T5 are transformer models."` — Technical terms
- `"The price is $1,234.56 USD."` — Numbers and symbols
- `"こんにちは世界"` — Non-English text

---

#### Part 2: Compare Input Length vs. Latency

**Goal:** Measure how token count (not character count) correlates with latency.

**Steps:**
1. Create prompts with similar character lengths but different token counts
2. Run inference on each prompt 5+ times
3. Record: character count, token count, mean latency
4. Plot or tabulate results to visualize the relationship

**Expected insight:** Latency correlates more strongly with token count than character count.

---

#### Part 3: Enable ONNX Profiling and Inspect Operator Runtimes

**Goal:** Understand where time is spent during inference.

**Steps:**
1. Enable profiling in `ort.SessionOptions`
2. Run inference
3. Read the generated JSON profile file
4. Parse and summarize operator-level timings
5. Identify the top 5 most time-consuming operators

**ONNX Profiling Setup:**
```python
options = ort.SessionOptions()
options.enable_profiling = True

session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

# Run inference
inputs = tokenizer(prompt, return_tensors="np")
_ = session.run(None, {"input_ids": inputs["input_ids"]})

# Get profile file
profile_file = session.end_profiling()
```

---

# 🤔 Section 4 — Paul-Elder Critical Thinking Questions

Apply the Paul-Elder framework to deepen your understanding:

### Question 1: EVIDENCE
**If two models have the same accuracy on a benchmark, but one uses a tokenizer that produces 20% more tokens on average, what implications does this have for production deployment?**

*Consider: Latency, cost, throughput, user experience.*

### Question 2: ASSUMPTIONS
**What assumptions are we making when we compare latency across prompts without normalizing by token count?**

*Consider: Fairness of comparison, real-world usage patterns, benchmark validity.*

### Question 3: IMPLICATIONS
**If an organization switches from CPUExecutionProvider to CUDAExecutionProvider, what second-order effects might this have beyond just latency reduction?**

*Consider: Infrastructure requirements, cost structure, scalability, maintenance complexity.*

---

# 🔄 Section 5 — Inversion Thinking: How Can Tokenization Mislead Evaluation?

Instead of asking "How does tokenization help evaluation?", we invert:

> **"How can tokenization mislead our evaluation results?"**

### Potential Pitfalls:

1. **Character-based comparisons:** Comparing prompts by character length ignores the actual computation (token count). A "short" prompt with many rare words may be slower than a "long" prompt with common words.

2. **Tokenizer mismatch:** Using a different tokenizer for evaluation than the model was trained with produces invalid results. Token boundaries affect attention patterns and model behavior.

3. **Vocabulary artifacts:** Some tokenizers have vocabulary gaps for certain domains (medical, legal, code). This can cause excessive tokenization and artificially inflate latency.

4. **Multilingual bias:** English-centric tokenizers produce more tokens for non-English text, making cross-language comparisons unfair.

5. **Special tokens hidden cost:** Prompts with special tokens (like `[SEP]`, `<pad>`) have hidden overhead that affects both token count and computation.

6. **Tokenization time ignored:** Most benchmarks measure only inference time, but tokenization itself has overhead—especially for long inputs or suboptimal tokenizers.

### Defensive Practice:
Always report **both** character count and token count in your evaluations. Normalize latency by tokens when comparing across different prompts or models.

---

# 📝 Section 6 — Mini-Project: Same Length, Different Tokens

### Task

Compare two prompts that have the **same surface length** (character count) but **different token counts**, and analyze the difference in latency.

### Instructions

1. **Create two prompts** with exactly the same character count (aim for ~100 characters):
   - **Prompt A:** Uses common English words (should tokenize into fewer tokens)
   - **Prompt B:** Uses technical terms, rare words, or non-English text (should tokenize into more tokens)

2. **Measure for each prompt:**
   - Character count
   - Token count
   - Mean latency over 10 runs
   - Standard deviation of latency

3. **Calculate:**
   - Latency per token for each prompt
   - Percentage difference in latency between prompts

4. **Analyze and report:**
   - Does the prompt with more tokens have higher latency?
   - Is the latency difference proportional to the token count difference?
   - What does this tell us about evaluation methodology?

### Example Prompt Pair

```
# Prompt A (common words, ~100 chars):
"The sun rises in the east and sets in the west. This happens every day without fail in our world."

# Prompt B (technical terms, ~100 chars):
"GPT-4's RLHF-based fine-tuning utilizes PPO algorithms with KL-divergence constraints for alignment."
```

### Submission Format

Create a markdown file `/examples/week02_mini_project.md`:

```markdown
# Week 2 Mini-Project: Tokenization Analysis

## Prompts
| ID | Prompt | Char Count | Token Count |
|----|--------|------------|-------------|
| A  | ...    | 100        | 20          |
| B  | ...    | 100        | 35          |

## Latency Results
| ID | Mean Latency (ms) | Std Dev (ms) | Latency/Token (ms) |
|----|-------------------|--------------|---------------------|
| A  | ...               | ...          | ...                 |
| B  | ...               | ...          | ...                 |

## Analysis
[Your interpretation of results]

## Key Takeaway
[One sentence summary of what this teaches about evaluation]
```

---

# ✔ Knowledge Mastery Checklist
- [ ] I can explain what a token is in simple terms
- [ ] I understand why token count matters more than character count for latency
- [ ] I can inspect tokenization using the transformers library
- [ ] I know what ONNX execution providers are and how they affect performance
- [ ] I can enable ONNX profiling and interpret operator timings
- [ ] I understand how tokenization can mislead evaluation
- [ ] I completed the mini-project

---

Week 2 complete.
Next: *Week 3 — Perplexity & Basic Benchmarks*.
