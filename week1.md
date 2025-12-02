# Week 1 — Foundations of LLM Evaluation & First Principles  
### LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives  
By the end of Week 1, you will:

1. Understand *what LLM evaluation is* and why it is the "currency" of model quality.  
2. Learn the *4 pillars of LLM evaluation*:  
   - Quantitative  
   - Qualitative  
   - Safety  
   - Performance  
3. Use Google Colab to run your first ONNX LLM benchmark.  
4. Understand evaluation using *First Principles, **Feynman Technique, **Design Thinking*,  
   *Paul–Elder Critical Thinking, **Inversion Thinking, and **Reflexion Loops*.  
5. Run your very first:  
   - ONNX model load  
   - First inference  
   - Basic latency measurement  

---

# 🧠 Section 1 — First Principles of LLM Evaluation  
LLM evaluation answers one core question:

> *"How good is this model—objectively, reproducibly, and safely?"*

Using first principles, any evaluation breaks into:

### *1. What does the model understand?*  
This is language capability.  
Measured with: perplexity, accuracy, benchmarks.

### *2. How does it behave?*  
This is quality and coherence.  
Measured with: LLM-as-judge scoring.

### *3. Is it safe?*  
This is risk / alignment.  
Measured with: TruthfulQA, toxicity, hallucination checks.

### *4. Is it usable?*  
This is performance and cost.  
Measured with: latency, throughput, memory.

Everything else (MMLU, HellaSwag, LAMBADA, BLEU, ROUGE) is just a tool layered over these fundamentals.

---

# 🧠 Section 2 — Feynman Technique Breakdown  
We reduce LLM evaluation to a simple explanation:

> **“Evaluating an LLM is like checking a car:  
> - How fast does it go?  
> - How well does it steer?  
> - How safe is it?  
> - How much fuel does it need?”**

- Fast → throughput, latency  
- Steer → correctness, reasoning  
- Safe → hallucination, toxicity  
- Fuel → memory, compute cost  

If you cannot explain the evaluation process in simple language, you do not truly understand it.

---

# 🧠 Section 3 — Paul-Elder Critical Thinking Framework  
For every evaluation, ask:

### *CLAIM*  
What quality does the model claim?

### *EVIDENCE*  
What metric proves or disproves it?

### *REASONING*  
Why do these results occur?

### *ASSUMPTIONS*  
Is the dataset biased?  
Did the model overfit?  

### *IMPLICATIONS*  
What would happen if we trust this model in production?

This framework is used in industry model audits.

---

# 🔄 Section 4 — Inversion Thinking  
Instead of:  
“Is my model good?”

We ask:

> *"How can my model fail?"*

Failures to anticipate include:

- Hallucinations  
- Slow inference  
- Factual inconsistency  
- Safety issues  
- Bias  
- Prompt sensitivity  

This mindset leads to superior benchmarks.

---

# 🔁 Section 5 — Reflexion Loop (LLM Self-Evaluation Philosophy)  
A reflexion loop asks:

> *“Given the result, how should I improve the evaluation design?”*

This prevents biased benchmarking and teaches scientific rigor.

---

# 🧪 Section 6 — Hands-on Lab (Google Colab)

### ✔ Step 1: Install requirements
python
!pip install onnxruntime transformers


### ✔ Step 2: Load your ONNX model from /tmp
python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

model_path = "/tmp/tinygpt.onnx"
tokenizer = AutoTokenizer.from_pretrained("gpt2")

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
print("Model loaded successfully")


### ✔ Step 3: First inference  
python
prompt = "Explain artificial intelligence."
inputs = tokenizer(prompt, return_tensors="np")
outputs = session.run(None, {"input_ids": inputs["input_ids"]})
result_ids = np.argmax(outputs[0], axis=-1)[0]
print(tokenizer.decode(result_ids))


### ✔ Step 4: Basic latency benchmark  
python
import time

def benchmark(model_session, text):
    t0 = time.time()
    inputs = tokenizer(text, return_tensors="np")
    _ = model_session.run(None, {"input_ids": inputs["input_ids"]})
    t1 = time.time()
    return (t1 - t0) * 1000  # ms

lat = benchmark(session, "Hello world")
print(f"Latency: {lat:.2f} ms")


---

# 📌 Section 7 — Industry Insights  
Week 1 teaches the same foundations used by:

- OpenAI evals  
- Anthropic robust eval  
- DeepMind/Google HELM  
- Meta LLaMA model card evaluations  
- Enterprise audits (finance, healthcare)  

Industry requires:  
✔ reproducibility  
✔ documented methodology  
✔ explainability  
✔ safety verification  

This is why we will build a full benchmark system in later weeks.

---

# 📝 Section 8 — Week 1 Mini Project  
*Task:*  
Evaluate tinyGPT on 3 prompts and measure latency:

1. Explain machine learning  
2. Summarize Singapore financial system  
3. Describe a robot to a child  

Create a table:

| Prompt | Output | Latency (ms) |

Upload to /examples/week01_results.md.

---

# ✔ Knowledge Mastery Checklist  
- [ ] I can explain what LLM evaluation is  
- [ ] I know the 4 pillars of evaluation  
- [ ] I ran an ONNX model in Colab  
- [ ] I measured latency  
- [ ] I understand Feynman, Paul-Elder, inversion thinking  
- [ ] I completed the mini-project  

---

Week 1 complete.  
Next: *Week 2 — Tokenization & ONNX Runtime Internals*.
