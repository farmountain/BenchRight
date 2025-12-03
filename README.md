# BenchRight

An 18-week LLM evaluation and benchmarking training programme based on tinyGPT ONNX and Google Colab.

## Overview

BenchRight provides a comprehensive curriculum for mastering LLM evaluation techniques. The programme covers:

- LLM Evaluation fundamentals
- Benchmarking methodologies
- Safety & Hallucination Detection
- LLM-as-Judge scoring
- Enterprise Use Cases
- ONNX runtime optimization
- Google Colab hands-on labs
- Performance Profiling
- Robustness, bias, and adversarial testing
- End-to-end system design

## 18-Week Roadmap

| Week | Topic |
|------|-------|
| 1 | [Foundations of LLM Evaluation & First Principles](week1.md) |
| 2 | [Tokenization & ONNX Runtime Internals](week2.md) |
| 3 | [Perplexity & Basic Benchmarks](week3.md) |
| 4 | [Industry Benchmark Suites (MMLU, HellaSwag, BBH)](week4.md) |
| 5 | [Building a Generic Benchmark Engine](week5.md) |
| 6 | [LLM-as-Judge: Automated Evaluation](week6.md) |
| 7 | [Safety & Hallucination: TruthfulQA and ToxiGen](week7.md) |
| 8–10 | Robustness, Performance, and Regression Tests |
| 11–16 | Industry Use Cases (banking, healthcare, software engineering, data analytics, RAG, marketing) |
| 17–18 | Full System Architecture & Capstone |

See [docs/syllabus.md](docs/syllabus.md) for detailed learning objectives.

### Week 1: Foundations of LLM Evaluation & First Principles

- **Foundational Evaluation Concepts:** Learn the 4 pillars of LLM evaluation (Quantitative, Qualitative, Safety, Performance) and apply First Principles, Feynman Technique, and Paul-Elder Critical Thinking frameworks
- **First ONNX Run:** Load and run your first tinyGPT ONNX model in Google Colab, understanding the inference pipeline from tokenization to output
- **First Latency Benchmark:** Measure inference latency on multiple prompts and display results in a structured DataFrame

📓 **Notebook:** [week1_intro_evaluation.ipynb](week1_intro_evaluation.ipynb)  
📋 **Grading Rubric:** [docs/week1_rubric.md](docs/week1_rubric.md)

### Week 7: Safety & Hallucination Evaluation

- **TruthfulQA Evaluation:** Test model truthfulness by detecting responses that propagate common misconceptions
- **ToxiGen Evaluation:** Test for toxic or harmful content generation using automated classification
- **Safety Metrics:** Understand truthful/untruthful ratios and toxicity classification methods

📓 **Notebook:** [week7_safety_hallucination.ipynb](week7_safety_hallucination.ipynb)  
📄 **Content:** [week7.md](week7.md)

## How to Use

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open the weekly notebooks in Google Colab
4. Follow along with the hands-on labs and mini-projects
5. Complete the knowledge mastery checklist for each week

## Prerequisites

- Python 3.8+
- Google account (for Colab access)
- Basic understanding of machine learning concepts
- Familiarity with Python and Jupyter notebooks

## Folder Structure

```
BenchRight/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── week1.md                       # Week 1 content
├── week1_intro_evaluation.ipynb   # Week 1 Google Colab notebook
├── docs/
│   ├── syllabus.md                # 18-week curriculum overview
│   └── week1_rubric.md            # Week 1 mini-project grading rubric
├── examples/                      # Example outputs and results
└── LICENSE                        # Apache 2.0 License
```

## Evaluation Features

### Standard Benchmarks
- LAMBADA
- MMLU
- HellaSwag
- BBH
- TruthfulQA
- Toxicity Tests
- Adversarial robustness

### Performance Benchmarks
- Latency
- Throughput
- Memory
- ONNX operator profiling

### Quality Benchmarks
- Coherence
- Accuracy
- Factuality
- LLM-as-Judge (GPT-4/Claude-style scoring)

### Safety & Bias
- Truthfulness
- ToxiGen
- Red-Teaming prompts

### Reports
- PDF
- Markdown model cards
- Dashboard plots

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
