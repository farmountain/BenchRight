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
| 2 | Tokenization & ONNX Runtime Internals |
| 3 | Perplexity & Basic Benchmarks |
| 4 | Industry Benchmark Suites (MMLU, HellaSwag, BBH) |
| 5–10 | Automated Evaluation Pipelines (LLM-as-judge, safety, robustness, performance, regression tests) |
| 11–16 | Industry Use Cases (banking, healthcare, software engineering, data analytics, RAG, marketing) |
| 17–18 | Full System Architecture & Capstone |

See [docs/syllabus.md](docs/syllabus.md) for detailed learning objectives.

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
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── week1.md               # Week 1 content
├── docs/
│   └── syllabus.md        # 18-week curriculum overview
├── examples/              # Example outputs and results
└── LICENSE                # Apache 2.0 License
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
