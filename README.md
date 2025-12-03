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
| 8 | [Robustness Tests](week8.md) |
| 9 | [Performance Profiling](week9.md) |
| 10 | [Regression & Version Comparison](week10.md) |
| 11 | [Banking & Compliance Evaluation](week11.md) |
| 12 | [Healthcare & Safety Evaluation](week12.md) |
| 13–16 | Industry Use Cases (software engineering, data analytics, RAG, marketing) |
| 17–18 | Full System Architecture & Capstone |

See [docs/syllabus.md](docs/syllabus.md) for detailed learning objectives.

### Week 1: Foundations of LLM Evaluation & First Principles

- **Foundational Evaluation Concepts:** Learn the 4 pillars of LLM evaluation (Quantitative, Qualitative, Safety, Performance) and apply First Principles, Feynman Technique, and Paul-Elder Critical Thinking frameworks
- **First ONNX Run:** Load and run your first tinyGPT ONNX model in Google Colab, understanding the inference pipeline from tokenization to output
- **First Latency Benchmark:** Measure inference latency on multiple prompts and display results in a structured DataFrame

📓 **Notebook:** [week1_intro_evaluation.ipynb](week1_intro_evaluation.ipynb)  
📋 **Grading Rubric:** [docs/week1_rubric.md](docs/week1_rubric.md)

### Week 2: Tokenization & ONNX Runtime Internals

- **Tokenization Fundamentals:** Understand what tokens are and how BPE, WordPiece, and SentencePiece algorithms break text into model-consumable pieces
- **Token vs. Character Analysis:** Learn why token count matters more than character count for latency and cost predictions
- **ONNX Runtime Profiling:** Enable profiling to inspect operator-level timing breakdowns and identify performance bottlenecks

📓 **Notebook:** [week2_tokenization_internals.ipynb](week2_tokenization_internals.ipynb)  
📄 **Content:** [week2.md](week2.md)

### Week 3: Perplexity & Basic Benchmarks

- **Perplexity Explained:** Understand perplexity as exponentiated average negative log-likelihood and what it measures about language models
- **Limitations of Perplexity:** Recognize when good perplexity scores can still lead to production failures
- **Pseudo-Perplexity Computation:** Compute approximate perplexity using ONNX models and analyze results across text domains

📓 **Notebook:** [week3_perplexity.ipynb](week3_perplexity.ipynb)  
📄 **Content:** [week3.md](week3.md)

### Week 4: Industry Benchmark Suites (MMLU, HellaSwag, BBH)

- **MMLU Evaluation:** Measure world knowledge and problem-solving ability across 57 subjects
- **HellaSwag Testing:** Test commonsense reasoning through adversarially-generated story completions
- **TruthfulQA & ToxiGen:** Evaluate truthfulness and toxicity detection capabilities

📓 **Notebook:** [week4_industry_benchmarks.ipynb](week4_industry_benchmarks.ipynb)  
📄 **Content:** [week4.md](week4.md)

### Week 5: Building a Generic Benchmark Engine

- **Generic Evaluation Loop:** Create reusable benchmark infrastructure that works with any model, dataset, and metric
- **Function Interface Pattern:** Use callable interfaces for models and metrics to enable flexible evaluation pipelines
- **Built-in Metrics:** Implement exact match, contains match, and custom metric functions for scoring

📓 **Notebook:** [week5_generic_benchmark_engine.ipynb](week5_generic_benchmark_engine.ipynb)  
📄 **Content:** [week5.md](week5.md)

### Week 6: LLM-as-Judge: Automated Evaluation

- **LLM-as-Judge Concept:** Use powerful language models to evaluate outputs from other models at scale
- **Evaluation Prompt Design:** Create effective system prompts that assess correctness, coherence, and helpfulness
- **Trade-offs Analysis:** Understand when to use automated evaluation vs. human evaluation

📓 **Notebook:** [week6_llm_as_judge.ipynb](week6_llm_as_judge.ipynb)  
📄 **Content:** [week6.md](week6.md)

### Week 7: Safety & Hallucination Evaluation

- **TruthfulQA Evaluation:** Test model truthfulness by detecting responses that propagate common misconceptions
- **ToxiGen Evaluation:** Test for toxic or harmful content generation using automated classification
- **Safety Metrics:** Understand truthful/untruthful ratios and toxicity classification methods

📓 **Notebook:** [week7_safety_hallucination.ipynb](week7_safety_hallucination.ipynb)  
📄 **Content:** [week7.md](week7.md)

### Week 8: Robustness Tests

- **Prompt Perturbation:** Apply typos, synonyms, and word reorderings to test model stability
- **Robustness Sweep:** Generate multiple perturbed variants and measure output consistency
- **Stability Metrics:** Calculate and interpret stability scores across input variations

📓 **Notebook:** [week8_robustness.ipynb](week8_robustness.ipynb)  
📄 **Content:** [week8.md](week8.md)

### Week 9: Performance Profiling

- **Latency Measurement:** Profile wall-clock inference time for ONNX models
- **Throughput Analysis:** Calculate tokens per second for performance comparisons
- **Memory Profiling:** Track memory usage during inference (when available)
- **Summary Statistics:** Generate mean, std, min, max for all performance metrics

📓 **Notebook:** [week9_performance_profiling.ipynb](week9_performance_profiling.ipynb)  
📄 **Content:** [week9.md](week9.md)

### Week 10: Regression & Version Comparison

- **Run Comparison:** Compare benchmark results across different model versions
- **Regression Detection:** Identify cases where the new model performs worse than the baseline
- **Severity Ranking:** Prioritize regressions by magnitude to focus on critical issues
- **Comprehensive Reporting:** Generate reports with summary statistics for multiple metrics

📓 **Notebook:** [week10_regression_comparison.ipynb](week10_regression_comparison.ipynb)  
📄 **Content:** [week10.md](week10.md)

### Week 11: Banking & Compliance Evaluation

- **Compliance Summarization:** Learn how LLMs can draft short summaries of regulations and KYC rules for banking and financial services
- **Correctness & Completeness Scoring:** Design evaluations that score model summaries on accuracy and whether critical details are included
- **Omission Detection:** Use LLM-as-Judge to identify what critical details (amounts, deadlines, penalties) were omitted from summaries
- **Critical Thinking:** Explore the risks of mis-summarizing regulations and when human review is essential

📓 **Notebook:** [week11_banking_compliance.ipynb](week11_banking_compliance.ipynb)  
📄 **Content:** [week11.md](week11.md)

### Week 12: Healthcare & Safety Evaluation

> ⚠️ **IMPORTANT:** This module is for **educational/synthetic purposes only**—not for real medical advice.

- **Healthcare Safety Evaluation:** Learn how to evaluate LLM responses to health questions with a focus on safety
- **Prescription Avoidance:** Design evaluations that check whether models avoid giving direct medication prescriptions
- **Professional Referral:** Assess whether models appropriately recommend consulting healthcare professionals
- **Safety Warnings:** Understand the unique risks of healthcare AI and the importance of appropriate caution

📓 **Notebook:** [week12_healthcare_safety.ipynb](week12_healthcare_safety.ipynb)  
📄 **Content:** [week12.md](week12.md)

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
