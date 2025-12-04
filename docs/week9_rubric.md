# Week 9 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 9 mini-project: **Performance Audit Report**.

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
| 3 | Code runs without errors, correctly uses profile_model on 10-20 prompts, captures latency, tokens/sec, and memory metrics, performs warmup runs |
| 2 | Code runs with minor issues but demonstrates understanding of performance profiling |
| 1 | Code has significant errors or profiles fewer than 10 prompts |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] 10-20 representative prompts of varied lengths are selected
- [ ] profile_model or PerformanceProfiler is used correctly
- [ ] Warmup runs are performed before measurement
- [ ] Multiple runs are used for statistical stability
- [ ] Latency, tokens/sec, and memory are captured

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows all prompts with metrics, includes summary statistics (mean, std, min, max), identifies correlations (e.g., input length vs latency) |
| 2 | Results table is mostly complete but may lack summary statistics |
| 1 | Results table is incomplete or missing key metrics |
| 0 | No results provided or results are unusable |

**Expected results format:**

### Performance Metrics
| Prompt | Input Tokens | Latency (ms) | Tokens/sec | Memory (MB) |
|--------|--------------|--------------|------------|-------------|
| ...    | ...          | ...          | ...        | ...         |

### Summary Statistics
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Latency (ms) | ??? | ??? | ??? | ??? |
| Tokens/sec | ??? | ??? | ??? | ??? |
| Memory (MB) | ??? | ??? | ??? | ??? |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides executive summary, key findings with 3-5 bullet points, analyzes correlations between metrics, provides actionable recommendations |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Identification of slowest/fastest prompts and reasons
- Correlation analysis (input length vs latency)
- Comparison to production requirements/SLAs
- Discussion of bottlenecks and optimization opportunities

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file following the specified format, includes all required sections, professional formatting |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Executive Summary
- [ ] Model Under Test description
- [ ] Test Configuration (hardware, warmup runs, measurement runs)
- [ ] Performance Metrics table
- [ ] Summary Statistics table
- [ ] Key Findings
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
# Week 9 Mini-Project: Performance Audit Report

## Executive Summary

TinyGPT demonstrates acceptable latency for interactive applications with a mean inference time of 48.5ms. Throughput averages 1,240 tokens/second with consistent memory usage around 180MB.

## Model Under Test

**Model:** tinyGPT (ONNX)  
**Parameters:** ~10M  
**Format:** ONNX Runtime (CPU)  
**Evaluation Date:** [Date]

## Test Configuration

- **Hardware:** Intel Core i7-10700 @ 2.9GHz, 32GB RAM
- **Python:** 3.10.12
- **ONNX Runtime:** 1.16.0
- **Warmup runs:** 3
- **Measurement runs:** 5 per prompt

## Performance Metrics

| Prompt | Input Tokens | Latency (ms) | Tokens/sec | Memory (MB) |
|--------|--------------|--------------|------------|-------------|
| What is 2+2? | 5 | 22.3 | 1,680 | 175 |
| Capital of France? | 6 | 25.8 | 1,550 | 176 |
| Explain AI briefly | 8 | 32.1 | 1,420 | 178 |
| Describe photosynthesis | 10 | 41.2 | 1,285 | 180 |
| Machine learning overview | 12 | 48.5 | 1,215 | 182 |
| History of computing | 15 | 58.9 | 1,102 | 185 |
| Climate change effects | 18 | 72.4 | 980 | 188 |
| Quantum physics intro | 20 | 85.6 | 892 | 191 |
| Full essay on economics | 35 | 142.3 | 645 | 205 |
| Long document summary | 50 | 198.7 | 512 | 218 |

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Input Tokens | 17.9 | 14.2 | 5 | 50 |
| Latency (ms) | 72.8 | 55.3 | 22.3 | 198.7 |
| Tokens/sec | 1,128 | 378 | 512 | 1,680 |
| Memory (MB) | 187.8 | 13.6 | 175 | 218 |

## Key Findings

1. **Strong correlation between input length and latency:** Linear relationship observed (R² ≈ 0.97). Each additional input token adds ~3.5ms of latency.

2. **Throughput degrades with longer inputs:** Short prompts achieve ~1,680 tokens/sec while long prompts drop to ~512 tokens/sec—a 3x difference.

3. **Consistent memory footprint:** Memory usage increases modestly with input length (~0.8MB per 10 tokens), indicating efficient memory management.

4. **P95 latency suitable for interactive use:** 95th percentile latency is ~165ms, acceptable for most chat applications.

5. **Cold start not significant:** After warmup, latency variance was minimal (std ~55ms mainly due to input length variation).

## Correlation Analysis

```
Input Tokens vs Latency:    r = 0.98 (strong positive)
Input Tokens vs Tokens/sec: r = -0.91 (strong negative)
Latency vs Memory:          r = 0.89 (moderate positive)
```

## Recommendations

1. **For Interactive Applications:**
   - ✅ Model meets <100ms SLA for prompts up to 20 tokens
   - ⚠️ Consider input length limits for real-time use

2. **For Batch Processing:**
   - Optimal throughput achieved with shorter prompts
   - Consider batching inputs of similar lengths

3. **Memory Optimization:**
   - Current usage is efficient
   - Can handle ~50 concurrent requests within 8GB

4. **Latency Reduction:**
   - Consider GPU acceleration for longer prompts
   - Implement input truncation for predictable latency

## Limitations

This evaluation did NOT test:
- GPU performance characteristics
- Batch inference throughput
- Concurrent request handling
- Long-running memory stability (memory leaks)
- Different quantization levels
- Network latency (for API deployment)
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Diversity of prompt lengths tested
2. Quality of statistical analysis
3. Identification of performance correlations
4. Practicality of recommendations for production use
