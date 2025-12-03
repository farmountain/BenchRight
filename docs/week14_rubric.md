# Week 14 Mini-Project Grading Rubric

## Overview

This rubric is used to evaluate the Week 14 mini-project: **SQL Evaluation Audit**.

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
| 3 | Code runs without errors, correctly uses SQLEvaluator with in-memory SQLite, evaluates 5+ natural language questions, computes execution rate and accuracy |
| 2 | Code runs with minor issues but demonstrates understanding of SQL evaluation |
| 1 | Code has significant errors or evaluates fewer than 5 questions |
| 0 | Code does not run or is missing essential components |

**Key checkpoints:**
- [ ] Database schema with at least 3 tables is defined
- [ ] Sample data is inserted (5-10 rows per table)
- [ ] 5+ natural language questions with reference SQL
- [ ] Execution success and result correctness are both evaluated
- [ ] Metrics are computed (execution rate, accuracy)

---

### 2. Quality of Results (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Results table shows all questions with difficulty, execution status, result match, and includes analysis by difficulty level |
| 2 | Results table is mostly complete but may lack difficulty breakdown |
| 1 | Results table is incomplete or missing key metrics |
| 0 | No results provided or results are unusable |

**Expected results format:**

| # | Question Type | Difficulty | Execution | Result Match |
|---|---------------|------------|-----------|--------------|
| 1 | Simple Count | Easy | ✅ | ✅ |
| 2 | Join Query | Medium | ✅ | ❌ |

---

### 3. Interpretation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Provides executive summary, analyzes performance by difficulty, identifies patterns in execution failures vs result mismatches, provides recommendations |
| 2 | Provides basic analysis with some insights but lacks depth |
| 1 | Minimal interpretation with little to no analysis |
| 0 | No interpretation provided |

**Strong interpretations typically include:**
- Analysis of easy vs medium vs hard query performance
- Distinction between execution failures and result mismatches
- Patterns in what causes each type of failure
- Recommendations for prompt engineering and evaluation design

---

### 4. Documentation (0–3)

| Score | Criteria |
|-------|----------|
| 3 | Well-organized markdown file following the specified format, includes database schema, all sections complete, professional formatting |
| 2 | Readable documentation with minor formatting issues |
| 1 | Poorly organized or difficult to read |
| 0 | No documentation or unreadable format |

**Documentation should include:**
- [ ] Executive Summary
- [ ] Database Schema (tables, columns, relationships)
- [ ] Test Cases Evaluated table
- [ ] Metrics (Execution Rate, Accuracy)
- [ ] Analysis by Difficulty
- [ ] Failure Analysis (execution vs result)
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
# Week 14 Mini-Project: SQL Evaluation Audit

## Executive Summary

Text-to-SQL evaluation shows 80% execution rate and 60% accuracy. Easy queries achieve 100% accuracy while complex multi-table joins drop to 20%. Schema understanding is the primary challenge.

## Database Schema

### Tables

| Table | Columns | Row Count |
|-------|---------|-----------|
| customers | customer_id, name, email, city, signup_date | 5 |
| products | product_id, name, category, price | 5 |
| orders | order_id, customer_id, order_date, total_amount | 6 |
| order_items | item_id, order_id, product_id, quantity, unit_price | 8 |

### Relationships

```
customers (1) ──── (*) orders
orders (1) ──── (*) order_items
products (1) ──── (*) order_items
```

### Sample Data

- **Customers:** Alice (NY), Bob (LA), Charlie (NY), Diana (Chicago), Eve (LA)
- **Products:** Laptop ($999), Headphones ($149), Coffee Maker ($79), Desk Chair ($249), Monitor ($399)
- **Orders:** 6 orders totaling $2,429.91

## Test Cases Evaluated

| # | Question | Type | Difficulty |
|---|----------|------|------------|
| 1 | How many customers? | Count | Easy |
| 2 | Customers from New York? | Filter | Easy |
| 3 | Total revenue? | Sum | Easy |
| 4 | Average product price? | Avg | Easy |
| 5 | Customers per city? | Group By | Medium |
| 6 | Customer names with order totals? | Join | Medium |
| 7 | Orders in 2024? | Date Filter | Medium |
| 8 | Total sales by category? | Multi-Join + Group | Hard |
| 9 | Customers with 2+ orders? | Subquery | Hard |
| 10 | Most popular product? | Complex Agg | Hard |

## Evaluation Results

| # | Question | Difficulty | Execution | Result Match | Notes |
|---|----------|------------|-----------|--------------|-------|
| 1 | Count customers | Easy | ✅ | ✅ | Perfect |
| 2 | NY customers | Easy | ✅ | ✅ | Perfect |
| 3 | Total revenue | Easy | ✅ | ✅ | Perfect |
| 4 | Avg price | Easy | ✅ | ✅ | Perfect |
| 5 | Customers/city | Medium | ✅ | ✅ | Perfect |
| 6 | Names + totals | Medium | ✅ | ❌ | Missing LEFT JOIN |
| 7 | 2024 orders | Medium | ❌ | ❌ | Wrong date function |
| 8 | Sales/category | Hard | ✅ | ❌ | Wrong join order |
| 9 | 2+ orders | Hard | ❌ | ❌ | Syntax error in HAVING |
| 10 | Popular product | Hard | ✅ | ❌ | Wrong aggregation |

## Metrics

| Metric | Value |
|--------|-------|
| Execution Rate | 80% (8/10) |
| Accuracy | 50% (5/10) |

## Analysis by Difficulty

### Easy Queries (4/4)

| Metric | Value |
|--------|-------|
| Execution Rate | 100% |
| Accuracy | 100% |
| Common Issues | None |

**Observation:** Simple SELECT, COUNT, SUM, AVG on single tables are handled perfectly.

### Medium Queries (1/3)

| Metric | Value |
|--------|-------|
| Execution Rate | 67% |
| Accuracy | 33% |
| Common Issues | Date functions, join types |

**Observation:** JOINs and date filtering introduce errors. Model uses wrong date syntax for SQLite.

### Hard Queries (0/3)

| Metric | Value |
|--------|-------|
| Execution Rate | 67% |
| Accuracy | 0% |
| Common Issues | Multi-join logic, subqueries, complex aggregation |

**Observation:** Complex queries execute but return wrong results. Model struggles with multi-table relationships.

## Failure Analysis

### Execution Failures (2 cases)

**Case 1: Date Filtering**
- **Question:** "How many orders in 2024?"
- **Generated:** `WHERE YEAR(order_date) = 2024`
- **Issue:** SQLite uses `strftime('%Y', ...)` not `YEAR()`
- **Root cause:** Model defaulting to MySQL/PostgreSQL syntax

**Case 2: Subquery Syntax**
- **Question:** "Customers with 2+ orders?"
- **Generated:** `HAVING COUNT(*) > 1` without GROUP BY
- **Issue:** HAVING requires GROUP BY
- **Root cause:** Incomplete query structure

### Result Mismatches (3 cases)

**Case 1: Join Type Error**
- **Question:** "Customer names with order totals"
- **Generated:** `INNER JOIN` when `LEFT JOIN` needed
- **Issue:** Customers without orders excluded
- **Root cause:** Didn't consider NULL cases

**Case 2: Multi-Join Order**
- **Question:** "Sales by category"
- **Generated:** Joined tables in wrong order, missed products without sales
- **Issue:** Incorrect aggregation grouping

**Case 3: Wrong Aggregation**
- **Question:** "Most popular product by quantity"
- **Generated:** Used MAX(quantity) instead of SUM(quantity)
- **Issue:** Misunderstood "most popular" = total quantity, not single purchase

## Recommendations

### For Prompt Engineering

1. **Include dialect specification:**
   - Explicitly state "Write SQLite query" in prompts
   - Provide dialect-specific function examples

2. **Add schema context:**
   - Include foreign key relationships in prompt
   - Show sample data to clarify data types

3. **Request query explanation:**
   - Ask model to explain logic before writing SQL
   - Catch conceptual errors early

### For Evaluation Design

1. **Test dialect awareness:**
   - Include date functions, string functions that vary by dialect
   
2. **Add NULL handling tests:**
   - Questions that require LEFT JOIN vs INNER JOIN decisions

3. **Semantic equivalence:**
   - Allow multiple correct SQL forms for same question

## Limitations

### What This Evaluation Cannot Assess

1. **Query performance:** No execution time limits or EXPLAIN analysis
2. **Large scale:** Only tested on small sample data
3. **Schema complexity:** Real databases have many more tables
4. **Cross-dialect:** Only tested SQLite

### Future Improvements

1. Add PostgreSQL and MySQL testing
2. Test with larger datasets (1000+ rows)
3. Include query optimization evaluation
4. Add security checks (SQL injection patterns)
```

---

## Feedback Guidelines

When providing feedback, focus on:
1. Quality of database schema design
2. Diversity of query difficulty levels
3. Distinction between execution and result failures
4. Practical recommendations for improvement
