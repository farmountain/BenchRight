# Week 14 — Data Analytics & SQL

### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 14, you will:

1. Understand *why SQL and analytics reasoning evaluation is critical* for data analytics use cases.
2. Learn how to design evaluation scenarios for *natural language to SQL translation*.
3. Implement an evaluation system using an *in-memory SQLite database* to verify query correctness.
4. Use a *two-metric evaluation approach*: query execution success and result correctness.
5. Apply critical thinking to evaluate the suitability of LLMs for data analytics tasks.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is SQL and Analytics Reasoning Evaluation?

### Simple Explanation

Imagine you're evaluating an AI assistant that helps data analysts write SQL queries:

- **The input** is a database schema plus a natural language question
- **A good response** is a SQL query that runs successfully and returns the correct answer
- **A bad response** is a query that either fails to execute or returns wrong results

> **SQL and analytics reasoning evaluation tests whether LLMs can translate natural language questions into correct SQL queries. Unlike general text generation, SQL evaluation has objective correctness criteria: the query must execute without errors AND produce the expected result set.**

This is different from other use cases because:
- **Binary syntax correctness:** SQL either parses or it doesn't—there's no "partially valid" syntax
- **Verifiable outputs:** We can run the query and compare results to expected answers
- **Schema dependency:** The model must understand table structures and relationships
- **Semantic precision:** Small changes in SQL (e.g., `>` vs `>=`) produce different results

### The Data Analytics Evaluation Domain

| Task Type | What To Evaluate | Metric |
|-----------|------------------|--------|
| **Text-to-SQL** | Generate SQL from natural language | Query execution + result match |
| **SQL Correction** | Fix buggy SQL queries | Query execution + result match |
| **Query Optimization** | Improve query performance | Execution time + result match |
| **Schema Understanding** | Answer questions about data model | Accuracy |
| **Result Interpretation** | Explain query results | LLM-as-Judge |

---

# 🧠 Section 2 — The Scenario: Natural Language to SQL Translation

### Scenario Description

We are evaluating how an LLM performs on text-to-SQL tasks using an in-memory SQLite database:
1. **Input:** Database schema + natural language question
2. **Output:** SQL query
3. **Evaluation:** Does the query run? Does it return the correct result?

### Designing the Text-to-SQL Dataset

A good text-to-SQL dataset contains:
1. **Schema Definition:** CREATE TABLE statements and sample data
2. **Natural Language Question:** What the user wants to know
3. **Reference SQL:** A correct query that answers the question
4. **Expected Result:** The correct answer to compare against

### Example Database: Sales Analytics

```sql
-- Schema Definition
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    city TEXT,
    signup_date DATE
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price REAL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    total_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    unit_price REAL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

### Example Text-to-SQL Cases

#### Example 1: Simple Aggregation
```
Question: "How many customers are there in total?"

Reference SQL:
SELECT COUNT(*) AS customer_count FROM customers;

Expected Result: A single value (the count of customers)
```

#### Example 2: Filtering with Conditions
```
Question: "What are the names of customers from New York?"

Reference SQL:
SELECT name FROM customers WHERE city = 'New York';

Expected Result: List of customer names from New York
```

#### Example 3: Join with Aggregation
```
Question: "What is the total revenue from orders?"

Reference SQL:
SELECT SUM(total_amount) AS total_revenue FROM orders;

Expected Result: A single sum value
```

#### Example 4: Complex Multi-Table Query
```
Question: "Which product category has the highest total sales?"

Reference SQL:
SELECT p.category, SUM(oi.quantity * oi.unit_price) AS total_sales
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.category
ORDER BY total_sales DESC
LIMIT 1;

Expected Result: Category name with highest sales
```

#### Example 5: Date-Based Filtering
```
Question: "How many orders were placed in 2024?"

Reference SQL:
SELECT COUNT(*) AS order_count 
FROM orders 
WHERE strftime('%Y', order_date) = '2024';

Expected Result: Count of orders in 2024
```

### What Makes SQL Evaluation Challenging

| Challenge | Why It's Hard | Risk if Failed |
|-----------|---------------|----------------|
| **Schema understanding** | Model must know table structures | Invalid column references |
| **Join logic** | Complex relationships between tables | Missing or duplicate data |
| **Aggregation** | GROUP BY, HAVING clauses | Incorrect calculations |
| **SQL dialects** | SQLite vs PostgreSQL vs MySQL | Syntax errors |

---

# 🧪 Section 3 — Designing the SQL Evaluation System

### Evaluation Design

For text-to-SQL tasks, we use a two-metric evaluation approach:

1. **Execution Success:** Does the query run without errors?
2. **Result Match:** Does the query return the correct result?

### The SQLEvaluator Class

```python
import sqlite3
from typing import Dict, List, Any, Tuple, Optional


class SQLEvaluator:
    """
    Evaluator for text-to-SQL tasks.
    
    Uses an in-memory SQLite database to verify:
    1. Query execution success
    2. Result correctness (compared to reference answer)
    """
    
    def __init__(self, schema_sql: str, data_sql: str):
        """
        Initialize the SQLEvaluator with a database schema and data.
        
        Args:
            schema_sql: SQL statements to create tables
            data_sql: SQL statements to insert sample data
        """
        self.schema_sql = schema_sql
        self.data_sql = data_sql
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the in-memory SQLite database."""
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        
        # Create schema
        for statement in self.schema_sql.split(';'):
            statement = statement.strip()
            if statement:
                self.cursor.execute(statement)
        
        # Insert data
        for statement in self.data_sql.split(';'):
            statement = statement.strip()
            if statement:
                self.cursor.execute(statement)
        
        self.conn.commit()
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Dictionary with:
            - success: bool indicating if query executed
            - result: query results if successful
            - error: error message if failed
        """
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description] if self.cursor.description else []
            
            return {
                "success": True,
                "result": results,
                "columns": columns,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "columns": [],
                "error": str(e),
            }
    
    def compare_results(
        self,
        generated_result: List[Tuple],
        reference_result: List[Tuple],
        order_matters: bool = False,
    ) -> bool:
        """
        Compare generated query results with reference results.
        
        Args:
            generated_result: Results from generated query
            reference_result: Expected reference results
            order_matters: Whether row order should match
            
        Returns:
            True if results match, False otherwise
        """
        if generated_result is None or reference_result is None:
            return False
        
        if order_matters:
            return generated_result == reference_result
        else:
            # Compare as sets of tuples
            return set(generated_result) == set(reference_result)
    
    def evaluate_query(
        self,
        generated_sql: str,
        reference_sql: str,
        order_matters: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a generated SQL query.
        
        Args:
            generated_sql: The SQL query to evaluate
            reference_sql: Reference SQL with correct answer
            order_matters: Whether result order should match
            
        Returns:
            Dictionary with:
            - execution_success: bool
            - result_match: bool
            - generated_result: results from generated query
            - reference_result: results from reference query
            - error: error message if any
        """
        # Execute reference query to get expected result
        ref_execution = self.execute_query(reference_sql)
        if not ref_execution["success"]:
            return {
                "execution_success": False,
                "result_match": False,
                "generated_result": None,
                "reference_result": None,
                "error": f"Reference query failed: {ref_execution['error']}",
            }
        
        # Execute generated query
        gen_execution = self.execute_query(generated_sql)
        
        if not gen_execution["success"]:
            return {
                "execution_success": False,
                "result_match": False,
                "generated_result": None,
                "reference_result": ref_execution["result"],
                "error": gen_execution["error"],
            }
        
        # Compare results
        results_match = self.compare_results(
            gen_execution["result"],
            ref_execution["result"],
            order_matters=order_matters,
        )
        
        return {
            "execution_success": True,
            "result_match": results_match,
            "generated_result": gen_execution["result"],
            "reference_result": ref_execution["result"],
            "error": None,
        }
    
    def compute_metrics(
        self, 
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics across multiple evaluations.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with:
            - execution_rate: proportion of queries that executed
            - accuracy: proportion of queries with correct results
        """
        if not results:
            return {"execution_rate": 0.0, "accuracy": 0.0}
        
        executed = sum(1 for r in results if r["execution_success"])
        correct = sum(1 for r in results if r["result_match"])
        
        return {
            "execution_rate": executed / len(results),
            "accuracy": correct / len(results),
        }
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
```

### Usage Example

```python
# Define schema
SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    total_amount REAL
)
"""

# Define sample data
DATA_SQL = """
INSERT INTO customers VALUES (1, 'Alice', 'New York');
INSERT INTO customers VALUES (2, 'Bob', 'Los Angeles');
INSERT INTO customers VALUES (3, 'Charlie', 'New York');
INSERT INTO orders VALUES (1, 1, 100.00);
INSERT INTO orders VALUES (2, 1, 150.00);
INSERT INTO orders VALUES (3, 2, 200.00);
"""

# Initialize evaluator
evaluator = SQLEvaluator(SCHEMA_SQL, DATA_SQL)

# Test a query
generated_sql = "SELECT COUNT(*) FROM customers WHERE city = 'New York'"
reference_sql = "SELECT COUNT(*) FROM customers WHERE city = 'New York'"

result = evaluator.evaluate_query(generated_sql, reference_sql)

print(f"Execution Success: {result['execution_success']}")
print(f"Result Match: {result['result_match']}")
print(f"Generated Result: {result['generated_result']}")
print(f"Reference Result: {result['reference_result']}")
```

---

# 🧪 Section 4 — Implementing the Text-to-SQL Evaluation Pipeline

### Complete Evaluation Pipeline

```python
from typing import Callable, Iterator, Tuple

# Prompt template for text-to-SQL
TEXT_TO_SQL_PROMPT = """You are a SQL expert. Given the following database schema and question, write a SQL query that answers the question.

## Database Schema
{schema}

## Question
{question}

## Instructions
1. Write a valid SQLite query
2. Return ONLY the SQL query, no explanations
3. Do not include markdown code blocks

## SQL Query:
"""


def create_text_to_sql_prompt(schema: str, question: str) -> str:
    """Create a prompt for text-to-SQL tasks."""
    return TEXT_TO_SQL_PROMPT.format(
        schema=schema,
        question=question,
    )


def run_text_to_sql_benchmark(
    model_fn: Callable[[str], str],
    evaluator: SQLEvaluator,
    dataset: List[Dict[str, str]],
    schema_description: str,
) -> Dict[str, Any]:
    """
    Run a text-to-SQL benchmark.
    
    Args:
        model_fn: Function that takes a prompt and returns SQL
        evaluator: SQLEvaluator instance
        dataset: List of {question, reference_sql} dictionaries
        schema_description: Human-readable schema description
        
    Returns:
        Benchmark results with metrics
    """
    results = []
    
    for item in dataset:
        # Create prompt
        prompt = create_text_to_sql_prompt(
            schema=schema_description,
            question=item["question"],
        )
        
        # Generate SQL
        generated_sql = model_fn(prompt)
        
        # Evaluate
        eval_result = evaluator.evaluate_query(
            generated_sql=generated_sql,
            reference_sql=item["reference_sql"],
            order_matters=item.get("order_matters", False),
        )
        
        results.append({
            "question": item["question"],
            "generated_sql": generated_sql,
            "reference_sql": item["reference_sql"],
            **eval_result,
        })
    
    # Compute metrics
    metrics = evaluator.compute_metrics(results)
    
    return {
        "results": results,
        "metrics": metrics,
        "total_examples": len(results),
    }
```

### Creating a Text-to-SQL Dataset

```python
# Example dataset for the sales analytics schema
TEXT_TO_SQL_DATASET = [
    {
        "question": "How many customers are there in total?",
        "reference_sql": "SELECT COUNT(*) FROM customers",
        "order_matters": False,
    },
    {
        "question": "What are the names of customers from New York?",
        "reference_sql": "SELECT name FROM customers WHERE city = 'New York'",
        "order_matters": False,
    },
    {
        "question": "What is the total revenue from all orders?",
        "reference_sql": "SELECT SUM(total_amount) FROM orders",
        "order_matters": False,
    },
    {
        "question": "How many orders did customer with ID 1 place?",
        "reference_sql": "SELECT COUNT(*) FROM orders WHERE customer_id = 1",
        "order_matters": False,
    },
    {
        "question": "What is the average order amount?",
        "reference_sql": "SELECT AVG(total_amount) FROM orders",
        "order_matters": False,
    },
]
```

---

# 🧪 Section 5 — Hands-on Lab: Evaluating SQL Generation

### Lab Overview

In this lab, you will:
1. Create an in-memory SQLite database with sample data
2. Define text-to-SQL test cases
3. Evaluate SQL queries for execution success and result correctness
4. Analyze which types of queries are hardest to generate correctly

### Step 1: Define the Database Schema and Data

```python
import sqlite3

# Sales analytics database schema
SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    city TEXT,
    signup_date DATE
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price REAL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    total_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    unit_price REAL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
"""

# Sample data
DATA_SQL = """
INSERT INTO customers VALUES (1, 'Alice Johnson', 'alice@example.com', 'New York', '2023-01-15');
INSERT INTO customers VALUES (2, 'Bob Smith', 'bob@example.com', 'Los Angeles', '2023-02-20');
INSERT INTO customers VALUES (3, 'Charlie Brown', 'charlie@example.com', 'New York', '2023-03-10');
INSERT INTO customers VALUES (4, 'Diana Lee', 'diana@example.com', 'Chicago', '2024-01-05');
INSERT INTO customers VALUES (5, 'Eve Wilson', 'eve@example.com', 'Los Angeles', '2024-02-15');

INSERT INTO products VALUES (1, 'Laptop', 'Electronics', 999.99);
INSERT INTO products VALUES (2, 'Headphones', 'Electronics', 149.99);
INSERT INTO products VALUES (3, 'Coffee Maker', 'Appliances', 79.99);
INSERT INTO products VALUES (4, 'Desk Chair', 'Furniture', 249.99);
INSERT INTO products VALUES (5, 'Monitor', 'Electronics', 399.99);

INSERT INTO orders VALUES (1, 1, '2024-01-10', 1149.98);
INSERT INTO orders VALUES (2, 1, '2024-02-15', 79.99);
INSERT INTO orders VALUES (3, 2, '2024-01-20', 399.99);
INSERT INTO orders VALUES (4, 3, '2024-02-01', 249.99);
INSERT INTO orders VALUES (5, 4, '2024-03-01', 549.98);
INSERT INTO orders VALUES (6, 2, '2024-03-15', 149.99);

INSERT INTO order_items VALUES (1, 1, 1, 1, 999.99);
INSERT INTO order_items VALUES (2, 1, 2, 1, 149.99);
INSERT INTO order_items VALUES (3, 2, 3, 1, 79.99);
INSERT INTO order_items VALUES (4, 3, 5, 1, 399.99);
INSERT INTO order_items VALUES (5, 4, 4, 1, 249.99);
INSERT INTO order_items VALUES (6, 5, 2, 2, 149.99);
INSERT INTO order_items VALUES (7, 5, 4, 1, 249.99);
INSERT INTO order_items VALUES (8, 6, 2, 1, 149.99);
"""

print("📊 Database schema and sample data defined!")
print("   Tables: customers, products, orders, order_items")
```

### Step 2: Define Test Cases

```python
# Comprehensive test cases with varying difficulty
test_cases = [
    {
        "name": "Simple Count",
        "question": "How many customers are there?",
        "reference_sql": "SELECT COUNT(*) FROM customers",
        "difficulty": "easy",
    },
    {
        "name": "Filter by City",
        "question": "What are the names of customers from New York?",
        "reference_sql": "SELECT name FROM customers WHERE city = 'New York'",
        "difficulty": "easy",
    },
    {
        "name": "Sum Aggregation",
        "question": "What is the total revenue from all orders?",
        "reference_sql": "SELECT SUM(total_amount) FROM orders",
        "difficulty": "easy",
    },
    {
        "name": "Average Calculation",
        "question": "What is the average product price?",
        "reference_sql": "SELECT AVG(price) FROM products",
        "difficulty": "easy",
    },
    {
        "name": "Group By with Count",
        "question": "How many customers are there in each city?",
        "reference_sql": "SELECT city, COUNT(*) FROM customers GROUP BY city",
        "difficulty": "medium",
    },
    {
        "name": "Join Two Tables",
        "question": "List customer names with their order totals.",
        "reference_sql": "SELECT c.name, o.total_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id",
        "difficulty": "medium",
    },
    {
        "name": "Date Filtering",
        "question": "How many orders were placed in 2024?",
        "reference_sql": "SELECT COUNT(*) FROM orders WHERE order_date >= '2024-01-01'",
        "difficulty": "medium",
    },
    {
        "name": "Group By with Sum",
        "question": "What is the total sales per product category?",
        "reference_sql": "SELECT p.category, SUM(oi.quantity * oi.unit_price) FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category",
        "difficulty": "hard",
    },
    {
        "name": "Subquery",
        "question": "Which customers have placed more than one order?",
        "reference_sql": "SELECT c.name FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 1)",
        "difficulty": "hard",
    },
    {
        "name": "Complex Aggregation",
        "question": "What is the most popular product by quantity sold?",
        "reference_sql": "SELECT p.name, SUM(oi.quantity) as total_qty FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.product_id ORDER BY total_qty DESC LIMIT 1",
        "difficulty": "hard",
    },
]

print(f"📋 Defined {len(test_cases)} test cases:")
for tc in test_cases:
    print(f"   [{tc['difficulty']}] {tc['name']}")
```

### Step 3: Run Evaluations

```python
# Initialize evaluator
evaluator = SQLEvaluator(SCHEMA_SQL, DATA_SQL)

# Run evaluations on reference queries (should all pass)
results = []
for tc in test_cases:
    result = evaluator.evaluate_query(
        generated_sql=tc["reference_sql"],
        reference_sql=tc["reference_sql"],
    )
    
    results.append({
        "name": tc["name"],
        "difficulty": tc["difficulty"],
        **result,
    })
    
    print(f"\n{'='*60}")
    print(f"Test: {tc['name']} [{tc['difficulty']}]")
    print(f"{'='*60}")
    print(f"Question: {tc['question']}")
    print(f"SQL: {tc['reference_sql']}")
    print(f"Execution: {'✅ Success' if result['execution_success'] else '❌ Failed'}")
    print(f"Result Match: {'✅ Yes' if result['result_match'] else '❌ No'}")
    print(f"Result: {result['reference_result']}")

# Compute metrics
metrics = evaluator.compute_metrics(results)
print(f"\n📊 Metrics:")
print(f"   Execution Rate: {metrics['execution_rate']:.0%}")
print(f"   Accuracy: {metrics['accuracy']:.0%}")

# Clean up
evaluator.close()
```

---

# 🤔 Section 6 — Paul-Elder Critical Thinking Questions

### Question 1: SQL DIALECT HANDLING
**What challenges arise when evaluating SQL generation across different database systems (SQLite, PostgreSQL, MySQL)?**

*Consider: Dialect-specific syntax (e.g., LIMIT vs TOP), function differences (strftime vs date_format), data type handling, and the tradeoff between testing on one dialect vs. multiple. How might you design an evaluation that works across dialects?*

### Question 2: RESULT EQUIVALENCE
**When comparing SQL query results, should two queries that return the same data but in different column orders be considered equivalent? What about different column names?**

*Consider: The difference between syntactic and semantic equivalence, the role of ORDER BY in result comparison, alias handling, and when exact matching is necessary vs. when fuzzy matching is acceptable.*

### Question 3: SCHEMA UNDERSTANDING
**How can we evaluate whether an LLM truly understands a database schema vs. making lucky guesses based on column name patterns?**

*Consider: The challenge of testing genuine understanding, the role of adversarial schemas with misleading names, the importance of join relationship understanding, and how evaluation design affects what we actually measure.*

---

# 🔄 Section 7 — Inversion Thinking: How Can SQL Generation Fail?

Instead of asking "How does LLM SQL generation help analysts?", let's invert:

> **"How can LLM-generated SQL queries cause problems in production systems?"**

### Failure Modes

1. **Syntax Errors**
   - Invalid SQL syntax for the target database
   - Missing or extra clauses
   - Consequence: Query fails to execute

2. **Logic Errors**
   - Wrong join type (INNER vs LEFT)
   - Incorrect filter conditions
   - Consequence: Wrong results that appear correct

3. **Performance Issues**
   - Missing indexes not considered
   - Cartesian products from missing joins
   - Consequence: Slow queries, database timeouts

4. **Security Vulnerabilities**
   - SQL injection if queries are built unsafely
   - Accessing unauthorized data
   - Consequence: Data breaches, compliance violations

5. **Schema Misunderstanding**
   - Referencing non-existent columns
   - Wrong assumptions about relationships
   - Consequence: Error or incorrect data access

### Defensive Practices

- **Schema Validation:** Verify generated queries reference valid tables/columns
- **Query Explanation:** Ask the model to explain its query logic
- **Result Sampling:** Compare small result samples before full execution
- **Execution Sandboxing:** Run generated queries in read-only test environments
- **Query Review:** Human review for production-critical queries
- **Timeout Limits:** Set strict timeouts for generated queries
- **Row Limits:** Add LIMIT clauses to prevent runaway queries

---

# 📝 Section 8 — Mini-Project: Build a Text-to-SQL Evaluator

### Task

Create a complete text-to-SQL evaluation pipeline that:
1. Uses an in-memory SQLite database with sample data
2. Processes at least 5 natural language questions
3. Evaluates both execution success and result correctness
4. Analyzes which query types are hardest to generate

### Instructions

1. **Create your database:**
   - Design a schema with at least 3 tables
   - Insert sample data (5-10 rows per table)
   - Include relationships between tables

2. **Create test cases:**
   - At least 5 natural language questions
   - Include variety: aggregations, joins, filters, grouping
   - Provide reference SQL for each

3. **Run evaluations:**
   - Use SQLEvaluator to test each generated query
   - Record execution success and result match
   - Track results by query difficulty

4. **Analyze results:**
   - Which query types have highest success rates?
   - What patterns cause failures?
   - What improvements could help?

### Submission Format

Create a markdown file `/examples/week14_sql_evaluation_audit.md`:

```markdown
# Week 14 Mini-Project: SQL Evaluation Audit

## Executive Summary
[2-3 sentences on overall findings]

## Database Schema

### Tables
| Table | Columns | Row Count |
|-------|---------|-----------|
| customers | customer_id, name, city | 5 |
| orders | order_id, customer_id, total | 6 |
| ... | ... | ... |

### Relationships
[Describe foreign key relationships]

## Test Cases Evaluated

| # | Question Type | Difficulty | Execution | Result Match |
|---|---------------|------------|-----------|--------------|
| 1 | Simple Count | Easy | ✅ | ✅ |
| 2 | Join Query | Medium | ✅ | ❌ |
| ... | ... | ... | ... | ... |

## Metrics

| Metric | Value |
|--------|-------|
| Execution Rate | 80% |
| Accuracy | 60% |

## Analysis by Difficulty

### Easy Queries
- Execution Rate: [value]
- Accuracy: [value]
- Common Issues: [patterns]

### Medium Queries
- Execution Rate: [value]
- Accuracy: [value]
- Common Issues: [patterns]

### Hard Queries
- Execution Rate: [value]
- Accuracy: [value]
- Common Issues: [patterns]

## Failure Analysis

### Execution Failures
[What caused queries to fail to execute?]

### Result Mismatches
[What caused queries to return wrong results?]

## Recommendations

### For Prompt Engineering
- [Recommendation 1]
- [Recommendation 2]

### For Evaluation Design
- [Recommendation 1]
- [Recommendation 2]

## Limitations

### What This Evaluation Cannot Assess
- [Limitation 1: e.g., query performance]
- [Limitation 2: e.g., cross-dialect compatibility]

### Future Improvements
- [Improvement 1]
- [Improvement 2]
```

---

# 🔧 Section 9 — Advanced: Extending the SQL Evaluator

### Adding Semantic Equivalence Checking

For production use, compare query semantics rather than exact results:

```python
def check_semantic_equivalence(
    sql1: str,
    sql2: str,
    evaluator: SQLEvaluator,
) -> Dict[str, Any]:
    """
    Check if two SQL queries are semantically equivalent.
    
    Two queries are semantically equivalent if they return
    the same data (ignoring order unless ORDER BY is used).
    
    Returns:
        Dictionary with:
        - equivalent: bool
        - sql1_result: results from first query
        - sql2_result: results from second query
        - difference: description of any differences
    """
    result1 = evaluator.execute_query(sql1)
    result2 = evaluator.execute_query(sql2)
    
    if not result1["success"] or not result2["success"]:
        return {
            "equivalent": False,
            "sql1_result": result1,
            "sql2_result": result2,
            "difference": "One or both queries failed to execute",
        }
    
    # Compare as sets for order-independent comparison
    set1 = set(result1["result"])
    set2 = set(result2["result"])
    
    if set1 == set2:
        return {
            "equivalent": True,
            "sql1_result": result1["result"],
            "sql2_result": result2["result"],
            "difference": None,
        }
    else:
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        return {
            "equivalent": False,
            "sql1_result": result1["result"],
            "sql2_result": result2["result"],
            "difference": f"Only in SQL1: {only_in_1}, Only in SQL2: {only_in_2}",
        }
```

### Adding Query Complexity Scoring

```python
def score_query_complexity(sql: str) -> Dict[str, Any]:
    """
    Score the complexity of a SQL query.
    
    Returns:
        Dictionary with:
        - complexity_score: 1-10 scale
        - features: list of detected features
        - difficulty: "easy", "medium", "hard"
    """
    sql_upper = sql.upper()
    
    features = []
    score = 1
    
    # Check for various SQL features
    if "JOIN" in sql_upper:
        features.append("JOIN")
        score += 2
    
    if "GROUP BY" in sql_upper:
        features.append("GROUP BY")
        score += 2
    
    if "HAVING" in sql_upper:
        features.append("HAVING")
        score += 1
    
    if "SUBQUERY" in sql_upper or sql.count("SELECT") > 1:
        features.append("SUBQUERY")
        score += 3
    
    if "UNION" in sql_upper:
        features.append("UNION")
        score += 2
    
    if "ORDER BY" in sql_upper:
        features.append("ORDER BY")
        score += 1
    
    if sql.upper().count("JOIN") > 1:
        features.append("MULTIPLE_JOINS")
        score += 2
    
    # Determine difficulty
    if score <= 2:
        difficulty = "easy"
    elif score <= 5:
        difficulty = "medium"
    else:
        difficulty = "hard"
    
    return {
        "complexity_score": min(score, 10),
        "features": features,
        "difficulty": difficulty,
    }
```

### Adding Query Validation

```python
def validate_query_against_schema(
    sql: str,
    schema: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Validate that a query only references valid tables and columns.
    
    Args:
        sql: SQL query to validate
        schema: Dictionary mapping table names to column lists
        
    Returns:
        Dictionary with:
        - valid: bool
        - issues: list of validation issues
    """
    import re
    
    issues = []
    
    # Extract table references (simplified)
    table_pattern = r'FROM\s+(\w+)|JOIN\s+(\w+)'
    table_matches = re.findall(table_pattern, sql, re.IGNORECASE)
    referenced_tables = [t[0] or t[1] for t in table_matches]
    
    # Check tables exist
    for table in referenced_tables:
        if table.lower() not in [t.lower() for t in schema.keys()]:
            issues.append(f"Unknown table: {table}")
    
    # TODO: Add column validation
    # This requires more sophisticated SQL parsing
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }
```

---

# ✔ Knowledge Mastery Checklist

Before moving to Week 15, ensure you can check all boxes:

- [ ] I understand why SQL evaluation requires both execution testing and result comparison
- [ ] I can create an in-memory SQLite database for testing SQL generation
- [ ] I can use the SQLEvaluator to test generated queries
- [ ] I understand the two-metric approach: execution success and result correctness
- [ ] I can design test cases with varying difficulty levels
- [ ] I know how to handle result comparison (order-sensitive vs. order-insensitive)
- [ ] I can identify different SQL query types and their evaluation challenges
- [ ] I understand the limitations of text-to-SQL evaluation
- [ ] I completed the mini-project SQL evaluation audit

---

Week 14 complete.
Next: *Week 15 — RAG (Retrieval-Augmented Generation) Use Cases*.
