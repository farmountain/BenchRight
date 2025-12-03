# Week 13 — Software Engineering

### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 13, you will:

1. Understand *why code generation and debugging evaluation is critical* for software engineering use cases.
2. Learn how to design a *synthetic bug-fix dataset* with function descriptions, buggy code, and expected fixes.
3. Implement *pass/fail metrics* using simple unit tests for code evaluation.
4. Describe how BenchRight's engine can *run unit tests automatically* to score model outputs.
5. Apply critical thinking to evaluate the suitability of LLMs for software engineering tasks.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is Code Generation Evaluation?

### Simple Explanation

Imagine you're evaluating an AI assistant that developers use to help write and fix code:

- **The input** is a description of what a function should do, or a piece of buggy code
- **A good response** is working code that passes unit tests
- **A bad response** is code that fails tests, has syntax errors, or doesn't solve the problem

> **Code generation evaluation tests whether LLMs can produce functionally correct code. Unlike natural language tasks where multiple phrasings are acceptable, code has a strict correctness criterion: it must pass all unit tests.**

This is different from general text evaluation because:
- **Binary correctness:** Code either works or it doesn't—there's no "partially correct" syntax
- **Testable outputs:** We can automatically verify correctness using unit tests
- **Precise semantics:** A single wrong character can break the entire program
- **Security implications:** Generated code may contain vulnerabilities

### The Software Engineering Evaluation Domain

| Task Type | What To Evaluate | Metric |
|-----------|------------------|--------|
| **Code Generation** | Generate function from description | Unit test pass rate |
| **Bug Fixing** | Fix buggy code given description | Unit test pass rate |
| **Code Completion** | Complete partial code | Unit test pass rate |
| **Code Translation** | Convert between languages | Unit test pass rate |
| **Test Generation** | Generate tests for code | Coverage, mutation score |

---

# 🧠 Section 2 — The Scenario: Synthetic Bug-Fix Dataset

### Scenario Description

We are evaluating how an LLM performs on code debugging tasks using a synthetic dataset:
1. **Input:** Function description + buggy code
2. **Output:** Fixed/corrected code
3. **Evaluation:** Automated unit tests determine pass/fail

### Designing the Synthetic Bug-Fix Dataset

A good bug-fix dataset contains:
1. **Function Description:** What the function should do
2. **Buggy Code:** Code with an intentional bug
3. **Expected Fix:** The corrected code
4. **Unit Tests:** Tests that verify the fix works

### Example Bug-Fix Cases

#### Example 1: Off-by-One Error
```python
# Function Description:
"""Calculate the sum of all numbers from 1 to n (inclusive)."""

# Buggy Code:
def sum_to_n(n):
    total = 0
    for i in range(n):  # Bug: should be range(1, n+1)
        total += i
    return total

# Expected Fix:
def sum_to_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total

# Unit Tests:
def test_sum_to_n():
    assert sum_to_n(5) == 15  # 1+2+3+4+5 = 15
    assert sum_to_n(1) == 1
    assert sum_to_n(10) == 55
```

#### Example 2: Wrong Operator
```python
# Function Description:
"""Check if a number is even."""

# Buggy Code:
def is_even(n):
    return n % 2 == 1  # Bug: should check == 0, not == 1

# Expected Fix:
def is_even(n):
    return n % 2 == 0

# Unit Tests:
def test_is_even():
    assert is_even(2) == True
    assert is_even(3) == False
    assert is_even(0) == True
    assert is_even(-4) == True
```

#### Example 3: Missing Return Statement
```python
# Function Description:
"""Find the maximum value in a list."""

# Buggy Code:
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    # Bug: missing return statement

# Expected Fix:
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val

# Unit Tests:
def test_find_max():
    assert find_max([1, 5, 3, 9, 2]) == 9
    assert find_max([42]) == 42
    assert find_max([]) == None
    assert find_max([-1, -5, -2]) == -1
```

#### Example 4: Wrong Comparison Direction
```python
# Function Description:
"""Return a list of numbers greater than a threshold."""

# Buggy Code:
def filter_greater_than(numbers, threshold):
    result = []
    for n in numbers:
        if n < threshold:  # Bug: should be > not <
            result.append(n)
    return result

# Expected Fix:
def filter_greater_than(numbers, threshold):
    result = []
    for n in numbers:
        if n > threshold:
            result.append(n)
    return result

# Unit Tests:
def test_filter_greater_than():
    assert filter_greater_than([1, 5, 10, 3], 4) == [5, 10]
    assert filter_greater_than([1, 2, 3], 10) == []
    assert filter_greater_than([], 5) == []
```

#### Example 5: String Concatenation Error
```python
# Function Description:
"""Reverse a string."""

# Buggy Code:
def reverse_string(s):
    result = ""
    for char in s:
        result = result + char  # Bug: should prepend, not append
    return result

# Expected Fix:
def reverse_string(s):
    result = ""
    for char in s:
        result = char + result
    return result

# Unit Tests:
def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("a") == "a"
    assert reverse_string("") == ""
    assert reverse_string("ab") == "ba"
```

### What Makes Code Evaluation Challenging

| Challenge | Why It's Hard | Risk if Failed |
|-----------|---------------|----------------|
| **Functional correctness** | Many solutions work; must test behavior | Broken code deployed |
| **Edge cases** | Need comprehensive tests | Bugs in production |
| **Security** | Code may have vulnerabilities | Security breaches |
| **Efficiency** | Correct but slow code | Performance issues |

---

# 🧪 Section 3 — Designing Pass/Fail Metrics with Unit Tests

### Evaluation Design

For code generation and bug fixing, we use a binary pass/fail metric:

1. **Pass:** The generated code passes ALL unit tests
2. **Fail:** The generated code fails ANY unit test (or has syntax errors)

### The CodeEvaluator Class

```python
import subprocess
import tempfile
import os
from typing import Dict, List, Any


class CodeEvaluator:
    """
    Evaluator for code generation and bug-fix tasks.
    
    Uses unit tests to determine pass/fail for generated code.
    """
    
    def __init__(self, timeout_seconds: int = 5):
        """
        Initialize the CodeEvaluator.
        
        Args:
            timeout_seconds: Maximum time allowed for test execution
        """
        self.timeout_seconds = timeout_seconds
    
    def evaluate_code(
        self,
        generated_code: str,
        test_code: str,
    ) -> Dict[str, Any]:
        """
        Evaluate generated code using unit tests.
        
        Args:
            generated_code: The code to evaluate
            test_code: Unit test code to verify correctness
            
        Returns:
            Dictionary with:
            - passed: bool indicating if all tests passed
            - error: str with error message if failed
            - output: str with test output
        """
        # Combine generated code with tests
        full_code = f"{generated_code}\n\n{test_code}\n\n"
        full_code += "if __name__ == '__main__':\n"
        full_code += "    import sys\n"
        full_code += "    # Run all test functions\n"
        full_code += "    test_functions = [name for name in dir() if name.startswith('test_')]\n"
        full_code += "    for test_name in test_functions:\n"
        full_code += "        try:\n"
        full_code += "            globals()[test_name]()\n"
        full_code += "            print(f'✓ {test_name} passed')\n"
        full_code += "        except AssertionError as e:\n"
        full_code += "            print(f'✗ {test_name} failed: {e}')\n"
        full_code += "            sys.exit(1)\n"
        full_code += "    print('All tests passed!')\n"
        
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(full_code)
                temp_path = f.name
            
            # Execute the code
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            # Clean up
            os.unlink(temp_path)
            
            # Check result
            if result.returncode == 0:
                return {
                    "passed": True,
                    "error": None,
                    "output": result.stdout,
                }
            else:
                return {
                    "passed": False,
                    "error": result.stderr or result.stdout,
                    "output": result.stdout,
                }
                
        except subprocess.TimeoutExpired:
            os.unlink(temp_path)
            return {
                "passed": False,
                "error": f"Execution timed out after {self.timeout_seconds} seconds",
                "output": "",
            }
        except SyntaxError as e:
            return {
                "passed": False,
                "error": f"Syntax error: {str(e)}",
                "output": "",
            }
        except Exception as e:
            return {
                "passed": False,
                "error": f"Execution error: {str(e)}",
                "output": "",
            }
    
    def compute_pass_rate(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute the pass rate across multiple evaluations.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Pass rate as a float between 0.0 and 1.0
        """
        if not results:
            return 0.0
        passed = sum(1 for r in results if r["passed"])
        return passed / len(results)
```

### Usage Example

```python
# Initialize the evaluator
evaluator = CodeEvaluator(timeout_seconds=5)

# Generated code to evaluate
generated_code = """
def sum_to_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
"""

# Unit tests
test_code = """
def test_sum_to_n():
    assert sum_to_n(5) == 15
    assert sum_to_n(1) == 1
    assert sum_to_n(10) == 55
"""

# Evaluate
result = evaluator.evaluate_code(generated_code, test_code)

print(f"Passed: {result['passed']}")
print(f"Output: {result['output']}")
if result['error']:
    print(f"Error: {result['error']}")
```

---

# 🧪 Section 4 — Integrating with BenchRight's Benchmark Engine

### How BenchRight Runs Unit Tests Automatically

BenchRight's generic benchmark engine can be extended to handle code evaluation:

```python
from typing import Callable, Iterator, Tuple, Any, Dict
from benchmark_engine.engine import run_benchmark


def code_pass_fail_metric(
    generated_code: str,
    test_code: str,
    evaluator: 'CodeEvaluator' = None
) -> float:
    """
    Metric function that returns 1.0 if code passes all tests, 0.0 otherwise.
    
    Args:
        generated_code: The generated code to evaluate
        test_code: Unit tests for verification
        evaluator: CodeEvaluator instance (optional, creates new if None)
        
    Returns:
        1.0 if all tests pass, 0.0 otherwise
    """
    if evaluator is None:
        evaluator = CodeEvaluator()
    
    result = evaluator.evaluate_code(generated_code, test_code)
    return 1.0 if result["passed"] else 0.0


def run_code_benchmark(
    model_fn: Callable[[str], str],
    dataset: Iterator[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Run a code generation/bug-fix benchmark.
    
    Args:
        model_fn: Function that takes a prompt and returns generated code
        dataset: Iterator of (prompt, test_code) tuples
        
    Returns:
        Benchmark results including pass rate
    """
    evaluator = CodeEvaluator(timeout_seconds=5)
    
    def metric_fn(generated_code: str, test_code: str) -> float:
        return code_pass_fail_metric(generated_code, test_code, evaluator)
    
    return run_benchmark(
        model_fn=model_fn,
        dataset=dataset,
        metric_fn=metric_fn,
        batch_size=1
    )
```

### Creating a Bug-Fix Prompt Template

```python
BUG_FIX_PROMPT_TEMPLATE = """You are a skilled Python programmer. Your task is to fix the bug in the following code.

## Function Description
{description}

## Buggy Code
```python
{buggy_code}
```

## Instructions
1. Identify the bug in the code above
2. Fix the bug while maintaining the same function signature
3. Return ONLY the fixed Python code, no explanations

## Fixed Code
```python
"""


def create_bug_fix_prompt(description: str, buggy_code: str) -> str:
    """Create a prompt for bug-fix tasks."""
    return BUG_FIX_PROMPT_TEMPLATE.format(
        description=description,
        buggy_code=buggy_code
    )
```

### Example: Running a Bug-Fix Benchmark

```python
# Define the synthetic bug-fix dataset
bug_fix_dataset = [
    {
        "description": "Calculate the sum of all numbers from 1 to n (inclusive).",
        "buggy_code": '''def sum_to_n(n):
    total = 0
    for i in range(n):
        total += i
    return total''',
        "test_code": '''def test_sum_to_n():
    assert sum_to_n(5) == 15
    assert sum_to_n(1) == 1
    assert sum_to_n(10) == 55''',
    },
    {
        "description": "Check if a number is even.",
        "buggy_code": '''def is_even(n):
    return n % 2 == 1''',
        "test_code": '''def test_is_even():
    assert is_even(2) == True
    assert is_even(3) == False
    assert is_even(0) == True''',
    },
    # ... more examples
]

# Create prompts and prepare dataset
def prepare_dataset():
    for item in bug_fix_dataset:
        prompt = create_bug_fix_prompt(
            item["description"],
            item["buggy_code"]
        )
        yield (prompt, item["test_code"])

# Run the benchmark
results = run_code_benchmark(
    model_fn=my_model,
    dataset=prepare_dataset()
)

print(f"Pass Rate: {results['mean_score']:.2%}")
print(f"Total Examples: {results['total_examples']}")
```

---

# 🧪 Section 5 — Hands-on Lab: Evaluating Bug-Fix Capabilities

### Lab Overview

In this lab, you will:
1. Create a synthetic bug-fix dataset with 5+ examples
2. Use the CodeEvaluator to test generated fixes
3. Compute pass rates for the model
4. Analyze which types of bugs are hardest to fix

### Step 1: Define Test Cases

```python
# Define comprehensive bug-fix test cases
test_cases = [
    {
        "name": "Off-by-One Error",
        "description": "Calculate the sum of all numbers from 1 to n (inclusive).",
        "buggy_code": '''def sum_to_n(n):
    total = 0
    for i in range(n):  # Bug: should be range(1, n+1)
        total += i
    return total''',
        "expected_fix": '''def sum_to_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total''',
        "test_code": '''def test_sum_to_n():
    assert sum_to_n(5) == 15
    assert sum_to_n(1) == 1
    assert sum_to_n(10) == 55''',
        "bug_type": "off-by-one",
    },
    {
        "name": "Wrong Operator",
        "description": "Check if a number is even.",
        "buggy_code": '''def is_even(n):
    return n % 2 == 1''',
        "expected_fix": '''def is_even(n):
    return n % 2 == 0''',
        "test_code": '''def test_is_even():
    assert is_even(2) == True
    assert is_even(3) == False
    assert is_even(0) == True
    assert is_even(-4) == True''',
        "bug_type": "wrong-operator",
    },
    {
        "name": "Missing Return",
        "description": "Find the maximum value in a list.",
        "buggy_code": '''def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item''',
        "expected_fix": '''def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val''',
        "test_code": '''def test_find_max():
    assert find_max([1, 5, 3, 9, 2]) == 9
    assert find_max([42]) == 42
    assert find_max([]) == None''',
        "bug_type": "missing-return",
    },
    {
        "name": "Wrong Comparison",
        "description": "Return a list of numbers greater than a threshold.",
        "buggy_code": '''def filter_greater_than(numbers, threshold):
    result = []
    for n in numbers:
        if n < threshold:
            result.append(n)
    return result''',
        "expected_fix": '''def filter_greater_than(numbers, threshold):
    result = []
    for n in numbers:
        if n > threshold:
            result.append(n)
    return result''',
        "test_code": '''def test_filter_greater_than():
    assert filter_greater_than([1, 5, 10, 3], 4) == [5, 10]
    assert filter_greater_than([1, 2, 3], 10) == []
    assert filter_greater_than([], 5) == []''',
        "bug_type": "wrong-comparison",
    },
    {
        "name": "String Order Error",
        "description": "Reverse a string.",
        "buggy_code": '''def reverse_string(s):
    result = ""
    for char in s:
        result = result + char
    return result''',
        "expected_fix": '''def reverse_string(s):
    result = ""
    for char in s:
        result = char + result
    return result''',
        "test_code": '''def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("a") == "a"
    assert reverse_string("") == ""''',
        "bug_type": "string-order",
    },
]

print(f"📊 Defined {len(test_cases)} bug-fix test cases:")
for i, tc in enumerate(test_cases, 1):
    print(f"   {i}. {tc['name']} (bug type: {tc['bug_type']})")
```

### Step 2: Run Evaluations

```python
# Evaluate expected fixes (should all pass)
evaluator = CodeEvaluator(timeout_seconds=5)
results = []

for tc in test_cases:
    result = evaluator.evaluate_code(
        tc["expected_fix"],
        tc["test_code"]
    )
    
    results.append({
        "name": tc["name"],
        "bug_type": tc["bug_type"],
        "passed": result["passed"],
        "error": result["error"],
    })
    
    print(f"\n{'='*60}")
    print(f"Test: {tc['name']}")
    print(f"Bug Type: {tc['bug_type']}")
    print(f"{'='*60}")
    print(f"Passed: {'✅ Yes' if result['passed'] else '❌ No'}")
    if result['error']:
        print(f"Error: {result['error']}")

# Calculate pass rate
pass_rate = evaluator.compute_pass_rate(results)
print(f"\n📊 Overall Pass Rate: {pass_rate:.2%}")
```

---

# 🤔 Section 6 — Paul-Elder Critical Thinking Questions

### Question 1: TEST COVERAGE
**What are the risks if an LLM-generated code fix passes all provided unit tests but contains a subtle bug that wasn't covered by the tests?**

*Consider: Test coverage vs. correctness, edge cases not tested, the difference between "passes tests" and "is correct," and how comprehensive test suites should be for safety-critical code.*

### Question 2: PROMPT ENGINEERING
**How does the quality of the bug description in the prompt affect the LLM's ability to fix the bug correctly? What information should always be included?**

*Consider: The importance of function specifications, the role of examples, whether showing the expected output helps, and the tradeoff between detailed prompts and model generalization.*

### Question 3: SECURITY IMPLICATIONS
**Should LLM-generated code be trusted in production systems? What safeguards should be in place before deploying generated code?**

*Consider: Code review requirements, automated security scanning, testing requirements, the potential for introducing vulnerabilities, and the difference between "works" and "safe."*

---

# 🔄 Section 7 — Inversion Thinking: How Can Code Generation Fail?

Instead of asking "How does LLM code generation help developers?", let's invert:

> **"How can LLM-generated code cause problems in production systems?"**

### Failure Modes

1. **Incorrect Logic**
   - Code passes simple tests but fails edge cases
   - Off-by-one errors, boundary condition failures
   - Consequence: Bugs discovered in production

2. **Security Vulnerabilities**
   - Generated code has SQL injection, XSS, or other vulnerabilities
   - No input validation or sanitization
   - Consequence: Security breaches, data leaks

3. **Performance Issues**
   - Code works but is inefficient (O(n²) when O(n) is possible)
   - Memory leaks, resource exhaustion
   - Consequence: Slow applications, outages

4. **Incomplete Solutions**
   - Code handles happy path but not error cases
   - Missing exception handling
   - Consequence: Crashes, undefined behavior

5. **Style and Maintainability**
   - Code is correct but unreadable
   - Poor naming, no documentation
   - Consequence: Technical debt, maintenance burden

### Defensive Practices

- **Comprehensive Testing:** Write tests for edge cases, not just happy paths
- **Security Review:** Scan generated code for common vulnerabilities
- **Code Review:** Human review before merging generated code
- **Performance Testing:** Benchmark generated code for efficiency
- **Documentation:** Require clear comments and docstrings
- **Regression Testing:** Ensure changes don't break existing functionality

---

# 📝 Section 8 — Mini-Project: Build a Bug-Fix Evaluator

### Task

Create a complete bug-fix evaluation pipeline that:
1. Uses the CodeEvaluator to test generated code
2. Processes at least 5 bug-fix examples
3. Computes pass rates by bug type
4. Analyzes which bug types are hardest to fix

### Instructions

1. **Create your bug-fix dataset:**
   - At least 5 buggy code examples
   - Include variety: off-by-one, wrong operator, missing return, etc.
   - Each should have comprehensive unit tests

2. **Simulate model outputs:**
   - Create both correct and incorrect fixes for each bug
   - Or use an LLM to generate fixes

3. **Run evaluations:**
   - Use CodeEvaluator to test each generated fix
   - Record pass/fail for each example
   - Track results by bug type

4. **Analyze results:**
   - Which bug types have highest pass rates?
   - What patterns do you observe in failures?
   - What improvements could help?

### Submission Format

Create a markdown file `/examples/week13_bug_fix_audit.md`:

```markdown
# Week 13 Mini-Project: Bug-Fix Evaluation Audit

## Executive Summary
[2-3 sentences on overall findings]

## Bug-Fix Examples Evaluated

| # | Bug Type | Description | Test Count |
|---|----------|-------------|------------|
| 1 | off-by-one | Sum to N | 3 |
| 2 | wrong-operator | Is Even | 4 |
| ... | ... | ... | ... |

## Evaluation Results

| # | Bug Type | Passed | Tests Run | Pass Rate |
|---|----------|--------|-----------|-----------|
| 1 | off-by-one | ✅ | 3/3 | 100% |
| 2 | wrong-operator | ❌ | 2/4 | 50% |
| ... | ... | ... | ... | ... |

## Analysis by Bug Type

### Bug Type Performance
| Bug Type | Examples | Pass Rate |
|----------|----------|-----------|
| off-by-one | 2 | 50% |
| wrong-operator | 1 | 100% |
| ... | ... | ... |

### Failure Analysis
[What caused failures? Common patterns?]

## Recommendations

### For Prompt Engineering
- [Recommendation 1]
- [Recommendation 2]

### For Test Design
- [Recommendation 1]
- [Recommendation 2]

## Limitations

### What This Evaluation Cannot Assess
- [Limitation 1: e.g., security vulnerabilities]
- [Limitation 2: e.g., code efficiency]

### Future Improvements
- [Improvement 1]
- [Improvement 2]
```

---

# 🔧 Section 9 — Advanced: Extending the Code Evaluator

### Adding Timeout and Resource Limits

For production use, add resource limits to prevent runaway code:

```python
import resource

def set_resource_limits():
    """Set resource limits for code execution."""
    # Limit CPU time to 5 seconds
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    # Limit memory to 100MB
    resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024))
```

### Adding Security Scanning

```python
def scan_for_vulnerabilities(code: str) -> List[str]:
    """
    Scan generated code for common security issues.
    
    Returns:
        List of identified security concerns
    """
    concerns = []
    
    # Check for dangerous functions
    dangerous_patterns = [
        ("eval(", "Use of eval() - potential code injection"),
        ("exec(", "Use of exec() - potential code injection"),
        ("os.system(", "Use of os.system() - potential command injection"),
        ("subprocess.call(", "Use of subprocess.call() - verify input sanitization"),
        ("__import__", "Dynamic import - verify module safety"),
    ]
    
    for pattern, warning in dangerous_patterns:
        if pattern in code:
            concerns.append(warning)
    
    return concerns
```

### Adding Code Style Checks

```python
def check_code_quality(code: str) -> Dict[str, Any]:
    """
    Check code quality using pylint or similar.
    
    Returns:
        Dictionary with quality metrics
    """
    # TODO: Integrate with pylint, flake8, or similar
    pass
```

### Adding Multi-Language Support

```python
class MultiLanguageEvaluator:
    """Evaluate code in multiple programming languages."""
    
    LANGUAGE_RUNNERS = {
        "python": ["python", "-c"],
        "javascript": ["node", "-e"],
        "go": ["go", "run"],
        # Add more languages as needed
    }
    
    def evaluate(
        self,
        code: str,
        tests: str,
        language: str
    ) -> Dict[str, Any]:
        """Evaluate code in the specified language."""
        pass
```

---

# ✔ Knowledge Mastery Checklist

Before moving to Week 14, ensure you can check all boxes:

- [ ] I understand why code generation evaluation requires unit tests rather than semantic comparison
- [ ] I can design a synthetic bug-fix dataset with descriptions, buggy code, expected fixes, and tests
- [ ] I can use the CodeEvaluator to automatically run unit tests on generated code
- [ ] I understand the pass/fail metric and how to compute pass rates
- [ ] I know how BenchRight's engine can integrate with code evaluation
- [ ] I can identify different bug types and analyze which are hardest to fix
- [ ] I understand the security implications of LLM-generated code
- [ ] I can articulate the limitations of unit test-based evaluation
- [ ] I completed the mini-project bug-fix evaluation audit

---

Week 13 complete.
Next: *Week 14 — Data Analytics Use Cases*.
