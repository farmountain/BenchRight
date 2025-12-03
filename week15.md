# Week 15 — RAG & Customer Service

### BenchRight LLM Evaluation Master Program (18 Weeks)

---

## 🎯 Learning Objectives
By the end of Week 15, you will:

1. Understand *what Retrieval-Augmented Generation (RAG) is* and why it's critical for customer service use cases.
2. Learn how to create a *tiny FAQ corpus* and index it with a simple vector store (e.g., FAISS).
3. Implement *RAG evaluation metrics* including retrieval quality and answer groundedness.
4. Evaluate answer groundedness qualitatively with an *LLM judge* and/or simple string matches.
5. Apply critical thinking to assess the strengths and limitations of RAG systems.

---

# 🧠 Section 1 — Feynman-Style Explanation: What is Retrieval-Augmented Generation?

### Simple Explanation

Imagine you're evaluating an AI customer service assistant:

- **The input** is a customer question
- **The retrieval step** finds relevant FAQ snippets from a knowledge base
- **The generation step** uses those snippets to produce a grounded answer
- **A good response** is accurate, uses the retrieved information, and doesn't hallucinate

> **Retrieval-Augmented Generation (RAG) combines information retrieval with language generation. Instead of relying solely on the model's internal knowledge, RAG retrieves relevant documents and uses them to generate more accurate, grounded responses.**

This is different from standard LLM generation because:
- **External knowledge:** RAG uses up-to-date information from a knowledge base
- **Groundedness:** Responses should be traceable to source documents
- **Reduced hallucination:** Retrieved context constrains the model's output
- **Domain adaptation:** Knowledge bases can be customized for specific use cases

### The RAG Pipeline

```
Customer Question
       │
       ▼
┌─────────────────┐
│   Embedding     │  ← Convert question to vector
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Vector Search  │  ← Find similar FAQ entries
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Retrieved      │  ← Top-k relevant snippets
│  Context        │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Generator      │  ← LLM generates answer
│  (tinyGPT)      │     using retrieved context
└─────────────────┘
       │
       ▼
    Answer
```

### What We Evaluate in RAG

| Component | What To Evaluate | Metric |
|-----------|------------------|--------|
| **Retrieval** | Are the right documents retrieved? | Recall@k, Precision@k |
| **Relevance** | Are retrieved docs relevant to query? | Relevance score |
| **Groundedness** | Is the answer based on retrieved context? | LLM-as-Judge, string match |
| **Faithfulness** | Does the answer contain hallucinations? | Hallucination rate |
| **Correctness** | Is the final answer correct? | Accuracy |

---

# 🧠 Section 2 — The Scenario: Customer Service FAQ Chatbot

### Scenario Description

We are evaluating a RAG-based customer service chatbot for an e-commerce company:
1. **Knowledge Base:** A small FAQ corpus with product and policy information
2. **Retrieval:** FAISS vector store for semantic search
3. **Generation:** tinyGPT as the generator given retrieved snippets
4. **Evaluation:** Groundedness assessment using LLM judge and string matching

### Designing the FAQ Corpus

A good FAQ corpus for customer service contains:
1. **Question:** The customer query
2. **Answer:** The official response
3. **Category:** Topic category for organization
4. **Keywords:** Important terms for retrieval

### Example FAQ Corpus

```python
FAQ_CORPUS = [
    {
        "id": "faq_001",
        "question": "What is your return policy?",
        "answer": "You can return any item within 30 days of purchase for a full refund. Items must be unused and in original packaging. Return shipping is free for defective items.",
        "category": "returns",
        "keywords": ["return", "refund", "30 days", "policy"]
    },
    {
        "id": "faq_002",
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. Free shipping is available on orders over $50.",
        "category": "shipping",
        "keywords": ["shipping", "delivery", "days", "free"]
    },
    {
        "id": "faq_003",
        "question": "How do I track my order?",
        "answer": "You can track your order by logging into your account and visiting the 'Order History' section. You will also receive tracking updates via email once your order ships.",
        "category": "orders",
        "keywords": ["track", "order", "status", "email"]
    },
    {
        "id": "faq_004",
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. All transactions are secured with SSL encryption.",
        "category": "payment",
        "keywords": ["payment", "credit card", "PayPal", "secure"]
    },
    {
        "id": "faq_005",
        "question": "How do I cancel my order?",
        "answer": "You can cancel your order within 1 hour of placing it by contacting customer support. After 1 hour, orders enter processing and cannot be canceled, but you can return the item once received.",
        "category": "orders",
        "keywords": ["cancel", "order", "support", "hour"]
    },
    {
        "id": "faq_006",
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to over 50 countries worldwide. International shipping rates vary by destination and typically take 10-14 business days.",
        "category": "shipping",
        "keywords": ["international", "worldwide", "countries", "global"]
    },
    {
        "id": "faq_007",
        "question": "What if my item arrives damaged?",
        "answer": "If your item arrives damaged, please contact us within 48 hours with photos of the damage. We will send a replacement at no additional cost and arrange free return shipping for the damaged item.",
        "category": "returns",
        "keywords": ["damaged", "broken", "replacement", "photos"]
    },
    {
        "id": "faq_008",
        "question": "How do I change my shipping address?",
        "answer": "You can update your shipping address in your account settings before placing an order. For orders already placed, contact customer support within 1 hour to request an address change.",
        "category": "shipping",
        "keywords": ["address", "change", "update", "account"]
    },
]
```

### Customer Questions for Evaluation

```python
TEST_QUESTIONS = [
    {
        "question": "Can I get a refund if I don't like the product?",
        "expected_faq_ids": ["faq_001"],
        "expected_answer_contains": ["30 days", "refund"],
    },
    {
        "question": "How fast is delivery?",
        "expected_faq_ids": ["faq_002"],
        "expected_answer_contains": ["5-7 business days", "express"],
    },
    {
        "question": "Where can I see my order status?",
        "expected_faq_ids": ["faq_003"],
        "expected_answer_contains": ["Order History", "account"],
    },
    {
        "question": "Can I pay with PayPal?",
        "expected_faq_ids": ["faq_004"],
        "expected_answer_contains": ["PayPal", "accept"],
    },
    {
        "question": "My package arrived broken, what do I do?",
        "expected_faq_ids": ["faq_007"],
        "expected_answer_contains": ["48 hours", "replacement", "photos"],
    },
]
```

### What Makes RAG Evaluation Challenging

| Challenge | Why It's Hard | Risk if Failed |
|-----------|---------------|----------------|
| **Retrieval quality** | Wrong docs retrieved = wrong answers | Misinformation |
| **Groundedness** | Model may ignore retrieved context | Hallucination |
| **Context window** | Too much/little context affects quality | Poor answers |
| **Semantic matching** | Questions may be phrased differently | Missed relevant docs |

---

# 🧪 Section 3 — Designing the RAG Evaluation System

### Evaluation Design

For RAG systems, we use a multi-component evaluation approach:

1. **Retrieval Evaluation:** Are the right documents retrieved?
2. **Groundedness Evaluation:** Is the answer based on retrieved context?
3. **Answer Quality:** Is the final answer correct and helpful?

### The SimpleVectorStore Class

```python
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class SimpleVectorStore:
    """
    A simple vector store using numpy for demonstration.
    
    In production, use FAISS, Pinecone, Weaviate, or similar.
    This implementation uses cosine similarity for retrieval.
    """
    
    def __init__(self):
        """Initialize the vector store."""
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embed_fn = None
    
    def set_embedding_function(self, embed_fn):
        """
        Set the embedding function.
        
        Args:
            embed_fn: Function that takes text and returns embedding vector
        """
        self.embed_fn = embed_fn
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'text' field
        """
        self.documents = documents
        
        if self.embed_fn is None:
            raise ValueError("Embedding function not set. Call set_embedding_function first.")
        
        # Generate embeddings for all documents
        texts = [doc.get("answer", doc.get("text", "")) for doc in documents]
        embeddings_list = [self.embed_fn(text) for text in texts]
        self.embeddings = np.array(embeddings_list)
    
    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Embed the query
        query_embedding = np.array(self.embed_fn(query))
        
        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return documents with scores
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def _cosine_similarity(
        self,
        query: np.ndarray,
        documents: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query: Query embedding (1D array)
            documents: Document embeddings (2D array)
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        doc_norms = documents / (np.linalg.norm(documents, axis=1, keepdims=True) + 1e-8)
        
        # Compute dot product
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
```

### The RAGEvaluator Class

```python
from typing import Dict, List, Any, Callable


class RAGEvaluator:
    """
    Evaluator for Retrieval-Augmented Generation systems.
    
    Evaluates:
    1. Retrieval quality (precision, recall)
    2. Answer groundedness (LLM judge or string match)
    3. Overall answer quality
    """
    
    def __init__(
        self,
        vector_store: 'SimpleVectorStore',
        generator_fn: Callable[[str, List[str]], str],
    ):
        """
        Initialize the RAGEvaluator.
        
        Args:
            vector_store: Vector store for retrieval
            generator_fn: Function that takes (question, context_list) and returns answer
        """
        self.vector_store = vector_store
        self.generator_fn = generator_fn
    
    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: Customer question
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        results = self.vector_store.search(question, top_k=top_k)
        return [{"document": doc, "score": score} for doc, score in results]
    
    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        """
        Generate an answer using retrieved context.
        
        Args:
            question: Customer question
            retrieved_docs: Retrieved documents
            
        Returns:
            Generated answer
        """
        # Extract context from retrieved documents
        context_list = [
            doc["document"].get("answer", doc["document"].get("text", ""))
            for doc in retrieved_docs
        ]
        
        return self.generator_fn(question, context_list)
    
    def evaluate_retrieval(
        self,
        retrieved_ids: List[str],
        expected_ids: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_ids: IDs of retrieved documents
            expected_ids: IDs of expected relevant documents
            
        Returns:
            Dictionary with precision, recall, and hit_rate
        """
        retrieved_set = set(retrieved_ids)
        expected_set = set(expected_ids)
        
        if len(retrieved_ids) == 0:
            return {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0}
        
        hits = len(retrieved_set & expected_set)
        
        precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
        recall = hits / len(expected_ids) if expected_ids else 0.0
        hit_rate = 1.0 if hits > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
        }
    
    def evaluate_groundedness_string_match(
        self,
        answer: str,
        expected_phrases: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate answer groundedness using string matching.
        
        Args:
            answer: Generated answer
            expected_phrases: Phrases that should appear in the answer
            
        Returns:
            Dictionary with groundedness score and matched phrases
        """
        answer_lower = answer.lower()
        matched = []
        unmatched = []
        
        for phrase in expected_phrases:
            if phrase.lower() in answer_lower:
                matched.append(phrase)
            else:
                unmatched.append(phrase)
        
        score = len(matched) / len(expected_phrases) if expected_phrases else 0.0
        
        return {
            "groundedness_score": score,
            "matched_phrases": matched,
            "unmatched_phrases": unmatched,
            "total_expected": len(expected_phrases),
        }
    
    def evaluate_groundedness_llm_judge(
        self,
        question: str,
        answer: str,
        context: List[str],
        judge_fn: Callable[[str], str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate answer groundedness using LLM-as-Judge.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context snippets
            judge_fn: Function that takes a prompt and returns judgment
            
        Returns:
            Dictionary with groundedness assessment
        """
        if judge_fn is None:
            # Return placeholder if no judge function provided
            return {
                "grounded": None,
                "explanation": "No judge function provided",
                "hallucination_detected": None,
            }
        
        # Create judgment prompt
        context_text = "\n".join([f"- {c}" for c in context])
        prompt = f"""You are evaluating whether an AI assistant's answer is grounded in the provided context.

## Context (Retrieved FAQ Snippets):
{context_text}

## Question:
{question}

## Answer:
{answer}

## Evaluation Criteria:
1. Is the answer factually supported by the context?
2. Does the answer contain information NOT found in the context (hallucination)?
3. Does the answer correctly address the question?

## Instructions:
Respond with a JSON object containing:
- "grounded": true if the answer is well-grounded in context, false otherwise
- "hallucination_detected": true if answer contains unsupported claims, false otherwise
- "explanation": brief explanation of your assessment

## Judgment:
"""
        
        judgment = judge_fn(prompt)
        
        # Parse judgment (simplified - in production use proper JSON parsing)
        grounded = "grounded\": true" in judgment.lower() or "grounded\":true" in judgment.lower()
        hallucination = "hallucination_detected\": true" in judgment.lower() or "hallucination_detected\":true" in judgment.lower()
        
        return {
            "grounded": grounded,
            "hallucination_detected": hallucination,
            "raw_judgment": judgment,
        }
    
    def run_evaluation(
        self,
        question: str,
        expected_faq_ids: List[str],
        expected_answer_contains: List[str],
        top_k: int = 3,
        judge_fn: Callable[[str], str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete RAG evaluation on a single question.
        
        Args:
            question: Customer question
            expected_faq_ids: IDs of FAQs that should be retrieved
            expected_answer_contains: Phrases that should appear in answer
            top_k: Number of documents to retrieve
            judge_fn: Optional LLM judge function
            
        Returns:
            Complete evaluation results
        """
        # Step 1: Retrieve
        retrieved = self.retrieve(question, top_k=top_k)
        retrieved_ids = [r["document"].get("id", "") for r in retrieved]
        
        # Step 2: Generate answer
        answer = self.generate_answer(question, retrieved)
        
        # Step 3: Evaluate retrieval
        retrieval_metrics = self.evaluate_retrieval(retrieved_ids, expected_faq_ids)
        
        # Step 4: Evaluate groundedness (string match)
        groundedness_string = self.evaluate_groundedness_string_match(
            answer, expected_answer_contains
        )
        
        # Step 5: Evaluate groundedness (LLM judge) if provided
        context = [r["document"].get("answer", "") for r in retrieved]
        groundedness_llm = self.evaluate_groundedness_llm_judge(
            question, answer, context, judge_fn
        )
        
        return {
            "question": question,
            "retrieved_docs": retrieved,
            "retrieved_ids": retrieved_ids,
            "answer": answer,
            "retrieval_metrics": retrieval_metrics,
            "groundedness_string": groundedness_string,
            "groundedness_llm": groundedness_llm,
        }
    
    def compute_aggregate_metrics(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics across multiple evaluations.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        # Aggregate retrieval metrics
        avg_precision = np.mean([r["retrieval_metrics"]["precision"] for r in results])
        avg_recall = np.mean([r["retrieval_metrics"]["recall"] for r in results])
        avg_hit_rate = np.mean([r["retrieval_metrics"]["hit_rate"] for r in results])
        
        # Aggregate groundedness (string match)
        avg_groundedness = np.mean([
            r["groundedness_string"]["groundedness_score"] for r in results
        ])
        
        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_hit_rate": avg_hit_rate,
            "avg_groundedness_score": avg_groundedness,
            "total_evaluated": len(results),
        }
```

### Usage Example

```python
# Simple bag-of-words embedding (for demonstration)
def simple_embedding(text: str) -> List[float]:
    """Create a simple word-frequency embedding."""
    words = text.lower().split()
    vocab = ["return", "refund", "shipping", "order", "payment", "track", 
             "cancel", "day", "hour", "free", "international", "damaged"]
    embedding = [words.count(w) for w in vocab]
    return embedding

# Mock generator function
def mock_generator(question: str, context_list: List[str]) -> str:
    """Simple generator that concatenates context."""
    if not context_list:
        return "I don't have information about that."
    return f"Based on our FAQ: {context_list[0]}"

# Create vector store and add FAQ
vector_store = SimpleVectorStore()
vector_store.set_embedding_function(simple_embedding)
vector_store.add_documents(FAQ_CORPUS)

# Create evaluator
evaluator = RAGEvaluator(
    vector_store=vector_store,
    generator_fn=mock_generator,
)

# Evaluate a question
result = evaluator.run_evaluation(
    question="Can I get a refund?",
    expected_faq_ids=["faq_001"],
    expected_answer_contains=["30 days", "refund"],
)

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Retrieval Precision: {result['retrieval_metrics']['precision']:.2f}")
print(f"Groundedness Score: {result['groundedness_string']['groundedness_score']:.2f}")
```

---

# 🧪 Section 4 — Implementing the RAG Evaluation Pipeline

### Complete Evaluation Pipeline

```python
from typing import Callable, List, Dict, Any

# Prompt template for RAG generation
RAG_PROMPT_TEMPLATE = """You are a helpful customer service assistant. Answer the customer's question based ONLY on the provided context. If the context doesn't contain the answer, say "I don't have information about that."

## Context:
{context}

## Customer Question:
{question}

## Instructions:
1. Answer based ONLY on the information in the context
2. Be concise and helpful
3. If you're not sure, admit uncertainty
4. Do not make up information

## Answer:
"""


def create_rag_prompt(question: str, context_list: List[str]) -> str:
    """Create a RAG prompt with question and context."""
    context = "\n".join([f"- {c}" for c in context_list])
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )


def run_rag_benchmark(
    evaluator: 'RAGEvaluator',
    test_cases: List[Dict[str, Any]],
    top_k: int = 3,
    judge_fn: Callable[[str], str] = None,
) -> Dict[str, Any]:
    """
    Run a complete RAG benchmark.
    
    Args:
        evaluator: RAGEvaluator instance
        test_cases: List of test cases with questions and expected values
        top_k: Number of documents to retrieve
        judge_fn: Optional LLM judge function
        
    Returns:
        Benchmark results with metrics
    """
    results = []
    
    for tc in test_cases:
        result = evaluator.run_evaluation(
            question=tc["question"],
            expected_faq_ids=tc.get("expected_faq_ids", []),
            expected_answer_contains=tc.get("expected_answer_contains", []),
            top_k=top_k,
            judge_fn=judge_fn,
        )
        results.append(result)
    
    # Compute aggregate metrics
    metrics = evaluator.compute_aggregate_metrics(results)
    
    return {
        "results": results,
        "metrics": metrics,
        "total_examples": len(results),
    }
```

### Creating a Test Dataset

```python
# Comprehensive test cases for customer service RAG
RAG_TEST_CASES = [
    {
        "question": "Can I get a refund if I don't like the product?",
        "expected_faq_ids": ["faq_001"],
        "expected_answer_contains": ["30 days", "refund"],
        "category": "returns",
    },
    {
        "question": "How fast is delivery?",
        "expected_faq_ids": ["faq_002"],
        "expected_answer_contains": ["5-7 business days", "express"],
        "category": "shipping",
    },
    {
        "question": "Where can I see my order status?",
        "expected_faq_ids": ["faq_003"],
        "expected_answer_contains": ["Order History", "account"],
        "category": "orders",
    },
    {
        "question": "Can I pay with PayPal?",
        "expected_faq_ids": ["faq_004"],
        "expected_answer_contains": ["PayPal", "accept"],
        "category": "payment",
    },
    {
        "question": "My package arrived broken, what do I do?",
        "expected_faq_ids": ["faq_007"],
        "expected_answer_contains": ["48 hours", "replacement", "photos"],
        "category": "returns",
    },
    {
        "question": "Do you ship to Canada?",
        "expected_faq_ids": ["faq_006"],
        "expected_answer_contains": ["international", "50 countries"],
        "category": "shipping",
    },
    {
        "question": "I want to cancel my order",
        "expected_faq_ids": ["faq_005"],
        "expected_answer_contains": ["1 hour", "cancel"],
        "category": "orders",
    },
]
```

---

# 🧪 Section 5 — Hands-on Lab: Evaluating RAG for Customer Service

### Lab Overview

In this lab, you will:
1. Create a tiny FAQ corpus and index it with a simple vector store
2. Implement retrieval and generation components
3. Benchmark tinyGPT as a generator given retrieved snippets
4. Evaluate answer groundedness qualitatively with an LLM judge and string matches

### Step 1: Define the FAQ Corpus

```python
# Customer service FAQ corpus
FAQ_CORPUS = [
    {
        "id": "faq_001",
        "question": "What is your return policy?",
        "answer": "You can return any item within 30 days of purchase for a full refund. Items must be unused and in original packaging. Return shipping is free for defective items.",
        "category": "returns",
    },
    {
        "id": "faq_002",
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. Free shipping is available on orders over $50.",
        "category": "shipping",
    },
    {
        "id": "faq_003",
        "question": "How do I track my order?",
        "answer": "You can track your order by logging into your account and visiting the 'Order History' section. You will also receive tracking updates via email once your order ships.",
        "category": "orders",
    },
    {
        "id": "faq_004",
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. All transactions are secured with SSL encryption.",
        "category": "payment",
    },
    {
        "id": "faq_005",
        "question": "How do I cancel my order?",
        "answer": "You can cancel your order within 1 hour of placing it by contacting customer support. After 1 hour, orders enter processing and cannot be canceled, but you can return the item once received.",
        "category": "orders",
    },
    {
        "id": "faq_006",
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to over 50 countries worldwide. International shipping rates vary by destination and typically take 10-14 business days.",
        "category": "shipping",
    },
    {
        "id": "faq_007",
        "question": "What if my item arrives damaged?",
        "answer": "If your item arrives damaged, please contact us within 48 hours with photos of the damage. We will send a replacement at no additional cost and arrange free return shipping for the damaged item.",
        "category": "returns",
    },
    {
        "id": "faq_008",
        "question": "How do I change my shipping address?",
        "answer": "You can update your shipping address in your account settings before placing an order. For orders already placed, contact customer support within 1 hour to request an address change.",
        "category": "shipping",
    },
]

print(f"📊 FAQ Corpus: {len(FAQ_CORPUS)} entries")
for faq in FAQ_CORPUS:
    print(f"   [{faq['category']}] {faq['id']}: {faq['question'][:40]}...")
```

### Step 2: Initialize the Vector Store

```python
import numpy as np

# Simple embedding function (for demonstration)
def simple_bow_embedding(text: str, vocab: List[str] = None) -> np.ndarray:
    """
    Create a simple bag-of-words embedding.
    
    Args:
        text: Text to embed
        vocab: Vocabulary list
        
    Returns:
        Embedding vector
    """
    if vocab is None:
        vocab = [
            "return", "refund", "shipping", "ship", "order", "payment", 
            "track", "cancel", "day", "days", "hour", "free", 
            "international", "damaged", "broken", "address", "account",
            "paypal", "visa", "credit", "policy", "replace"
        ]
    
    words = text.lower().split()
    embedding = np.array([words.count(w) for w in vocab], dtype=np.float32)
    return embedding

# Create and populate vector store
vector_store = SimpleVectorStore()
vector_store.set_embedding_function(simple_bow_embedding)
vector_store.add_documents(FAQ_CORPUS)

print("✅ Vector store initialized with FAQ corpus!")
print(f"   Embedding dimension: {len(simple_bow_embedding('test'))}")
```

### Step 3: Create the Generator Function

```python
# Mock generator function (simulates tinyGPT behavior)
class MockTinyGPT:
    """
    Mock generator that simulates tinyGPT behavior.
    
    For demonstration, it uses template-based generation
    based on retrieved context.
    """
    
    def __init__(self):
        """Initialize the mock generator."""
        pass
    
    def generate(
        self,
        question: str,
        context_list: List[str],
    ) -> str:
        """
        Generate an answer given question and context.
        
        Args:
            question: Customer question
            context_list: List of retrieved context snippets
            
        Returns:
            Generated answer
        """
        if not context_list:
            return "I don't have information about that topic."
        
        # Use the first (most relevant) context
        main_context = context_list[0]
        
        # Simple template-based response
        return f"Based on our policies: {main_context}"


# Create generator
mock_generator = MockTinyGPT()

def generator_fn(question: str, context_list: List[str]) -> str:
    """Wrapper function for the generator."""
    return mock_generator.generate(question, context_list)

print("✅ Mock tinyGPT generator created!")
```

### Step 4: Run Retrieval Tests

```python
# Test retrieval quality
print("🔍 Testing Retrieval Quality...")
print("=" * 70)

for tc in RAG_TEST_CASES[:3]:  # Test first 3
    results = vector_store.search(tc["question"], top_k=3)
    
    print(f"\nQuestion: {tc['question']}")
    print(f"Expected FAQs: {tc['expected_faq_ids']}")
    print(f"Retrieved:")
    for doc, score in results:
        retrieved_id = doc.get("id", "N/A")
        is_expected = "✓" if retrieved_id in tc["expected_faq_ids"] else "✗"
        print(f"   {is_expected} [{retrieved_id}] Score: {score:.3f} - {doc['question'][:30]}...")
```

### Step 5: Run Full RAG Evaluation

```python
# Create evaluator
rag_evaluator = RAGEvaluator(
    vector_store=vector_store,
    generator_fn=generator_fn,
)

# Run evaluation
print("🔄 Running Full RAG Evaluation...")
print("=" * 70)

all_results = []
for tc in RAG_TEST_CASES:
    result = rag_evaluator.run_evaluation(
        question=tc["question"],
        expected_faq_ids=tc["expected_faq_ids"],
        expected_answer_contains=tc["expected_answer_contains"],
    )
    all_results.append(result)
    
    print(f"\n{'='*60}")
    print(f"Question: {tc['question']}")
    print(f"Category: {tc['category']}")
    print(f"{'='*60}")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"\nRetrieval Metrics:")
    print(f"   Precision: {result['retrieval_metrics']['precision']:.2f}")
    print(f"   Recall: {result['retrieval_metrics']['recall']:.2f}")
    print(f"   Hit Rate: {result['retrieval_metrics']['hit_rate']:.2f}")
    print(f"\nGroundedness (String Match):")
    print(f"   Score: {result['groundedness_string']['groundedness_score']:.2f}")
    print(f"   Matched: {result['groundedness_string']['matched_phrases']}")
    print(f"   Unmatched: {result['groundedness_string']['unmatched_phrases']}")

# Compute aggregate metrics
metrics = rag_evaluator.compute_aggregate_metrics(all_results)
print(f"\n{'='*70}")
print("📊 Aggregate Metrics")
print(f"{'='*70}")
print(f"Average Precision: {metrics['avg_precision']:.2%}")
print(f"Average Recall: {metrics['avg_recall']:.2%}")
print(f"Average Hit Rate: {metrics['avg_hit_rate']:.2%}")
print(f"Average Groundedness: {metrics['avg_groundedness_score']:.2%}")
```

---

# 🤔 Section 6 — Paul-Elder Critical Thinking Questions

### Question 1: RETRIEVAL vs. GENERATION
**In a RAG system, which component—retrieval or generation—has a greater impact on answer quality? How should evaluation effort be allocated between them?**

*Consider: If the wrong documents are retrieved, even a perfect generator will fail. If retrieval is perfect but generation is poor, the context is wasted. How do you design evaluations that identify the bottleneck in your RAG system?*

### Question 2: GROUNDEDNESS vs. HELPFULNESS
**When evaluating customer service RAG, should we prioritize groundedness (answers strictly based on context) or helpfulness (answers that solve the customer's problem)? When might these conflict?**

*Consider: A perfectly grounded answer might miss important context not in the FAQ. A helpful answer might include inferences or suggestions not explicitly in the retrieved documents. How do you balance these in evaluation metrics?*

### Question 3: EVALUATION FAITHFULNESS
**How can we ensure that an LLM judge accurately assesses groundedness? What biases might an LLM judge have, and how could these affect evaluation reliability?**

*Consider: LLM judges may have their own knowledge that influences assessments. They may be lenient or strict depending on prompting. How do you validate that your LLM judge's assessments align with human judgments?*

---

# 🔄 Section 7 — Inversion Thinking: How Can RAG Systems Fail?

Instead of asking "How does RAG improve customer service?", let's invert:

> **"How can RAG systems cause problems in customer service?"**

### Failure Modes

1. **Retrieval Failures**
   - Relevant documents not retrieved
   - Wrong documents retrieved due to semantic mismatch
   - Consequence: Incorrect or irrelevant answers

2. **Context Window Overflow**
   - Too many documents retrieved
   - Important information truncated
   - Consequence: Missed critical details

3. **Hallucination Despite Context**
   - Model ignores retrieved context
   - Model adds information not in context
   - Consequence: Misinformation, policy violations

4. **Outdated Knowledge Base**
   - FAQ not updated with policy changes
   - Model gives old information
   - Consequence: Customer frustration, legal issues

5. **Poor Question Understanding**
   - Customer question misinterpreted
   - Retrieves documents for wrong intent
   - Consequence: Off-topic responses

6. **Conflicting Information**
   - Multiple FAQs with conflicting answers
   - Model confused by contradictions
   - Consequence: Inconsistent responses

### Defensive Practices

- **Retrieval Evaluation:** Regularly test that FAQs are retrieved correctly
- **Groundedness Checks:** Verify answers cite retrieved content
- **Confidence Thresholds:** Escalate to human when retrieval scores are low
- **Regular KB Updates:** Keep knowledge base current
- **A/B Testing:** Compare RAG answers to human agent answers
- **Fallback Paths:** Provide "contact support" option for edge cases
- **Citation Display:** Show customers which FAQ the answer is based on

---

# 📝 Section 8 — Mini-Project: Build a RAG Evaluation System

### Task

Create a complete RAG evaluation system for customer service that:
1. Uses a tiny FAQ corpus with at least 8 entries
2. Implements retrieval with a simple vector store
3. Evaluates both retrieval quality and answer groundedness
4. Uses both string matching and (optionally) LLM judge for groundedness

### Instructions

1. **Create your FAQ corpus:**
   - At least 8 FAQ entries
   - Cover at least 3 categories (returns, shipping, payments, etc.)
   - Include question, answer, and metadata

2. **Implement retrieval:**
   - Use the SimpleVectorStore or integrate FAISS
   - Test retrieval with at least 5 questions
   - Measure precision, recall, and hit rate

3. **Implement generation:**
   - Create a generator function (mock or real LLM)
   - Generate answers using retrieved context
   - Test on the same 5+ questions

4. **Evaluate groundedness:**
   - Use string matching for key phrases
   - Optionally use an LLM judge for qualitative assessment
   - Report groundedness scores

5. **Analyze results:**
   - Which questions have best retrieval?
   - Which answers are most grounded?
   - What patterns cause failures?

### Submission Format

Create a markdown file `/examples/week15_rag_evaluation_audit.md`:

```markdown
# Week 15 Mini-Project: RAG Evaluation Audit

## Executive Summary
[2-3 sentences on overall RAG system performance]

## FAQ Corpus

### Categories and Coverage
| Category | # FAQs | Example Question |
|----------|--------|------------------|
| returns | 2 | What is your return policy? |
| shipping | 3 | How long does shipping take? |
| ... | ... | ... |

### Total FAQs: [number]

## Retrieval Evaluation

### Test Questions
| # | Question | Expected FAQ | Retrieved FAQ | Hit? |
|---|----------|--------------|---------------|------|
| 1 | Can I get a refund? | faq_001 | faq_001 | ✅ |
| 2 | Where's my order? | faq_003 | faq_002 | ❌ |
| ... | ... | ... | ... | ... |

### Retrieval Metrics
| Metric | Value |
|--------|-------|
| Average Precision | 80% |
| Average Recall | 75% |
| Hit Rate | 85% |

## Generation Evaluation

### Sample Outputs
| Question | Answer Excerpt | Grounded? |
|----------|----------------|-----------|
| Can I get a refund? | "Within 30 days..." | ✅ |
| ... | ... | ... |

## Groundedness Analysis

### String Match Results
| # | Question | Expected Phrases | Matched | Score |
|---|----------|------------------|---------|-------|
| 1 | Refund? | [30 days, refund] | [30 days, refund] | 100% |
| 2 | Shipping? | [5-7 days, express] | [5-7 days] | 50% |
| ... | ... | ... | ... | ... |

### Overall Groundedness Score: [value]%

### LLM Judge Results (if used)
| # | Grounded | Hallucination | Notes |
|---|----------|---------------|-------|
| 1 | ✅ | ❌ | Well-grounded |
| 2 | ❌ | ✅ | Added unsupported claim |
| ... | ... | ... | ... |

## Failure Analysis

### Retrieval Failures
[What caused retrieval failures?]

### Groundedness Failures
[What caused groundedness issues?]

## Recommendations

### For Retrieval
- [Recommendation 1]
- [Recommendation 2]

### For Generation
- [Recommendation 1]
- [Recommendation 2]

### For Knowledge Base
- [Recommendation 1]
- [Recommendation 2]

## Limitations

### What This Evaluation Cannot Assess
- [Limitation 1: e.g., real user satisfaction]
- [Limitation 2: e.g., edge case handling]

### Future Improvements
- [Improvement 1]
- [Improvement 2]
```

---

# 🔧 Section 9 — Advanced: Extending the RAG Evaluator

### Adding FAISS Integration

For production use, integrate with FAISS for efficient similarity search:

```python
# Note: Requires `pip install faiss-cpu` or `pip install faiss-gpu`
import faiss
import numpy as np

class FAISSVectorStore:
    """
    Vector store using FAISS for efficient similarity search.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize the FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[Dict[str, Any]] = []
        self.embed_fn = None
    
    def set_embedding_function(self, embed_fn):
        """Set the embedding function."""
        self.embed_fn = embed_fn
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the index."""
        self.documents = documents
        
        texts = [doc.get("answer", doc.get("text", "")) for doc in documents]
        embeddings = np.array([self.embed_fn(text) for text in texts], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for similar documents."""
        query_embedding = np.array([self.embed_fn(query)], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # -1 indicates no result
                # Convert L2 distance to similarity (closer to 0 = more similar)
                similarity = 1.0 / (1.0 + dist)
                results.append((self.documents[idx], similarity))
        
        return results
```

### Adding Sentence Transformer Embeddings

For better semantic search, use sentence transformers:

```python
# Note: Requires `pip install sentence-transformers`
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder:
    """
    Embedding function using SentenceTransformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, text: str) -> np.ndarray:
        """Embed text."""
        return self.model.encode(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        return self.model.encode(texts)
```

### Adding Retrieval Quality Analysis

```python
def analyze_retrieval_quality(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze retrieval quality across multiple queries.
    
    Returns:
        Dictionary with retrieval analysis
    """
    # Compute MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for r in results:
        expected = set(r.get("expected_faq_ids", []))
        retrieved = r.get("retrieved_ids", [])
        
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in expected:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # Compute success rate at different k values
    success_at_k = {}
    for k in [1, 3, 5]:
        successes = 0
        for r in results:
            expected = set(r.get("expected_faq_ids", []))
            retrieved = r.get("retrieved_ids", [])[:k]
            if expected & set(retrieved):
                successes += 1
        success_at_k[f"success@{k}"] = successes / len(results) if results else 0.0
    
    return {
        "mrr": mrr,
        **success_at_k,
        "total_queries": len(results),
    }
```

### Adding Hallucination Detection

```python
def detect_hallucination(
    answer: str,
    context_list: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect potential hallucinations in the answer.
    
    Uses a simple heuristic: check if answer sentences
    have some overlap with context.
    
    Args:
        answer: Generated answer
        context_list: Retrieved context snippets
        threshold: Minimum word overlap ratio for grounded sentence
        
    Returns:
        Dictionary with hallucination analysis
    """
    # Combine all context
    context_words = set()
    for c in context_list:
        context_words.update(c.lower().split())
    
    # Split answer into sentences (simplified)
    sentences = answer.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    
    grounded_sentences = []
    hallucinated_sentences = []
    
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        if not sentence_words:
            continue
        
        overlap = len(sentence_words & context_words)
        ratio = overlap / len(sentence_words)
        
        if ratio >= threshold:
            grounded_sentences.append(sentence)
        else:
            hallucinated_sentences.append(sentence)
    
    hallucination_rate = len(hallucinated_sentences) / len(sentences) if sentences else 0.0
    
    return {
        "hallucination_rate": hallucination_rate,
        "grounded_sentences": grounded_sentences,
        "potentially_hallucinated": hallucinated_sentences,
        "total_sentences": len(sentences),
    }
```

---

# ✔ Knowledge Mastery Checklist

Before moving to Week 16, ensure you can check all boxes:

- [ ] I understand what RAG is and why it's useful for customer service
- [ ] I can create a tiny FAQ corpus and explain its structure
- [ ] I can implement a simple vector store for semantic retrieval
- [ ] I understand how to evaluate retrieval quality (precision, recall, hit rate)
- [ ] I can evaluate answer groundedness using string matching
- [ ] I understand how to use an LLM as a judge for groundedness assessment
- [ ] I can identify common RAG failure modes (retrieval errors, hallucination)
- [ ] I understand the trade-off between groundedness and helpfulness
- [ ] I completed the mini-project RAG evaluation audit

---

Week 15 complete.
Next: *Week 16 — Marketing & Content Use Cases*.
