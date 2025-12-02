"""
LLM-as-Judge scoring module for BenchRight evaluation.

This module provides a generic LLM-based scoring mechanism that uses
a carefully designed system prompt to evaluate correctness, coherence,
and helpfulness of model outputs.

The LLMJudge class is designed to be reusable across multiple evaluation
scenarios and can work with any OpenAI-compatible client.

Example usage:
    >>> from openai import OpenAI
    >>> from benchmark_engine.llm_judge import LLMJudge
    >>> client = OpenAI()
    >>> judge = LLMJudge(client)
    >>> result = judge.score_answer(
    ...     question="What is the capital of France?",
    ...     answer="Paris",
    ...     reference="Paris"
    ... )
    >>> print(result['score'], result['rationale'])
"""

from typing import Any, Dict, Optional
import json
import re


# System prompt designed for general-purpose evaluation
# Reusable across Weeks 6, 11-16 evaluation scenarios
EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator for language model outputs. Your task is to evaluate the quality of an answer given a question and an optional reference answer.

Evaluate the answer on three criteria:
1. **Correctness** (0-10): How factually accurate and correct is the answer? Does it match the reference if provided?
2. **Coherence** (0-10): Is the answer well-structured, logical, and easy to understand?
3. **Helpfulness** (0-10): Does the answer adequately address the question? Is it informative and useful?

For each criterion, provide a score from 0 to 10 where:
- 0-3: Poor (incorrect, incoherent, or unhelpful)
- 4-6: Moderate (partially correct/coherent/helpful with some issues)
- 7-9: Good (mostly correct/coherent/helpful with minor issues)
- 10: Excellent (fully correct/coherent/helpful)

Compute an overall score as the weighted average: (Correctness * 0.5) + (Coherence * 0.25) + (Helpfulness * 0.25)

Respond ONLY with a valid JSON object in this exact format:
{
    "correctness": <score>,
    "coherence": <score>,
    "helpfulness": <score>,
    "score": <overall_score>,
    "rationale": "<brief explanation of the scores>"
}

Do not include any text before or after the JSON object."""


class LLMJudge:
    """
    A judge class that uses an LLM to evaluate answer quality.

    This class wraps an OpenAI-compatible client and uses a carefully crafted
    system prompt to evaluate answers based on correctness, coherence, and
    helpfulness.

    Attributes:
        client: An OpenAI-compatible client object with a chat.completions.create method.
        model: The model name to use for evaluation (default: "gpt-4").
        system_prompt: The system prompt used for evaluation.

    Example:
        >>> from openai import OpenAI
        >>> judge = LLMJudge(OpenAI())
        >>> result = judge.score_answer("What is 2+2?", "4", "4")
        >>> print(result['score'])
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4",
        system_prompt: Optional[str] = None
    ) -> None:
        """
        Initialize the LLM Judge.

        Args:
            client: An OpenAI-compatible client object. Must have a
                    chat.completions.create method that accepts model and messages.
            model: The model name to use for evaluation. Defaults to "gpt-4".
            system_prompt: Optional custom system prompt. If not provided,
                          uses the default evaluation prompt.
        """
        self.client = client
        self.model = model
        self.system_prompt = system_prompt or EVALUATION_SYSTEM_PROMPT

    def score_answer(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Score an answer using the LLM judge.

        Args:
            question: The original question or prompt.
            answer: The model-generated answer to evaluate.
            reference: Optional reference/expected answer for comparison.
                      If not provided, evaluation is based solely on the
                      question and answer.

        Returns:
            A dictionary containing:
                - 'score': float - Overall score (0-10) as weighted average
                - 'rationale': str - Brief explanation of the scoring
                - 'correctness': float - Correctness score (0-10)
                - 'coherence': float - Coherence score (0-10)
                - 'helpfulness': float - Helpfulness score (0-10)
                - 'raw_response': str - The raw LLM response (for debugging)

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
            Exception: If the LLM API call fails.

        Example:
            >>> result = judge.score_answer(
            ...     question="What is the capital of France?",
            ...     answer="The capital of France is Paris.",
            ...     reference="Paris"
            ... )
            >>> print(f"Score: {result['score']}/10")
            >>> print(f"Rationale: {result['rationale']}")
        """
        # Construct the user message
        user_message = self._format_user_message(question, answer, reference)

        # Call the LLM
        response = self._call_llm(user_message)

        # Parse and validate the response
        result = self._parse_response(response)
        result['raw_response'] = response

        return result

    def _format_user_message(
        self,
        question: str,
        answer: str,
        reference: Optional[str]
    ) -> str:
        """
        Format the user message for the LLM.

        Args:
            question: The original question.
            answer: The answer to evaluate.
            reference: Optional reference answer.

        Returns:
            Formatted user message string.
        """
        message = f"""Please evaluate the following:

**Question:** {question}

**Answer to evaluate:** {answer}
"""
        if reference:
            message += f"""
**Reference answer:** {reference}
"""
        return message

    def _call_llm(self, user_message: str) -> str:
        """
        Call the LLM API with the evaluation request.

        Args:
            user_message: The formatted user message.

        Returns:
            The LLM's response text.

        Raises:
            Exception: If the API call fails.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0  # Use deterministic output for consistency
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured result.

        Args:
            response: The raw LLM response string.

        Returns:
            Parsed dictionary with score and rationale.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        # Try to extract JSON from the response
        try:
            # First, try direct JSON parsing
            result = json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse JSON from LLM response: {e}\n"
                        f"Response was: {response}"
                    )
            else:
                raise ValueError(
                    f"No valid JSON found in LLM response: {response}"
                )

        # Validate required fields
        required_fields = ['score', 'rationale']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field '{field}' in response")

        # Ensure score is a float
        result['score'] = float(result['score'])

        # Add optional fields with defaults if missing
        result.setdefault('correctness', result['score'])
        result.setdefault('coherence', result['score'])
        result.setdefault('helpfulness', result['score'])

        return result


class MockLLMClient:
    """
    A mock LLM client for testing purposes.

    This class simulates an OpenAI-compatible client without making
    actual API calls. Useful for unit testing and demonstrations.

    Example:
        >>> mock_client = MockLLMClient()
        >>> judge = LLMJudge(mock_client)
        >>> result = judge.score_answer("What is 2+2?", "4", "4")
    """

    class MockChoice:
        """Mock choice object."""
        def __init__(self, content: str):
            self.message = type('Message', (), {'content': content})()

    class MockResponse:
        """Mock response object."""
        def __init__(self, content: str):
            self.choices = [MockLLMClient.MockChoice(content)]

    class MockChatCompletions:
        """Mock chat completions object."""
        def create(self, **kwargs) -> 'MockLLMClient.MockResponse':
            """Return a mock response with evaluation scores."""
            mock_result = {
                "correctness": 8.0,
                "coherence": 9.0,
                "helpfulness": 8.0,
                "score": 8.25,
                "rationale": "The answer is correct and well-structured, providing the expected information clearly."
            }
            return MockLLMClient.MockResponse(json.dumps(mock_result))

    class MockChat:
        """Mock chat object."""
        def __init__(self):
            self.completions = MockLLMClient.MockChatCompletions()

    def __init__(self):
        """Initialize mock client."""
        self.chat = self.MockChat()


if __name__ == "__main__":
    # Demonstration of LLM Judge
    print("=" * 60)
    print("BenchRight LLM-as-Judge Demo")
    print("=" * 60)

    # Create a mock client for demonstration
    mock_client = MockLLMClient()
    judge = LLMJudge(mock_client, model="gpt-4")

    # Test cases
    test_cases = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "reference": "Paris"
        },
        {
            "question": "Explain photosynthesis in simple terms.",
            "answer": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
            "reference": "Plants use sunlight to make food from water and carbon dioxide."
        },
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "reference": "4"
        },
    ]

    print("\n📝 Running LLM-as-Judge evaluation...")
    print("-" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}:")
        print(f"   Question: {test['question']}")
        print(f"   Answer: {test['answer'][:50]}...")
        print(f"   Reference: {test['reference']}")

        result = judge.score_answer(
            question=test['question'],
            answer=test['answer'],
            reference=test['reference']
        )

        print(f"\n   📊 Scores:")
        print(f"      Overall: {result['score']:.2f}/10")
        print(f"      Correctness: {result.get('correctness', 'N/A')}/10")
        print(f"      Coherence: {result.get('coherence', 'N/A')}/10")
        print(f"      Helpfulness: {result.get('helpfulness', 'N/A')}/10")
        print(f"   📋 Rationale: {result['rationale']}")

    print("\n" + "=" * 60)
    print("Demo complete!")
