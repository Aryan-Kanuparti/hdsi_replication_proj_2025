import json
import os
from typing import Any, Dict, List

from anthropic import Anthropic


class EducationalEnhancement:
    """Wraps query results with pedagogical features"""

    def __init__(self, anthropic_client: Anthropic):
        """
        Initialize with Anthropic client

        Args:
            anthropic_client: Instance of Anthropic() client
        """
        self.client = anthropic_client
        self.model = "claude-sonnet-4-20250514"

    def enhance_response(
        self, state: Dict[str, Any], educational_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Add educational layers to response

        Returns dict with: answer, skill_level, decomposition, justification,
        limitations, next_questions, vocabulary, confidence
        """
        if not educational_mode:
            return {}

        print("[Educational Agent] Enhancing response...")

        return {
            "answer": state.get("final_answer", ""),
            "skill_level": self._classify_difficulty(state.get("user_question", "")),
            "decomposition": self._decompose_query(state),
            "justification": self._generate_justification(state),
            "limitations": self._identify_limitations(state),
            "next_questions": self._suggest_followups(state),
            "vocabulary": self._extract_vocabulary(state),
            "confidence": self._assess_confidence(state),
        }

    def _call_llm(self, prompt: str) -> str:
        """Helper to call Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"[Educational Agent] LLM call failed: {e}")
            return ""

    def _classify_difficulty(self, question: str) -> str:
        """Classify as beginner/intermediate/advanced"""
        if not question:
            return "beginner"

        prompt = f"""Classify this biomedical question's difficulty level.

Question: {question}

Levels:
- beginner: Single relationship (e.g., "What does TP53 do?")
- intermediate: Aggregation/filtering (e.g., "How many diseases link to TP53?")
- advanced: Multi-hop/comparison (e.g., "Compare TP53 vs BRCA1")

Return ONLY one word: beginner, intermediate, or advanced"""

        response = self._call_llm(prompt).strip().lower()

        # Validate response
        if response in ["beginner", "intermediate", "advanced"]:
            return response
        return "beginner"

    def _decompose_query(self, state: Dict) -> List[Dict]:
        """Break complex queries into steps"""
        question_type = state.get("question_type", "")

        if question_type not in ["comparative_analysis", "statistical_analysis"]:
            return None

        prompt = f"""Break this question into step-by-step sub-questions.

Question: {state.get('user_question', '')}
Question Type: {question_type}

Return ONLY a JSON array (no other text):
[
  {{
    "step": 1,
    "question": "What information do we need first?",
    "purpose": "Why this step matters",
    "result": "What we expect to find"
  }}
]

Maximum 3 steps. Be concise."""

        response = self._call_llm(prompt)

        try:
            # Try to parse JSON
            parsed = json.loads(response)
            return parsed if isinstance(parsed, list) else None
        except:
            print(f"[Educational Agent] Failed to parse decomposition")
            return None

    def _generate_justification(self, state: Dict) -> str:
        """Explain reasoning"""
        prompt = f"""Explain how we arrived at this answer in 3-4 sentences for a student.

Question: {state.get('user_question', '')}
Type: {state.get('question_type', '')}
Entities: {state.get('entities', [])}
Results: {len(state.get('results', []))} found

Include:
1. Why we classified it this way
2. How we identified entities  
3. What the query does

Be concise and educational."""

        response = self._call_llm(prompt)
        return response if response else "No justification available"

    def _identify_limitations(self, state: Dict) -> List[str]:
        """What doesn't this answer tell us?"""
        prompt = f"""What are 2-3 limitations of this answer?

Question: {state.get('user_question', '')}
Answer: {state.get('final_answer', '')}

Return ONLY a JSON array: ["limitation 1", "limitation 2"]"""

        response = self._call_llm(prompt)

        try:
            parsed = json.loads(response)
            return parsed if isinstance(parsed, list) else []
        except:
            return []

    def _suggest_followups(self, state: Dict) -> List[str]:
        """Suggest next questions"""
        prompt = f"""Suggest 3 follow-up questions to deepen learning.

Question: {state.get('user_question', '')}
Answer: {state.get('final_answer', '')}

Make them progressively harder. Return ONLY JSON: ["q1", "q2", "q3"]"""

        response = self._call_llm(prompt)

        try:
            parsed = json.loads(response)
            return parsed if isinstance(parsed, list) else []
        except:
            return []

    def _extract_vocabulary(self, state: Dict) -> Dict[str, str]:
        """Define medical terms"""
        prompt = f"""Extract 2-3 medical/technical terms and define them simply.

Question: {state.get('user_question', '')}
Answer: {state.get('final_answer', '')}

Return ONLY JSON: {{"term": "simple definition"}}"""

        response = self._call_llm(prompt)

        try:
            parsed = json.loads(response)
            return parsed if isinstance(parsed, dict) else {}
        except:
            return {}

    def _assess_confidence(self, state: Dict) -> int:
        """Return confidence score 0-100"""
        score = 50  # Base

        if state.get("results") and len(state["results"]) > 0:
            score += 20
        if state.get("entities"):
            score += 15
        if state.get("error"):
            score -= 30
        if state.get("question_type") in ["gene_disease", "drug_treatment"]:
            score += 15

        return max(0, min(100, score))


# Test code
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    enhancer = EducationalEnhancement(client)

    # Mock state
    test_state = {
        "user_question": "How many diseases is TP53 linked to?",
        "question_type": "statistical_analysis",
        "entities": ["TP53"],
        "final_answer": "TP53 is linked to 37 diseases.",
        "results": [{"count": 37}],
    }

    print("Testing Educational Enhancement...")
    result = enhancer.enhance_response(test_state, educational_mode=True)
    print("\n=== RESULTS ===")
    print(json.dumps(result, indent=2))
