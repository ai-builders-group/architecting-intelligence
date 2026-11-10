# --8<-- [start:lst-7.24]
# src/agent/supervision/multi_head_verifier.py
from typing import Literal
from pydantic import BaseModel, Field
import asyncio

class VerificationResult(BaseModel):
    """Result from a single verifier model."""
    model_id: str
    decision: dict  # The model's output
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)

class MultiHeadVerifier:
    """
    Ensemble verification using independent Tier 1 models to validate Tier 3 output.
    
    This pattern replaces fragile logit calibration with the physics of
    independent sampling: if 3 diverse models agree, the probability of
    collective hallucination is P_error^3.
    """
    
    def __init__(
        self,
        verifier_models: list[str],  # e.g., ["claude-haiku", "gpt-4o-mini", "llama-3.1-8b"]
        consensus_threshold: float = 1.0,  # Require unanimous agreement
    ):
        self.verifiers = verifier_models
        self.consensus_threshold = consensus_threshold
    
    async def verify(
        self,
        primary_output: dict,
        context: str,
    ) -> tuple[bool, float, str]:
        """
        Verify primary model output using ensemble voting.
        
        Args:
            primary_output: The Tier 3 model's decision
            context: Original query + context
        
        Returns:
            (should_escalate, consensus_score, reason)
        """
        # Run all verifiers in parallel
        verification_tasks = [
            self._verify_single(model_id, primary_output, context)
            for model_id in self.verifiers
        ]
        results = await asyncio.gather(
            *verification_tasks, return_exceptions=True
        )
        
        # Consensus: fraction of verifiers that agree; an exception is a
        # disagreement
        agreements = [
            isinstance(result, VerificationResult)
            and self._decisions_match(primary_output, result.decision)
            for result in results
        ]
        consensus = sum(agreements) / len(agreements) if agreements else 0.0
        
        # Escalate if consensus falls below threshold, or if there are no
        # verifiers
        should_escalate = not agreements or consensus < self.consensus_threshold
        
        reason = (
            f"Consensus: {consensus:.1%} "
            f"({sum(agreements)}/{len(agreements)} agree). "
            f"Threshold: {self.consensus_threshold:.1%}"
        )
        
        return should_escalate, consensus, reason
    
    async def _verify_single(
        self, model_id: str, primary_output: dict, context: str
    ) -> VerificationResult:
        """Ask a verifier model to independently solve the same task."""
        # Prompt the verifier to generate its own decision
        verifier_prompt = f"""
        Original task: {context}
        
        Another model proposed this decision:
        {primary_output}
        
        Independently solve this task. Do NOT be biased by the proposal.
        Output your own decision in the same format.
        """
        
        # Call verifier model (implementation depends on LLM client)
        # response = await llm_client.generate(model=model_id, prompt=verifier_prompt)
        
        # For demonstration, return placeholder
        return VerificationResult(
            model_id=model_id,
            decision=primary_output,  # Replace with actual verifier output
            reasoning="Verifier reasoning trace",
            confidence=0.9,
        )
    
    def _decisions_match(self, decision_a: dict, decision_b: dict) -> bool:
        """
        Check semantic equivalence of decisions.
        
        Implementation depends on decision schema. For structured output:
        - Compare action names
        - Compare critical parameters (e.g., amounts, IDs)
        - Allow tolerance for cosmetic differences (wording, formatting)
        """
        # Example: exact match on action type
        return decision_a.get("action") == decision_b.get("action")
# --8<-- [end:lst-7.24]
