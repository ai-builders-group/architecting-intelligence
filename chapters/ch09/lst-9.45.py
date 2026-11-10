# --8<-- [start:lst-9.45]
# chapter_07/verified_summarization_nli.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from openai import OpenAI
import re
from datetime import datetime, UTC

client = OpenAI()

class AuthoritativeFact(BaseModel):
    """A specific fact that must be preserved in summaries."""
    fact_type: Literal["structured", "semantic"]
    value: str
    source_message_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    verification_method: Literal["regex", "nli"]

class SummaryVerification(BaseModel):
    """Result of fact-preservation verification."""
    is_valid: bool
    preserved_facts: List[AuthoritativeFact]
    missing_facts: List[AuthoritativeFact]
    hallucinated_content: List[str] = Field(default_factory=list)
    verification_score: float = Field(ge=0.0, le=1.0)
    entailment_score: float = Field(ge=0.0, le=1.0, description="NLI semantic check")

class TopicSegment(BaseModel):
    """A coherent topic segment within a conversation."""
    messages: List[Message]
    summary: str | None = None
    authoritative_facts: List[AuthoritativeFact] = Field(default_factory=list)
    verification: SummaryVerification | None = None
    start_turn: int
    end_turn: int
    fallback_used: bool = Field(default=False, description="True if using raw history")

class FactExtractor:
    """
    Extract authoritative facts using hybrid regex + semantic patterns.
    
    Facts are concrete, verifiable data points that must survive summarization.
    """
    
    @staticmethod
    def extract_structured_facts(messages: List[Message]) -> List[AuthoritativeFact]:
        """
        Extract structured facts using deterministic regex patterns.
        
        Returns only facts from tool outputs and user-provided data.
        """
        facts = []
        
        # Patterns for different fact types
        patterns = {
            "id": r'\b(?:ID|id|#)[\s:]*([\w-]{8,})\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "date": r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            "number": r'\$[\d,]+\.?\d*|\b\d+\.?\d*%',
        }
        
        for msg in messages:
            # Only extract from authoritative sources
            if not (msg.is_authoritative or msg.role == "user" or msg.role == "tool"):
                continue
            
            for fact_type, pattern in patterns.items():
                matches = re.findall(pattern, msg.content)
                for match in matches:
                    facts.append(AuthoritativeFact(
                        fact_type="structured",
                        value=match if isinstance(match, str) else match[0],
                        source_message_id=msg.id,
                        confidence=1.0 if msg.role == "tool" else 0.8,
                        verification_method="regex"
                    ))
        
        # Deduplicate while preserving order
        seen = set()
        unique_facts = []
        for fact in facts:
            key = (fact.fact_type, fact.value)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
        
        return unique_facts
    
    @staticmethod
    def extract_semantic_facts(messages: List[Message]) -> List[AuthoritativeFact]:
        """
        Extract semantic facts using LLM-based extraction.
        
        Captures user-provided context that is semantically critical
        but not regex-matchable (e.g., "my dog died", "I'm pregnant").
        """
        facts = []
        
        # Filter to user messages (where semantic context lives)
        user_messages = [m for m in messages if m.role == "user"]
        
        if not user_messages:
            return []
        
        # Concatenate user messages
        user_text = "\n".join([m.content for m in user_messages])
        
        # Extract semantic facts using cheap model
        response = client.chat.completions.create(
            model="gemini-3-flash",
            messages=[{
                "role": "system",
                "content": (
                    "Extract critical semantic facts from user messages that an agent must remember. "
                    "Focus on: life events, personal circumstances, emotional context, problem descriptions. "
                    "Output as a JSON object: {\"facts\": [{\"fact\": \"...\", \"importance\": 0.0-1.0}]}"
                )
            }, {
                "role": "user",
                "content": user_text
            }],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        import json
        try:
            extracted = json.loads(response.choices[0].message.content)
            for item in extracted.get("facts", []):
                if item.get("importance", 0) >= 0.7:  # High importance threshold
                    facts.append(AuthoritativeFact(
                        fact_type="semantic",
                        value=item["fact"],
                        source_message_id=user_messages[0].id,  # Approximate
                        confidence=item.get("importance", 0.8),
                        verification_method="nli"
                    ))
        except json.JSONDecodeError:
            pass  # Extraction failed - return empty list
        
        return facts

class VerifiedSummarizer:
    """
    Summarize conversation segments with mandatory fact verification.
    
    Uses dual-layer verification (regex + NLI) and graceful degradation
    to prevent memory poisoning from hallucinated or lossy compression.
    """
    
    def __init__(self):
        self.client = OpenAI()
        self.fact_extractor = FactExtractor()
    
    def summarize_segment(
        self, 
        segment: TopicSegment,
        structured_facts: List[AuthoritativeFact],
        semantic_facts: List[AuthoritativeFact]
    ) -> str:
        """
        Generate a summary with explicit fact preservation instructions.
        """
        # Build fact list for the prompt
        structured_list = "\n".join([
            f"- {fact.value} ({fact.fact_type})"
            for fact in structured_facts
        ])
        
        semantic_list = "\n".join([
            f"- {fact.value}"
            for fact in semantic_facts
        ])
        
        # Construct the conversation text
        transcript = "\n".join([
            f"{m.role.upper()}: {m.content}"
            for m in segment.messages
        ])
        
        # Generate summary with explicit fact preservation
        response = self.client.chat.completions.create(
            model="gemini-3-flash",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conversation summarizer. Generate a 150-200 token summary "
                        "that preserves ALL the facts listed below.\n\n"
                        "STRUCTURED FACTS (must appear verbatim):\n"
                        f"{structured_list}\n\n"
                        "SEMANTIC FACTS (must be preserved in meaning):\n"
                        f"{semantic_list}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Summarize this conversation segment:\n\n{transcript}"
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def verify_summary_hybrid(
        self, 
        summary: str, 
        structured_facts: List[AuthoritativeFact],
        semantic_facts: List[AuthoritativeFact],
        original_transcript: str
    ) -> SummaryVerification:
        """
        Cross-examine the summary using regex + NLI.
        
        Structured facts verified via regex (exact match).
        Semantic facts verified via NLI entailment check.
        
        Returns:
            Verification result with preserved/missing facts
        """
        preserved = []
        missing = []
        
        # Layer 1: Regex verification for structured facts
        for fact in structured_facts:
            if re.search(rf"(?<!\w){re.escape(fact.value)}(?!\w)", summary):
                preserved.append(fact)
            else:
                missing.append(fact)
        
        # Layer 2: NLI verification for semantic facts
        if semantic_facts:
            # Use NLI model to check if summary entails semantic facts
            entailment_response = self.client.chat.completions.create(
                model="gemini-3-flash",
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a Natural Language Inference (NLI) judge. "
                        "Determine if the SUMMARY preserves the semantic meaning "
                        "of each FACT. "
                        "Output JSON: {\"results\": [{\"fact\": \"...\", "
                        "\"entailed\": true/false}]}"
                    )
                }, {
                    "role": "user",
                    "content": (
                        f"SUMMARY:\n{summary}\n\n"
                        f"FACTS TO CHECK:\n" +
                        "\n".join([f"- {f.value}" for f in semantic_facts])
                    )
                }],
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                nli_results = json.loads(entailment_response.choices[0].message.content)
                for fact in semantic_facts:
                    # Check if this fact was entailed
                    fact_result = next(
                        (r for r in nli_results.get("results", []) if r["fact"] == fact.value),
                        None
                    )
                    
                    if fact_result and fact_result.get("entailed", False):
                        preserved.append(fact)
                    else:
                        missing.append(fact)
            except (json.JSONDecodeError, StopIteration):
                # NLI check failed - assume semantic facts are missing
                # (conservative)
                missing.extend(semantic_facts)
        
        # Calculate verification score
        total_facts = len(structured_facts) + len(semantic_facts)
        if total_facts == 0:
            score = 1.0  # No facts to preserve
        else:
            score = len(preserved) / total_facts
        
        # Calculate entailment score (semantic only)
        if semantic_facts:
            entailment_score = len([f for f in preserved if f.fact_type == "semantic"]) / len(semantic_facts)
        else:
            entailment_score = 1.0
        
        # Summary is valid only if ALL high-confidence facts are preserved
        high_conf_facts = [f for f in (structured_facts + semantic_facts) if f.confidence >= 0.9]
        high_conf_missing = [f for f in missing if f.confidence >= 0.9]
        
        is_valid = len(high_conf_missing) == 0
        
        return SummaryVerification(
            is_valid=is_valid,
            preserved_facts=preserved,
            missing_facts=missing,
            verification_score=score,
            entailment_score=entailment_score
        )
    
    def summarize_with_graceful_degradation(
        self, 
        segment: TopicSegment,
        max_retries: int = 2
    ) -> tuple[str, SummaryVerification, bool]:
        """
        Generate and verify summary with GRACEFUL DEGRADATION.
        
        If verification fails after max retries, FALLBACK to raw history
        and emit a cost warning rather than raising an exception.
        
        Returns:
            (content, verification_result, fallback_used)
        """
        # Extract facts (hybrid approach)
        structured_facts = self.fact_extractor.extract_structured_facts(segment.messages)
        semantic_facts = self.fact_extractor.extract_semantic_facts(segment.messages)
        
        segment.authoritative_facts = structured_facts + semantic_facts
        
        original_transcript = "\n".join([
            f"{m.role.upper()}: {m.content}"
            for m in segment.messages
        ])
        
        summary = self.summarize_segment(segment, structured_facts, semantic_facts)
        for attempt in range(max_retries + 1):
            verification = self.verify_summary_hybrid(
                summary, 
                structured_facts, 
                semantic_facts,
                original_transcript
            )
            
            if verification.is_valid:
                return summary, verification, False
            
            # If verification failed, regenerate with explicit missing facts
            if attempt < max_retries:
                missing_list = "\n".join([
                    f"- MISSING {f.fact_type.upper()}: {f.value}"
                    for f in verification.missing_facts
                ])
                
                # Retry with explicit correction
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"The previous summary was rejected because it dropped these facts:\n"
                                f"{missing_list}\n\n"
                                "Regenerate the summary ensuring these facts are included."
                            )
                        },
                        {
                            "role": "user",
                            "content": summary  # Show the failed attempt
                        }
                    ],
                    max_tokens=300
                )
                summary = response.choices[0].message.content
        
        # GRACEFUL DEGRADATION: Max retries exhausted
        # Do NOT raise exception - fallback to raw history
        import logging
        logging.warning(
            f"Summary verification failed after {max_retries} attempts. "
            f"Missing {len(verification.missing_facts)} authoritative facts. "
            f"Falling back to raw history. Cost impact: +${len(original_transcript.split()) * 0.000002:.4f}"
        )
        
        # Return original transcript as "summary"
        segment.fallback_used = True
        return original_transcript, verification, True
# --8<-- [end:lst-9.45]
