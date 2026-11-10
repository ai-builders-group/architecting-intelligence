# --8<-- [start:lst-16.3]
# src/ch10_weight_context_continuum/synthetic_data_factory.py
from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, UTC
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field, computed_field, model_validator


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class ModelTier(str, Enum):
    """Capability tiers for synthetic data generation."""
    TEACHER = "teacher"   # Frontier: Opus, GPT-4, Llama-70B
    JUDGE   = "judge"     # Strong: Sonnet, GPT-4o
    STUDENT = "student"   # Efficient: Haiku, Llama-8B, Mistral-7B


class JudgeDimension(str, Enum):
    """Rubric dimensions for candidate scoring."""
    CORRECTNESS  = "correctness"   # Factual accuracy
    COHERENCE    = "coherence"     # Internal logical consistency
    COMPLETENESS = "completeness"  # Coverage of the prompt's requirements
    SAFETY       = "safety"        # Absence of harmful content
    FORMAT       = "format"        # Adherence to structural requirements


class DistillationStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


# ─────────────────────────────────────────────
# CORE SCHEMAS
# ─────────────────────────────────────────────

class TrainingPrompt(BaseModel):
    """A single input prompt from the seed dataset."""
    prompt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    domain: str
    difficulty: float = Field(ge=0.0, le=1.0, description="Normalized difficulty [0, 1]")
    expected_format: str | None = Field(default=None)


class GenerationCandidate(BaseModel):
    """A single candidate response from the teacher model."""
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str
    response: str
    teacher_model: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    input_tokens: int
    output_tokens: int

    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA-256 fingerprint for deduplication."""
        return hashlib.sha256(self.response.encode()).hexdigest()


class JudgeScore(BaseModel):
    """Structured rubric assessment from the judge model."""
    score_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id: str
    prompt_id: str
    dimensions: dict[JudgeDimension, float] = Field(
        description="Per-dimension scores in [0.0, 1.0]"
    )
    judge_model: str
    judge_reasoning: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def composite_score(self) -> float:
        """
        Weighted harmonic mean of dimension scores.
        Safety is a hard gate: any safety score < 0.8 floors the composite to 0.
        """
        if self.dimensions.get(JudgeDimension.SAFETY, 1.0) < 0.8:
            return 0.0
        weights = {
            JudgeDimension.CORRECTNESS:  0.40,
            JudgeDimension.COHERENCE:    0.25,
            JudgeDimension.COMPLETENESS: 0.20,
            JudgeDimension.FORMAT:       0.10,
            JudgeDimension.SAFETY:       0.05,
        }
        weighted_sum = sum(
            weights.get(dim, 0.0) * score
            for dim, score in self.dimensions.items()
        )
        return round(min(1.0, weighted_sum), 6)

    @model_validator(mode="after")
    def validate_dimension_bounds(self) -> "JudgeScore":
        for dim, score in self.dimensions.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Dimension {dim} score {score} outside [0.0, 1.0]"
                )
        return self


class DistillationRecord(BaseModel):
    """
    A filtered, high-quality (prompt, response) pair ready for student training.
    Persisted with optimistic locking to prevent concurrent write corruption.
    """
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str
    prompt_content: str
    chosen_response: str
    composite_score: float = Field(ge=0.0, le=1.0)
    teacher_model: str
    judge_model: str
    n_candidates_generated: int
    percentile_rank: float = Field(
        ge=0.0,
        le=1.0,
        description="Score rank within the batch (1.0 = top score)"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Optimistic locking: version column prevents lost-update races
    # (anti-pitfall: no blind UPSERTs)
    version: int = Field(default=0, description="Optimistic lock version")

    @computed_field
    @property
    def record_hash(self) -> str:
        """Deterministic content hash for idempotent ingestion."""
        payload = f"{self.prompt_id}:{self.chosen_response}"
        return hashlib.sha256(payload.encode()).hexdigest()


class DistillationDataset(BaseModel):
    """
    The output artifact of a distillation run.
    Uses delta logging semantics: records are appended, never overwritten.
    """
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    teacher_model: str
    judge_model: str
    student_model: str
    quality_threshold: float
    records: list[DistillationRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: DistillationStatus = DistillationStatus.PENDING

    @computed_field
    @property
    def acceptance_rate(self) -> float:
        """Fraction of generated candidates that passed the quality gate."""
        if not self.records:
            return 0.0
        total_candidates = sum(r.n_candidates_generated for r in self.records)
        return len(self.records) / total_candidates if total_candidates > 0 else 0.0

    @computed_field
    @property
    def mean_quality_score(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.composite_score for r in self.records) / len(self.records)

    @computed_field
    @property
    def corpus_sha256(self) -> str:
        """Deterministic fingerprint of the full dataset for provenance."""
        content = "|".join(sorted(r.record_hash for r in self.records))
        return hashlib.sha256(content.encode()).hexdigest()


# ─────────────────────────────────────────────
# PIPELINE CONFIGURATION
# ─────────────────────────────────────────────

class RejectionSamplingConfig(BaseModel):
    """
    Configuration for a single distillation run.
    Encodes the teacher → judge → student topology.
    """
    teacher_model: str = Field(
        description="High-capability generator (e.g., claude-opus-4-5, gpt-4o)"
    )
    judge_model: str = Field(
        description="Rubric evaluator, equal or higher capability than teacher"
    )
    student_model: str = Field(
        description="Distillation target (e.g., claude-haiku-3-5, llama-3-8b)"
    )
    n_candidates_per_prompt: int = Field(
        default=10,
        ge=2,
        le=100,
        description="N in rejection sampling. Returns scale logarithmically."
    )
    quality_percentile_threshold: float = Field(
        default=0.99,
        ge=0.5,
        le=1.0,
        description="Minimum composite score percentile for dataset inclusion"
    )
    absolute_score_floor: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Hard minimum composite score regardless of percentile rank"
    )
    max_concurrent_requests: int = Field(
        default=8,
        description="Semaphore bound for parallel teacher/judge calls"
    )
    temperature: float = Field(
        default=0.9,
        ge=0.0,
        le=2.0,
        description="Teacher sampling temperature; diversity requires T > 0.7"
    )
    dimension_weights: dict[JudgeDimension, float] = Field(
        default_factory=lambda: {
            JudgeDimension.CORRECTNESS:  0.40,
            JudgeDimension.COHERENCE:    0.25,
            JudgeDimension.COMPLETENESS: 0.20,
            JudgeDimension.FORMAT:       0.10,
            JudgeDimension.SAFETY:       0.05,
        }
    )

    @model_validator(mode="after")
    def validate_weight_sum(self) -> "RejectionSamplingConfig":
        total = sum(self.dimension_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"dimension_weights must sum to 1.0; got {total:.6f}"
            )
        return self


class RejectionSamplingPipeline(BaseModel):
    """
    Named pattern: Synthetic Data Pipeline.
    Orchestrates: Teacher generates → Judge filters → Top K% enters training set.

    DISTINCTION from Section 10.1 (Inference-Time RS):
    - 10.1: Per-query, real-time, returns best response to user, no weight update.
    - 10.6: Per-dataset, offline, writes to training corpus, triggers student fine-tuning.
    """
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: RejectionSamplingConfig
    dataset_version: str = Field(
        description="Semantic version for the output training artifact (e.g., v1.2.0)"
    )

    class Config:
        arbitrary_types_allowed = True


# ─────────────────────────────────────────────
# CLIENT PROTOCOLS
# ─────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    """Minimal protocol for any LLM backend (Anthropic, OpenAI, local Ollama)."""
    model_id: str

    async def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)."""
        ...


@runtime_checkable
class JudgeClient(Protocol):
    """Protocol for structured rubric scoring."""
    model_id: str

    async def score(
        self,
        prompt: str,
        response: str,
        dimensions: list[JudgeDimension],
    ) -> JudgeScore:
        """Returns a structured JudgeScore with per-dimension scores."""
        ...


# ─────────────────────────────────────────────
# PIPELINE LOGIC
# ─────────────────────────────────────────────

async def _generate_candidates(
    prompt: TrainingPrompt,
    teacher: LLMClient,
    config: RejectionSamplingConfig,
    semaphore: asyncio.Semaphore,
) -> list[GenerationCandidate]:
    """
    Generate N diverse candidates for a single training prompt.
    Concurrent under semaphore to bound API cost spikes.
    """
    async def _single_generation() -> GenerationCandidate:
        async with semaphore:
            response_text, input_tokens, output_tokens = await teacher.generate(
                prompt=prompt.content,
                temperature=config.temperature,
                max_tokens=2048,
            )
            return GenerationCandidate(
                prompt_id=prompt.prompt_id,
                response=response_text,
                teacher_model=teacher.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    tasks = [_single_generation() for _ in range(config.n_candidates_per_prompt)]
    return await asyncio.gather(*tasks)


async def _score_candidates(
    prompt: TrainingPrompt,
    candidates: list[GenerationCandidate],
    judge: JudgeClient,
    config: RejectionSamplingConfig,
    semaphore: asyncio.Semaphore,
) -> list[tuple[GenerationCandidate, JudgeScore]]:
    """
    Score all candidates under the rubric.
    Returns (candidate, score) pairs sorted by composite_score descending.
    """
    async def _single_score(candidate: GenerationCandidate) -> JudgeScore:
        async with semaphore:
            return await judge.score(
                prompt=prompt.content,
                response=candidate.response,
                dimensions=list(config.dimension_weights.keys()),
            )

    tasks = [_single_score(c) for c in candidates]
    scores = await asyncio.gather(*tasks)
    paired = list(zip(candidates, scores))
    return sorted(paired, key=lambda x: x[1].composite_score, reverse=True)


def _apply_quality_gate(
    scored_pairs: list[tuple[GenerationCandidate, JudgeScore]],
    prompt: TrainingPrompt,
    config: RejectionSamplingConfig,
) -> DistillationRecord | None:
    """
    Rejection sampling gate: accept the top candidate iff it exceeds
    both the percentile threshold and the absolute score floor.

    The percentile threshold is computed within the current batch of N
    candidates, not globally—this is a per-prompt relative filter.
    Absolute floor prevents a high percentile on a universally poor batch
    from contaminating the training set.
    """
    if not scored_pairs:  # judge or teacher returned nothing
        return None
    scored_pairs.sort(key=lambda p: p[1].composite_score, reverse=True)

    all_scores = [score.composite_score for _, score in scored_pairs]
    top_candidate, top_score = scored_pairs[0]

    # Compute percentile rank within this prompt's candidates
    n_below = sum(1 for s in all_scores if s < top_score.composite_score)
    percentile_rank = n_below / len(all_scores)

    # Apply dual gate: percentile AND absolute floor
    if (
        percentile_rank < config.quality_percentile_threshold
        or top_score.composite_score < config.absolute_score_floor
    ):
        return None  # Rejected: entire batch below quality threshold

    return DistillationRecord(
        prompt_id=prompt.prompt_id,
        prompt_content=prompt.content,
        chosen_response=top_candidate.response,
        composite_score=top_score.composite_score,
        teacher_model=top_candidate.teacher_model,
        judge_model=top_score.judge_model,
        n_candidates_generated=len(scored_pairs),
        percentile_rank=percentile_rank,
    )


async def distill(
    pipeline: RejectionSamplingPipeline,
    prompts: Sequence[TrainingPrompt],
    teacher: LLMClient,
    judge: JudgeClient,
) -> DistillationDataset:
    """
    Execute the Synthetic Data Pipeline.

    Architecture:
        For each prompt in the seed dataset:
        1. Teacher generates N candidates (parallel).
        2. Judge scores all N candidates (parallel).
        3. Quality gate accepts top candidate if score >= threshold.
        4. Accepted records are appended to the training dataset (delta log).

    Delta Logging:
        Records are appended, never overwritten, to avoid O(N^2) write
        amplification. The dataset is a persistent, append-only log;
        the corpus_sha256 fingerprints the current materialized view.

    Returns:
        DistillationDataset with acceptance_rate and mean_quality_score
        for downstream training pipeline ingestion.
    """
    dataset = DistillationDataset(
        teacher_model=teacher.model_id,
        judge_model=judge.model_id,
        student_model=pipeline.config.student_model,
        quality_threshold=pipeline.config.quality_percentile_threshold,
        status=DistillationStatus.RUNNING,
    )

    semaphore = asyncio.Semaphore(pipeline.config.max_concurrent_requests)

    for prompt in prompts:
        # Phase 1: Generation
        candidates = await _generate_candidates(
            prompt=prompt,
            teacher=teacher,
            config=pipeline.config,
            semaphore=semaphore,
        )

        # Phase 2: Scoring
        scored_pairs = await _score_candidates(
            prompt=prompt,
            candidates=candidates,
            judge=judge,
            config=pipeline.config,
            semaphore=semaphore,
        )

        # Phase 3: Quality Gate (Rejection Sampling)
        record = _apply_quality_gate(
            scored_pairs=scored_pairs,
            prompt=prompt,
            config=pipeline.config,
        )

        # Phase 4: Delta Append (only accepted records are persisted)
        if record is not None:
            dataset.records.append(record)

    dataset.status = DistillationStatus.COMPLETED
    return dataset


# ─────────────────────────────────────────────
# EVAL: TEACHER VS. STUDENT COMPARISON
# ─────────────────────────────────────────────

class DistillationEvalResult(BaseModel):
    """
    Post-training evaluation: does the student approach teacher capability?
    Used to determine whether to promote the student to production traffic.
    """
    eval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str
    teacher_model: str
    student_model: str
    teacher_mean_score: float
    student_mean_score: float
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def capability_retention_pct(self) -> float:
        """
        Fraction of teacher capability transferred to student.
        100% = student matches teacher; <80% typically warrants re-distillation.
        """
        if self.teacher_mean_score == 0.0:
            return 0.0
        return round(
            (self.student_mean_score / self.teacher_mean_score) * 100.0, 2
        )

    @computed_field
    @property
    def is_production_ready(self) -> bool:
        """
        Promote to production if student retains ≥ 85% of teacher capability.
        Below this threshold, the weak-to-strong capability gap is too large.
        """
        return self.capability_retention_pct >= 85.0
# --8<-- [end:lst-16.3]
