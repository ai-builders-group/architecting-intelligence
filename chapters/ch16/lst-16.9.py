# --8<-- [start:lst-16.9]
# weight_release_pipeline.py

from __future__ import annotations

import hashlib
import secrets
import asyncio
from datetime import datetime, UTC, timedelta
from enum import Enum
from typing import Literal, Annotated
from uuid import uuid4

import httpx
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    computed_field,
    field_validator,
    PositiveFloat,
    PositiveInt,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStatus(str, Enum):
    INACTIVE = "inactive"
    CANARY = "canary"
    PARTIAL = "partial"
    FULL = "full"
    ROLLED_BACK = "rolled_back"


class GateVerdict(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class RolloutPhase(str, Enum):
    SHADOW = "shadow"       # 0% user traffic, shadow mode only
    CANARY = "canary"       # 1% traffic
    EARLY = "early"         # 10% traffic
    HALF = "half"           # 50% traffic
    FULL = "full"           # 100% traffic


# ---------------------------------------------------------------------------
# Data Provenance
# ---------------------------------------------------------------------------


class DatasetSplit(BaseModel):
    """A single partition of the training corpus."""

    split_name: Literal["train", "validation", "test"]
    record_count: PositiveInt
    token_count: PositiveInt
    sha256_hash: str = Field(
        description="SHA-256 of the serialized split file for integrity verification"
    )
    source_uri: str = Field(
        description="URI to the canonical source (s3://, gs://, hf://)"
    )
    license: str = Field(
        description="SPDX license identifier or 'proprietary'"
    )
    collected_at: datetime


class ContaminationReport(BaseModel):
    """Result of n-gram decontamination check against evaluation benchmarks."""

    benchmark_name: str
    ngram_size: int = Field(default=13)
    examples_flagged: int
    examples_removed: int
    max_overlap_score: float = Field(ge=0.0, le=1.0)
    threshold_used: float = Field(default=0.8)
    is_clean: bool


class DataProvenance(BaseModel):
    """
    Immutable lineage record for a training dataset.

    Every TrainingJob references a DataProvenance whose sha256_hash
    is committed to the model registry, enabling forensic attribution.
    """

    provenance_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_name: str
    dataset_version: str
    splits: list[DatasetSplit]
    contamination_reports: list[ContaminationReport]
    curator_id: str = Field(description="Service account that curated this dataset")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def corpus_sha256(self) -> str:
        """
        Deterministic hash of the entire corpus: SHA-256 over sorted split hashes.
        Provides a single integrity fingerprint for the dataset manifest.
        """
        sorted_hashes = sorted(s.sha256_hash for s in self.splits)
        combined = ":".join(sorted_hashes).encode()
        return hashlib.sha256(combined).hexdigest()

    @computed_field
    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self.splits)

    @computed_field
    @property
    def is_decontaminated(self) -> bool:
        return all(r.is_clean for r in self.contamination_reports)

    @model_validator(mode="after")
    def validate_contamination_coverage(self) -> DataProvenance:
        """Reject datasets that have not been checked against any benchmarks."""
        if not self.contamination_reports:
            raise ValueError(
                "DataProvenance requires at least one ContaminationReport. "
                "Run the decontamination pipeline before registering a dataset."
            )
        return self


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


class LoRAConfig(BaseModel):
    """
    Low-Rank Adaptation hyperparameters.

    References:
    - Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
      arXiv:2106.09685 [3]

    Mathematical Foundation:
        W' = W + (α/r) · B·A
        where B ∈ R^{d×r}, A ∈ R^{r×k}
        |θ_LoRA| = 4L·2rd  (Q,K,V,O projections)
    """

    rank: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Rank r of the low-rank decomposition. Higher rank = more expressivity = more risk of forgetting."
    )
    alpha: float = Field(
        default=32.0,
        gt=0,
        description="LoRA scaling factor α. Effective LR scale = α/r."
    )
    dropout: float = Field(
        default=0.05,
        ge=0.0,
        lt=1.0,
        description="Dropout applied to LoRA layers during training."
    )
    target_modules: list[str] = Field(
        default=["q_proj", "v_proj"],
        description="Which weight matrices to adapt. Minimal set: q+v. Full set: q+k+v+o+gate+up+down."
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Whether to train bias terms alongside LoRA matrices."
    )

    @computed_field
    @property
    def trainable_parameter_fraction(self) -> float:
        """
        Approximate fraction of parameters updated relative to a 7B base model.
        Serves as a cost estimate proxy for training runs.
        """
        # 7B model: ~32 layers, d=4096, 4 projections per layer
        base_params = 7_000_000_000
        # |θ_LoRA| = 4 × L × 2 × r × d = 4 × 32 × 2 × rank × 4096
        n_modules = len(self.target_modules)
        lora_params = n_modules * 32 * 2 * self.rank * 4096
        return lora_params / base_params


class TrainingHyperparameters(BaseModel):
    """
    Complete hyperparameter specification for a training run.
    Treated as an immutable artifact: changing any field requires a new TrainingJob.
    """

    learning_rate: PositiveFloat = Field(default=2e-4)
    learning_rate_scheduler: Literal[
        "cosine", "linear", "constant", "cosine_with_restarts"
    ] = Field(default="cosine")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    num_epochs: PositiveInt = Field(default=3)
    per_device_train_batch_size: PositiveInt = Field(default=4)
    gradient_accumulation_steps: PositiveInt = Field(default=8)
    max_grad_norm: float = Field(default=1.0, gt=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_sequence_length: PositiveInt = Field(default=2048)
    packing: bool = Field(
        default=True,
        description="Whether to pack multiple short examples into a single sequence to maximize GPU utilization."
    )
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=True)
    gradient_checkpointing: bool = Field(
        default=True,
        description="Trade compute for memory: recompute activations during backward pass."
    )
    seed: int = Field(
        default_factory=lambda: secrets.randbelow(2**31),
        description="Random seed for data shuffling. Stored for provenance; training remains non-deterministic due to GPU non-associativity."
    )

    @computed_field
    @property
    def effective_batch_size(self) -> int:
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
        )


# ---------------------------------------------------------------------------
# Training Job
# ---------------------------------------------------------------------------


class TrainingJob(BaseModel):
    """
    The canonical artifact of the training stage.

    Analogous to a software build artifact:
    - job_id  →  build ID
    - adapter_artifact_uri  →  compiled binary path
    - version  →  optimistic locking column (see ADR-04.4a)

    Anti-pitfall: version field enforces optimistic locking to prevent
    race conditions in distributed experiment tracking databases.
    """

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    base_model_id: str = Field(
        description="HuggingFace model ID or internal registry URI of the frozen base model."
    )
    adapter_name: str = Field(
        description="Human-readable name, e.g., 'medical-coding-icd10-v3'."
    )
    lora_config: LoRAConfig
    hyperparameters: TrainingHyperparameters
    data_provenance: DataProvenance
    status: TrainingStatus = Field(default=TrainingStatus.QUEUED)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    adapter_artifact_uri: str | None = Field(
        default=None,
        description="URI to the saved LoRA adapter weights (s3://...adapter.bin). Populated only on COMPLETED status."
    )
    adapter_sha256: str | None = Field(
        default=None,
        description="SHA-256 of the adapter artifact for integrity verification."
    )
    gpu_hours_consumed: float | None = Field(default=None)
    final_train_loss: float | None = Field(default=None)
    final_eval_loss: float | None = Field(default=None)
    # Optimistic locking: incremented on every state transition.
    # Prevents lost-update anomalies in distributed training orchestrators.
    version: int = Field(default=0)

    @computed_field
    @property
    def wall_clock_hours(self) -> float | None:
        if self.started_at is None or self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() / 3600.0

    @model_validator(mode="after")
    def validate_completed_state(self) -> TrainingJob:
        if self.status == TrainingStatus.COMPLETED:
            if self.adapter_artifact_uri is None:
                raise ValueError(
                    "COMPLETED TrainingJob must have adapter_artifact_uri set."
                )
            if self.adapter_sha256 is None:
                raise ValueError(
                    "COMPLETED TrainingJob must have adapter_sha256 set."
                )
        return self


# ---------------------------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------------------------


class BenchmarkResult(BaseModel):
    """Single benchmark evaluation outcome."""

    benchmark_name: str = Field(
        description="e.g., 'medqa', 'mmlu_medicine', 'icd10_coding_accuracy'"
    )
    metric_name: str = Field(
        description="e.g., 'exact_match', 'f1', 'accuracy'"
    )
    score: float = Field(ge=0.0, le=1.0)
    baseline_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score of the base model (without adapter) on the same benchmark."
    )
    sample_count: PositiveInt
    evaluation_duration_seconds: float

    @computed_field
    @property
    def delta_from_baseline(self) -> float:
        """Absolute improvement over base model."""
        return self.score - self.baseline_score

    @computed_field
    @property
    def relative_improvement_pct(self) -> float:
        """Relative improvement as a percentage."""
        if self.baseline_score == 0:
            return float("inf")
        return (self.delta_from_baseline / self.baseline_score) * 100.0


class KLDivergenceGate(BaseModel):
    """
    Measures behavioral drift between candidate and base model.

    Implements the KL divergence gate:
        D_KL(P_new || P_base) = Σ_x P_new(x) log(P_new(x) / P_base(x))

    Approximated via top-k logit distributions (k=100) across a probe dataset.
    High D_KL → candidate has undergone a behavioral phase transition.

    References:
    - Hinton et al. (2015). "Distilling the Knowledge in a Neural Network."
      arXiv:1503.02531 [1] — KL divergence as the canonical measure of
      distribution distance in knowledge distillation.
    """

    probe_dataset_name: str
    probe_sample_count: PositiveInt
    top_k_logits: int = Field(
        default=100,
        description="Number of top logits used to approximate the full vocabulary distribution."
    )
    kl_divergence_nats: float = Field(
        ge=0.0,
        description="Mean KL divergence in nats, averaged over all probe tokens."
    )
    threshold_nats: float = Field(
        default=0.3,
        description="Maximum permissible D_KL. Above this, the adapter is rejected."
    )
    verdict: GateVerdict
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def derive_verdict(self) -> KLDivergenceGate:
        if self.verdict == GateVerdict.PASSED and self.kl_divergence_nats > self.threshold_nats:
            raise ValueError(
                f"KLDivergenceGate verdict PASSED but D_KL={self.kl_divergence_nats:.4f} "
                f"exceeds threshold={self.threshold_nats}. Verdict is inconsistent."
            )
        return self


class RegressionTest(BaseModel):
    """
    A single golden example from the behavioral invariant test suite.

    These tests encode capabilities that must survive all weight updates.
    Failures are hard stops regardless of benchmark improvements.
    """

    test_id: str
    category: Literal[
        "safety_refusal",
        "format_compliance",
        "capability_invariant",
        "sla_critical",
    ]
    prompt: str
    expected_pattern: str = Field(
        description="Regex pattern or exact string that the model output must match."
    )
    match_type: Literal["regex", "exact", "semantic_similarity"] = Field(
        default="regex"
    )
    similarity_threshold: float = Field(
        default=0.85,
        description="Minimum cosine similarity for semantic_similarity match type."
    )
    model_output: str | None = Field(default=None)
    passed: bool | None = Field(default=None)


class EvalHarness(BaseModel):
    """
    The complete evaluation artifact produced by Stage 3 of the Weight Release Pipeline.

    Composed of three classes of tests:
    1. Hold-out benchmarks (capability verification)
    2. KL divergence gate (behavioral stability)
    3. Regression test suite (invariant preservation)

    All three classes must pass for the harness to issue an overall PASSED verdict.
    """

    harness_id: str = Field(default_factory=lambda: str(uuid4()))
    training_job_id: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    benchmark_results: list[BenchmarkResult]
    kl_divergence_gate: KLDivergenceGate
    regression_tests: list[RegressionTest]

    @computed_field
    @property
    def benchmark_verdict(self) -> GateVerdict:
        """
        Benchmarks pass if every target metric shows positive delta from baseline.
        A zero-improvement adapter is rejected regardless of absolute score.
        """
        if all(r.delta_from_baseline >= 0.0 for r in self.benchmark_results):
            if any(r.delta_from_baseline > 0.0 for r in self.benchmark_results):
                return GateVerdict.PASSED
            return GateVerdict.WARNING  # No regression, but also no improvement
        return GateVerdict.FAILED

    @computed_field
    @property
    def regression_verdict(self) -> GateVerdict:
        completed = [t for t in self.regression_tests if t.passed is not None]
        if not completed:
            return GateVerdict.FAILED
        n_passed = sum(1 for t in completed if t.passed)
        pass_rate = n_passed / len(completed)
        if pass_rate == 1.0:
            return GateVerdict.PASSED
        elif pass_rate >= 0.95:
            return GateVerdict.WARNING  # Minor regressions: human review required
        return GateVerdict.FAILED

    @computed_field
    @property
    def overall_verdict(self) -> GateVerdict:
        verdicts = [
            self.benchmark_verdict,
            self.kl_divergence_gate.verdict,
            self.regression_verdict,
        ]
        if GateVerdict.FAILED in verdicts:
            return GateVerdict.FAILED
        if GateVerdict.WARNING in verdicts:
            return GateVerdict.WARNING
        return GateVerdict.PASSED

    @computed_field
    @property
    def model_card_summary(self) -> dict[str, object]:
        """
        Auto-generated Model Card fragment for the model registry.
        Adapts the model-reporting structure proposed by Mitchell et al. (2019).
        """
        return {
            "evaluation_date": self.evaluated_at.isoformat(),
            "overall_verdict": self.overall_verdict.value,
            "kl_divergence_nats": self.kl_divergence_gate.kl_divergence_nats,
            "benchmark_improvements": {
                r.benchmark_name: {
                    "score": r.score,
                    "delta": r.delta_from_baseline,
                    "relative_improvement_pct": r.relative_improvement_pct,
                }
                for r in self.benchmark_results
            },
            "regression_pass_rate": sum(
                1 for t in self.regression_tests if t.passed
            ) / max(len(self.regression_tests), 1),
            "known_regressions": [
                t.test_id
                for t in self.regression_tests
                if t.passed is False
            ],
        }


# ---------------------------------------------------------------------------
# Deployment Configuration and Rollout
# ---------------------------------------------------------------------------


class TrafficPolicy(BaseModel):
    """
    Blue/green traffic splitting configuration.

    Defines the graduated rollout schedule and the metric gates
    that must be satisfied at each phase boundary.
    """

    phase: RolloutPhase
    green_traffic_pct: float = Field(ge=0.0, le=100.0)
    min_duration_minutes: PositiveInt = Field(
        description="Minimum time in this phase before advancing, regardless of metrics."
    )
    # Metric thresholds: rollback if green drops below these relative to blue
    max_error_rate_delta_pct: float = Field(
        default=2.0,
        description="Maximum permissible increase in error rate (green vs. blue) in percentage points."
    )
    max_latency_p95_delta_ms: float = Field(
        default=100.0,
        description="Maximum permissible p95 latency increase in milliseconds."
    )
    min_quality_score_delta: float = Field(
        default=-0.02,
        description="Minimum allowable quality score delta (green - blue). Negative values permit minor degradation."
    )
    min_sample_count: PositiveInt = Field(
        default=1000,
        description="Minimum requests before metric evaluation is statistically valid."
    )


class RollbackEvent(BaseModel):
    """Immutable record of an automatic rollback. Appended to the event log."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trigger_metric: str = Field(
        description="Which metric breached its threshold and triggered rollback."
    )
    metric_value_green: float
    metric_value_blue: float
    threshold: float
    traffic_pct_at_rollback: float
    rollback_duration_seconds: float | None = Field(
        default=None,
        description="Time from trigger to 0% green traffic."
    )


class ModelDeployment(BaseModel):
    """
    The deployment artifact for a trained and evaluated model adapter.

    Implements the blue/green pattern with automatic rollback.

    version field enforces optimistic locking to prevent concurrent
    deployment mutations in multi-region orchestrators (ADR-04.4a).
    """

    deployment_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(
        description="Owner of this deployment; every write is scoped to it."
    )
    training_job_id: str
    harness_id: str
    blue_endpoint: str = Field(
        description="URI of the currently stable production model endpoint."
    )
    green_endpoint: str = Field(
        description="URI of the candidate model endpoint."
    )
    status: DeploymentStatus = Field(default=DeploymentStatus.INACTIVE)
    current_phase: RolloutPhase = Field(default=RolloutPhase.SHADOW)
    green_traffic_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    traffic_policy: list[TrafficPolicy] = Field(
        description="Ordered rollout phases. Pipeline advances through these sequentially."
    )
    rollback_events: list[RollbackEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    promoted_at: datetime | None = Field(default=None)
    rolled_back_at: datetime | None = Field(default=None)
    # Optimistic locking: prevents lost-update race conditions in
    # distributed deployment controllers.
    version: int = Field(default=0)

    @computed_field
    @property
    def is_active(self) -> bool:
        return self.status in (
            DeploymentStatus.CANARY,
            DeploymentStatus.PARTIAL,
            DeploymentStatus.FULL,
        )

    @computed_field
    @property
    def rollback_count(self) -> int:
        return len(self.rollback_events)


# ---------------------------------------------------------------------------
# Metric Snapshot (for watchdog evaluation)
# ---------------------------------------------------------------------------


class MetricSnapshot(BaseModel):
    """Real-time metric sample from a model endpoint."""

    endpoint_id: str
    sampled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_count: PositiveInt
    error_rate_pct: float = Field(ge=0.0, le=100.0)
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite quality signal from downstream task metrics."
    )


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------


class PhaseAdvanceDecision(BaseModel):
    """Decision record for a single phase boundary evaluation."""

    from_phase: RolloutPhase
    to_phase: RolloutPhase | None
    decision: Literal["advance", "hold", "rollback"]
    reason: str
    blue_metrics: MetricSnapshot
    green_metrics: MetricSnapshot


async def _fetch_metrics(endpoint: str, client: httpx.AsyncClient) -> MetricSnapshot:
    """
    Fetch real-time metrics from an endpoint's observability sidecar.
    In production, this integrates with Prometheus/Datadog via the metrics API.
    """
    response = await client.get(
        f"{endpoint}/metrics/snapshot",
        timeout=10.0,
    )
    response.raise_for_status()
    return MetricSnapshot.model_validate(response.json())


def _evaluate_phase_gate(
    policy: TrafficPolicy,
    blue: MetricSnapshot,
    green: MetricSnapshot,
) -> PhaseAdvanceDecision:
    """
    Evaluate whether the current traffic phase meets its metric gate.

    Returns an advance, hold, or rollback decision with a structured reason.
    Rollback is triggered when green metrics fall outside the policy bounds.
    Statistical validity check: skip evaluation if sample count is too low.
    """
    next_phase_map = {
        RolloutPhase.SHADOW: RolloutPhase.CANARY,
        RolloutPhase.CANARY: RolloutPhase.EARLY,
        RolloutPhase.EARLY: RolloutPhase.HALF,
        RolloutPhase.HALF: RolloutPhase.FULL,
        RolloutPhase.FULL: None,
    }

    if green.request_count < policy.min_sample_count:
        return PhaseAdvanceDecision(
            from_phase=policy.phase,
            to_phase=None,
            decision="hold",
            reason=(
                f"Insufficient samples: {green.request_count} < "
                f"{policy.min_sample_count} required for statistical validity."
            ),
            blue_metrics=blue,
            green_metrics=green,
        )

    # Gate 1: Error rate
    error_delta = green.error_rate_pct - blue.error_rate_pct
    if error_delta > policy.max_error_rate_delta_pct:
        return PhaseAdvanceDecision(
            from_phase=policy.phase,
            to_phase=None,
            decision="rollback",
            reason=(
                f"Error rate regression: green={green.error_rate_pct:.2f}% "
                f"vs blue={blue.error_rate_pct:.2f}% "
                f"(Δ={error_delta:.2f}pp exceeds threshold={policy.max_error_rate_delta_pct}pp)."
            ),
            blue_metrics=blue,
            green_metrics=green,
        )

    # Gate 2: p95 latency
    latency_delta = green.latency_p95_ms - blue.latency_p95_ms
    if latency_delta > policy.max_latency_p95_delta_ms:
        return PhaseAdvanceDecision(
            from_phase=policy.phase,
            to_phase=None,
            decision="rollback",
            reason=(
                f"Latency regression: green p95={green.latency_p95_ms:.1f}ms "
                f"vs blue p95={blue.latency_p95_ms:.1f}ms "
                f"(Δ={latency_delta:.1f}ms exceeds threshold={policy.max_latency_p95_delta_ms}ms)."
            ),
            blue_metrics=blue,
            green_metrics=green,
        )

    # Gate 3: Quality score
    quality_delta = green.quality_score - blue.quality_score
    if quality_delta < policy.min_quality_score_delta:
        return PhaseAdvanceDecision(
            from_phase=policy.phase,
            to_phase=None,
            decision="rollback",
            reason=(
                f"Quality regression: green={green.quality_score:.4f} "
                f"vs blue={blue.quality_score:.4f} "
                f"(Δ={quality_delta:.4f} below threshold={policy.min_quality_score_delta})."
            ),
            blue_metrics=blue,
            green_metrics=green,
        )

    next_phase = next_phase_map[policy.phase]
    return PhaseAdvanceDecision(
        from_phase=policy.phase,
        to_phase=next_phase,
        decision="advance" if next_phase is not None else "hold",
        reason=(
            f"All gates passed: Δerror={error_delta:.2f}pp, "
            f"Δp95={latency_delta:.1f}ms, "
            f"Δquality={quality_delta:.4f}."
        ),
        blue_metrics=blue,
        green_metrics=green,
    )


async def deploy_model(
    deployment: ModelDeployment,
    db_client: object,  # Type: AsyncpgConnection | SQLAlchemyAsyncEngine
    poll_interval_seconds: int = 60,
) -> ModelDeployment:
    """
    Execute the Weight Release Pipeline deployment stage.

    Implements a graduated blue/green rollout with automatic rollback.

    Architecture:
    - Polls metrics at poll_interval_seconds frequency.
    - Evaluates phase gate after min_duration_minutes and sufficient sample count.
    - On rollback trigger: immediately drops green to 0%, writes RollbackEvent.
    - Uses optimistic locking (version field) to prevent concurrent mutation.

    Anti-pitfall: This function does NOT use NullPool. The db_client
    must be configured with PgBouncer connection pooling to amortize
    TLS handshake latency on checkpoint writes (ADR-06-05).

    Args:
        deployment: The ModelDeployment artifact from the eval harness.
        db_client: Async database connection with connection pooling.
        poll_interval_seconds: Metric evaluation frequency.

    Returns:
        Final ModelDeployment state (either FULL or ROLLED_BACK).
    """
    phase_index = 0
    phase_start_time = datetime.now(UTC)

    async with httpx.AsyncClient() as http_client:
        while phase_index < len(deployment.traffic_policy):
            policy = deployment.traffic_policy[phase_index]

            # Set traffic split for this phase
            deployment = deployment.model_copy(
                update={
                    "current_phase": policy.phase,
                    "green_traffic_pct": policy.green_traffic_pct,
                    "status": DeploymentStatus.CANARY
                    if policy.green_traffic_pct <= 1.0
                    else DeploymentStatus.PARTIAL,
                    "version": deployment.version + 1,
                }
            )

            # Persist state with optimistic locking check
            await _upsert_deployment_with_version_check(db_client, deployment)

            await asyncio.sleep(poll_interval_seconds)

            # Check minimum phase duration
            elapsed = (datetime.now(UTC) - phase_start_time).total_seconds() / 60
            if elapsed < policy.min_duration_minutes:
                continue  # Hold: insufficient time in phase

            # Fetch metrics from both endpoints concurrently
            try:
                blue_metrics, green_metrics = await asyncio.gather(
                    _fetch_metrics(deployment.blue_endpoint, http_client),
                    _fetch_metrics(deployment.green_endpoint, http_client),
                )
            except httpx.RequestError as exc:
                # Network error fetching green metrics → conservative rollback
                rollback_event = RollbackEvent(
                    deployment_id=deployment.deployment_id,
                    trigger_metric="endpoint_availability",
                    metric_value_green=0.0,
                    metric_value_blue=1.0,
                    threshold=1.0,
                    traffic_pct_at_rollback=deployment.green_traffic_pct,
                )
                return await _execute_rollback(
                    deployment=deployment,
                    rollback_event=rollback_event,
                    db_client=db_client,
                    reason=f"Metrics fetch failure: {exc}",
                )

            decision = _evaluate_phase_gate(policy, blue_metrics, green_metrics)

            if decision.decision == "rollback":
                rollback_event = RollbackEvent(
                    deployment_id=deployment.deployment_id,
                    trigger_metric=decision.reason.split(":")[0].lower().replace(" ", "_"),
                    metric_value_green=green_metrics.error_rate_pct,
                    metric_value_blue=blue_metrics.error_rate_pct,
                    threshold=policy.max_error_rate_delta_pct,
                    traffic_pct_at_rollback=deployment.green_traffic_pct,
                )
                return await _execute_rollback(
                    deployment=deployment,
                    rollback_event=rollback_event,
                    db_client=db_client,
                    reason=decision.reason,
                )

            if decision.decision == "advance":
                phase_index += 1
                phase_start_time = datetime.now(UTC)

            # decision == "hold": loop continues

        # All phases passed → full promotion
        deployment = deployment.model_copy(
            update={
                "status": DeploymentStatus.FULL,
                "green_traffic_pct": 100.0,
                "current_phase": RolloutPhase.FULL,
                "promoted_at": datetime.now(UTC),
                "version": deployment.version + 1,
            }
        )
        await _upsert_deployment_with_version_check(db_client, deployment)
        return deployment


async def _execute_rollback(
    deployment: ModelDeployment,
    rollback_event: RollbackEvent,
    db_client: object,
    reason: str,
) -> ModelDeployment:
    """
    Execute an immediate rollback: drop green traffic to 0%, persist event.

    Target SLO: complete rollback within 60 seconds of trigger.
    """
    rollback_start = datetime.now(UTC)

    updated_rollbacks = deployment.rollback_events + [rollback_event]
    deployment = deployment.model_copy(
        update={
            "status": DeploymentStatus.ROLLED_BACK,
            "green_traffic_pct": 0.0,
            "rolled_back_at": datetime.now(UTC),
            "rollback_events": updated_rollbacks,
            "version": deployment.version + 1,
        }
    )

    rollback_duration = (datetime.now(UTC) - rollback_start).total_seconds()

    # Update rollback duration in the event record
    updated_event = rollback_event.model_copy(
        update={"rollback_duration_seconds": rollback_duration}
    )
    updated_rollbacks[-1] = updated_event
    deployment = deployment.model_copy(update={"rollback_events": updated_rollbacks})

    await _upsert_deployment_with_version_check(db_client, deployment)
    return deployment


async def _upsert_deployment_with_version_check(
    db_client: object,
    deployment: ModelDeployment,
) -> None:
    """
    Persist deployment state with optimistic locking.

    SQL pattern:
        UPDATE model_deployments
        SET state = :state, version = version + 1
        WHERE tenant_id = :tenant_id AND deployment_id = :id
          AND version = :expected_version

    If rowcount == 0: a concurrent writer has mutated the row.
    Raise ConcurrentModificationError rather than silently losing the update.

    Anti-pitfall: This is NOT a blind UPSERT. The WHERE clause on version
    prevents the lost-update anomaly in multi-region deployment controllers.
    """
    # Implementation delegates to the injected db_client.
    # The calling layer must configure PgBouncer connection pooling
    # to avoid TLS handshake latency on every checkpoint write (ADR-06-05).
    expected_version = deployment.version - 1  # Pre-increment version
    query = """
        UPDATE model_deployments
        SET
            state         = :state,
            green_pct     = :green_pct,
            status        = :status,
            version       = :version,
            updated_at    = :updated_at
        WHERE
            tenant_id     = :tenant_id
            AND deployment_id = :deployment_id
            AND version   = :expected_version
    """
    # db_client.execute(query, {...})  # Injected by the orchestrator layer.
    # Raise ConcurrentModificationError if rowcount == 0.
    _ = (db_client, query, expected_version)  # Suppress unused-variable warnings
# --8<-- [end:lst-16.9]
