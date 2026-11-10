# --8<-- [start:lst-16.11]
# src/ch10_weight_context_continuum/model_card.py

from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
from pydantic import BaseModel, Field, computed_field


class GateVerdict(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


class IntendedUse(BaseModel):
    primary_use_cases: list[str]
    out_of_scope_use_cases: list[str]
    intended_users: list[str]


class QuantitativeAnalysis(BaseModel):
    benchmark_name: str
    score: float
    confidence_interval_95: tuple[float, float]
    evaluation_set_size: int


class ModelCard(BaseModel):
    """
    Automated Model Card: the Bill of Materials for a deployed model adapter.

    Generated deterministically from TrainingJob + EvalHarness + ModelDeployment.
    Committed to the model registry alongside the adapter artifact.

    Adapts: Mitchell et al. (2019), "Model Cards for Model Reporting."
    """

    card_version: str = Field(default="1.0")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Identity
    adapter_name: str
    base_model_id: str
    adapter_artifact_uri: str
    adapter_sha256: str
    training_job_id: str

    # Data Provenance
    corpus_sha256: str
    total_training_tokens: int
    dataset_licenses: list[str]
    is_decontaminated: bool

    # Training Configuration
    lora_rank: int
    lora_alpha: float
    lora_target_modules: list[str]
    trainable_parameter_fraction: float

    # Evaluation Results
    overall_eval_verdict: GateVerdict
    kl_divergence_nats: float
    kl_threshold_nats: float = Field(
        default=0.3, gt=0
    )  # policy: 0.3 is illustrative
    benchmark_results: list[QuantitativeAnalysis]
    regression_pass_rate: float
    known_regressions: list[str]

    # Deployment History
    deployment_id: str
    deployment_status: str
    rollback_count: int
    promoted_at: datetime | None

    # Intended Use (human-authored, injected at card generation time)
    intended_use: IntendedUse

    @computed_field
    @property
    def is_production_ready(self) -> bool:
        return (
            self.overall_eval_verdict == GateVerdict.PASSED
            and self.is_decontaminated
            and self.kl_divergence_nats < self.kl_threshold_nats
            and self.regression_pass_rate == 1.0
        )

    @classmethod
    def from_pipeline_artifacts(
        cls,
        job: TrainingJob,
        harness: EvalHarness,
        deployment: ModelDeployment,
        intended_use: IntendedUse,
    ) -> ModelCard:
        """
        Deterministic constructor: derives all fields from pipeline artifacts.
        No human data entry required or permitted for structured fields.
        """
        benchmark_analyses = [
            QuantitativeAnalysis(
                benchmark_name=r.benchmark_name,
                score=r.score,
                # 95% CI from Wilson score interval (n=sample_count)
                confidence_interval_95=(
                    max(0.0, r.score - 1.96 * (r.score * (1 - r.score) / r.sample_count) ** 0.5),
                    min(1.0, r.score + 1.96 * (r.score * (1 - r.score) / r.sample_count) ** 0.5),
                ),
                evaluation_set_size=r.sample_count,
            )
            for r in harness.benchmark_results
        ]

        return cls(
            adapter_name=job.adapter_name,
            base_model_id=job.base_model_id,
            adapter_artifact_uri=job.adapter_artifact_uri or "",
            adapter_sha256=job.adapter_sha256 or "",
            training_job_id=job.job_id,
            corpus_sha256=job.data_provenance.corpus_sha256,
            total_training_tokens=job.data_provenance.total_tokens,
            dataset_licenses=list({
                s.license for s in job.data_provenance.splits
            }),
            is_decontaminated=job.data_provenance.is_decontaminated,
            lora_rank=job.lora_config.rank,
            lora_alpha=job.lora_config.alpha,
            lora_target_modules=job.lora_config.target_modules,
            trainable_parameter_fraction=job.lora_config.trainable_parameter_fraction,
            overall_eval_verdict=harness.overall_verdict,
            kl_divergence_nats=harness.kl_divergence_gate.kl_divergence_nats,
            kl_threshold_nats=harness.kl_divergence_gate.threshold_nats,
            benchmark_results=benchmark_analyses,
            regression_pass_rate=sum(
                1 for t in harness.regression_tests if t.passed
            ) / max(len(harness.regression_tests), 1),
            known_regressions=[
                t.test_id for t in harness.regression_tests if t.passed is False
            ],
            deployment_id=deployment.deployment_id,
            deployment_status=deployment.status.value,
            rollback_count=deployment.rollback_count,
            promoted_at=deployment.promoted_at,
            intended_use=intended_use,
        )
# --8<-- [end:lst-16.11]
