# --8<-- [start:lst-15.5]
# src/ch10/optimization_boundary/adr.py
from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class OptimizationStrategy(str, Enum):
    PROMPT_ENGINEERING = "prompt_engineering"
    RAG = "rag"
    FINE_TUNING_LORA = "fine_tuning_lora"
    FINE_TUNING_FULL = "fine_tuning_full"
    RLHF = "rlhf"


class KnowledgeType(str, Enum):
    FACTUAL = "factual"          # Entity attributes, world state → REVERSAL CURSE risk
    BEHAVIORAL = "behavioral"    # Tone, format, style → Fine-tune target
    SYNTACTIC = "syntactic"      # SQL, JSON, domain grammar → Fine-tune target
    SAFETY = "safety"            # Refusal posture → RLHF target
    TEMPORAL = "temporal"        # Time-sensitive → Context mandatory


class BreakevenAnalysis(BaseModel):
    breakeven_days: float = Field(gt=0, description="Days until fine-tuning cost breaks even.")


class FailureModeAnalysis(BaseModel):
    """
    Mandatory failure mode assessment for the chosen strategy.
    Asymmetric risk must be documented before commitment.
    """

    primary_failure_mode: str = Field(
        description="The most likely failure mode for the chosen strategy."
    )
    failure_probability: float = Field(
        ge=0.0, le=1.0,
        description="Estimated probability of primary failure mode (0–1).",
    )
    mitigation: str = Field(
        description="Deterministic mitigation strategy for the primary failure mode.",
    )

    # Strategy-specific failure modes
    retrieval_failure_description: str | None = Field(
        default=None,
        description="For RAG: Precision/recall failure mode (e.g., embedding drift, "
        "missing coverage). Null for non-RAG strategies.",
    )
    catastrophic_forgetting_risk: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="For fine-tuning: Probability of capability regression on held-out "
        "benchmark suite. Null for non-fine-tuning strategies.",
    )
    rank_collapse_risk: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="For LoRA: Probability of adapter rank collapse (all singular values "
        "converge to near-zero). Measured via adapter singular value monitoring.",
    )


class ObservabilityContract(BaseModel):
    """
    Mandatory observability requirements for the chosen strategy.
    A decision without an observability contract is a production liability.
    """

    golden_dataset_size: int = Field(
        ge=50,
        description="Minimum number of (input, expected_output) pairs in the Golden "
        "Dataset for CI/CD regression gating. Floor: 50 examples.",
    )
    golden_dataset_update_frequency_days: int = Field(
        description="Maximum days between Golden Dataset refresh cycles. "
        "Behavioral adapters: ≤30 days. Safety adapters: ≤7 days.",
    )
    regression_threshold: float = Field(
        ge=0.0, le=1.0,
        description="Maximum acceptable performance drop on Golden Dataset vs. baseline "
        "before deployment is blocked. Recommended: ≤0.05 (5% regression).",
    )
    drift_monitoring_metric: str = Field(
        description="Metric used for production drift detection. "
        "Examples: embedding cosine distance, output distribution KL divergence, "
        "refusal rate, format parse failure rate.",
    )
    drift_alert_threshold: float = Field(
        description="Threshold on drift_monitoring_metric that triggers re-evaluation. "
        "Strategy-specific; must be empirically calibrated on historical data.",
    )
    adapter_singular_value_monitoring: bool = Field(
        default=False,
        description="If True, monitor LoRA adapter singular values in production "
        "to detect rank collapse. Required for LoRA strategies.",
    )


class PhysicsConstraint(BaseModel):
    """
    Hardware constraints that must be satisfied before a strategy is viable.
    This section translates the PCIe/HBM analysis into go/no-go criteria.
    """

    requires_multi_tenant_lora: bool = Field(
        description="If True, adapter serving requires S-LoRA or Punica-style "
        "multiplexing. Single-tenant serving is architecturally disqualified.",
    )
    min_tenants_for_viability: int = Field(
        default=1,
        description="Minimum S-LoRA tenant count for economic viability. "
        "Derived from breakeven analysis.",
    )
    swap_latency_budget_ms: float = Field(
        default=float("inf"),
        description="Maximum tolerable PCIe adapter swap latency (ms). "
        "Must exceed T_swap = M_LoRA / BW_PCIe.",
    )
    hbm_resident_adapters: int = Field(
        default=1,
        description="Number of adapters maintained in HBM hot-pool. "
        "Determines LRU eviction frequency.",
    )
    adapter_size_gb: float = Field(
        default=0.0168, gt=0, description="LoRA adapter size (GB).")
    pcie_bandwidth_gb_s: float = Field(
        default=25.0, gt=0, description="Effective PCIe throughput (GB/s).")

# ... (remaining PhysicsConstraint fields, then class FineTuningADR)

class FineTuningADR(BaseModel):
    """
    Architectural Decision Record: Fine-Tuning vs. RAG.

    Named Pattern: Fine-Tuning Decision Tree
    Purpose: Deterministic classification of knowledge-to-substrate mapping
    to prevent entry into the Valley of Death cost state.
    """

    # --- Identity ---
    adr_id: str = Field(
        description="Unique ADR identifier. Format: ADR-10.2-{system}-{YYYYMMDD}.",
        pattern=r"^ADR-10\.2-.+-\d{8}$",
    )
    system_name: str = Field(description="The AI system or feature this ADR governs.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="ADR creation timestamp (UTC, timezone-aware).",
    )
    author: str = Field(description="Principal engineer responsible for this ADR.")
    status: Literal["proposed", "accepted", "superseded", "deprecated"] = Field(
        default="proposed",
    )

    # --- Knowledge Characterization ---
    knowledge_type: KnowledgeType = Field(
        description="Classification of the knowledge domain. "
        "FACTUAL knowledge is subject to the Reversal Curse [Berglund et al. 2023].",
    )
    knowledge_description: str = Field(
        description="Precise description of the knowledge domain. "
        "Include examples of forward and reverse query forms if FACTUAL.",
    )
    is_user_specific: bool = Field(
        description="True if knowledge contains PII or is scoped to individual users. "
        "Forces CONTEXT substrate; weights are shared global state.",
    )
    update_frequency_days: float = Field(
        description="Expected days between knowledge updates. "
        "< 90 days (quarterly) → prohibits WEIGHTS substrate.",
        gt=0,
    )
    queries_per_day: float = Field(
        description="Observed or projected daily query volume.",
        gt=0,
    )
    average_context_tokens: int = Field(
        description="Typical token footprint of this knowledge domain per query.",
        gt=0,
    )

    # --- Decision ---
    recommended_strategy: OptimizationStrategy = Field(
        description="The chosen optimization strategy.",
    )
    rationale: str = Field(
        description="Decision rationale referencing the four decision gates: "
        "Privacy, Volatility, Knowledge-Type, and Infrastructure.",
    )

    # --- Economic Model ---
    breakeven_analysis: BreakevenAnalysis | None = Field(
        default=None,
        description="Required when recommended_strategy is FINE_TUNING_*. "
        "Must demonstrate positive breakeven_days within planning horizon.",
    )

    # --- Risk & Observability ---
    failure_mode_analysis: FailureModeAnalysis = Field(
        description="Mandatory failure mode assessment.",
    )
    observability_contract: ObservabilityContract = Field(
        description="Mandatory observability and regression-testing requirements.",
    )

    # --- Physics ---
    physics_constraint: PhysicsConstraint = Field(
        description="Hardware constraints for the chosen strategy.",
    )

    # --- Cross-References ---
    cross_references: list[str] = Field(
        default_factory=list,
        description="Related ADRs, sections, and external references.",
    )

    @model_validator(mode="after")
    def _require_breakeven(self) -> "FineTuningADR":
        is_ft = self.recommended_strategy.name.startswith("FINE_TUNING_")
        if is_ft and self.breakeven_analysis is None:
            raise ValueError("FINE_TUNING_* requires breakeven_analysis")
        return self

    @model_validator(mode="after")
    def _require_sv_monitoring(self) -> "FineTuningADR":
        lora = self.recommended_strategy is OptimizationStrategy.FINE_TUNING_LORA
        if lora and not (
            self.observability_contract.adapter_singular_value_monitoring):
            raise ValueError("LoRA requires adapter_singular_value_monitoring")
        return self

    @model_validator(mode="after")
    def _check_swap_budget(self) -> "FineTuningADR":
        pc = self.physics_constraint
        t_swap_ms = pc.adapter_size_gb / pc.pcie_bandwidth_gb_s * 1000
        if pc.swap_latency_budget_ms <= t_swap_ms:
            raise ValueError("swap_latency_budget_ms must exceed T_swap")
        return self
# --8<-- [end:lst-15.5]
