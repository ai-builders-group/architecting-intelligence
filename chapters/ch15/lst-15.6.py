# --8<-- [start:lst-15.6]
# src/ch10/optimization_boundary/decision_tree.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class KnowledgeType(str, Enum):
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    SYNTACTIC = "syntactic"
    SAFETY = "safety"


class OptimizationStrategy(str, Enum):
    RAG = "rag"
    FINE_TUNING_LORA = "fine_tuning_lora"
    RLHF = "rlhf"


class InfrastructureProfile(BaseModel):
    gpu_hourly_cost_usd: float
    tenants_sharing_gpu: int = 1
    adapter_size_mb: float = 200.0
    pcie_gb_per_s: float = 25.0

    @property
    def swap_latency_ms(self) -> float:
        return self.adapter_size_mb / self.pcie_gb_per_s

    @property
    def idle_cost_per_day_usd(self) -> float:
        return self.gpu_hourly_cost_usd * 24 / self.tenants_sharing_gpu


class BreakevenAnalysis(BaseModel):
    fine_tuning_capex_usd: float
    infra: InfrastructureProfile
    rag_cost_per_query_usd: float
    queries_per_day: float
    ft_cost_per_query_usd: float = 0.0002

    @property
    def rag_cost_per_day_usd(self) -> float:
        return self.rag_cost_per_query_usd * self.queries_per_day

    @property
    def ft_variable_cost_per_day_usd(self) -> float:
        return self.ft_cost_per_query_usd * self.queries_per_day

    @property
    def breakeven_days(self) -> float | Literal["NEVER"]:
        savings = (
            self.rag_cost_per_day_usd
            - self.ft_variable_cost_per_day_usd
            - self.infra.idle_cost_per_day_usd
        )
        if savings <= 0:
            return "NEVER"
        return self.fine_tuning_capex_usd / savings


@dataclass(frozen=True)
class DecisionGateResult:
    """Immutable record of a single gate evaluation in the decision tree."""

    gate_name: Literal["privacy", "volatility", "knowledge_type", "infrastructure", "economic"]
    triggered: bool
    recommended_strategy: OptimizationStrategy | None
    reasoning: str


class DecisionTreeInput(BaseModel):
    """Input specification for the Fine-Tuning Decision Tree."""

    knowledge_type: KnowledgeType
    is_user_specific: bool
    update_frequency_days: float = Field(gt=0)
    queries_per_day: float = Field(gt=0)
    average_context_tokens: int = Field(gt=0)
    latency_sla_ms: float = Field(default=2000.0, gt=0)
    planning_horizon_days: int = Field(default=365, gt=0)

    # Infrastructure availability
    slora_available: bool = Field(
        default=False,
        description="True if S-LoRA or Punica is available in the serving stack.",
    )
    slora_tenant_pool_size: int = Field(
        default=1,
        description="Number of tenants sharing the GPU pool in S-LoRA deployment.",
        ge=1,
    )
    gpu_hourly_cost_usd: float = Field(default=3.0, gt=0)

    # Cost parameters
    rag_cost_per_query_usd: float = Field(default=0.00125, gt=0)
    fine_tuning_capex_usd: float = Field(default=500.0, gt=0)


class DecisionTreeOutput(BaseModel):
    """Output of the Fine-Tuning Decision Tree, including the full gate trace."""

    recommended_strategy: OptimizationStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    gate_trace: list[DecisionGateResult]
    economic_model: BreakevenAnalysis | None = None
    swap_latency_ms: float | None = None
    warnings: list[str] = Field(default_factory=list)


def fine_tuning_decision_tree(inp: DecisionTreeInput) -> DecisionTreeOutput:
    """
    Named Pattern: Fine-Tuning Decision Tree

    Classifies a knowledge domain into the optimal optimization strategy
    via four sequential gates. Each gate is a hard constraint; later gates
    are not evaluated if an earlier gate triggers.

    Gate 1 — Privacy:
        User-specific knowledge MUST use CONTEXT. Weights are global state;
        per-user data encoded in weights leaks across the user population.

    Gate 2 — Volatility:
        Knowledge that changes more frequently than the training-plus-deployment
        cycle (empirically ~90 days) MUST use CONTEXT. No fine-tuning pipeline
        can match sub-quarterly update cadences.

    Gate 3 — Knowledge Type:
        FACTUAL knowledge is subject to the Reversal Curse [Berglund et al. 2023].
        P(B|A) ≠ P(A|B) in weight space. CONTEXT (RAG) is mandatory for facts.
        BEHAVIORAL, SYNTACTIC, and SAFETY knowledge targets the global output prior
        and is appropriate for fine-tuning.

    Gate 4 — Infrastructure:
        Fine-tuning without S-LoRA/Punica is economically prohibited below
        ~10K queries/day (single-tenant GPU idle cost dominates). This gate
        verifies infrastructure availability and economic viability.
    """
    gates: list[DecisionGateResult] = []
    warnings: list[str] = []

    # ── Gate 1: Privacy ────────────────────────────────────────────────────
    if inp.is_user_specific:
        gates.append(DecisionGateResult(
            gate_name="privacy",
            triggered=True,
            recommended_strategy=OptimizationStrategy.RAG,
            reasoning=(
                "User-specific knowledge is PII-scoped. Model weights are shared "
                "global state across all users; encoding per-user data in weights "
                "constitutes a data leakage vector. CONTEXT (RAG) with per-session "
                "isolation is mandatory."
            ),
        ))
        return DecisionTreeOutput(
            recommended_strategy=OptimizationStrategy.RAG,
            confidence=0.97,
            gate_trace=gates,
        )

    gates.append(DecisionGateResult(
        gate_name="privacy",
        triggered=False,
        recommended_strategy=None,
        reasoning="Knowledge is not user-specific. Privacy gate cleared.",
    ))

    # ── Gate 2: Volatility ─────────────────────────────────────────────────
    # Training + deployment cycle: ~30 days minimum (data curation + training +
    # eval + shadow deployment). Knowledge updating faster than quarterly (90 days)
    # will be stale before the fine-tuned model reaches production.
    VOLATILITY_THRESHOLD_DAYS = 90.0

    if inp.update_frequency_days < VOLATILITY_THRESHOLD_DAYS:
        gates.append(DecisionGateResult(
            gate_name="volatility",
            triggered=True,
            recommended_strategy=OptimizationStrategy.RAG,
            reasoning=(
                f"Update frequency ({inp.update_frequency_days:.0f} days) is below "
                f"the training-plus-deployment cycle floor ({VOLATILITY_THRESHOLD_DAYS} days). "
                "Fine-tuned weights will be stale at deployment. CONTEXT is mandatory."
            ),
        ))
        return DecisionTreeOutput(
            recommended_strategy=OptimizationStrategy.RAG,
            confidence=0.93,
            gate_trace=gates,
        )

    gates.append(DecisionGateResult(
        gate_name="volatility",
        triggered=False,
        recommended_strategy=None,
        reasoning=(
            f"Update frequency ({inp.update_frequency_days:.0f} days) exceeds "
            "volatility threshold. Volatility gate cleared."
        ),
    ))

    # ── Gate 3: Knowledge Type ─────────────────────────────────────────────
    # FACTUAL → Reversal Curse risk; RAG mandatory.
    # TEMPORAL → By definition volatile; should have been caught by Gate 2,
    #            but included here as belt-and-suspenders.
    # BEHAVIORAL / SYNTACTIC / SAFETY → Fine-tune target.

    if inp.knowledge_type in (KnowledgeType.FACTUAL, KnowledgeType.TEMPORAL):
        warnings.append(
            "REVERSAL CURSE WARNING [Berglund et al. 2023]: Factual knowledge "
            "encoded in weights exhibits P(B|A) ≠ P(A|B). Forward queries train "
            "correctly; reverse queries produce systematic hallucinations. "
            "RAG is the correct substrate."
        )
        gates.append(DecisionGateResult(
            gate_name="knowledge_type",
            triggered=True,
            recommended_strategy=OptimizationStrategy.RAG,
            reasoning=(
                "Temporal knowledge goes stale between fine-tuning cycles; "
                "weights cannot track it. CONTEXT is mandatory."
                if inp.knowledge_type == KnowledgeType.TEMPORAL
                else
                f"Knowledge type '{inp.knowledge_type.value}' is subject to the "
                "Reversal Curse. Fine-tuning encodes directional P(B|A) but does "
                "not reliably induce the inverse P(A|B). CONTEXT is mandatory."
            ),
        ))
        return DecisionTreeOutput(
            recommended_strategy=OptimizationStrategy.RAG,
            confidence=0.91,
            gate_trace=gates,
            warnings=warnings,
        )

    # Determine target fine-tuning strategy by knowledge type
    target_ft_strategy = {
        KnowledgeType.BEHAVIORAL: OptimizationStrategy.FINE_TUNING_LORA,
        KnowledgeType.SYNTACTIC: OptimizationStrategy.FINE_TUNING_LORA,
        KnowledgeType.SAFETY: OptimizationStrategy.RLHF,
    }.get(inp.knowledge_type, OptimizationStrategy.FINE_TUNING_LORA)

    gates.append(DecisionGateResult(
        gate_name="knowledge_type",
        triggered=False,
        recommended_strategy=target_ft_strategy,
        reasoning=(
            f"Knowledge type '{inp.knowledge_type.value}' targets the global output "
            "prior — a legitimate fine-tuning objective. Knowledge-type gate cleared."
        ),
    ))

    # ── Gate 4: Infrastructure & Economic Viability ────────────────────────
    infra = InfrastructureProfile(
        gpu_hourly_cost_usd=inp.gpu_hourly_cost_usd,
        tenants_sharing_gpu=inp.slora_tenant_pool_size if inp.slora_available else 1,
    )

    breakeven = BreakevenAnalysis(
        fine_tuning_capex_usd=inp.fine_tuning_capex_usd,
        infra=infra,
        rag_cost_per_query_usd=inp.rag_cost_per_query_usd,
        queries_per_day=inp.queries_per_day,
    )

    swap_latency_ms = infra.swap_latency_ms

    if infra.tenants_sharing_gpu > 1 and swap_latency_ms > inp.latency_sla_ms:
        warnings.append(
            "PCIe WALL: Shared-GPU adapter swapping detected. "
            f"Adapter swap latency: T_swap = {swap_latency_ms:.1f}ms "
            f"exceeds latency_sla_ms ({inp.latency_sla_ms:.0f}ms). "
            "At scale, per-request swaps destroy TTFT. "
            "Shrink adapters or tenants per GPU before committing to fine-tuning."
        )

    be_result = breakeven.breakeven_days
    if be_result == "NEVER" or be_result > inp.planning_horizon_days:
        if be_result == "NEVER":
            warnings.append(
                "ECONOMIC VETO: Fine-tuning NEVER breaks even at current query volume. "
                f"C_idle (${infra.idle_cost_per_day_usd:.2f}/day) > "
                f"RAG savings (${breakeven.rag_cost_per_day_usd - breakeven.ft_variable_cost_per_day_usd:.2f}/day). "
                "Increase tenant pool size (S-LoRA) or query volume before proceeding."
            )
        else:
            warnings.append(
                f"ECONOMIC WARNING: Breakeven at {be_result:.0f} days exceeds "
                f"planning horizon ({inp.planning_horizon_days} days). "
                "Fine-tuning will not recover CapEx within the planning window."
            )
        gates.append(DecisionGateResult(
            gate_name="infrastructure",
            triggered=True,
            recommended_strategy=OptimizationStrategy.RAG,
            reasoning=(
                "Economic veto: CapEx is not recovered "
                "within the planning horizon."
            ),
        ))
        return DecisionTreeOutput(
            recommended_strategy=OptimizationStrategy.RAG,
            confidence=0.88,
            gate_trace=gates,
            economic_model=breakeven,
            swap_latency_ms=swap_latency_ms,
            warnings=warnings,
        )

    gates.append(DecisionGateResult(
        gate_name="infrastructure",
        triggered=False,
        recommended_strategy=target_ft_strategy,
        reasoning=(
            f"Infrastructure gate cleared. Breakeven: {be_result:.0f} days. "
            f"S-LoRA tenant pool: {inp.slora_tenant_pool_size}. "
            f"Adapter swap latency: {swap_latency_ms:.1f}ms."
        ),
    ))

    confidence = 0.85 if warnings else 0.92
    return DecisionTreeOutput(
        recommended_strategy=target_ft_strategy,
        confidence=confidence,
        gate_trace=gates,
        economic_model=breakeven,
        swap_latency_ms=swap_latency_ms,
        warnings=warnings,
    )
# --8<-- [end:lst-15.6]
