# --8<-- [start:lst-15.7]
# src/ch10/optimization_boundary/examples/sql_dialect.py
"""
Reference Case: Fine-tuning a 7B model to generate proprietary SQL dialect
(e.g., BigQuery STRUCT/ARRAY syntax vs. standard SQL).

This is the canonical legitimate fine-tuning use case: syntactic output
distribution shift, not factual knowledge encoding.
"""
from enum import Enum
from types import SimpleNamespace
from datetime import datetime, UTC

KnowledgeType = Enum("KnowledgeType", "SYNTACTIC")
OptimizationStrategy = Enum("OptimizationStrategy", "FINE_TUNING_LORA")
_Record = type("_Record", (), {"__init__": lambda self, **kw: self.__dict__.update(kw)})
DecisionTreeInput = FailureModeAnalysis = FineTuningADR = _Record
ObservabilityContract = PhysicsConstraint = BreakevenAnalysis = InfrastructureProfile = _Record


def fine_tuning_decision_tree(inp):
    days = inp.fine_tuning_capex_usd / (inp.queries_per_day * inp.rag_cost_per_query_usd - 3.85)
    return SimpleNamespace(recommended_strategy=OptimizationStrategy.FINE_TUNING_LORA,
                           economic_model=SimpleNamespace(breakeven_days=days), swap_latency_ms=0.0168 / 25.0 * 1000)


def build_sql_dialect_adr() -> FineTuningADR:
    """Build a complete ADR for SQL dialect fine-tuning."""

    inp = DecisionTreeInput(
        knowledge_type=KnowledgeType.SYNTACTIC,
        is_user_specific=False,
        update_frequency_days=180,      # 180-day refresh interval, above the
        # 90-day volatility gate
        queries_per_day=50_000,         # High-volume analytics platform
        average_context_tokens=512,
        latency_sla_ms=1500.0,
        planning_horizon_days=365,
        slora_available=True,
        slora_tenant_pool_size=20,      # 20 tenants on one A100
        gpu_hourly_cost_usd=3.0,
        rag_cost_per_query_usd=0.00125,
        fine_tuning_capex_usd=600.0,
    )

    result = fine_tuning_decision_tree(inp)
    # result.recommended_strategy → FINE_TUNING_LORA
    # result.economic_model.breakeven_days → ~10.2 days before retrieval
    #   (600 / (62.50 - 3.85); RAG = 50K QPD × $0.00125 = $62.5/day)
    # result.swap_latency_ms → ~0.67ms (default 16.8 MB adapter; acceptable for
    # 50K QPD batched serving)

    infra = InfrastructureProfile(
        gpu_hourly_cost_usd=3.0,
        tenants_sharing_gpu=20,
    )

    return FineTuningADR(
        adr_id="ADR-10.2-sql-dialect-adapter-20260101",
        system_name="Analytics Query Generator",
        created_at=datetime.now(UTC),
        author="platform-ml-team",
        status="accepted",
        knowledge_type=KnowledgeType.SYNTACTIC,
        knowledge_description=(
            "BigQuery-specific SQL syntax: STRUCT construction, UNNEST operations, "
            "ARRAY_AGG, PARTITION BY with QUALIFY. Base model generates standard "
            "ANSI SQL; fine-tuning shifts output distribution toward BigQuery dialect. "
            "NOT factual knowledge — no Reversal Curse risk."
        ),
        is_user_specific=False,
        update_frequency_days=180,
        queries_per_day=50_000,
        average_context_tokens=512,
        recommended_strategy=result.recommended_strategy,
        rationale=(
            "Gate 1 (Privacy): Not user-specific. Cleared. "
            "Gate 2 (Volatility): 180-day update cycle > 90-day threshold. Cleared. "
            "Gate 3 (Knowledge Type): SYNTACTIC — shifts global output distribution "
            "toward BigQuery grammar. No Reversal Curse risk. Fine-tuning is correct tool. "
            "Gate 4 (Infrastructure): S-LoRA pool (20 tenants); "
            "C_idle = $3.60/day. "
            "Breakeven at ~13 days on 50K QPD, retrieval included. "
            "Economically justified."
        ),
        breakeven_analysis=BreakevenAnalysis(
            fine_tuning_capex_usd=600.0,
            infra=infra,
            rag_cost_per_query_usd=0.00125,
            queries_per_day=50_000,
        ),
        failure_mode_analysis=FailureModeAnalysis(
            primary_failure_mode=(
                "Catastrophic forgetting of general SQL reasoning during BigQuery "
                "dialect specialization."
            ),
            failure_probability=0.15,
            mitigation=(
                "LoRA rank r=16 constrains update magnitude. Replay buffer with 20% "
                "general SQL examples interleaved during training. "
                "Evaluate on SPIDER benchmark post-fine-tune; block deployment if "
                "score drops > 5% from baseline."
            ),
            retrieval_failure_description=None,
            catastrophic_forgetting_risk=0.15,
            rank_collapse_risk=0.05,
        ),
        observability_contract=ObservabilityContract(
            golden_dataset_size=200,
            golden_dataset_update_frequency_days=30,
            regression_threshold=0.05,
            drift_monitoring_metric="sql_parse_failure_rate",
            drift_alert_threshold=0.02,
            adapter_singular_value_monitoring=True,
        ),
        physics_constraint=PhysicsConstraint(
            requires_multi_tenant_lora=True,
            min_tenants_for_viability=10,
            swap_latency_budget_ms=10.0,
            hbm_resident_adapters=5,
        ),
        cross_references=[
            "ADR-10.0-substrate-router",
            "Chapter 16: Catastrophic Forgetting",
            "Chapter 16: LoRA Full Treatment",
            "Chapter 5: Attention SNR (Lost in the Middle)",
            "[1] Berglund et al. (2023) — Reversal Curse",
            "[2] Hu et al. (2021) — LoRA",
        ],
    )
# --8<-- [end:lst-15.7]
