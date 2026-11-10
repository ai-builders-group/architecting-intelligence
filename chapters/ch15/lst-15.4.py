# --8<-- [start:lst-15.4]
# src/ch10/optimization_boundary/breakeven.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class InfrastructureProfile(BaseModel):
    """Physical deployment constraints for fine-tuned adapter serving."""

    gpu_hourly_cost_usd: float = Field(
        description="On-demand GPU reservation cost (USD/hour). "
        "A100-80GB: ~$3.00/hr single-tenant. Use $3.00/M for S-LoRA pooled.",
        gt=0,
    )
    tenants_sharing_gpu: int = Field(
        default=1,
        description="Number of tenants multiplexed on the same GPU. "
        "S-LoRA/Punica enables M >> 1. Single-tenant: M=1.",
        ge=1,
    )
    pcie_bandwidth_gb_per_s: float = Field(
        default=25.0,
        description="Effective PCIe Gen4 x16 throughput (GB/s). "
        "Theoretical: 32 GB/s. Practical: 22-28 GB/s.",
        gt=0,
    )
    lora_adapter_size_mb: float = Field(
        default=134.0,
        description="LoRA adapter size in MB (rank=16, 7B model, FP16).",
        gt=0,
    )

    @property
    def idle_cost_per_day_usd(self) -> float:
        """Daily GPU reservation cost, amortized across tenants."""
        return (self.gpu_hourly_cost_usd * 24) / self.tenants_sharing_gpu

    @property
    def swap_latency_ms(self) -> float:
        """PCIe transfer time for one adapter swap (ms)."""
        return (self.lora_adapter_size_mb / (self.pcie_bandwidth_gb_per_s * 1024)) * 1000


class BreakevenAnalysis(BaseModel):
    """
    Economic model for the CapEx (fine-tune) vs. OpEx (RAG) decision.

    ANTI-PATTERN CORRECTION: The naive model C_breakeven = C_capex / C_opex_per_query
    is delusional. It omits C_idle, which dominates at low query volumes.

    The correct model:
        C_total_FT = C_capex + (N_days × C_idle) + (N_queries × C_token)
        C_total_RAG = N_queries × C_rag_per_query

    Breakeven exists only when C_total_FT < C_total_RAG, solved for N_queries.
    When C_idle > C_rag_per_query × queries_per_day, breakeven is at N_days → ∞.
    """

    # CapEx components
    fine_tuning_capex_usd: float = Field(
        description="One-time fine-tuning cost (GPU compute + data curation). "
        "Typical 7B model on 1M tokens: $300-700.",
        gt=0,
    )
    infra: InfrastructureProfile = Field(
        description="Deployment infrastructure profile.",
    )

    # OpEx components
    rag_cost_per_query_usd: float = Field(
        description="Marginal RAG cost per query (retrieval + LLM tokens). "
        "Typical: $0.001-0.005.",
        gt=0,
    )
    ft_token_cost_per_query_usd: float = Field(
        default=0.000005,
        description="Marginal token cost on fine-tuned model (amortized GPU). "
        "Substantially lower than RAG at high volume.",
        ge=0,
    )

    # Query volume
    queries_per_day: float = Field(
        description="Observed or projected daily query volume.",
        gt=0,
    )

    @property
    def rag_cost_per_day_usd(self) -> float:
        return self.queries_per_day * self.rag_cost_per_query_usd

    @property
    def ft_variable_cost_per_day_usd(self) -> float:
        return self.queries_per_day * self.ft_token_cost_per_query_usd

    @property
    def ft_total_cost_per_day_usd(self) -> float:
        """Fine-tune daily cost = idle + variable token cost."""
        return (
            self.infra.idle_cost_per_day_usd + self.ft_variable_cost_per_day_usd
        )

    @property
    def breakeven_days(self) -> float | Literal["NEVER"]:
        """
        Solve for N_days where C_total_FT == C_total_RAG.

        C_capex + N × (C_idle + C_ft_token × QPD)  =  N × (C_rag × QPD)
        C_capex  =  N × (C_rag × QPD - C_idle - C_ft_token × QPD)
        N  =  C_capex / (daily_rag_savings)

        If daily_rag_savings ≤ 0, fine-tuning NEVER breaks even.
        """
        daily_rag_savings = (
            self.rag_cost_per_day_usd - self.ft_total_cost_per_day_usd
        )
        if daily_rag_savings <= 0:
            return "NEVER"
        return self.fine_tuning_capex_usd / daily_rag_savings

    def total_cost_after_n_days(self, n_days: int) -> dict[str, float]:
        """Return total cost for each strategy after n_days."""
        rag_total = n_days * self.rag_cost_per_day_usd
        ft_total = (
            self.fine_tuning_capex_usd + n_days * self.ft_total_cost_per_day_usd
        )
        return {
            "rag_usd": round(rag_total, 2),
            "fine_tune_usd": round(ft_total, 2),
            "delta_usd": round(ft_total - rag_total, 2),
            "breakeven_days": self.breakeven_days,
        }


# ---------------------------------------------------------------------------
# Worked Example A: Single-Tenant (the "delusional" case)
# ---------------------------------------------------------------------------
_single_tenant = BreakevenAnalysis(
    fine_tuning_capex_usd=500.0,
    infra=InfrastructureProfile(
        gpu_hourly_cost_usd=3.0,
        tenants_sharing_gpu=1,  # Dedicated GPU
    ),
    rag_cost_per_query_usd=0.00125,
    queries_per_day=1_000,
)
# C_idle = $72/day; RAG saves $1.25/day → breakeven = NEVER
# _single_tenant.breakeven_days → "NEVER"
# _single_tenant.total_cost_after_n_days(365) →
#   {"rag_usd": 456.25, "fine_tune_usd": 26780.0, "delta_usd": 26323.75}

# ---------------------------------------------------------------------------
# Worked Example B: S-LoRA Multi-Tenant (M=100)
# ---------------------------------------------------------------------------
_slora_pooled = BreakevenAnalysis(
    fine_tuning_capex_usd=500.0,
    infra=InfrastructureProfile(
        gpu_hourly_cost_usd=3.0,
        tenants_sharing_gpu=100,  # S-LoRA: 100 adapters, one GPU
    ),
    rag_cost_per_query_usd=0.00125,
    queries_per_day=1_000,
)
# C_idle = $72/100 = $0.72/day; RAG cost = $1.25/day → saves $0.53/day
# _slora_pooled.breakeven_days → 943 days ≈ 2.6 years
# _slora_pooled.total_cost_after_n_days(365) →
#   {"rag_usd": 456.25, "fine_tune_usd": 762.8, "delta_usd": 306.55}
# --8<-- [end:lst-15.4]
