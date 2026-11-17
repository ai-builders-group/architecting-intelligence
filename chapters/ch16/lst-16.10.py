# --8<-- [start:lst-16.10]
# default_policies.py

from dataclasses import dataclass
from enum import Enum


class RolloutPhase(str, Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    EARLY = "early"
    HALF = "half"
    FULL = "full"


@dataclass(frozen=True)
class TrafficPolicy:
    phase: RolloutPhase
    green_traffic_pct: float
    min_duration_minutes: int
    max_error_rate_delta_pct: float
    max_latency_p95_delta_ms: float
    min_quality_score_delta: float
    min_sample_count: int


DEFAULT_ROLLOUT_POLICY: list[TrafficPolicy] = [
    TrafficPolicy(
        phase=RolloutPhase.SHADOW,
        green_traffic_pct=0.0,
        min_duration_minutes=60,        # 1 hour shadow mode: replay-only
        max_error_rate_delta_pct=0.0,   # any rise over blue stops the rollout
        max_latency_p95_delta_ms=200.0,
        min_quality_score_delta=0.0,
        min_sample_count=500,
    ),
    TrafficPolicy(
        phase=RolloutPhase.CANARY,
        green_traffic_pct=1.0,
        min_duration_minutes=60,        # 1 hour at 1% before advancing
        max_error_rate_delta_pct=1.0,
        max_latency_p95_delta_ms=100.0,
        min_quality_score_delta=-0.01,
        min_sample_count=1000,
    ),
    TrafficPolicy(
        phase=RolloutPhase.EARLY,
        green_traffic_pct=10.0,
        min_duration_minutes=360,       # 6 hours at 10%
        max_error_rate_delta_pct=2.0,
        max_latency_p95_delta_ms=100.0,
        min_quality_score_delta=-0.02,
        min_sample_count=5000,
    ),
    TrafficPolicy(
        phase=RolloutPhase.HALF,
        green_traffic_pct=50.0,
        min_duration_minutes=1440,      # 24 hours at 50%: full diurnal cycle
        max_error_rate_delta_pct=2.0,
        max_latency_p95_delta_ms=75.0,
        min_quality_score_delta=-0.02,
        min_sample_count=50_000,
    ),
    TrafficPolicy(
        phase=RolloutPhase.FULL,
        green_traffic_pct=100.0,
        min_duration_minutes=0,
        max_error_rate_delta_pct=5.0,  # Illustrative: wider tolerance at full
        # traffic (noise)
        max_latency_p95_delta_ms=50.0,
        min_quality_score_delta=-0.03,
        min_sample_count=100_000,
    ),
]
# --8<-- [end:lst-16.10]
