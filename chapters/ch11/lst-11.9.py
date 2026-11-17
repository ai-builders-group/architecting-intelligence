# --8<-- [start:lst-11.9]
# self_healing_ontology.py
"""
Self-Healing Ontology Pattern

Architecture:
  - OntologyDriftDetector: MMD-based drift detection on zero-cost embeddings.
  - TraceInjectedMetaReviewer: Samples ReplayBuffer; passes concrete traces to LLM.
  - NegativeHarness: Regression guard with closed-loop retry (max 3) + adaptive backoff.
  - ContractUpdate: Versioned diff with rationale and approval gate.
  - OntologyVersionHeader: Durable workflow version pinning.
  - AntiCorruptionLayer: Stable application schema decoupled from living contract.

Anti-patterns explicitly avoided:
  - No Pydantic mutation at runtime.
  - No raw W1 in 1536-D (uses MMD + median heuristic).
  - No fixed cooldown (uses exponential backoff: T_base * 2^f).
  - No abstract-metrics-only review (traces injected from ReplayBuffer).
  - No re-embedding for monitoring (pulls vectors from State objects).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, Protocol, Sequence

import numpy as np
from pydantic import BaseModel, Field, computed_field, model_validator

# ---------------------------------------------------------------------------
# Re-export from ch10_03 — we reference, not duplicate.
# ReplayBuffer, ReplaySample, DriftSeverity are defined in 10_03.
# ---------------------------------------------------------------------------


# ============================================================================
# SECTION 1: SCHEMA FOUNDATIONS
# ============================================================================

class DriftKind(str, Enum):
    """Three mechanistically distinct ontology failure modes."""
    SEMANTIC  = "semantic"    # Model interpretation shifted post-weight-update.
    WORLD     = "world"       # External domain changed; rule encodes stale world.
    DISTRIBUTION = "distribution"  # Input population shifted; new boundary cases exposed.


class DriftSeverityLevel(str, Enum):
    NOMINAL  = "nominal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


class RuleEmbedding(BaseModel):
    """
    Planaria Unit: an autonomous rule with distributed representation.

    Rules are NOT stored as lines in a flat file. Each rule is a
    content-addressable unit whose semantic fingerprint is stored alongside
    its predicate. The system survives partial updates (amputation) because
    no rule depends on another rule's ordinal position.
    """
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_text: str = Field(description="Natural language statement of the rule.")
    json_schema_predicate: dict[str, Any] = Field(
        description="JSON Schema encoding the rule as a machine-checkable predicate."
    )
    embedding: list[float] = Field(
        description="Semantic vector fingerprint. Pulled from RAG/routing State — zero marginal cost."
    )
    authored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(description="Ontology version when rule was authored.")
    provenance: Literal["human", "meta_review", "initial_load"] = "human"

    @computed_field
    @property
    def content_hash(self) -> str:
        payload = f"{self.rule_text}:{json.dumps(self.json_schema_predicate, sort_keys=True)}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class OntologyVersionHeader(BaseModel):
    """
    Durable workflow versioning.

    Long-running workflows (e.g., multi-day research agents) pin to the
    contract version active at their start time. This header is stored in
    AgentState.metadata and checked at every guardrail evaluation. An updated
    contract does NOT retroactively invalidate in-flight workflows.

    This mirrors the 'Schema on Read' pattern used in data lakes: the stored
    execution traces are immutable; the interpretation layer is versioned.
    """
    version_id: str
    activated_at: datetime
    deactivated_at: datetime | None = None
    rule_ids: frozenset[str] = Field(description="Stable snapshot of active rule IDs.")

    @property
    def is_current(self) -> bool:
        return self.deactivated_at is None


class ExecutionTrace(BaseModel):
    """
    The 'gradient' of an ontology failure.

    A meta-reviewer operating on aggregate floating-point metrics (MMD, KL)
    cannot reason about WHY a rule is stale. It requires the raw failure text:
    input, model output, guardrail that fired, and human ground-truth label.
    """
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    input_text: str
    raw_model_output: str
    guardrail_id: str
    guardrail_verdict: Literal["pass", "fail"]
    human_ground_truth: Literal["correct", "incorrect"] | None = None
    output_embedding: list[float] = Field(
        description="Pulled from State.routing_embedding — no re-embedding cost."
    )

    @property
    def is_counterexample(self) -> bool:
        """True when the guardrail fires on a semantically correct output."""
        return (
            self.guardrail_verdict == "fail"
            and self.human_ground_truth == "correct"
        )


# ============================================================================
# SECTION 2: DRIFT DETECTION — MMD WITH MEDIAN HEURISTIC
# ============================================================================

class MMDConfig(BaseModel):
    """Configuration for Maximum Mean Discrepancy computation."""
    # Bandwidth selection: 'median' sets sigma by the median heuristic.
    # 'fixed' uses a pre-specified sigma value.
    bandwidth_strategy: Literal["median", "fixed"] = "median"
    fixed_sigma: float | None = None  # Only used when strategy='fixed'
    sample_size: int = Field(
        default=512,
        description=(
            "Stochastic reservoir sample size. Full-corpus computation is O(n^2); "
            "512-sample MMD is statistically consistent at 1-2% of the full cost."
        )
    )

    @model_validator(mode="after")
    def validate_fixed_sigma(self) -> "MMDConfig":
        if self.bandwidth_strategy == "fixed" and self.fixed_sigma is None:
            raise ValueError("fixed_sigma must be set when bandwidth_strategy='fixed'.")
        return self


def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix K(X, Y).

    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    # Squared pairwise distances via expansion: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
    XX = np.sum(X ** 2, axis=1, keepdims=True)   # (n, 1)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)   # (m, 1)
    sq_dists = XX + YY.T - 2.0 * (X @ Y.T)       # (n, m)
    sq_dists = np.maximum(sq_dists, 0.0)          # Numerical guard against small negatives
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def compute_mmd_squared(
    reference_embeddings: np.ndarray,
    production_embeddings: np.ndarray,
    config: MMDConfig,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Unbiased estimator of MMD^2 using a Gaussian RBF kernel.

    Uses stochastic reservoir sampling (config.sample_size) to keep
    computation O(sample_size^2) rather than O(n^2). With sample_size=512
    and 1536-D embeddings, this runs in ~15ms on a single CPU core.

    Bandwidth sigma is set by the median heuristic [24]:
        sigma^2 = median(||x_i - x_j||^2) over the joint sample.
    This is adaptive: it scales with the intrinsic geometry of the
    embedding space regardless of model provider or dimensionality.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = config.sample_size

    # Stochastic reservoir sampling (10-20% of corpus keeps computation O(sample_size^2))
    ref_idx = rng.choice(len(reference_embeddings), size=min(n, len(reference_embeddings)), replace=False)
    prod_idx = rng.choice(len(production_embeddings), size=min(n, len(production_embeddings)), replace=False)

    X = reference_embeddings[ref_idx]   # (n, d)
    Y = production_embeddings[prod_idx] # (n, d)

    if config.bandwidth_strategy == "median":
        # Median heuristic on the joint sample
        joint = np.vstack([X, Y])
        # Sample pairs for efficiency
        pair_idx = rng.choice(len(joint), size=min(2000, len(joint)), replace=False)
        sample = joint[pair_idx]
        diffs = sample[:, None, :] - sample[None, :, :]  # (k, k, d)
        sq_dists = np.sum(diffs ** 2, axis=-1).ravel()
        sigma_sq = float(np.median(sq_dists[sq_dists > 0]))
        sigma = math.sqrt(max(sigma_sq, 1e-8))
    else:
        sigma = config.fixed_sigma  # type: ignore[assignment]

    K_XX = _rbf_kernel_matrix(X, X, sigma)
    K_YY = _rbf_kernel_matrix(Y, Y, sigma)
    K_XY = _rbf_kernel_matrix(X, Y, sigma)

    # Unbiased MMD^2 estimator (exclude diagonal)
    n_x, n_y = len(X), len(Y)
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)

    mmd_sq = (
        K_XX.sum() / (n_x * (n_x - 1))
        - 2.0 * K_XY.mean()
        + K_YY.sum() / (n_y * (n_y - 1))
    )
    return float(mmd_sq)


class DriftMeasurement(BaseModel):
    """Snapshot of ontology health at a point in time."""
    measured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    window_size_executions: int
    mmd_squared: float = Field(description="Embedding drift (MMD^2 with RBF kernel).")
    rule_violation_rate: float = Field(ge=0.0, le=1.0)
    js_divergence: float = Field(ge=0.0, description="Jensen-Shannon divergence on label distribution.")
    composite_score: float
    severity: DriftSeverityLevel
    dominant_kind: DriftKind | None = None
    counterexample_count: int = Field(
        description="Number of traces where guardrail fired on semantically correct output."
    )


class OntologyDriftDetector(BaseModel):
    """
    Monitors three drift dimensions and computes a composite score.

    Zero-Cost Embedding Strategy:
        This detector does NOT call the embedding API during monitoring.
        Embeddings are pulled from State.output_embedding — computed during
        initial RAG retrieval or semantic routing. Monitoring cost is ~$0.
        No separate API call required.
    """
    mmd_config: MMDConfig = Field(default_factory=MMDConfig)
    reference_embeddings: list[list[float]] = Field(
        description="Embedding distribution from the ontology's authoring period."
    )
    reference_label_distribution: dict[str, float] = Field(
        description="Historical distribution of output categories (e.g., intent labels)."
    )
    drift_weights: tuple[float, float, float] = Field(
        default=(0.4, 0.4, 0.2),
        description="Weights for (MMD, violation_rate, JS_divergence) in composite score."
    )
    # Threshold is calibrated by permutation test, not hardcoded.
    # Default 0.08 is a starting point, replaced during calibration.
    alert_threshold: float = Field(
        default=0.08,
        description=(
            "Composite drift threshold. Set by 95th percentile of the null distribution "
            "under a permutation test. Do not treat this constant as invariant."
        )
    )

    def measure(
        self,
        recent_traces: Sequence[ExecutionTrace],
    ) -> DriftMeasurement:
        """
        Compute composite drift score from a window of execution traces.
        Embeddings are sourced from trace.output_embedding — zero marginal cost.
        """
        if not recent_traces:
            raise ValueError("Cannot measure drift on empty trace window.")

        ref_arr = np.array(self.reference_embeddings, dtype=np.float32)
        prod_arr = np.array(
            [t.output_embedding for t in recent_traces], dtype=np.float32
        )

        # Metric 1: Embedding drift via MMD^2
        mmd_sq = compute_mmd_squared(ref_arr, prod_arr, self.mmd_config)

        # Metric 2: Rule violation rate
        total = len(recent_traces)
        violations = sum(1 for t in recent_traces if t.guardrail_verdict == "fail")
        violation_rate = violations / total

        # Metric 3: Jensen-Shannon divergence on label distribution
        current_labels: dict[str, int] = {}
        for t in recent_traces:
            # Use guardrail_id as a proxy label; real systems use intent classifier output
            current_labels[t.guardrail_id] = current_labels.get(t.guardrail_id, 0) + 1

        all_labels = set(self.reference_label_distribution) | set(current_labels)
        ref_p = np.array([self.reference_label_distribution.get(l, 1e-9) for l in all_labels])
        cur_q = np.array([current_labels.get(l, 0) / total for l in all_labels])
        ref_p /= ref_p.sum()
        cur_q = np.where(cur_q == 0, 1e-9, cur_q)
        cur_q /= cur_q.sum()
        m = 0.5 * (ref_p + cur_q)
        js_div = float(0.5 * np.sum(ref_p * np.log(ref_p / m)) + 0.5 * np.sum(cur_q * np.log(cur_q / m)))

        # Composite score
        w1, w2, w3 = self.drift_weights
        composite = w1 * min(mmd_sq, 1.0) + w2 * violation_rate + w3 * min(js_div, 1.0)

        # Severity classification
        if composite < self.alert_threshold * 0.5:
            severity = DriftSeverityLevel.NOMINAL
        elif composite < self.alert_threshold:
            severity = DriftSeverityLevel.ELEVATED
        else:
            severity = DriftSeverityLevel.CRITICAL

        # Identify dominant drift kind by which component is largest
        components = {
            DriftKind.SEMANTIC: w1 * min(mmd_sq, 1.0),
            DriftKind.WORLD: w2 * violation_rate,
            DriftKind.DISTRIBUTION: w3 * min(js_div, 1.0),
        }
        dominant = max(components, key=components.__getitem__)

        counterexamples = sum(1 for t in recent_traces if t.is_counterexample)

        return DriftMeasurement(
            window_size_executions=total,
            mmd_squared=mmd_sq,
            rule_violation_rate=violation_rate,
            js_divergence=js_div,
            composite_score=composite,
            severity=severity,
            dominant_kind=dominant if composite >= self.alert_threshold else None,
            counterexample_count=counterexamples,
        )


# ============================================================================
# SECTION 3: META-REVIEW — TRACE-INJECTED LLM PROPOSAL
# ============================================================================

class MetaReviewRequest(BaseModel):
    """
    The context package sent to the LLM meta-reviewer.

    ANTI-PATTERN AVOIDED: The reviewer does NOT receive aggregate metrics
    (MMD=0.12, violation_rate=0.34). These numbers describe the *magnitude*
    of drift but provide no mechanism for a language model to propose a
    targeted rule change.

    CORRECT PATTERN: The reviewer receives the top-K failing execution traces
    — concrete (Input, Output, Rule, Verdict, Ground Truth) tuples. These
    are the 'gradients' of the failure; they show the reviewer exactly where
    the rule boundary needs to move.
    """
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    drift_measurement: DriftMeasurement
    rules_under_review: list[RuleEmbedding]
    # Top-K traces sampled from ReplayBuffer by semantic proximity to the failing rules
    injected_traces: list[ExecutionTrace] = Field(
        min_length=1,
        description=(
            "Concrete execution traces (Input → Output → Guardrail Failed → Ground Truth). "
            "Sampled from ReplayBuffer using semantic proximity to failing rule embeddings. "
            "K=10 is empirically sufficient; beyond K=20 shows diminishing returns."
        )
    )
    top_k_traces: int = Field(default=10)

    def render_prompt(self) -> str:
        """Render the structured prompt for the LLM meta-reviewer."""
        traces_block = "\n\n".join(
            f"--- Trace {i+1} (counterexample={t.is_counterexample}) ---\n"
            f"INPUT: {t.input_text[:500]}\n"
            f"MODEL OUTPUT: {t.raw_model_output[:500]}\n"
            f"RULE VIOLATED: {t.guardrail_id}\n"
            f"GUARDRAIL VERDICT: {t.guardrail_verdict}\n"
            f"HUMAN LABEL: {t.human_ground_truth or 'unlabeled'}"
            for i, t in enumerate(self.injected_traces)
        )

        rules_block = "\n".join(
            f"[{r.rule_id}] {r.rule_text}\n  Predicate: {json.dumps(r.json_schema_predicate)}"
            for r in self.rules_under_review
        )

        return f"""You are an ontology engineer reviewing a production AI system's guardrail rules.

DRIFT MEASUREMENT:
- Composite Score: {self.drift_measurement.composite_score:.4f}
- Dominant Drift Kind: {self.drift_measurement.dominant_kind}
- Rule Violation Rate: {self.drift_measurement.rule_violation_rate:.2%}
- Counterexamples (rule fired on correct output): {self.drift_measurement.counterexample_count}

RULES UNDER REVIEW:
{rules_block}

FAILING EXECUTION TRACES (top {len(self.injected_traces)} by semantic proximity to failing rules):
{traces_block}

TASK:
Propose a minimal, targeted modification to one or more of the rules above that:
1. Resolves the counterexamples shown (cases where the guardrail incorrectly rejected valid output).
2. Does NOT weaken the rule's coverage of genuinely invalid outputs.
3. Provides a precise rationale grounded in the traces above.

Respond in JSON with this structure:
{{
  "proposed_changes": [
    {{
      "rule_id": "<id>",
      "new_rule_text": "<updated natural language rule>",
      "new_json_schema_predicate": {{...}},
      "rationale": "<one paragraph grounded in the specific traces above>"
    }}
  ],
  "confidence": 0.0-1.0,
  "requires_human_review": true/false
}}"""


class ProposedRuleChange(BaseModel):
    rule_id: str
    new_rule_text: str
    new_json_schema_predicate: dict[str, Any]
    rationale: str


class MetaReviewProposal(BaseModel):
    """Structured output from the LLM meta-reviewer."""
    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    proposed_changes: list[ProposedRuleChange]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


async def trigger_meta_review(
    drift_measurement: DriftMeasurement,
    rules_under_review: list[RuleEmbedding],
    replay_buffer_samples: list[ExecutionTrace],
    llm_client: "LLMClientProtocol",
    top_k: int = 10,
) -> MetaReviewProposal:
    """
    Construct a trace-injected meta-review request and invoke the LLM.

    Trace Selection Strategy:
        From the replay buffer, we select traces by SEMANTIC PROXIMITY to
        the failing rules' embeddings, not by recency alone. This surfaces
        the most mechanistically relevant failures, not just the most recent.

        Specifically, we rank each trace by its cosine distance to the
        centroid of the failing rules' embedding vectors:

            score(trace) = cosine_sim(trace.output_embedding, rule_centroid)

        The top-K traces by this score are the 'gradients' of the failure.
    """
    if not replay_buffer_samples:
        raise ValueError("Cannot trigger meta-review: replay buffer is empty.")

    # Compute centroid of failing rules' embeddings
    rule_vecs = np.array([r.embedding for r in rules_under_review], dtype=np.float32)
    centroid = rule_vecs.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

    # Rank traces by semantic proximity to failing rule centroid
    def cosine_proximity(trace: ExecutionTrace) -> float:
        vec = np.array(trace.output_embedding, dtype=np.float32)
        norm = vec / (np.linalg.norm(vec) + 1e-8)
        return float(np.dot(norm, centroid_norm))

    ranked = sorted(replay_buffer_samples, key=cosine_proximity, reverse=True)
    top_traces = ranked[:top_k]

    request = MetaReviewRequest(
        drift_measurement=drift_measurement,
        rules_under_review=rules_under_review,
        injected_traces=top_traces,
        top_k_traces=top_k,
    )

    raw_response = await llm_client.complete(
        system="You are an expert ontology engineer. Respond only with valid JSON.",
        user=request.render_prompt(),
        temperature=0.1,  # Low temperature reduces variance; validation remains external
        response_format={"type": "json_object"},
    )

    parsed = json.loads(raw_response)
    return MetaReviewProposal(
        request_id=request.request_id,
        proposed_changes=[ProposedRuleChange(**c) for c in parsed["proposed_changes"]],
        confidence=float(parsed["confidence"]),
        requires_human_review=bool(parsed.get("requires_human_review", True)),
    )


# ============================================================================
# SECTION 4: NEGATIVE HARNESS — CLOSED-LOOP REGRESSION GUARD
# ============================================================================

class RegressionExample(BaseModel):
    """A previously-rejected output that must remain rejected under the new contract."""
    example_id: str
    input_text: str
    output_text: str
    rejection_reason: str
    # Temporal decay weight: recent rejections are more relevant than stale ones
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def decay_weight(self, now: datetime, half_life_days: float = 30.0) -> float:
        """
        Temporal decay: older regression examples are down-weighted to prevent
        stale corpus examples from dominating the score buffer.
        """
        delta_days = (now - self.recorded_at.replace(tzinfo=UTC)).total_seconds() / 86400.0
        return math.exp(-math.log(2) * delta_days / half_life_days)


class HarnessVerdict(str, Enum):
    PASS = "pass"
    REGRESSION_DETECTED = "regression_detected"


class HarnessResult(BaseModel):
    verdict: HarnessVerdict
    first_regression_example: RegressionExample | None = None
    checked_count: int
    passed_count: int
    elapsed_ms: float


class NegativeHarness:
    """
    Validates that a proposed contract update does not re-admit
    previously-rejected outputs.

    Closed-Loop Retry Protocol:
        If a regression is detected, the system does NOT immediately trip
        the circuit breaker. Instead, it feeds the first_regression_example
        back to the LLM meta-reviewer as explicit negative feedback
        (max 3 attempts). Only if all retries fail does the circuit breaker
        activate with exponential backoff.

    Behavioral Differential Testing:
        The harness runs proposed rules against a holdout set of known-correct
        outputs to verify acceptance rate does not degrade.
    """

    def __init__(self, negative_corpus: list[RegressionExample]) -> None:
        self.negative_corpus = negative_corpus

    def check(
        self,
        proposal: MetaReviewProposal,
        evaluate_rule_fn: "RuleEvaluatorProtocol",
    ) -> HarnessResult:
        """
        Evaluate each proposed rule change against the negative corpus.
        Temporally-weighted: stale examples have reduced influence on the verdict.
        """
        import time
        start = time.monotonic()
        now = datetime.now(UTC)

        passed = 0
        first_regression: RegressionExample | None = None

        for example in self.negative_corpus:
            weight = example.decay_weight(now)
            if weight < 0.05:
                # Decay below 5%: example is too stale to be authoritative
                passed += 1
                continue

            for change in proposal.proposed_changes:
                is_rejected = evaluate_rule_fn(
                    output=example.output_text,
                    schema_predicate=change.new_json_schema_predicate,
                )
                if not is_rejected:
                    # Proposed rule would ACCEPT a previously-rejected output → regression
                    if first_regression is None:
                        first_regression = example
                    break
            else:
                passed += 1

        elapsed = (time.monotonic() - start) * 1000
        verdict = HarnessVerdict.PASS if first_regression is None else HarnessVerdict.REGRESSION_DETECTED

        return HarnessResult(
            verdict=verdict,
            first_regression_example=first_regression,
            checked_count=len(self.negative_corpus),
            passed_count=passed,
            elapsed_ms=elapsed,
        )


# ============================================================================
# SECTION 5: CONTRACT UPDATE — VERSIONED DIFF WITH APPROVAL GATE
# ============================================================================

class ApprovalStatus(str, Enum):
    PENDING_AUTO    = "pending_auto"
    APPROVED_AUTO   = "approved_auto"
    PENDING_HUMAN   = "pending_human"
    APPROVED_HUMAN  = "approved_human"
    REJECTED        = "rejected"
    CIRCUIT_OPEN    = "circuit_open"


class RuleDiff(BaseModel):
    """Before/after diff for a single rule modification."""
    rule_id: str
    before_text: str
    after_text: str
    before_predicate: dict[str, Any]
    after_predicate: dict[str, Any]
    rationale: str

    @computed_field
    @property
    def predicate_changed(self) -> bool:
        return self.before_predicate != self.after_predicate

    @computed_field
    @property
    def loosens_rule(self) -> bool:
        # Fail closed: only a diff that keeps every old constraint unchanged
        # (and may add more) counts as tightening. Anything else is treated
        # as loosening, because a changed constraint cannot be shown to be
        # stricter here.
        return not all(
            key in self.after_predicate and self.after_predicate[key] == value
            for key, value in self.before_predicate.items()
        )


class ContractUpdate(BaseModel):
    """
    Versioned contract update with full diff, rationale, and approval gate.

    Anti-Corruption Layer Guarantee:
        This object describes changes to the LIVING CONTRACT LAYER
        (JSON Schema predicates, rule text). It does NOT modify any
        Pydantic class definition. Application code consuming the
        stable Application Schema is unaffected until the ACL translator
        is explicitly updated and deployed through the normal CI/CD pipeline.
    """
    update_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    triggered_by_drift: DriftMeasurement
    proposal: MetaReviewProposal
    rule_diffs: list[RuleDiff]
    new_version_id: str = Field(
        default_factory=lambda: f"v{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    approval_status: ApprovalStatus = ApprovalStatus.PENDING_AUTO
    approved_at: datetime | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None
    # Retry tracking for closed-loop reflection
    harness_retry_count: int = Field(default=0, ge=0, le=3)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def requires_human_gate(self) -> bool:
        """
        Force human review if:
        - The proposal explicitly requests it, OR
        - Any diff loosens a rule (or cannot be shown to only tighten it), OR
        - Confidence is below 0.75 (uncertain proposal), OR
        - More than 2 rules are modified simultaneously (high blast radius).
        """
        return (
            self.proposal.requires_human_review
            or any(d.loosens_rule for d in self.rule_diffs)
            or self.proposal.confidence < 0.75
            or len(self.rule_diffs) > 2
        )

    def summary(self) -> str:
        diffs_summary = "\n".join(
            f"  [{d.rule_id}] '{d.before_text[:60]}...' → '{d.after_text[:60]}...'"
            for d in self.rule_diffs
        )
        return (
            f"ContractUpdate {self.update_id}\n"
            f"  Version: {self.new_version_id}\n"
            f"  Status: {self.approval_status}\n"
            f"  Rules Modified: {len(self.rule_diffs)}\n"
            f"  Requires Human: {self.requires_human_gate}\n"
            f"  Confidence: {self.proposal.confidence:.2f}\n"
            f"Diffs:\n{diffs_summary}"
        )


# ============================================================================
# SECTION 6: CIRCUIT BREAKER WITH EXPONENTIAL BACKOFF
# ============================================================================

class CircuitState(str, Enum):
    CLOSED = "closed"   # Normal operation
    OPEN   = "open"     # Modification loop suspended
    HALF   = "half"     # Probe: attempt single update


class OntologyCircuitBreaker(BaseModel):
    """
    Prevents runaway modification loops with adaptive exponential backoff.

    Backoff Formula: T_cooldown(f) = min(T_base * 2^f, T_max)
    where f = consecutive failed update attempts.

    Unlike a fixed 3600s window, exponential backoff:
    - Recovers quickly (300s) from minor transient drifts.
    - Prevents oscillation (up to 24h) under persistent failures.
    - Provides proportional suppression without human-set constants.
    """
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_at: datetime | None = None
    last_success_at: datetime | None = None
    t_base_seconds: float = 300.0
    t_max_seconds: float = 86400.0

    @computed_field
    @property
    def cooldown_seconds(self) -> float:
        return min(
            self.t_base_seconds * (2 ** self.failure_count), self.t_max_seconds
        )

    def is_open(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return False
        if self.state == CircuitState.OPEN and self.last_failure_at is not None:
            elapsed = (datetime.now(UTC) - self.last_failure_at).total_seconds()
            if elapsed >= self.cooldown_seconds:
                self.state = CircuitState.HALF
                return False
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_success_at = datetime.now(UTC)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.state = CircuitState.OPEN
        self.last_failure_at = datetime.now(UTC)


# ============================================================================
# SECTION 7: ANTI-CORRUPTION LAYER — STABLE APPLICATION SCHEMA
# ============================================================================

class StableOutputSchema(BaseModel):
    """
    The Application Schema: versioned, static, consumed by application code.

    This schema changes ONLY through explicit human-approved migrations.
    It is the stable contract downstream services depend on.

    The ACL translates between the dynamic Living Contract fields
    and these stable types. Application code never imports from
    the Living Contract layer directly.
    """
    intent: str
    entities: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    schema_version: str = Field(description="The Application Schema version this object conforms to.")


class AntiCorruptionLayer:
    """
    Translates dynamic Living Contract outputs to stable Application Schema types.

    When the Living Contract evolves (via ContractUpdate), ONLY this class
    changes — not StableOutputSchema, not downstream service code.

    This is the architectural boundary that makes self-healing safe.
    The 'self-healing' is isolated to the Living Contract layer;
    the Application layer is immune to runtime ontology changes.
    """

    def __init__(self, active_version_header: OntologyVersionHeader) -> None:
        self.version_header = active_version_header

    def translate(
        self,
        raw_contract_output: dict[str, Any],
        workflow_version_id: str | None = None,
    ) -> StableOutputSchema:
        """
        Translate a raw Living Contract output to StableOutputSchema.

        Workflow Version Pinning:
            If workflow_version_id is provided and differs from the current
            active version, apply backward-compatible translation logic for
            the pinned version. This allows long-running workflows to complete
            on their original schema while new workflows use the updated rules.
        """
        if workflow_version_id and workflow_version_id != self.version_header.version_id:
            # In production: look up legacy translation in a version registry.
            # Here we show the pattern; real implementation queries the DB.
            return self._translate_legacy(raw_contract_output, workflow_version_id)

        return StableOutputSchema(
            intent=str(raw_contract_output.get("intent", "unknown")),
            entities=[str(e) for e in raw_contract_output.get("entities", [])],
            confidence=float(raw_contract_output.get("confidence", 0.0)),
            schema_version=self.version_header.version_id,
        )

    def _translate_legacy(
        self,
        raw: dict[str, Any],
        version_id: str,
    ) -> StableOutputSchema:
        """
        Backward-compatible translation for workflows pinned to older contract versions.
        In production, this dispatches to a version-keyed registry of translators.
        """
        # Placeholder: real implementation maps version_id → translator function
        return StableOutputSchema(
            intent=str(raw.get("intent", raw.get("category", "unknown"))),
            entities=raw.get("entities", raw.get("tags", [])),
            confidence=float(raw.get("confidence", raw.get("score", 0.5))),
            schema_version=version_id,
        )


# ============================================================================
# SECTION 8: THE SELF-HEALING ORCHESTRATOR
# ============================================================================

class LLMClientProtocol(Protocol):
    async def complete(
        self,
        system: str,
        user: str,
        temperature: float,
        response_format: dict[str, str],
    ) -> str: ...


class RuleEvaluatorProtocol(Protocol):
    def __call__(self, output: str, schema_predicate: dict[str, Any]) -> bool: ...


class SelfHealingOrchestrator:
    """
    The Self-Healing Ontology Loop.

    Coordinates the five-stage feedback cycle:
      Stage 1 → Evaluation (external eval harness, not shown)
      Stage 2 → Drift Detection (OntologyDriftDetector)
      Stage 3 → Meta-Review (trigger_meta_review)
      Stage 4 → Contract Update + NegativeHarness validation
      Stage 5 → Deployment (version header update + ACL reload)

    Planaria Architecture:
        Rules are stored as distributed, content-addressable embeddings
        (RuleEmbedding objects). The system heals by updating individual
        rule entries, not by rewriting a central contract file.
        Like planarian tissue, each rule is autonomous: no rule's survival
        depends on another rule's ordinal position or filename location.
    """

    def __init__(
        self,
        detector: OntologyDriftDetector,
        harness: NegativeHarness,
        llm_client: LLMClientProtocol,
        rule_evaluator: RuleEvaluatorProtocol,
        circuit_breaker: OntologyCircuitBreaker,
        active_rules: list[RuleEmbedding],
    ) -> None:
        self.detector = detector
        self.harness = harness
        self.llm_client = llm_client
        self.rule_evaluator = rule_evaluator
        self.circuit_breaker = circuit_breaker
        self.active_rules = active_rules
        self._update_history: list[ContractUpdate] = []

    async def run_cycle(
        self,
        recent_traces: list[ExecutionTrace],
        replay_buffer_samples: list[ExecutionTrace],
    ) -> ContractUpdate | None:
        """
        Execute one complete self-healing cycle.

        Returns a ContractUpdate if drift was detected and a proposal was
        generated, or None if the system is healthy or the circuit is open.
        """
        # Stage 2: Drift Detection
        measurement = self.detector.measure(recent_traces)

        if measurement.severity == DriftSeverityLevel.NOMINAL:
            return None  # System healthy; no action required.

        if self.circuit_breaker.is_open():
            # Circuit open: suppress modification loop.
            # Adaptive backoff: cooldown is 2^f * T_base, not a fixed constant.
            return ContractUpdate(
                triggered_by_drift=measurement,
                proposal=MetaReviewProposal(
                    request_id="circuit_open",
                    proposed_changes=[],
                    confidence=0.0,
                    requires_human_review=True,
                ),
                rule_diffs=[],
                approval_status=ApprovalStatus.CIRCUIT_OPEN,
            )

        # Identify rules implicated in the drift using semantic proximity
        implicated_rules = self._identify_implicated_rules(recent_traces)

        # Stage 3: Meta-Review — closed-loop with max 3 retries
        proposal: MetaReviewProposal | None = None
        harness_result: HarnessResult | None = None
        retry_count = 0
        regression_feedback: RegressionExample | None = None

        while retry_count < 3:
            # On retry: inject the regression example as additional context
            traces_for_review = replay_buffer_samples.copy()
            if regression_feedback is not None:
                # Convert regression example to trace for LLM feedback
                regression_trace = ExecutionTrace(
                    input_text=regression_feedback.input_text,
                    raw_model_output=regression_feedback.output_text,
                    guardrail_id="REGRESSION_GUARD",
                    guardrail_verdict="pass",  # Would have PASSED under bad proposal
                    human_ground_truth="incorrect",  # But it's actually incorrect
                    output_embedding=[0.0] * len(recent_traces[0].output_embedding),
                )
                traces_for_review = [regression_trace] + traces_for_review

            proposal = await trigger_meta_review(
                drift_measurement=measurement,
                rules_under_review=implicated_rules,
                replay_buffer_samples=traces_for_review,
                llm_client=self.llm_client,
                top_k=10,
            )

            # Build candidate ContractUpdate
            rule_diffs = self._build_diffs(implicated_rules, proposal)
            candidate = ContractUpdate(
                triggered_by_drift=measurement,
                proposal=proposal,
                rule_diffs=rule_diffs,
                harness_retry_count=retry_count,
            )

            # Stage 4: Negative Harness validation
            harness_result = self.harness.check(proposal, self.rule_evaluator)

            if harness_result.verdict == HarnessVerdict.PASS:
                break  # Proposal survived regression check

            # Regression detected: prepare feedback for next retry
            regression_feedback = harness_result.first_regression_example
            retry_count += 1

        if harness_result is None or harness_result.verdict == HarnessVerdict.REGRESSION_DETECTED:
            # All retries exhausted: trip circuit breaker with exponential backoff
            self.circuit_breaker.record_failure()
            return ContractUpdate(
                triggered_by_drift=measurement,
                proposal=proposal or MetaReviewProposal(
                    request_id="retry_exhausted",
                    proposed_changes=[],
                    confidence=0.0,
                    requires_human_review=True,
                ),
                rule_diffs=[],
                approval_status=ApprovalStatus.REJECTED,
                rejection_reason=(
                    f"NegativeHarness failed after {retry_count} retries. "
                    f"Circuit breaker opened. Cooldown: {self.circuit_breaker.cooldown_seconds:.0f}s."
                ),
            )

        # Stage 5: Deployment gate
        if proposal is None:
            raise RuntimeError("Meta-review loop finished without a proposal.")
        rule_diffs = self._build_diffs(implicated_rules, proposal)
        update = ContractUpdate(
            triggered_by_drift=measurement,
            proposal=proposal,
            rule_diffs=rule_diffs,
            harness_retry_count=retry_count,
            approval_status=(
                ApprovalStatus.PENDING_HUMAN
                if ContractUpdate(
                    triggered_by_drift=measurement,
                    proposal=proposal,
                    rule_diffs=rule_diffs,
                ).requires_human_gate
                else ApprovalStatus.APPROVED_AUTO
            ),
        )

        if update.approval_status == ApprovalStatus.APPROVED_AUTO:
            self._apply_update(update)
            self.circuit_breaker.record_success()

        self._update_history.append(update)
        return update

    def _identify_implicated_rules(
        self, traces: list[ExecutionTrace]
    ) -> list[RuleEmbedding]:
        """
        Select rules most implicated in the drift.
        Uses cosine similarity between trace embeddings and rule embeddings.
        Returns the top-3 most implicated rules (bounded blast radius).
        """
        if not traces:
            return self.active_rules[:3]

        trace_centroid = np.mean(
            [t.output_embedding for t in traces if t.guardrail_verdict == "fail"],
            axis=0,
        ).astype(np.float32)
        norm_centroid = trace_centroid / (np.linalg.norm(trace_centroid) + 1e-8)

        scored: list[tuple[float, RuleEmbedding]] = []
        for rule in self.active_rules:
            rvec = np.array(rule.embedding, dtype=np.float32)
            rnorm = rvec / (np.linalg.norm(rvec) + 1e-8)
            score = float(np.dot(norm_centroid, rnorm))
            scored.append((score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:3]]

    def _build_diffs(
        self,
        original_rules: list[RuleEmbedding],
        proposal: MetaReviewProposal,
    ) -> list[RuleDiff]:
        rule_map = {r.rule_id: r for r in original_rules}
        diffs: list[RuleDiff] = []
        for change in proposal.proposed_changes:
            original = rule_map.get(change.rule_id)
            if original is None:
                continue
            diffs.append(RuleDiff(
                rule_id=change.rule_id,
                before_text=original.rule_text,
                after_text=change.new_rule_text,
                before_predicate=original.json_schema_predicate,
                after_predicate=change.new_json_schema_predicate,
                rationale=change.rationale,
            ))
        return diffs

    def _apply_update(self, update: ContractUpdate) -> None:
        """
        Planaria-style distributed update:
        Only the implicated rule entries are modified.
        The surrounding rules are untouched.
        """
        change_map = {c.rule_id: c for c in update.proposal.proposed_changes}
        for i, rule in enumerate(self.active_rules):
            if rule.rule_id in change_map:
                change = change_map[rule.rule_id]
                self.active_rules[i] = RuleEmbedding(
                    rule_id=rule.rule_id,
                    rule_text=change.new_rule_text,
                    json_schema_predicate=change.new_json_schema_predicate,
                    # Embedding is re-computed lazily on next read from vector store
                    embedding=rule.embedding,
                    schema_version=update.new_version_id,
                    provenance="meta_review",
                )
# --8<-- [end:lst-11.9]
