# --8<-- [start:lst-16.2]
# kl_divergence_monitor.py
from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Enums ───────────────────────────────────────────────────────────────────


class DriftSeverity(str, Enum):
    """
    Operational classification of KL divergence drift.

    Thresholds are empirically calibrated; a D_KL > 0.5 on held-out
    pre-training samples is a strong signal of catastrophic forgetting
    in the interference zone. See ADR-10.3-03 for threshold calibration.
    """

    NOMINAL = "nominal"       # D_KL < 0.1: Within expected fine-tuning variance
    ELEVATED = "elevated"     # 0.1 ≤ D_KL < 0.3: Monitor closely
    WARNING = "warning"       # 0.3 ≤ D_KL < 0.5: Consider reducing learning rate
    CRITICAL = "critical"     # D_KL ≥ 0.5: Halt fine-tuning; replay ratio too low


# ─── Protocols ───────────────────────────────────────────────────────────────


class ModelDistributionProtocol(Protocol):
    """
    Interface contract for a model that can return token log-probabilities.

    Implementations must be fault-tolerant: transient errors (OOM,
    HTTP 5xx from vLLM, CUDA device assertion) must raise
    InferenceTransientError, allowing the circuit breaker to retry
    without propagating failures to the training loop.
    """

    async def get_logits(
        self,
        text: str,
        max_new_tokens: int = 1,
    ) -> torch.Tensor:
        """
        Returns raw logits tensor of shape [vocab_size] for the next-token
        distribution over the input text.

        Raises:
            InferenceTransientError: For retryable errors (OOM, timeout, HTTP 5xx).
            InferencePermanentError: For non-retryable errors (schema mismatch, auth).
        """
        ...


class InferenceTransientError(Exception):
    """Retryable inference error (e.g., OOM, HTTP 5xx, CUDA timeout)."""


class InferencePermanentError(Exception):
    """Non-retryable inference error (e.g., auth failure, schema mismatch)."""


# ─── Schemas ─────────────────────────────────────────────────────────────────


class SharedSupportConfig(BaseModel):
    """
    Configuration for shared-support KL divergence computation.

    Mathematical Motivation:
    Computing D_KL(P_base || P_tuned) on disjoint support sets produces
    undefined values when P_base(i) > 0 and P_tuned(i) ≈ 0.
    Epsilon fallbacks introduce systematic bias and produce artifact spikes.

    The correct solution: define the support as the top-K tokens of P_base.
    Both distributions are then evaluated over the SAME K indices.
    Since P_tuned is a softmax over the full vocabulary, P_tuned(i) > 0
    for all i by construction, eliminating the support disjoint problem.
    """

    top_k_support: int = Field(
        default=1000,
        ge=100,
        le=50_000,
        description=(
            "Number of top tokens from P_base to define the shared support V^K_base. "
            "K=1000 captures >95% of the base model's probability mass for most "
            "natural language sequences while keeping computation tractable."
        ),
    )
    reference_corpus_size: int = Field(
        default=512,
        ge=32,
        description="Number of reference sequences used to compute the average D_KL.",
    )


class DriftMeasurement(BaseModel):
    """A single KL divergence measurement from the evaluation sidecar."""

    training_step: int
    kl_divergence: float = Field(ge=0.0, description="D_KL(P_base ‖ P_tuned) on shared support.")
    severity: DriftSeverity
    support_size_actual: int = Field(description="Actual |V^K_base| used (may be < top_k if vocab smaller).")
    evaluation_latency_ms: float
    measured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    circuit_breaker_open: bool = Field(
        default=False,
        description="True if measurement was skipped due to open circuit breaker.",
    )

    @classmethod
    def classify_severity(cls, kl: float) -> DriftSeverity:
        if kl < 0.1:
            return DriftSeverity.NOMINAL
        if kl < 0.3:
            return DriftSeverity.ELEVATED
        if kl < 0.5:
            return DriftSeverity.WARNING
        return DriftSeverity.CRITICAL


class CircuitBreakerConfig(BaseModel):
    """Configuration for the evaluation circuit breaker."""

    failure_threshold: int = Field(
        default=3,
        description="Consecutive failures before the circuit opens.",
    )
    recovery_timeout_seconds: float = Field(
        default=60.0,
        description="Time the circuit remains open before attempting recovery.",
    )
    max_retries_per_call: int = Field(
        default=2,
        description="Retry attempts before a single call counts as a failure.",
    )
    retry_backoff_seconds: float = Field(
        default=2.0,
        description="Exponential backoff base for retries.",
    )


# ─── KL_DivergenceMonitor ────────────────────────────────────────────────────


class KL_DivergenceMonitor:
    """
    Asynchronous KL divergence monitor for detecting catastrophic forgetting.

    Architecture:
        This monitor runs in the Asynchronous Evaluation Sidecar (separate
        inference cluster from the training process). It is NOT called
        inside the training loop. The training loop emits LoRA checkpoints
        to a queue; this class consumes them asynchronously.

    Mathematical Foundation:
        D_KL(P_base ‖ P_tuned) = Σ_{i ∈ V^K_base} P_base(i) log(P_base(i)/P_tuned(i))

        Support V^K_base is derived EXCLUSIVELY from the base model's top-K
        distribution over the reference corpus. This guarantees mathematical
        stability (no disjoint support, no epsilon hacks).

    Circuit Breaker:
        Transient inference failures (OOM, HTTP 5xx) open the circuit breaker.
        During open-circuit periods, the training job continues uninterrupted;
        observability degradation is logged but does NOT constitute a training
        failure. A missed measurement is preferable to a killed training run.
    """

    def __init__(
        self,
        base_model: ModelDistributionProtocol,
        tuned_model: ModelDistributionProtocol,
        support_config: SharedSupportConfig,
        circuit_config: CircuitBreakerConfig,
    ) -> None:
        self._base = base_model
        self._tuned = tuned_model
        self._support_config = support_config
        self._circuit_config = circuit_config

        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False
        self._circuit_opened_at: float = 0.0

        # Cached base model support (populated lazily on first evaluation)
        self._base_support_indices: torch.Tensor | None = None

    async def _call_with_retry(
        self,
        model: ModelDistributionProtocol,
        text: str,
    ) -> torch.Tensor:
        """
        Execute a model inference call with exponential backoff retry.
        Raises InferenceTransientError after max_retries to trigger circuit breaker logic.
        """
        last_error: Exception | None = None
        for attempt in range(self._circuit_config.max_retries_per_call + 1):
            try:
                return await model.get_logits(text)
            except InferenceTransientError as e:
                last_error = e
                if attempt < self._circuit_config.max_retries_per_call:
                    backoff = self._circuit_config.retry_backoff_seconds * (2**attempt)
                    logger.warning(
                        "Inference transient error (attempt %d/%d); retrying in %.1fs. Error: %s",
                        attempt + 1,
                        self._circuit_config.max_retries_per_call,
                        backoff,
                        str(e),
                    )
                    await asyncio.sleep(backoff)
            except InferencePermanentError:
                raise  # Do not retry permanent errors

        raise InferenceTransientError(
            f"Exhausted {self._circuit_config.max_retries_per_call} retries."
        ) from last_error

    def _is_circuit_open(self) -> bool:
        """Check circuit state, recovering after timeout."""
        if not self._circuit_open:
            return False
        elapsed = time.monotonic() - self._circuit_opened_at
        if elapsed >= self._circuit_config.recovery_timeout_seconds:
            logger.info(
                "Circuit breaker entering HALF-OPEN state after %.0fs.",
                elapsed,
            )
            self._circuit_open = False
            self._consecutive_failures = 0
            return False
        return True

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_config.failure_threshold:
            self._circuit_open = True
            self._circuit_opened_at = time.monotonic()
            logger.error(
                "Circuit breaker OPEN after %d consecutive inference failures. "
                "KL divergence monitoring suspended for %.0fs. "
                "Training run continues unaffected.",
                self._consecutive_failures,
                self._circuit_config.recovery_timeout_seconds,
            )

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open = False

    async def _build_base_support(
        self, reference_texts: list[str]
    ) -> torch.Tensor:
        """
        Compute the shared support V^K_base by averaging top-K indices
        across the reference corpus.

        Returns: LongTensor of shape [top_k_support] containing the
                 most probable token indices under the base model.
        """
        all_logits: list[torch.Tensor] = []
        for text in reference_texts[: self._support_config.reference_corpus_size]:
            try:
                logits = await self._call_with_retry(self._base, text)
                all_logits.append(logits)
            except InferenceTransientError:
                logger.warning("Skipping reference text during support construction.")

        if not all_logits:
            raise RuntimeError("No base model logits available to construct shared support.")

        # Average log-probabilities across reference corpus
        stacked = torch.stack(all_logits, dim=0)          # [N_ref, vocab_size]
        avg_log_probs = torch.log_softmax(stacked, dim=-1).mean(dim=0)  # [vocab_size]
        top_k_indices = torch.topk(
            avg_log_probs,
            k=min(self._support_config.top_k_support, avg_log_probs.shape[0]),
        ).indices
        logger.info(
            "Shared support V^K_base constructed: K=%d tokens from base model distribution.",
            top_k_indices.shape[0],
        )
        return top_k_indices

    async def measure(
        self,
        reference_texts: list[str],
        training_step: int,
    ) -> DriftMeasurement:
        """
        Compute D_KL(P_base ‖ P_tuned) on the shared support V^K_base.

        This method is called from the Asynchronous Evaluation Sidecar,
        NOT from the training loop. It does not block gradient computation.

        Mathematical guarantee:
            For all i ∈ V^K_base:
                P_base(i) > 0  (by definition: these are top-K of base)
                P_tuned(i) > 0 (softmax over full vocab is always positive)
            Therefore D_KL is finite and well-defined for all inputs.
        """
        t_start = time.monotonic()

        # Check circuit breaker
        if self._is_circuit_open():
            return DriftMeasurement(
                training_step=training_step,
                kl_divergence=float("nan"),
                severity=DriftSeverity.NOMINAL,  # Assume nominal when blind
                support_size_actual=0,
                evaluation_latency_ms=0.0,
                circuit_breaker_open=True,
            )

        try:
            # Lazily build shared support on first evaluation
            if self._base_support_indices is None:
                self._base_support_indices = await self._build_base_support(reference_texts)

            support = self._base_support_indices
            K = support.shape[0]

            kl_accumulator = 0.0
            n_evaluated = 0

            for text in reference_texts[: self._support_config.reference_corpus_size]:
                base_logits = await self._call_with_retry(self._base, text)
                tuned_logits = await self._call_with_retry(self._tuned, text)

                # Extract probabilities at SHARED SUPPORT indices only.
                # Both tensors are indexed by the SAME support, so the
                # K entries line up token for token.
                base_support_logits = base_logits[support]    # [K]
                tuned_support_logits = tuned_logits[support]  # [K]

                # Renormalize over the support slice to get valid distributions
                p_base = F.softmax(base_support_logits, dim=0)   # [K], sums to 1
                # over support
                log_p_tuned = F.log_softmax(tuned_support_logits, dim=0)  # [K]

                # D_KL(P_base || P_tuned) = Σ p_base * log(p_base / p_tuned)
                # F.kl_div expects log-probabilities as input (log_target=False)
                kl = F.kl_div(
                    log_p_tuned,
                    p_base,
                    reduction="sum",
                    log_target=False,
                ).item()

                kl_accumulator += kl
                n_evaluated += 1

            mean_kl = kl_accumulator / max(n_evaluated, 1)
            latency_ms = (time.monotonic() - t_start) * 1000.0

            self._record_success()
            severity = DriftMeasurement.classify_severity(mean_kl)

            if severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
                logger.warning(
                    "Drift severity=%s at step=%d: D_KL=%.4f. "
                    "Consider increasing replay_fraction or reducing learning rate.",
                    severity,
                    training_step,
                    mean_kl,
                )

            return DriftMeasurement(
                training_step=training_step,
                kl_divergence=mean_kl,
                severity=severity,
                support_size_actual=K,
                evaluation_latency_ms=latency_ms,
            )

        except InferenceTransientError as e:
            self._record_failure()
            logger.error(
                "KL divergence measurement failed at step=%d: %s. "
                "Circuit breaker failure count: %d/%d.",
                training_step,
                str(e),
                self._consecutive_failures,
                self._circuit_config.failure_threshold,
            )
            return DriftMeasurement(
                training_step=training_step,
                kl_divergence=float("nan"),
                severity=DriftSeverity.NOMINAL,
                support_size_actual=0,
                evaluation_latency_ms=(time.monotonic() - t_start) * 1000.0,
                circuit_breaker_open=self._circuit_open,
            )
# --8<-- [end:lst-16.2]
