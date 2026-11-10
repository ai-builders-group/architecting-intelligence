# --8<-- [start:lst-13.1]
from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Annotated, Literal, Protocol, Self, TypeAlias

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    model_validator,
)

Identifier = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:/-]+$"),
]
Digest = Annotated[str, Field(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
Version = Annotated[int, Field(strict=True, ge=1)]
Signature = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
# H8: authorization time comes from an injected clock. The integrator who wires
# the boundary chooses the time source; no field of the request can carry a time.
Clock: TypeAlias = Callable[[], datetime]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class SubjectRef(FrozenModel):
    subject_id: Identifier
    identity_version: Version


def _one_version_per_subject(refs: frozenset[SubjectRef]) -> frozenset[SubjectRef]:
    if len({ref.subject_id for ref in refs}) != len(refs):
        raise ValueError("each affected subject appears once")
    return refs


# H13: affected subjects are a set of (id, identity version) pairs, serialized
# sorted, so digests never depend on set order.
SubjectIds = Annotated[
    frozenset[SubjectRef],
    Field(min_length=1),
    AfterValidator(_one_version_per_subject),
    PlainSerializer(
        lambda refs: [r.model_dump() for r in sorted(refs, key=lambda r: r.subject_id)],
        return_type=list[dict[str, object]],
    ),
]


def _digest(value: FrozenModel) -> Digest:
    """The one canonicalization (H12): sorted-key compact JSON, sha256 prefixed."""
    encoded = json.dumps(
        value.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


class ToolOperation(str, Enum):
    SEND_MESSAGE = "send_message"
    CHANGE_PAYMENT_DETAILS = "change_payment_details"


class EffectClass(str, Enum):
    MESSAGE = "message"
    PAYMENT_MUTATION = "payment_mutation"


class ReversibilityClass(str, Enum):
    REVERSIBLE = "reversible"
    COMPENSABLE = "compensable"
    IRREVERSIBLE = "irreversible"


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


class ConditionKind(str, Enum):
    FINANCE_DUAL_CONTROL = "finance_dual_control"


class EscalationDestination(str, Enum):
    FINANCE_APPROVAL = "finance_approval"


class IdentityRole(str, Enum):
    REQUESTER = "requester"
    AUTHORIZING_PRINCIPAL = "authorizing_principal"
    RESOURCE_OWNER = "resource_owner"
    AFFECTED_SUBJECT = "affected_subject"


class Reason(str, Enum):
    EXPLICIT_AUTHORITY_MATCH = "explicit_authority_match"
    TRUSTED_TIME_UNAVAILABLE = "trusted_time_unavailable"
    REQUEST_NOT_CURRENT = "request_not_current"
    STATE_NOT_CURRENT = "state_not_current"
    STATE_UNAVAILABLE = "state_unavailable"
    STATE_REVOKED = "state_revoked"
    STATE_BINDING_MISMATCH = "state_binding_mismatch"
    AUTHORITY_NOT_CURRENT = "authority_not_current"
    SCOPE_MISMATCH = "scope_mismatch"
    POLICY_MISMATCH = "policy_mismatch"
    REVERSIBILITY_UNSUPPORTED = "reversibility_unsupported"
    REQUIRED_CONDITION_MISSING = "required_condition_missing"
    ESCALATION_UNAVAILABLE = "escalation_unavailable"
    ESCALATION_ROUTE_MISMATCH = "escalation_route_mismatch"
    ESCALATION_EXPIRED = "escalation_expired"
    DIGEST_MISMATCH = "digest_mismatch"
    SIGNATURE_INVALID = "signature_invalid"
    CONDITION_AMBIGUOUS = "condition_ambiguous"
    CAPABILITY_REFUSED = "capability_refused"
    MALFORMED_INPUT = "malformed_input"
    STATE_ISSUER_NOT_ADMITTED = "state_issuer_not_admitted"


class SendMessageArgs(FrozenModel):
    kind: Literal["send_message"] = "send_message"
    recipient_id: Identifier
    channel_id: Identifier
    content_digest: Digest


class PaymentDetailChangeArgs(FrozenModel):
    kind: Literal["change_payment_details"] = "change_payment_details"
    vendor_id: Identifier
    payment_token_digest: Digest
    expected_payment_profile_version: Version


ToolArguments: TypeAlias = Annotated[
    SendMessageArgs | PaymentDetailChangeArgs,
    Field(discriminator="kind"),
]


class RequestLineage(FrozenModel):
    """The delegation chain the request claims; lineage_digest covers exactly this."""

    authority_ref: Identifier
    authority_generation: Version
    revocation_generation: Version
    grant_id: Identifier
    grant_version: Version
    requester_id: Identifier
    requester_identity_version: Version
    claimed_principal_id: Identifier
    principal_identity_version: Version


_UNSEALED: Digest = "sha256:" + "0" * 64


class EffectRequest(FrozenModel):
    request_id: Identifier
    request_digest: Digest
    payload_digest: Digest
    lineage_digest: Digest
    authority_ref: Identifier
    authority_generation: Version
    revocation_generation: Version
    requester_id: Identifier
    requester_identity_version: Version
    claimed_principal_id: Identifier
    principal_identity_version: Version
    resource_owner_id: Identifier
    resource_owner_identity_version: Version
    affected_subject_ids: SubjectIds
    grant_id: Identifier
    grant_version: Version
    requested_capability: Identifier
    resource_id: Identifier
    expected_resource_version: Version
    operation: ToolOperation
    arguments: ToolArguments
    transaction_id: Identifier
    purpose: Identifier
    destination_id: Identifier
    effect_class: EffectClass
    context_digest: Digest
    reversibility: ReversibilityClass
    declared_at: AwareDatetime
    deadline: AwareDatetime

    @model_validator(mode="after")
    def operation_contract(self) -> Self:
        expected: tuple[type[FrozenModel], EffectClass]
        if self.operation is ToolOperation.SEND_MESSAGE:
            expected = (SendMessageArgs, EffectClass.MESSAGE)
        else:
            expected = (PaymentDetailChangeArgs, EffectClass.PAYMENT_MUTATION)

        if not isinstance(self.arguments, expected[0]):
            raise ValueError("operation and argument variant do not match")
        if self.effect_class is not expected[1]:
            raise ValueError("operation and effect class do not match")
        if self.declared_at >= self.deadline:
            raise ValueError("request deadline must follow declared time")
        return self

    @classmethod
    def sealed(cls, **fields: object) -> Self:
        """Build a request whose three digests are computed, not supplied
        (H10)."""
        lineage = RequestLineage(
            **{n: fields[n] for n in RequestLineage.model_fields}
        )
        draft = cls(
            **fields,
            request_digest=_UNSEALED,
            payload_digest=_digest(fields["arguments"]),
            lineage_digest=_digest(lineage),
        )
        return draft.model_copy(update={"request_digest": _digest(draft)})

    def digests_match(self) -> bool:
        lineage = RequestLineage(
            **{n: getattr(self, n) for n in RequestLineage.model_fields}
        )
        unsealed = self.model_copy(update={"request_digest": _UNSEALED})
        return (
            self.payload_digest == _digest(self.arguments)
            and self.lineage_digest == _digest(lineage)
            and self.request_digest == _digest(unsealed)
        )


class AuthorityGrant(FrozenModel):
    grant_id: Identifier
    version: Version
    authority_ref: Identifier
    authority_generation: Version
    revocation_generation: Version
    authorizing_principal_id: Identifier
    principal_identity_version: Version
    delegate_id: Identifier
    delegate_identity_version: Version
    resource_owner_id: Identifier
    resource_owner_identity_version: Version
    affected_subject_ids: SubjectIds
    capability: Identifier
    resource_id: Identifier
    expected_resource_version: Version
    operation: ToolOperation
    arguments_digest: Digest
    transaction_id: Identifier
    purpose: Identifier
    destination_id: Identifier
    effect_class: EffectClass
    context_digest: Digest
    not_before: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def active_window(self) -> Self:
        if self.not_before >= self.expires_at:
            raise ValueError("grant expiry must follow activation")
        return self


class ConsequencePolicy(FrozenModel):
    policy_id: Identifier
    version: Version
    operation: ToolOperation
    effect_class: EffectClass
    admitted_reversibility: tuple[ReversibilityClass, ...]
    required_condition_kind: ConditionKind | None
    admitted_condition_issuer_id: Identifier | None
    missing_condition_decision: Decision  # DENY or ESCALATE; ALLOW is rejected below
    escalation_destination: EscalationDestination | None
    escalation_deadline: AwareDatetime | None
    not_before: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def coherent_branch(self) -> Self:
        if self.not_before >= self.expires_at:
            raise ValueError("policy expiry must follow activation")
        if self.missing_condition_decision is Decision.ALLOW:
            raise ValueError("a missing condition can deny or escalate, never allow")
        if not self.admitted_reversibility:
            raise ValueError("policy must admit at least one reversibility class")
        ordered = tuple(sorted(set(self.admitted_reversibility), key=lambda item: item.value))
        if ordered != self.admitted_reversibility:
            raise ValueError("reversibility classes must be unique and canonically ordered")

        if self.required_condition_kind is None:
            if self.admitted_condition_issuer_id is not None:
                raise ValueError("an issuer requires a condition kind")
            if self.missing_condition_decision is not Decision.DENY:
                raise ValueError("cannot escalate when no condition is required")
            if self.escalation_destination is not None or self.escalation_deadline is not None:
                raise ValueError("unexpected escalation fields")
            return self

        if self.admitted_condition_issuer_id is None:
            raise ValueError("a required condition needs an admitted issuer")

        if self.missing_condition_decision is Decision.ESCALATE:
            if self.escalation_destination is None or self.escalation_deadline is None:
                raise ValueError("escalation requires destination and deadline")
            if self.escalation_deadline <= self.not_before:
                raise ValueError("escalation deadline must follow policy activation")
        elif self.escalation_destination is not None or self.escalation_deadline is not None:
            raise ValueError("deny policy cannot carry escalation fields")
        return self


class IndependentCondition(FrozenModel):
    condition_id: Identifier
    version: Version
    kind: ConditionKind
    issuer_id: Identifier
    request_record_digest: Digest
    grant_digest: Digest
    resource_id: Identifier
    operation: ToolOperation
    arguments_digest: Digest
    transaction_id: Identifier
    purpose: Identifier
    destination_id: Identifier
    effect_class: EffectClass
    context_digest: Digest
    issued_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def active_window(self) -> Self:
        if self.issued_at >= self.expires_at:
            raise ValueError("condition expiry must follow issuance")
        return self


class SignedCondition(FrozenModel):
    """H9: a condition counts only if its admitted issuer signed it (None =
    unsigned)."""

    condition: IndependentCondition
    signature: Signature | None = None


class IndependentConditionSet(FrozenModel):
    condition_set_id: Identifier
    version: Version
    conditions: tuple[SignedCondition, ...]

    @model_validator(mode="after")
    def canonical_order(self) -> Self:
        keys = tuple(
            (item.condition.condition_id, item.condition.version) for item in self.conditions
        )
        if len(keys) != len(set(keys)):
            raise ValueError("condition identities must be unique")
        if keys != tuple(sorted(keys)):
            raise ValueError("conditions must use canonical identity order")
        return self


class IdentityState(FrozenModel):
    role: IdentityRole
    identity_id: Identifier
    version: Version
    active: bool
    available: bool


class CurrentAuthorizationState(FrozenModel):
    snapshot_id: Identifier
    version: Version
    issuer_id: Identifier
    consistency_point_id: Identifier
    observed_at: AwareDatetime
    valid_until: AwareDatetime
    available: bool
    authority_generation: Version
    revocation_generation: Version
    authority_revoked: bool
    identities: tuple[IdentityState, ...]
    resource_id: Identifier
    resource_version: Version
    resource_available: bool
    active_grant_id: Identifier
    active_grant_version: Version
    active_grant_digest: Digest
    grant_available: bool
    active_policy_id: Identifier
    active_policy_version: Version
    active_policy_digest: Digest
    policy_available: bool
    active_condition_set_id: Identifier
    active_condition_set_version: Version
    active_condition_set_digest: Digest
    condition_set_available: bool
    escalation_service_available: bool
    active_escalation_destination: EscalationDestination | None

    @model_validator(mode="after")
    def snapshot_invariants(self) -> Self:
        if self.observed_at >= self.valid_until:
            raise ValueError("state lease must follow observation")
        # One row per role, except AFFECTED_SUBJECT: one row per affected subject.
        roles = tuple(item.role for item in self.identities)
        single = tuple(role for role in roles if role is not IdentityRole.AFFECTED_SUBJECT)
        if set(roles) != set(IdentityRole) or len(single) != len(set(single)):
            raise ValueError("snapshot must contain each identity role")
        keys = tuple((item.role.value, item.identity_id) for item in self.identities)
        if len(keys) != len(set(keys)) or keys != tuple(sorted(keys)):
            raise ValueError("identity states must be unique and in canonical order")
        return self


class SignedState(FrozenModel):
    """H9: the state is issued by the authority service, not supplied by the
    caller."""

    state: CurrentAuthorizationState
    signature: Signature | None = None


class SignatureVerifier(Protocol):
    def verify(self, issuer_id: str, kind: str, digest: str, signature: str) -> bool: ...


class HmacSignatures:
    """Issuer keys held by the verifier. Production would use asymmetric keys;
    HMAC keeps this listing standard-library only. `kind` separates the roles.
    HMAC is symmetric: whoever can verify can also sign, so this is not
    non-repudiable."""

    def __init__(self, keys: dict[str, bytes]) -> None:
        self._keys = dict(keys)

    def sign(self, issuer_id: str, kind: str, digest: str) -> str:
        message = f"{kind}\n{digest}".encode()
        return hmac.new(self._keys[issuer_id], message, sha256).hexdigest()

    def verify(self, issuer_id: str, kind: str, digest: str, signature: str) -> bool:
        if issuer_id not in self._keys:
            return False
        return hmac.compare_digest(self.sign(issuer_id, kind, digest), signature)


class EscalationDirective(FrozenModel):
    required_condition_kind: ConditionKind
    admitted_issuer_id: Identifier
    destination: EscalationDestination
    deadline: AwareDatetime


class DecisionEvidence(FrozenModel):
    request_id: Identifier
    request_record_digest: Digest
    grant_id: Identifier
    grant_version: Version
    grant_digest: Digest
    policy_id: Identifier
    policy_version: Version
    policy_digest: Digest
    condition_set_id: Identifier
    condition_set_version: Version
    condition_set_digest: Digest
    state_snapshot_id: Identifier
    state_snapshot_version: Version
    state_snapshot_digest: Digest
    evaluated_at: AwareDatetime | None


class DecisionIdentityMaterial(FrozenModel):
    canonicalization_version: Literal["ai.effect-decision.v1"] = "ai.effect-decision.v1"
    evidence: DecisionEvidence
    outcome: Decision
    reason: Reason
    escalation_directive_digest: Digest | None


class CapabilityIdentityMaterial(FrozenModel):
    canonicalization_version: Literal["ai.effect-capability.v1"] = "ai.effect-capability.v1"
    decision_id: Digest
    request_record_digest: Digest
    state_snapshot_digest: Digest
    evidence_digest: Digest
    expires_at: AwareDatetime


class SingleUseCapability(FrozenModel):
    """A use-limit claim, not a bearer token (H6). Its id is a digest of public
    fields, so anyone can recompute it. Single use is enforced only by
    CapabilityStore.consume, a server-side compare-and-swap."""

    capability_id: Digest
    decision_id: Digest
    request_record_digest: Digest
    state_snapshot_digest: Digest
    evidence_digest: Digest
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def identity_is_bound(self) -> Self:
        material = CapabilityIdentityMaterial(
            decision_id=self.decision_id,
            request_record_digest=self.request_record_digest,
            state_snapshot_digest=self.state_snapshot_digest,
            evidence_digest=self.evidence_digest,
            expires_at=self.expires_at,
        )
        if self.capability_id != _digest(material):
            raise ValueError("capability identity does not match its binding")
        return self


class AuthorizationDecision(FrozenModel):
    decision_id: Digest
    outcome: Decision
    reason: Reason
    evidence: DecisionEvidence
    escalation: EscalationDirective | None
    escalation_directive_digest: Digest | None
    capability: SingleUseCapability | None

    @model_validator(mode="after")
    def terminal_invariants(self) -> Self:
        directive_digest = _digest(self.escalation) if self.escalation is not None else None
        if directive_digest != self.escalation_directive_digest:
            raise ValueError("escalation directive digest mismatch")

        material = DecisionIdentityMaterial(
            evidence=self.evidence,
            outcome=self.outcome,
            reason=self.reason,
            escalation_directive_digest=self.escalation_directive_digest,
        )
        if self.decision_id != _digest(material):
            raise ValueError("decision identity does not match its evidence")

        if self.outcome is Decision.ALLOW:
            if self.capability is None or self.escalation is not None:
                raise ValueError("allow requires exactly one capability")
            if self.capability.decision_id != self.decision_id:
                raise ValueError("capability is bound to another decision")
            if self.capability.request_record_digest != self.evidence.request_record_digest:
                raise ValueError("capability is bound to another request")
            if self.capability.state_snapshot_digest != self.evidence.state_snapshot_digest:
                raise ValueError("capability is bound to another state snapshot")
            if self.capability.evidence_digest != _digest(self.evidence):
                raise ValueError("capability evidence binding mismatch")
        elif self.outcome is Decision.ESCALATE:
            if self.capability is not None or self.escalation is None:
                raise ValueError("escalate requires a directive and no capability")
        elif self.capability is not None or self.escalation is not None:
            raise ValueError("deny carries neither capability nor escalation")
        return self


class UseLimitRecord(FrozenModel):
    """The server-side record that makes a capability single-use (H6)."""

    capability_id: Digest
    max_uses: Version
    uses: Annotated[int, Field(strict=True, ge=0)]
    expires_at: AwareDatetime


class CapabilityStore(Protocol):
    def register(self, capability: SingleUseCapability, transaction_id: Identifier) -> bool:
        """Create the use-limit record (max_uses=1, uses=0). Refuse (False) a
        second capability for the same request or transaction."""

    def consume(self, capability_id: Digest, now: datetime) -> bool:
        """Atomic compare-and-swap uses: n -> n + 1. True at most once per limit."""


class InMemoryCapabilityStore:
    """Demo store: records are never evicted. Production uses an external store
    with expiry. Single use means one consume per capability, and one capability
    per request and per transaction; it does not make the effect itself idempotent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[Digest, UseLimitRecord] = {}
        self._issued: set[tuple[str, str]] = set()  # ("request"|"transaction", key)

    def register(self, capability: SingleUseCapability, transaction_id: Identifier) -> bool:
        keys = {("request", capability.request_record_digest), ("transaction", transaction_id)}
        with self._lock:
            if capability.capability_id in self._records:
                return True  # the same capability again; its use count is untouched
            if keys & self._issued:
                return False
            self._issued |= keys
            self._records[capability.capability_id] = UseLimitRecord(
                capability_id=capability.capability_id,
                max_uses=1,
                uses=0,
                expires_at=capability.expires_at,
            )
        return True

    def consume(self, capability_id: Digest, now: datetime) -> bool:
        with self._lock:  # compare and swap happen under one lock
            record = self._records.get(capability_id)
            if (
                record is None
                or record.uses >= record.max_uses
                or now >= record.expires_at
            ):
                return False
            self._records[capability_id] = record.model_copy(
                update={"uses": record.uses + 1}
            )
            return True


def consume_capability(
    capability: SingleUseCapability, store: CapabilityStore, clock: Clock
) -> bool:
    """The enforcement point calls this before the effect; False means do not act."""
    now = clock()
    if not _is_aware(now):
        return False
    return store.consume(capability.capability_id, now)


def _is_aware(value: datetime | None) -> bool:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


def _identity_state_matches(
    request: EffectRequest,
    grant: AuthorityGrant,
    state: CurrentAuthorizationState,
) -> bool:
    expected = {
        IdentityRole.REQUESTER: (
            request.requester_id,
            request.requester_identity_version,
            grant.delegate_id,
            grant.delegate_identity_version,
        ),
        IdentityRole.AUTHORIZING_PRINCIPAL: (
            request.claimed_principal_id,
            request.principal_identity_version,
            grant.authorizing_principal_id,
            grant.principal_identity_version,
        ),
        IdentityRole.RESOURCE_OWNER: (
            request.resource_owner_id,
            request.resource_owner_identity_version,
            grant.resource_owner_id,
            grant.resource_owner_identity_version,
        ),
    }
    by_role = {
        item.role: item
        for item in state.identities
        if item.role is not IdentityRole.AFFECTED_SUBJECT
    }
    subjects = {
        SubjectRef(subject_id=item.identity_id, identity_version=item.version): item
        for item in state.identities
        if item.role is IdentityRole.AFFECTED_SUBJECT
    }
    if not (request.affected_subject_ids == grant.affected_subject_ids == frozenset(subjects)):
        return False
    if not all(item.available and item.active for item in subjects.values()):
        return False
    for role, binding in expected.items():
        current = by_role[role]
        request_id, request_version, grant_id, grant_version = binding
        if not current.available or not current.active:
            return False
        if request_id != grant_id or current.identity_id != request_id:
            return False
        if request_version != grant_version or current.version != request_version:
            return False
    return True


def _scope_matches(
    request: EffectRequest,
    grant: AuthorityGrant,
    arguments_digest: Digest,
) -> bool:
    return (
        request.grant_id == grant.grant_id
        and request.grant_version == grant.version
        and request.authority_ref == grant.authority_ref
        and request.authority_generation == grant.authority_generation
        and request.revocation_generation == grant.revocation_generation
        and request.claimed_principal_id == grant.authorizing_principal_id
        and request.requester_id == grant.delegate_id
        and request.resource_owner_id == grant.resource_owner_id
        and request.affected_subject_ids == grant.affected_subject_ids
        and request.requested_capability == grant.capability
        and request.resource_id == grant.resource_id
        and request.expected_resource_version == grant.expected_resource_version
        and request.operation is grant.operation
        and arguments_digest == grant.arguments_digest
        and request.transaction_id == grant.transaction_id
        and request.purpose == grant.purpose
        and request.destination_id == grant.destination_id
        and request.effect_class is grant.effect_class
        and request.context_digest == grant.context_digest
    )


def _condition_matches(
    condition: IndependentCondition,
    request: EffectRequest,
    request_record_digest: Digest,
    grant_digest: Digest,
    arguments_digest: Digest,
    issuer_id: Identifier,
    now: datetime,
) -> bool:
    return (
        condition.issuer_id == issuer_id
        and condition.request_record_digest == request_record_digest
        and condition.grant_digest == grant_digest
        and condition.resource_id == request.resource_id
        and condition.operation is request.operation
        and condition.arguments_digest == arguments_digest
        and condition.transaction_id == request.transaction_id
        and condition.purpose == request.purpose
        and condition.destination_id == request.destination_id
        and condition.effect_class is request.effect_class
        and condition.context_digest == request.context_digest
        and condition.issued_at <= now < condition.expires_at
    )


def _signed_by(
    verifier: SignatureVerifier,
    issuer_id: Identifier,
    kind: str,
    digest: Digest,
    signature: Signature | None,
) -> bool:
    return signature is not None and verifier.verify(issuer_id, kind, digest, signature)


def _record(
    evidence: DecisionEvidence,
    outcome: Decision,
    reason: Reason,
    escalation: EscalationDirective | None = None,
    capability_expires_at: datetime | None = None,
) -> AuthorizationDecision:
    directive_digest = _digest(escalation) if escalation is not None else None
    decision_id = _digest(
        DecisionIdentityMaterial(
            evidence=evidence,
            outcome=outcome,
            reason=reason,
            escalation_directive_digest=directive_digest,
        )
    )

    capability: SingleUseCapability | None = None
    if outcome is Decision.ALLOW:
        if capability_expires_at is None:
            raise ValueError("allow requires a capability expiry")
        capability_material = CapabilityIdentityMaterial(
            decision_id=decision_id,
            request_record_digest=evidence.request_record_digest,
            state_snapshot_digest=evidence.state_snapshot_digest,
            evidence_digest=_digest(evidence),
            expires_at=capability_expires_at,
        )
        capability = SingleUseCapability(
            capability_id=_digest(capability_material),
            decision_id=decision_id,
            request_record_digest=evidence.request_record_digest,
            state_snapshot_digest=evidence.state_snapshot_digest,
            evidence_digest=_digest(evidence),
            expires_at=capability_expires_at,
        )

    return AuthorizationDecision(
        decision_id=decision_id,
        outcome=outcome,
        reason=reason,
        evidence=evidence,
        escalation=escalation,
        escalation_directive_digest=directive_digest,
        capability=capability,
    )


def _decide_effect_at(
    request: EffectRequest,
    grant: AuthorityGrant,
    policy: ConsequencePolicy,
    condition_set: IndependentConditionSet,
    signed_state: SignedState,
    state_issuer_id: Identifier,
    verifier: SignatureVerifier,
    boundary_now: datetime | None,
) -> AuthorizationDecision:
    state = signed_state.state
    request_record_digest = _digest(request)
    grant_digest = _digest(grant)
    policy_digest = _digest(policy)
    condition_set_digest = _digest(condition_set)
    state_digest = _digest(state)

    evidence = DecisionEvidence(
        request_id=request.request_id,
        request_record_digest=request_record_digest,
        grant_id=grant.grant_id,
        grant_version=grant.version,
        grant_digest=grant_digest,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        policy_digest=policy_digest,
        condition_set_id=condition_set.condition_set_id,
        condition_set_version=condition_set.version,
        condition_set_digest=condition_set_digest,
        state_snapshot_id=state.snapshot_id,
        state_snapshot_version=state.version,
        state_snapshot_digest=state_digest,
        evaluated_at=boundary_now if _is_aware(boundary_now) else None,
    )

    if not _is_aware(boundary_now):
        return _record(evidence, Decision.DENY, Reason.TRUSTED_TIME_UNAVAILABLE)
    now = boundary_now

    # H10: the three request digests are recomputable by anyone, so this seal only
    # catches an inconsistent request, not a caller who re-seals. The binding is the
    # signed state pinning the grant digest, and the condition's request_record_digest.
    if not request.digests_match():
        return _record(evidence, Decision.DENY, Reason.DIGEST_MISMATCH)
    # H9: the state's issuer is the one the operator admitted (never the state's
    # own
    # claim), and its signature must verify under that issuer's key.
    if state.issuer_id != state_issuer_id:
        return _record(evidence, Decision.DENY, Reason.STATE_ISSUER_NOT_ADMITTED)
    if not _signed_by(
        verifier, state_issuer_id, "state", state_digest, signed_state.signature
    ):
        return _record(evidence, Decision.DENY, Reason.SIGNATURE_INVALID)

    if request.declared_at > now or now >= request.deadline:
        return _record(evidence, Decision.DENY, Reason.REQUEST_NOT_CURRENT)
    if state.observed_at > now or now >= state.valid_until:
        return _record(evidence, Decision.DENY, Reason.STATE_NOT_CURRENT)
    if not (
        state.available
        and state.resource_available
        and state.grant_available
        and state.policy_available
        and state.condition_set_available
    ):
        return _record(evidence, Decision.DENY, Reason.STATE_UNAVAILABLE)
    if state.authority_revoked:
        return _record(evidence, Decision.DENY, Reason.STATE_REVOKED)
    if not (grant.not_before <= now < grant.expires_at):
        return _record(evidence, Decision.DENY, Reason.AUTHORITY_NOT_CURRENT)
    if not (policy.not_before <= now < policy.expires_at):
        return _record(evidence, Decision.DENY, Reason.POLICY_MISMATCH)

    state_binding_matches = (
        state.authority_generation == request.authority_generation
        and state.authority_generation == grant.authority_generation
        and state.revocation_generation == request.revocation_generation
        and state.revocation_generation == grant.revocation_generation
        and state.resource_id == request.resource_id
        and state.resource_version == request.expected_resource_version
        and state.active_grant_id == grant.grant_id
        and state.active_grant_version == grant.version
        and state.active_grant_digest == grant_digest
        and state.active_policy_id == policy.policy_id
        and state.active_policy_version == policy.version
        and state.active_policy_digest == policy_digest
        and state.active_condition_set_id == condition_set.condition_set_id
        and state.active_condition_set_version == condition_set.version
        and state.active_condition_set_digest == condition_set_digest
    )
    if not state_binding_matches or not _identity_state_matches(request, grant, state):
        return _record(evidence, Decision.DENY, Reason.STATE_BINDING_MISMATCH)

    arguments_digest = _digest(request.arguments)
    if not _scope_matches(request, grant, arguments_digest):
        return _record(evidence, Decision.DENY, Reason.SCOPE_MISMATCH)
    if policy.operation is not request.operation or policy.effect_class is not request.effect_class:
        return _record(evidence, Decision.DENY, Reason.POLICY_MISMATCH)
    if request.reversibility not in policy.admitted_reversibility:
        return _record(evidence, Decision.DENY, Reason.REVERSIBILITY_UNSUPPORTED)

    condition_expiry: datetime | None = None
    if policy.required_condition_kind is not None:
        issuer = policy.admitted_condition_issuer_id
        if issuer is None:
            return _record(evidence, Decision.DENY, Reason.POLICY_MISMATCH)

        relevant = tuple(
            item
            for item in condition_set.conditions
            if item.condition.kind is policy.required_condition_kind
        )
        if len(relevant) > 1:  # H11: ambiguity denies; it never reaches ESCALATE
            return _record(evidence, Decision.DENY, Reason.CONDITION_AMBIGUOUS)
        signed = relevant[0] if relevant else None
        condition = signed.condition if signed is not None else None
        if signed is not None and not _signed_by(
            verifier,
            issuer,  # the policy's admitted issuer, not the condition's own claim
            "condition",
            _digest(signed.condition),
            signed.signature,
        ):
            return _record(evidence, Decision.DENY, Reason.SIGNATURE_INVALID)
        condition_valid = (
            condition is not None
            and _condition_matches(
                condition,
                request,
                request_record_digest,
                grant_digest,
                arguments_digest,
                issuer,
                now,
            )
        )

        if not condition_valid:
            if policy.missing_condition_decision is Decision.DENY:
                return _record(
                    evidence,
                    Decision.DENY,
                    Reason.REQUIRED_CONDITION_MISSING,
                )

            destination = policy.escalation_destination
            escalation_limit = policy.escalation_deadline
            if destination is None or escalation_limit is None:
                return _record(evidence, Decision.DENY, Reason.POLICY_MISMATCH)
            if not state.escalation_service_available:
                return _record(evidence, Decision.DENY, Reason.ESCALATION_UNAVAILABLE)
            if state.active_escalation_destination is not destination:
                return _record(
                    evidence,
                    Decision.DENY,
                    Reason.ESCALATION_ROUTE_MISMATCH,
                )

            deadline = min(
                request.deadline,
                grant.expires_at,
                policy.expires_at,
                state.valid_until,
                escalation_limit,
            )
            if now >= deadline:
                return _record(evidence, Decision.DENY, Reason.ESCALATION_EXPIRED)

            directive = EscalationDirective(
                required_condition_kind=policy.required_condition_kind,
                admitted_issuer_id=issuer,
                destination=destination,
                deadline=deadline,
            )
            return _record(
                evidence,
                Decision.ESCALATE,
                Reason.REQUIRED_CONDITION_MISSING,
                escalation=directive,
            )

        condition_expiry = condition.expires_at

    expiry_candidates = [
        request.deadline,
        grant.expires_at,
        policy.expires_at,
        state.valid_until,
    ]
    if condition_expiry is not None:
        expiry_candidates.append(condition_expiry)

    capability_expiry = min(expiry_candidates)
    if now >= capability_expiry:
        return _record(evidence, Decision.DENY, Reason.AUTHORITY_NOT_CURRENT)

    return _record(
        evidence,
        Decision.ALLOW,
        Reason.EXPLICIT_AUTHORITY_MATCH,
        capability_expires_at=capability_expiry,
    )


def decide_effect(
    request: EffectRequest,
    grant: AuthorityGrant,
    policy: ConsequencePolicy,
    condition_set: IndependentConditionSet,
    signed_state: SignedState,
    *,
    state_issuer_id: Identifier,
    clock: Clock,
    verifier: SignatureVerifier,
    store: CapabilityStore,
) -> AuthorizationDecision:
    """Public boundary. Time comes from the injected `clock` and the admitted
    `state_issuer_id` from the operator's configuration, both wired by the
    integrator; neither is a field of the request. State and conditions must
    carry their issuers' signatures. A clock that raises propagates: no decision
    exists, so nothing is allowed."""
    decision = _decide_effect_at(
        request, grant, policy, condition_set, signed_state, state_issuer_id,
        verifier, clock()
    )
    if decision.outcome is Decision.ALLOW:
        capability = decision.capability
        if capability is None or not store.register(
            capability, request.transaction_id
        ):
            return _record(
                decision.evidence, Decision.DENY, Reason.CAPABILITY_REFUSED
            )
    return decision


class DecisionInputs(FrozenModel):
    request: EffectRequest
    grant: AuthorityGrant
    policy: ConsequencePolicy
    condition_set: IndependentConditionSet
    signed_state: SignedState


class MalformedInput(FrozenModel):
    outcome: Literal[Decision.DENY] = Decision.DENY
    reason: Reason = Reason.MALFORMED_INPUT
    locations: tuple[str, ...]


def decide_effect_from_json(
    raw: str | bytes,
    *,
    state_issuer_id: Identifier,
    clock: Clock,
    verifier: SignatureVerifier,
    store: CapabilityStore,
) -> AuthorizationDecision | MalformedInput:
    try:
        inputs = DecisionInputs.model_validate_json(raw)
    except ValidationError as error:  # H16: validation errors only, never Exception
        return MalformedInput(
            locations=tuple(".".join(str(part) for part in e["loc"]) for e in error.errors())
        )
    return decide_effect(
        inputs.request,
        inputs.grant,
        inputs.policy,
        inputs.condition_set,
        inputs.signed_state,
        state_issuer_id=state_issuer_id,
        clock=clock,
        verifier=verifier,
        store=store,
    )
# --8<-- [end:lst-13.1]
