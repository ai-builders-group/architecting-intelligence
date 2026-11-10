# --8<-- [start:lst-10.21]
# architecting_intelligence/ch10/graph_writer.py
from datetime import datetime, UTC
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
import hashlib
import json
EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TECHNOLOGY"]
RelationType = Literal["WORKS_AT", "FOUNDED", "ACQUIRED", "LOCATED_IN",
                       "REPORTED_BY", "LEADS_TO", "CONFLICTS_WITH"]
class TripleWrite(BaseModel):
    model_config = ConfigDict(frozen=True)  # (3)
    tenant_id: str
    subject_id: str
    subject_label: EntityType  # (1)
    predicate: RelationType  # (1)
    object_id: str
    object_label: EntityType  # (1)
    chunk_id: str
    chunk_text: str
    evidence_hash: str = Field(description="SHA256 of source text")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    def to_cypher(self) -> tuple[str, dict]:
        parts = [self.tenant_id, self.subject_id, self.predicate,
                 self.object_id, self.evidence_hash]
        key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
        query = f"""MERGE (s:Entity:{self.subject_label}
                {{tenant_id: $tenant_id, id: $subject_id}})
        MERGE (o:Entity:{self.object_label}
                {{tenant_id: $tenant_id, id: $object_id}})
        MERGE (c:Chunk {{tenant_id: $tenant_id, id: $chunk_id}})
        MERGE (s)-[:MENTIONED_IN]->(c)
        MERGE (o)-[:MENTIONED_IN]->(c)
        SET s.name = $subject_id, o.name = $object_id, c.text = $chunk_text
        MERGE (s)-[r:{self.predicate} {{idempotency_key: $key}}]->(o)
        ON CREATE SET r.evidence_hash = $evidence_hash,
                      r.created_at = $timestamp"""  # (2)
        params = self.model_dump(include={
            "tenant_id", "subject_id", "object_id", "chunk_id",
            "chunk_text", "evidence_hash", "timestamp"})
        return query, params | {"key": key}
# --8<-- [end:lst-10.21]
