# --8<-- [start:lst-9.4]
# src/ingestion/table_verifier.py
from pydantic import BaseModel, Field
from typing import Literal

class VerificationReport(BaseModel):
    """
    Output of deterministic table verification.
    """
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    discrepancies: list[tuple[int, int, float]] = Field(
        default_factory=list,
        description="List of (row, col, discrepancy_amount)"
    )
    requires_reflection: bool
    requires_human: bool

class TableVerifier:
    """
    Deterministic verification of table arithmetic.
    """
    
    def verify(self, table: StructuredTable) -> VerificationReport:
        """
        Checks row sums and column sums against extracted totals.
        """
        errors = []
        discrepancies = []
        
        # Detect totals row (heuristic: last row or contains keyword "total")
        totals_row = self._detect_totals_row(table)
        
        if totals_row is not None:
            # Verify each column sum
            for col in range(table.num_cols):
                is_valid, discrepancy = table.verify_column_sum(col, totals_row)
                
                if not is_valid:
                    errors.append(
                        f"Column {col} sum mismatch: discrepancy={discrepancy:.2f}"
                    )
                    discrepancies.append((totals_row, col, discrepancy))
        
        # Detect totals column (heuristic: last column or contains keyword)
        totals_col = self._detect_totals_column(table)
        
        if totals_col is not None:
            for row in range(table.num_rows):
                is_valid, discrepancy = table.verify_row_sum(row, totals_col)
                
                if not is_valid:
                    errors.append(
                        f"Row {row} sum mismatch: discrepancy={discrepancy:.2f}"
                    )
                    discrepancies.append((row, totals_col, discrepancy))
        
        # Decision logic
        max_discrepancy = max([d for _, _, d in discrepancies], default=0.0)
        
        requires_reflection = len(errors) > 0 and max_discrepancy < 100.0
        requires_human = len(errors) > 0 and max_discrepancy >= 100.0
        
        return VerificationReport(
            is_valid=(len(errors) == 0),
            errors=errors,
            discrepancies=discrepancies,
            requires_reflection=requires_reflection,
            requires_human=requires_human
        )
    
    def _detect_totals_row(self, table: StructuredTable) -> int | None:
        """
        Heuristic: last row or first row containing 'total' keyword.
        """
        for row in range(table.num_rows):
            for col in range(table.num_cols):
                cell = table.get_cell(row, col)
                if cell and "total" in cell.value.lower():
                    return row
        
        # Default: last row
        return table.num_rows - 1 if table.num_rows > 0 else None
    
    def _detect_totals_column(self, table: StructuredTable) -> int | None:
        """
        Heuristic: last column or first column containing 'total' keyword.
        """
        for col in range(table.num_cols):
            for row in range(table.num_rows):
                cell = table.get_cell(row, col)
                if cell and "total" in cell.value.lower():
                    return col
        
        return None  # No totals column detected

async def extract_with_verification(
    image_bytes: bytes,
    page_num: int,
    image_quality: "ImageQualityMetrics",
    consensus_extractor: ConsensusExtractor,
    verifier: TableVerifier,
    reflection_fn: callable
) -> StructuredTable:
    """
    Full pipeline: cascade extraction → verification → reflection → escalation.
    """
    # Step 1: Active Cascade extraction
    consensus_result = await consensus_extractor.extract_with_cascade(
        image_bytes, page_num, image_quality
    )
    
    # Step 2: Deterministic verification
    verification = verifier.verify(consensus_result.consensus_table)
    
    # Step 3: Self-correction loop
    if verification.requires_reflection:
        # Trigger reflection prompt with error details
        corrected_table = await reflection_fn(
            image_bytes,
            consensus_result.consensus_table,
            verification.errors
        )
        
        # Re-verify
        verification = verifier.verify(corrected_table)
        
        if (verification.is_valid
                and consensus_result.agreement_level != "escalated"):
            return corrected_table
    
    # Step 4: HITL escalation
    if (
        not verification.is_valid
        or consensus_result.agreement_level == "escalated"
    ):
        raise HumanEscalationRequired(
            reason=f"Table verification failed on page {page_num}",
            errors=verification.errors,
            consensus_uncertainty=consensus_result.uncertainty_score,
            table_candidate=consensus_result.consensus_table
        )
    
    return consensus_result.consensus_table

class HumanEscalationRequired(Exception):
    """Raised when verification and reflection both fail."""
    def __init__(
        self,
        reason: str,
        errors: list[str],
        consensus_uncertainty: float,
        table_candidate: StructuredTable
    ):
        self.reason = reason
        self.errors = errors
        self.consensus_uncertainty = consensus_uncertainty
        self.table_candidate = table_candidate
        super().__init__(reason)
# --8<-- [end:lst-9.4]
