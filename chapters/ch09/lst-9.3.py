# --8<-- [start:lst-9.3]
# src/ingestion/consensus_extractor.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
import asyncio
import numpy as np
from scipy.optimize import linear_sum_assignment

class ConsensusResult(BaseModel):
    """
    Output of the Active Cascade consensus algorithm.
    """
    consensus_table: StructuredTable
    uncertainty_score: float = Field(ge=0.0, description="Weighted std dev across models")
    agreement_level: Literal["unanimous", "majority", "escalated"]
    used_expensive_model: bool = Field(description="True if Tier 3 was invoked")
    divergent_cells: list[tuple[int, int]] = Field(
        default_factory=list,
        description="List of (row, col) tuples with high uncertainty"
    )
    cost_usd: float = Field(description="Total inference cost for this page")

class ConsensusExtractor:
    """
    Active Cascade architecture for vision extraction.
    Runs 2× Llama-3.2-Vision (cheap, self-hosted) first.
    Only invokes GPT-4o (expensive, tiebreaker) on disagreement.
    """
    
    def __init__(
        self,
        llama_client_a: callable,
        llama_client_b: callable,
        gpt4o_client: callable,
        ambiguity_threshold: float = 0.10,
        iou_threshold: float = 0.85,
        calibrator: Optional["ConfidenceCalibrator"] = None
    ):
        self.llama_a = llama_client_a
        self.llama_b = llama_client_b
        self.gpt4o = gpt4o_client
        self.ambiguity_threshold = ambiguity_threshold
        self.iou_threshold = iou_threshold
        self.calibrator = calibrator
    
    async def extract_with_cascade(
        self,
        image_bytes: bytes,
        page_num: int,
        image_quality: "ImageQualityMetrics"
    ) -> ConsensusResult:
        """
        Orchestrates the Active Cascade: cheap models first, expensive on demand.
        """
        cost = 0.0
        
        # Step 1: Parallel extraction with cheap models
        task_a = self.llama_a(image_bytes, page_num)
        task_b = self.llama_b(image_bytes, page_num)
        
        result_a, result_b = await asyncio.gather(task_a, task_b)
        cost += 2 * 0.001  # $0.001 per self-hosted inference
        
        # Step 2: Calibrate confidence scores based on image quality
        if self.calibrator:
            result_a.extraction_confidence = self.calibrator.calibrate_confidence(
                result_a.extraction_confidence, image_quality
            )
            result_b.extraction_confidence = self.calibrator.calibrate_confidence(
                result_b.extraction_confidence, image_quality
            )
        
        # Step 3: Spatial alignment using Hungarian Algorithm
        if result_a.tables and result_b.tables:
            table_a = result_a.tables[0]
            table_b = result_b.tables[0]
            
            aligned_pairs, mean_iou = self._align_cells_spatial(table_a, table_b)
            
            # Step 4: Check for agreement
            consensus_table, uncertainty, divergent_cells = self._compute_consensus_spatial(
                aligned_pairs, table_a, table_b, result_a.extraction_confidence, result_b.extraction_confidence
            )
            
            # Step 5: Deterministic verification (cross-footing)
            verifier = TableVerifier()
            verification = verifier.verify(consensus_table)
            
            # Step 6: Decision gate - do we need the expensive model?
            needs_escalation = (
                mean_iou < self.iou_threshold or
                not verification.is_valid or
                uncertainty > self.ambiguity_threshold or
                min(
                    result_a.extraction_confidence, result_b.extraction_confidence
                ) < 0.7
            )
            
            if needs_escalation:
                # Step 7: Invoke expensive tiebreaker
                result_gpt = await self.gpt4o(image_bytes, page_num)
                cost += self.config.c_vlm
                
                if self.calibrator:
                    result_gpt.extraction_confidence = self.calibrator.calibrate_confidence(
                        result_gpt.extraction_confidence, image_quality
                    )
                
                # Re-compute consensus with all three models
                table_gpt = result_gpt.tables[0]
                aligned_all = self._align_three_models(table_a, table_b, table_gpt)
                consensus_table, uncertainty, divergent_cells = self._compute_consensus_three(
                    aligned_all,
                    table_a,
                    result_a.extraction_confidence,
                    result_b.extraction_confidence,
                    result_gpt.extraction_confidence
                )
                
                agreement_level = "escalated"
                used_expensive = True
            else:
                agreement_level = "unanimous"
                used_expensive = False
        else:
            # Fallback: no tables detected
            consensus_table = StructuredTable(
                table_id=f"page_{page_num}",
                page_number=page_num,
                cells=[],
                num_rows=0,
                num_cols=0,
                table_bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
                table_type="other",
                extracted_by="consensus_cascade"
            )
            uncertainty = 0.0
            divergent_cells = []
            agreement_level = "unanimous"
            used_expensive = False
        
        return ConsensusResult(
            consensus_table=consensus_table,
            uncertainty_score=uncertainty,
            agreement_level=agreement_level,
            used_expensive_model=used_expensive,
            divergent_cells=divergent_cells,
            cost_usd=cost
        )
    
    def _align_cells_spatial(
        self,
        table_a: StructuredTable,
        table_b: StructuredTable
    ) -> tuple[list[tuple[TableCell, TableCell]], float]:
        """
        Uses the Hungarian Algorithm to find optimal cell matching via IoU.
        Returns (aligned_pairs, mean_iou).
        """
        cells_a = table_a.cells
        cells_b = table_b.cells
        
        # Build cost matrix (1 - IoU)
        n_a = len(cells_a)
        n_b = len(cells_b)
        cost_matrix = np.zeros((n_a, n_b))
        
        for i, cell_a in enumerate(cells_a):
            for j, cell_b in enumerate(cells_b):
                iou = cell_a.bbox.iou(cell_b.bbox)
                cost_matrix[i, j] = 1.0 - iou
        
        # Apply Hungarian Algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract aligned pairs with sufficient IoU
        aligned_pairs = []
        iou_scores = []
        
        for i, j in zip(row_indices, col_indices):
            iou = 1.0 - cost_matrix[i, j]
            if iou >= 0.5:  # Minimum threshold for valid match
                aligned_pairs.append((cells_a[i], cells_b[j]))
                iou_scores.append(iou)
        
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0
        
        return aligned_pairs, mean_iou
    
    def _compute_consensus_spatial(
        self,
        aligned_pairs: list[tuple[TableCell, TableCell]],
        table_a: StructuredTable,
        table_b: StructuredTable,
        conf_a: float,
        conf_b: float
    ) -> tuple[StructuredTable, float, list[tuple[int, int]]]:
        """
        Computes weighted consensus for spatially aligned cells.
        """
        consensus_cells = []
        divergent_cells = []
        uncertainties = []
        
        for cell_a, cell_b in aligned_pairs:
            # Weighted consensus for numeric values
            if cell_a.value_type in ("number", "currency") and cell_b.value_type in ("number", "currency"):
                val_a = cell_a.parse_numeric()
                val_b = cell_b.parse_numeric()
                
                if val_a is not None and val_b is not None:
                    # Weighted mean
                    consensus_val = (conf_a * val_a + conf_b * val_b) / (conf_a + conf_b)
                    
                    # Uncertainty (weighted variance)
                    variance = (conf_a * (val_a - consensus_val)**2 + conf_b * (val_b - consensus_val)**2) / (conf_a + conf_b)
                    uncertainty = np.sqrt(variance)
                    
                    relative_uncertainty = uncertainty / (abs(consensus_val) + 1e-6)
                    
                    if relative_uncertainty > self.ambiguity_threshold:
                        divergent_cells.append((cell_a.row_index, cell_a.col_index))
                    
                    uncertainties.append(uncertainty)
                    
                    # Use average bounding box
                    consensus_bbox = BoundingBox(
                        x1=(cell_a.bbox.x1 + cell_b.bbox.x1) / 2,
                        y1=(cell_a.bbox.y1 + cell_b.bbox.y1) / 2,
                        x2=(cell_a.bbox.x2 + cell_b.bbox.x2) / 2,
                        y2=(cell_a.bbox.y2 + cell_b.bbox.y2) / 2
                    )
                    
                    consensus_cells.append(TableCell(
                        row_index=cell_a.row_index,
                        col_index=cell_a.col_index,
                        value=str(consensus_val),
                        value_type="number",
                        numeric_value=consensus_val,
                        confidence=(conf_a + conf_b) / 2,
                        bbox=consensus_bbox,
                        is_header=cell_a.is_header
                    ))
                else:
                    # Use the one with non-None value
                    consensus_cells.append(cell_a if val_a is not None else cell_b)
            else:
                # Text consensus: use the one with higher confidence
                consensus_cells.append(cell_a if conf_a > conf_b else cell_b)
        
        # Construct consensus table
        consensus_table = StructuredTable(
            table_id=table_a.table_id,
            page_number=table_a.page_number,
            cells=consensus_cells,
            num_rows=max(c.row_index for c in consensus_cells) + 1 if consensus_cells else 0,
            num_cols=max(c.col_index for c in consensus_cells) + 1 if consensus_cells else 0,
            table_bbox=table_a.table_bbox,
            table_type=table_a.table_type,
            extracted_by="consensus_cascade"
        )
        
        global_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.0
        
        return consensus_table, global_uncertainty, divergent_cells
    
    def _align_three_models(
        self,
        table_a: StructuredTable,
        table_b: StructuredTable,
        table_c: StructuredTable
    ) -> list[tuple[Optional[TableCell], Optional[TableCell], Optional[TableCell]]]:
        """
        Three-way alignment: align A-B, then align consensus to C.
        """
        # Align A and B
        pairs_ab, _ = self._align_cells_spatial(table_a, table_b)
        
        # Create virtual "consensus" cells from A-B pairs
        consensus_cells_ab = []
        for cell_a, cell_b in pairs_ab:
            avg_bbox = BoundingBox(
                x1=(cell_a.bbox.x1 + cell_b.bbox.x1) / 2,
                y1=(cell_a.bbox.y1 + cell_b.bbox.y1) / 2,
                x2=(cell_a.bbox.x2 + cell_b.bbox.x2) / 2,
                y2=(cell_a.bbox.y2 + cell_b.bbox.y2) / 2
            )
            consensus_cells_ab.append((cell_a, cell_b, avg_bbox))
        
        # Align consensus_ab to C
        cells_c = table_c.cells
        n_ab = len(consensus_cells_ab)
        n_c = len(cells_c)
        cost_matrix = np.zeros((n_ab, n_c))
        
        for i, (_, _, bbox_ab) in enumerate(consensus_cells_ab):
            for j, cell_c in enumerate(cells_c):
                iou = bbox_ab.iou(cell_c.bbox)
                cost_matrix[i, j] = 1.0 - iou
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        aligned_triples = []
        for i, j in zip(row_indices, col_indices):
            iou = 1.0 - cost_matrix[i, j]
            if iou >= 0.5:
                cell_a, cell_b, _ = consensus_cells_ab[i]
                cell_c = cells_c[j]
                aligned_triples.append((cell_a, cell_b, cell_c))
        
        return aligned_triples
    
    def _compute_consensus_three(
        self,
        aligned_triples: list[tuple[TableCell, TableCell, TableCell]],
        table_a: StructuredTable,
        conf_a: float,
        conf_b: float,
        conf_c: float
    ) -> tuple[StructuredTable, float, list[tuple[int, int]]]:
        """
        Three-way weighted consensus.
        """
        consensus_cells = []
        divergent_cells = []
        uncertainties = []
        
        for cell_a, cell_b, cell_c in aligned_triples:
            if all(c.value_type in ("number", "currency") for c in [cell_a, cell_b, cell_c]):
                val_a = cell_a.parse_numeric()
                val_b = cell_b.parse_numeric()
                val_c = cell_c.parse_numeric()
                
                if all(v is not None for v in [val_a, val_b, val_c]):
                    # Weighted mean
                    total_conf = conf_a + conf_b + conf_c
                    consensus_val = (conf_a * val_a + conf_b * val_b + conf_c * val_c) / total_conf
                    
                    # Uncertainty
                    variance = (
                        conf_a * (val_a - consensus_val)**2 +
                        conf_b * (val_b - consensus_val)**2 +
                        conf_c * (val_c - consensus_val)**2
                    ) / total_conf
                    uncertainty = np.sqrt(variance)
                    
                    relative_uncertainty = uncertainty / (abs(consensus_val) + 1e-6)
                    
                    if relative_uncertainty > self.ambiguity_threshold:
                        divergent_cells.append((cell_a.row_index, cell_a.col_index))
                    
                    uncertainties.append(uncertainty)
                    
                    consensus_bbox = BoundingBox(
                        x1=(cell_a.bbox.x1 + cell_b.bbox.x1 + cell_c.bbox.x1) / 3,
                        y1=(cell_a.bbox.y1 + cell_b.bbox.y1 + cell_c.bbox.y1) / 3,
                        x2=(cell_a.bbox.x2 + cell_b.bbox.x2 + cell_c.bbox.x2) / 3,
                        y2=(cell_a.bbox.y2 + cell_b.bbox.y2 + cell_c.bbox.y2) / 3
                    )
                    
                    consensus_cells.append(TableCell(
                        row_index=cell_a.row_index,
                        col_index=cell_a.col_index,
                        value=str(consensus_val),
                        value_type="number",
                        numeric_value=consensus_val,
                        confidence=total_conf / 3,
                        bbox=consensus_bbox,
                        is_header=cell_a.is_header
                    ))
                else:
                    # Unparseable value: keep the most confident cell
                    consensus_cells.append(max(
                        ((cell_a, conf_a), (cell_b, conf_b), (cell_c, conf_c)),
                        key=lambda pair: pair[1]
                    )[0])
            else:
                # Text consensus: use the one with highest confidence
                consensus_cells.append(max(
                    ((cell_a, conf_a), (cell_b, conf_b), (cell_c, conf_c)),
                    key=lambda pair: pair[1]
                )[0])
        
        consensus_table = StructuredTable(
            table_id=table_a.table_id,
            page_number=table_a.page_number,
            cells=consensus_cells,
            num_rows=max(c.row_index for c in consensus_cells) + 1 if consensus_cells else 0,
            num_cols=max(c.col_index for c in consensus_cells) + 1 if consensus_cells else 0,
            table_bbox=table_a.table_bbox,
            table_type=table_a.table_type,
            extracted_by="consensus_cascade_3way"
        )
        
        global_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.0
        
        return consensus_table, global_uncertainty, divergent_cells
# --8<-- [end:lst-9.3]
