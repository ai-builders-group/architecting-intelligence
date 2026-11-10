# --8<-- [start:lst-9.2]
# src/ingestion/vision_schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, UTC

class BoundingBox(BaseModel):
    """
    Normalized spatial coordinates for a cell or table.
    Coordinates are in [0, 1] relative to page dimensions.
    """
    x1: float = Field(ge=0.0, le=1.0, description="Left edge (normalized)")
    y1: float = Field(ge=0.0, le=1.0, description="Top edge (normalized)")
    x2: float = Field(ge=0.0, le=1.0, description="Right edge (normalized)")
    y2: float = Field(ge=0.0, le=1.0, description="Bottom edge (normalized)")
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: "BoundingBox") -> float:
        """
        Compute Intersection over Union (IoU) with another bounding box.
        Critical for cell alignment across models.
        """
        # Intersection coordinates
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)
        
        # No intersection
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        # Intersection area
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Union area
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0

class TableCell(BaseModel):
    """
    Atomic representation of a single table cell with spatial grounding.
    """
    row_index: int = Field(ge=0, description="0-indexed row position")
    col_index: int = Field(ge=0, description="0-indexed column position")
    value: str = Field(description="Raw text content")
    value_type: Literal["text", "number", "date", "currency", "empty"]
    is_header: bool = Field(default=False)
    confidence: float = Field(ge=0.0, le=1.0, description="OCR/VLM confidence")
    
    # Spatial grounding (CRITICAL for alignment)
    bbox: BoundingBox = Field(description="Normalized bounding box")
    
    # Computed numeric value (if applicable)
    numeric_value: Optional[float] = Field(default=None, description="Parsed number")
    
    def parse_numeric(self) -> Optional[float]:
        """
        Deterministic parsing with financial notation support.
        """
        if self.value_type not in ("number", "currency"):
            return None
        
        # Remove currency symbols, commas, whitespace
        cleaned = self.value.replace("$", "").replace(",", "").replace(" ", "")
        
        try:
            return float(cleaned)
        except ValueError:
            return None

class StructuredTable(BaseModel):
    """
    Complete table representation with validation logic.
    """
    table_id: str
    page_number: int
    cells: list[TableCell]
    num_rows: int = Field(ge=1)
    num_cols: int = Field(ge=1)
    
    # Table-level bounding box
    table_bbox: BoundingBox = Field(description="Enclosing bounding box for entire table")
    
    # Metadata
    table_type: Literal["financial", "inventory", "schedule", "comparison", "other"]
    has_totals_row: bool = Field(default=False, description="Contains computed sum row")
    extracted_by: str = Field(description="VLM model identifier")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """Retrieve cell by grid coordinates."""
        for cell in self.cells:
            if cell.row_index == row and cell.col_index == col:
                return cell
        return None
    
    def get_cell_by_bbox(
        self, bbox: BoundingBox, iou_threshold: float = 0.5
    ) -> Optional[TableCell]:
        """
        Retrieve cell by spatial coordinates using IoU matching.
        Critical for alignment when grid indices are unreliable.
        """
        best_match = None
        best_iou = 0.0
        
        for cell in self.cells:
            iou = cell.bbox.iou(bbox)
            if iou > iou_threshold and iou > best_iou:
                best_match = cell
                best_iou = iou
        
        return best_match
    
    def verify_row_sum(self, row_index: int, sum_col_index: int) -> tuple[bool, float]:
        """
        Deterministic verification: checks if a 'Total' cell equals sum of row.
        Returns (is_valid, discrepancy).
        """
        sum_cell = self.get_cell(row_index, sum_col_index)
        if not sum_cell or sum_cell.numeric_value is None:
            return (False, 0.0)
        
        # Sum all numeric cells in the row (excluding the sum cell itself)
        calculated_sum = 0.0
        for col in range(self.num_cols):
            if col == sum_col_index:
                continue
            cell = self.get_cell(row_index, col)
            if cell and cell.numeric_value is not None:
                calculated_sum += cell.numeric_value
        
        discrepancy = abs(sum_cell.numeric_value - calculated_sum)
        tolerance = 0.01 * max(abs(sum_cell.numeric_value), abs(calculated_sum), 1.0)
        
        return (discrepancy <= tolerance, discrepancy)
    
    def verify_column_sum(self, col_index: int, sum_row_index: int) -> tuple[bool, float]:
        """
        Deterministic verification: checks if a 'Total' cell equals sum of column.
        """
        sum_cell = self.get_cell(sum_row_index, col_index)
        if not sum_cell or sum_cell.numeric_value is None:
            return (False, 0.0)
        
        calculated_sum = 0.0
        for row in range(self.num_rows):
            if row == sum_row_index:
                continue
            cell = self.get_cell(row, col_index)
            if cell and cell.numeric_value is not None:
                calculated_sum += cell.numeric_value
        
        discrepancy = abs(sum_cell.numeric_value - calculated_sum)
        tolerance = 0.01 * max(abs(sum_cell.numeric_value), abs(calculated_sum), 1.0)
        
        return (discrepancy <= tolerance, discrepancy)

class PageExtraction(BaseModel):
    """
    Complete extraction result for a single page.
    """
    page_number: int
    content_type: Literal["prose", "table", "chart", "form", "mixed"]
    
    # Text content (for prose pages)
    text: Optional[str] = Field(default=None)
    
    # Structured tables
    tables: list[StructuredTable] = Field(default_factory=list)
    
    # Chart descriptions (for non-tabular visuals)
    chart_description: Optional[str] = Field(default=None)
    
    # Confidence and provenance
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extracted_by: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
# --8<-- [end:lst-9.2]
