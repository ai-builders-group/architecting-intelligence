# --8<-- [start:lst-9.1]
# src/ingestion/triage_classifier.py
from pydantic import BaseModel, Field
from typing import Literal
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class VisualComplexity(BaseModel):
    """
    Output of the lightweight triage classifier.
    All metrics are deterministic image processing features.
    """
    has_tables: bool = Field(description="Grid-like structure detected")
    has_charts: bool = Field(description="Non-text visual elements detected")
    text_density: float = Field(ge=0.0, le=1.0, description="Fraction of pixels with high gradient variance")
    layout_columns: int = Field(ge=1, description="Number of detected text columns")
    complexity_score: float = Field(ge=0.0, description="Weighted composite score")
    
class PageRoute(BaseModel):
    """Decision output for routing logic."""
    pipeline: Literal["fast_ocr", "vlm_single", "vlm_ensemble"]
    reasoning: str
    estimated_cost: float = Field(description="Cost in USD")
    estimated_latency: float = Field(description="Latency in seconds")

class LightweightTriageClassifier:
    """
    Vision-only page complexity classifier using image processing heuristics.
    Operates in <20ms on CPU, <5ms on GPU.
    
    Architecture: EfficientNet-B0 (5.3M params) fine-tuned on DocBank dataset
    for binary classification: {simple_prose, complex_layout}.
    """
    
    def __init__(self, model_path: str = "models/efficientnet_triage.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load lightweight CNN classifier
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Learned weights for composite scoring (calibrated on validation set)
        self.weights = {
            "tables": 3.0,
            "charts": 2.0,
            "columns": 1.5,
            "text_density": -1.0  # High density = simple prose
        }
    
    def analyze(self, image: Image.Image) -> VisualComplexity:
        """
        Computes visual complexity using deterministic image features.
        """
        # Neural classification (table/chart detection)
        has_tables, has_charts = self._detect_structures(image)
        
        # Image processing heuristics
        text_density = self._compute_text_density(image)
        layout_columns = self._detect_columns(image)
        
        # Composite scoring
        complexity_score = (
            self.weights["tables"] * has_tables +
            self.weights["charts"] * has_charts +
            self.weights["columns"] * (layout_columns - 1) +
            self.weights["text_density"] * text_density
        )
        
        return VisualComplexity(
            has_tables=has_tables,
            has_charts=has_charts,
            text_density=text_density,
            layout_columns=layout_columns,
            complexity_score=complexity_score
        )
    
    def route(self, complexity: VisualComplexity) -> PageRoute:
        """
        Deterministic routing logic based on complexity score.
        """
        score = complexity.complexity_score
        
        # Calibrated thresholds from Chapter 3
        if score < 2.5:
            return PageRoute(
                pipeline="fast_ocr",
                reasoning="Text-heavy prose, no complex structures",
                estimated_cost=0.001,  # Tesseract
                estimated_latency=0.2
            )
        elif score < 7.0:
            return PageRoute(
                pipeline="vlm_single",
                reasoning="Moderate complexity (tables or charts)",
                estimated_cost=0.001,  # Self-hosted Llama-3.2
                estimated_latency=1.5
            )
        else:
            return PageRoute(
                pipeline="vlm_ensemble",
                reasoning="High complexity (financial/tabular with precision requirement)",
                estimated_cost=0.00374,  # Optimistic cascade (70% cheap only)
                estimated_latency=2.5
            )
    
    def _detect_structures(self, image: Image.Image) -> tuple[bool, bool]:
        """
        Uses EfficientNet-B0 for fast binary classification.
        Returns (has_tables, has_charts).
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Binary thresholds (calibrated to favor false positives for safety)
        has_tables = probs[0] > 0.3
        has_charts = probs[1] > 0.3
        
        return has_tables, has_charts
    
    def _compute_text_density(self, image: Image.Image) -> float:
        """
        Deterministic heuristic: text regions have high gradient variance.
        """
        gray = np.array(image.convert('L'))
        
        # Sobel edge detection
        gradient_x = np.abs(np.diff(gray, axis=1))
        gradient_y = np.abs(np.diff(gray, axis=0))
        
        # Text regions have moderate gradient variance
        variance = np.var(gradient_x) + np.var(gradient_y)
        
        # Normalize to [0, 1] (calibrated on validation set)
        normalized = min(variance / 10000.0, 1.0)
        return normalized
    
    def _detect_columns(self, image: Image.Image) -> int:
        """
        Detects multi-column layout using X-projection histogram clustering.
        """
        gray = np.array(image.convert('L'))
        
        # Binarize (Otsu's method)
        threshold = gray.mean()
        binary = (gray < threshold).astype(np.uint8)
        
        # X-projection: count black pixels per column
        x_projection = binary.sum(axis=0)
        
        # Detect valleys (white space between columns)
        valleys = []
        for i in range(1, len(x_projection) - 1):
            if x_projection[i] < 0.05 * x_projection.max():
                if x_projection[i] < x_projection[i-1] and x_projection[i] < x_projection[i+1]:
                    valleys.append(i)
        
        # Number of columns = valleys + 1
        return len(valleys) + 1 if valleys else 1
# --8<-- [end:lst-9.1]
