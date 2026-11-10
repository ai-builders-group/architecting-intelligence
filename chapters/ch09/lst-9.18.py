# --8<-- [start:lst-9.18]
# path/to/relevance_calibrator.py
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np
from typing import Literal

class CalibratedRelevance(BaseModel):
    """
    Calibrated relevance score representing P(relevant | query, chunk).
    """
    raw_similarity: float = Field(ge=-1.0, le=1.0)
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    calibration_method: Literal["platt", "isotonic"]

class RelevanceCalibrator:
    """
    Calibrates vector similarity scores into probabilities using Platt Scaling.
    Trained on labeled relevance judgments from eval set.
    """
    def __init__(self, method: Literal["platt", "isotonic"] = "platt"):
        self.method = method
        self.calibration_model = None
        self.is_trained = False
    
    def train(
        self,
        raw_similarities: np.ndarray,
        relevance_labels: np.ndarray
    ):
        """
        Train calibration model on labeled data.
        
        Args:
            raw_similarities: Cosine similarity scores from vector search
            relevance_labels: Binary labels (1=relevant, 0=not relevant)
        """
        if self.method == "platt":
            # Platt form: one calibrator for the chapter, see lst-9.29
            # Transform similarities to "logits" (unbounded scale)
            # Using arctan to map [-1, 1] -> [-π/2, π/2]
            logits = np.arctan(raw_similarities) * 2  # Scale to ~[-π, π]
            
            self.calibration_model = LogisticRegression()
            self.calibration_model.fit(logits.reshape(-1, 1), relevance_labels)
        
        elif self.method == "isotonic":
            # Isotonic Regression: non-parametric monotonic calibration
            self.calibration_model = IsotonicRegression(out_of_bounds="clip")
            self.calibration_model.fit(raw_similarities, relevance_labels)
        
        self.is_trained = True
    
    def calibrate(self, raw_similarity: float) -> CalibratedRelevance:
        """
        Transform raw cosine similarity into calibrated probability.
        """
        if not self.is_trained:
            # Fallback: sigmoid transformation (uncalibrated)
            calibrated = 1 / (1 + np.exp(-5 * raw_similarity))
        else:
            if self.method == "platt":
                logit = np.arctan(raw_similarity) * 2
                calibrated = self.calibration_model.predict_proba([[logit]])[0, 1]
            else:  # isotonic
                calibrated = self.calibration_model.predict([raw_similarity])[0]
        
        return CalibratedRelevance(
            raw_similarity=raw_similarity,
            calibrated_probability=float(calibrated),
            calibration_method=self.method
        )
    
    def calibrate_batch(self, raw_similarities: np.ndarray) -> list[CalibratedRelevance]:
        """Batch calibration for efficiency."""
        if not self.is_trained:
            calibrated = 1 / (1 + np.exp(-5 * raw_similarities))
        else:
            if self.method == "platt":
                logits = np.arctan(raw_similarities) * 2
                calibrated = self.calibration_model.predict_proba(logits.reshape(-1, 1))[:, 1]
            else:
                calibrated = self.calibration_model.predict(raw_similarities)
        
        return [
            CalibratedRelevance(
                raw_similarity=float(raw_sim),
                calibrated_probability=float(cal_prob),
                calibration_method=self.method
            )
            for raw_sim, cal_prob in zip(raw_similarities, calibrated)
        ]
# --8<-- [end:lst-9.18]
