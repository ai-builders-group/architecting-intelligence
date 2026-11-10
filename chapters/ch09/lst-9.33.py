# --8<-- [start:lst-9.33]
# path/to/offline_calibrator.py
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import pickle

class CalibrationParameters(BaseModel):
    """
    Static calibration parameters learned offline on Golden Dataset.
    Applied at runtime without re-training.
    """
    method: Literal["platt", "isotonic"] = "platt"
    
    # Platt Scaling parameters: P(rel) = sigmoid(A * sim + B)
    platt_A: float = Field(default=5.0, description="Slope parameter")
    platt_B: float = Field(default=-2.5, description="Intercept parameter")
    
    # Isotonic mapping (if method="isotonic")
    isotonic_x: list[float] = Field(default_factory=list)
    isotonic_y: list[float] = Field(default_factory=list)
    
    # Metadata
    trained_on_dataset: str = Field(description="Golden Dataset ID")
    trained_at: str = Field(description="Training timestamp")
    eval_metrics: dict = Field(default_factory=dict)

class StaticCalibrator:
    """
    Applies OFFLINE-trained calibration parameters at runtime.
    No training occurs during inference.
    """
    def __init__(self, params: CalibrationParameters):
        self.params = params
    
    def calibrate(self, raw_similarity: float) -> float:
        """
        Transform raw cosine similarity using STATIC parameters.
        
        Args:
            raw_similarity: Cosine similarity from vector search
        
        Returns:
            Calibrated probability P(relevant | similarity)
        """
        if self.params.method == "platt":
            # Sigmoid transformation: P = 1 / (1 + exp(-(A*x + B)))
            logit = self.params.platt_A * raw_similarity + self.params.platt_B
            return float(1 / (1 + np.exp(-logit)))
        
        elif self.params.method == "isotonic":
            # Piecewise linear interpolation
            return float(np.interp(
                raw_similarity,
                self.params.isotonic_x,
                self.params.isotonic_y
            ))
        
        raise ValueError(f"unknown method: {self.params.method}")

class CalibrationTrainer:
    """
    OFFLINE trainer: Run once on Golden Dataset, export parameters.
    This runs in your CI/CD pipeline, NOT in production.
    """
    @staticmethod
    def train_on_golden_dataset(
        similarities: np.ndarray,
        labels: np.ndarray,
        method: Literal["platt", "isotonic"] = "platt"
    ) -> CalibrationParameters:
        """
        Train calibration model on labeled evaluation set.
        
        Args:
            similarities: Raw cosine similarities from vector search
            labels: Binary relevance labels (1=relevant, 0=not)
            method: Calibration algorithm
        
        Returns:
            Frozen parameters for runtime use
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        from datetime import datetime, UTC
        
        if method == "platt":
            # Fit logistic regression
            model = LogisticRegression()
            model.fit(similarities.reshape(-1, 1), labels)
            
            return CalibrationParameters(
                method="platt",
                platt_A=float(model.coef_[0][0]),
                platt_B=float(model.intercept_[0]),
                trained_on_dataset="golden_v2.0",
                trained_at=datetime.now(UTC).isoformat(),
                eval_metrics={}  # Add precision/recall from validation
            )
        
        elif method == "isotonic":
            # Fit isotonic regression
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(similarities, labels)
            
            return CalibrationParameters(
                method="isotonic",
                isotonic_x=model.X_thresholds_.tolist(),
                isotonic_y=model.y_thresholds_.tolist(),
                trained_on_dataset="golden_v2.0",
                trained_at=datetime.now(UTC).isoformat(),
                eval_metrics={}
            )
# --8<-- [end:lst-9.33]
