# --8<-- [start:lst-9.5]
# src/ingestion/confidence_calibrator.py
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image, ImageStat
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class ImageQualityMetrics(BaseModel):
    """
    Deterministic image quality features.
    """
    resolution: tuple[int, int]
    dpi: int
    sharpness: float = Field(ge=0.0, description="Variance of Laplacian")
    contrast: float = Field(ge=0.0, le=1.0)
    noise_level: float = Field(ge=0.0, le=1.0)
    is_color: bool
    quality_score: float = Field(ge=0.0, le=1.0, description="Composite quality")

class ConfidenceCalibrator:
    """
    Adjusts VLM confidence scores based on:
    1. Input image quality (deterministic)
    2. Empirical accuracy on validation set (Platt Scaling)
    
    Prevents overconfidence on degraded inputs (The Fax Machine Problem).
    """
    
    def __init__(self, calibration_method: str = "platt"):
        self.calibration_method = calibration_method
        self.calibration_model = None
        self.is_trained = False
    
    def train_calibration(
        self,
        raw_confidences: np.ndarray,
        ground_truth_labels: np.ndarray
    ):
        """
        Trains the calibration model on a validation set.
        
        Args:
            raw_confidences: Model-reported confidence scores [0, 1]
            ground_truth_labels: Binary labels (1 = correct, 0 = incorrect)
        """
        if self.calibration_method == "platt":
            # Platt Scaling: fit logistic regression to map logits to probabilities
            self.calibration_model = LogisticRegression()
            # Convert confidences to logits
            logits = np.log(raw_confidences / (1 - raw_confidences + 1e-10))
            self.calibration_model.fit(logits.reshape(-1, 1), ground_truth_labels)
        
        elif self.calibration_method == "isotonic":
            # Isotonic Regression: non-parametric calibration
            self.calibration_model = IsotonicRegression(out_of_bounds="clip")
            self.calibration_model.fit(raw_confidences, ground_truth_labels)
        
        self.is_trained = True
    
    def analyze_image_quality(self, image: Image.Image) -> ImageQualityMetrics:
        """
        Computes deterministic quality metrics.
        """
        # Resolution
        width, height = image.size
        
        # DPI (if available)
        dpi = image.info.get("dpi", (72, 72))[0] if "dpi" in image.info else 72
        
        # Sharpness (Laplacian variance)
        gray = np.array(image.convert('L'))
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        # Use scipy.signal.convolve2d for proper 2D convolution
        from scipy.signal import convolve2d
        convolved = convolve2d(gray, laplacian, mode='same', boundary='symm')
        sharpness = float(np.var(convolved))
        
        # Contrast (std dev of pixel intensities)
        stat = ImageStat.Stat(image.convert('L'))
        contrast = stat.stddev[0] / 255.0
        
        # Noise level (high-frequency component)
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # High-frequency energy (outer 20% of spectrum)
        h, w = magnitude.shape
        mask = np.ones_like(magnitude)
        mask[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)] = 0
        high_freq_energy = np.sum(magnitude * mask)
        total_energy = np.sum(magnitude)
        noise_level = min(high_freq_energy / (total_energy + 1e-9), 1.0)
        
        # Composite quality score (tuned on validation set)
        quality_score = (
            0.3 * min(dpi / 300.0, 1.0) +
            0.3 * min(sharpness / 1000.0, 1.0) +
            0.2 * contrast +
            0.2 * (1.0 - noise_level)
        )
        
        return ImageQualityMetrics(
            resolution=(width, height),
            dpi=dpi,
            sharpness=sharpness,
            contrast=contrast,
            noise_level=noise_level,
            is_color=(image.mode == 'RGB'),
            quality_score=quality_score
        )
    
    def calibrate_confidence(
        self,
        raw_confidence: float,
        quality: ImageQualityMetrics
    ) -> float:
        """
        Two-stage calibration:
        1. Image quality penalty (deterministic)
        2. Logit calibration (learned from validation data)
        """
        # Stage 1: Image quality penalties
        if quality.dpi < 150:
            dpi_penalty = 0.5  # Severe penalty for low-DPI scans
        elif quality.dpi < 250:
            dpi_penalty = 0.8
        else:
            dpi_penalty = 1.0
        
        if quality.sharpness < 100:
            sharpness_penalty = 0.6  # Blurry images
        elif quality.sharpness < 500:
            sharpness_penalty = 0.85
        else:
            sharpness_penalty = 1.0
        
        if quality.noise_level > 0.5:
            noise_penalty = 0.7  # Noisy/faxed documents
        else:
            noise_penalty = 1.0
        
        # Apply quality penalties
        adjusted_confidence = (
            raw_confidence * dpi_penalty * sharpness_penalty * noise_penalty
        )
        adjusted_confidence = np.clip(adjusted_confidence, 1e-6, 1 - 1e-6)
        
        # Stage 2: Logit calibration (if trained)
        if self.is_trained and self.calibration_model:
            if self.calibration_method == "platt":
                # Convert to logit, apply calibration
                logit = np.log(
                    adjusted_confidence / (1 - adjusted_confidence + 1e-10))
                calibrated = self.calibration_model.predict_proba([[logit]])[0, 1]
            else:  # isotonic
                calibrated = self.calibration_model.predict([adjusted_confidence])[0]
            
            return min(float(calibrated), 1.0)
        
        return min(adjusted_confidence, 1.0)
# --8<-- [end:lst-9.5]
