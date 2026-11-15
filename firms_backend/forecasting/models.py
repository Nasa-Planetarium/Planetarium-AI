"""Save and load locally trained FIRMS forecast models without pickled code."""
from dataclasses import dataclass
import json
from pathlib import Path

from lightgbm import Booster
import numpy as np

MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"
MODEL_KIND = {"daily", "annual"}


@dataclass(frozen=True)
class ProbabilityCalibration:
    coefficients: np.ndarray
    intercept: float

    def predict_proba(self, features):
        logit = features @ self.coefficients + self.intercept
        probability = np.exp(-np.logaddexp(0, -logit))
        return np.column_stack((1 - probability, probability))


def save_bundle(root, kind, detector, intensity, calibration, trained_through):
    output = Path(root) / kind
    output.mkdir(parents=True, exist_ok=True)
    detector.save_model(output / "detection.txt")
    intensity.save_model(output / "frp.txt")
    metadata = {
        "kind": kind,
        "trained_through_utc": trained_through,
        "method": "lightgbm_calibrated",
        "calibration_features": ["raw_detection_logit", "forecast_lead"],
        "coefficients": calibration.coef_.ravel().tolist(),
        "intercept": float(calibration.intercept_[0]),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_bundle(kind, root=MODEL_DIRECTORY):
    if kind not in MODEL_KIND:
        raise ValueError("kind must be daily or annual")
    directory = Path(root) / kind
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["kind"] != kind or metadata["calibration_features"] != ["raw_detection_logit", "forecast_lead"]:
        raise ValueError(f"Unexpected calibration schema for {kind}")
    calibration = ProbabilityCalibration(np.asarray(metadata["coefficients"]), metadata["intercept"])
    return (Booster(model_file=str(directory / "detection.txt")),
            Booster(model_file=str(directory / "frp.txt")),
            calibration)
