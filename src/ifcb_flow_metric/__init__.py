"""IFCB Flow Metric - Anomaly detection toolkit for IFCB data."""

__version__ = "1.1.0"

# Import key classes for convenient access
from ifcb_flow_metric.models.feature_extractor import FeatureExtractor
from ifcb_flow_metric.models.trainer import ModelTrainer
from ifcb_flow_metric.models.inference import Inferencer

__all__ = [
    "FeatureExtractor",
    "ModelTrainer",
    "Inferencer",
]
