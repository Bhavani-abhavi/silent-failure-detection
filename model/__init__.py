"""The monitored model. Trained once on a time prefix, then frozen."""

from model.baseline import BaselineModel, evaluate, train_baseline

__all__ = ["BaselineModel", "train_baseline", "evaluate"]
