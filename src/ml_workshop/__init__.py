"""Init for ``ml_workshop`` package."""

# Handle import of torch with extensions for XPUs (Intel GPUs).
from ml_workshop._xpu import torch

from ml_workshop._penguins import PenguinDataset
from ml_workshop._ellipse import EllipseDataset
