"""Ellipse-drawing dataset."""

from torch.utils.data import Dataset
from torchvision.transforms import Compose


class EllipseDataset(Dataset):
    """Random ellipse-drawing data set."""

    def __init__(self):
        """Build ``EllipseDataset.``"""
        self._num_items = 500
        self._img_size = 128
