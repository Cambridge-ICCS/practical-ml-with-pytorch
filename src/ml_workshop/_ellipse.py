"""Ellipse-drawing dataset."""

from typing import Tuple, Optional, Any

from numpy.random import default_rng
from numpy.random import Generator
from numpy import uint8, ndarray, float32, array, full


from torch.utils.data import Dataset
from torchvision.transforms import Compose

from skimage.draw import ellipse  # pylint: disable=no-name-in-module


class EllipseDataset(Dataset):
    """Random ellipse-drawing data set.

    Parameters
    ----------
    training : bool, optional
        Are we training, or validating.

    """

    def __init__(
        self,
        input_tfms: Optional[Compose] = None,
        target_tfms: Optional[Compose] = None,
        training: bool = False,
    ):
        """Build ``EllipseDataset``."""
        self._num_items = 1000
        self._img_size = 128
        self._rng = default_rng(123 if training is True else 666)
        self._x_tfms = input_tfms
        self._y_tfms = target_tfms

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self._num_items

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return input-target pair ``idx``.

        Parameters
        ----------
        idx : int
            The index of the item in the dataset to be returned.

        Returns
        -------

        """
        image, target = _draw_ellipse(self._rng, self._img_size)

        if self._x_tfms is not None:
            image = self._x_tfms(image)

        if self._y_tfms is not None:
            target = self._y_tfms(target)

        return image, target


def _draw_ellipse(rng: Generator, img_size: int) -> Tuple[ndarray, ndarray]:
    """Draw an ellipse.

    Parameters
    ----------
    rng : Generator
        Numpy's default random number generator.
    img_size : int
        The size of the square image.

    Returns
    -------
    img : ndarray
        The image of the ellipse.
    coord_info : ndarray
        The coordinate information: [row, col, rad_row, rad_col]. The ellipse
        is centred at ``(row, col)``, and the vertical and horizontal raddii
        are ``(rad_row, rad_col)``.

    """
    row_rad, col_rad = rng.integers(4, img_size // 4, size=2)

    row, col = rng.integers(
        max(row_rad, col_rad),
        img_size - max(row_rad, col_rad),
        size=2,
    )

    rows, cols = ellipse(
        row,
        col,
        row_rad,
        col_rad,
        shape=(img_size, img_size),
    )

    img = full((img_size, img_size, 3), rng.integers(256, size=3), dtype=uint8)

    img[rows, cols] = rng.integers(256, size=3)
    coord_info = array([row, col, row_rad, col_rad], dtype=float32) / img_size

    return img, coord_info
