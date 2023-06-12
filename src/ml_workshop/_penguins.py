"""Penguins dataset."""
from torch.utils.data import Dataset

from pandas import DataFrame

from palmerpenguins import load_penguins


class PenguinDataset(Dataset):
    """Penguin dataset class.

    Parameters
    ----------
    input_keys : Sequence[str]
        The column titles to use in the input feature vectors.
    target_keys : Sequnce[str]
        The column titles to use in the target feature vectors.

    """

    def __init__(self):
        """Build ``PenguinDataset``."""
        self._data = _load_penguin_data()
        self._valid_split = self._data.groupby(b="species", n=20)


def _load_penguin_data() -> DataFrame:
    """Return the cleaned penguin data.

    Returns
    -------
    DataFrame
        The penguin dataset, with rows containing ``NaN``s dropped.

    """
    data = load_penguins()
    return (
        data.loc[~data.isna().any(axis=1)]
        .sort_values(by=sorted(data.keys()))
        .reset_index(drop=True)
    )
