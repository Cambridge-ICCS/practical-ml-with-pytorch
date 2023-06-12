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
        self._valid = self._data.groupby(by=["species", "sex"]).sample(
            n=10,
            random_state=123,
        )
        self._train = self._data.loc[~self._data.index.isin(self._valid.index)]


def _load_penguin_data() -> DataFrame:
    """Return the cleaned penguin data.

    Returns
    -------
    DataFrame
        The penguin dataset, with rows containing ``NaN``s dropped.

    """
    data = load_penguins()
    data = (
        data.loc[~data.isna().any(axis=1)]
        .sort_values(by=sorted(data.keys()))
        .reset_index(drop=True)
    )
    data.sex = (data.sex == "male").astype(float)
    return data
