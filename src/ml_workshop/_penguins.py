"""Penguins dataset."""
from typing import Optional, List, Dict, Tuple, Any

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from pandas import DataFrame


from palmerpenguins import load_penguins

# pylint: disable=too-many-arguments


class PenguinDataset(Dataset):
    """Penguin dataset class.

    Parameters
    ----------
    input_keys : Sequence[str]
        The column titles to use in the input feature vectors.
    target_keys : Sequnce[str]
        The column titles to use in the target feature vectors.
    train : bool
        If ``True``, this object will serve as the training set, and if
        ``False``, the validation set.
    x_tfms : Compose, optional
        A composition of transforms to apply to the inputs.
    y_tfms : Compose, optional
        A composition of transfroms to apply to the targets.

    """

    def __init__(
        self,
        input_keys: List[str],
        target_keys: str,
        train: bool,
        x_tfms: Optional[Compose] = None,
        y_tfms: Optional[Compose] = None,
    ):
        """Build ``PenguinDataset``."""
        self.input_keys = input_keys
        self.target_keys = target_keys

        self.full_df = _load_penguin_data()
        self.split = _split_data(self.full_df)["train" if train is True else "valid"]

        self.x_tfms = x_tfms
        self.y_tfms = y_tfms

    def __len__(self) -> int:
        """Return the length of requested split.

        Returns
        -------
        int
            The number of items in the dataset.

        """
        return len(self.split)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return an input-target pair.

        Parameters
        ----------
        idx : int
            Index of the input-target pair to return.

        Returns
        -------
        in_feats : Any
            Inputs.
        target : Any
            Targets.

        """
        feats = tuple(self.split.iloc[idx][self.input_keys])

        tgts = tuple(self.split.iloc[idx][self.target_keys])

        if self.x_tfms is not None:
            feats = self.x_tfms(feats)

        if self.y_tfms is not None:
            tgts = self.y_tfms(tgts)

        return feats, tgts


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


def _split_data(penguin_df: DataFrame) -> Dict[str, DataFrame]:
    """Split the ``penguin_df`` into a training and validation set.

    Parameters
    ----------
    penguin_df : DataFrame
        The full penguin data set.

    Returns
    -------
    Dict[str, DataFrame]
        Dictionary holding the ``"train"`` and ``"valid"`` splits. The valid
        split has 10 females and 10 males of each species, and the training
        split contains the rest of the dataset.

    """
    valid_df = penguin_df.groupby(by=["species", "sex"]).sample(
        n=10,
        random_state=123,
    )

    # The training items are simply the items *not* in the valid split
    train_df = penguin_df.loc[~penguin_df.index.isin(valid_df.index)]

    return {"train": train_df, "valid": valid_df}
