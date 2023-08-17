import pytest
import pandas as pd

from torch.utils.data import DataLoader
from diffab_pytorch.data import DiffAbDataset


def test_DiffAbDataset_single_batch_size():
    meta_df = pd.read_csv("data/minisabdab/meta.csv")
    ds = DiffAbDataset(meta_df=meta_df, data_dir="data/minisabdab/imgt")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    for batch in loader:
        pass
