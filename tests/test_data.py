import pytest
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from diffab_pytorch import DiffAb
from diffab_pytorch.data import DiffAbDataset, collate_fn


def test_DiffAbDataset_batch_size_1():
    meta_df = pd.read_csv("data/minisabdab/meta.csv")
    ds = DiffAbDataset(
        meta_df=meta_df, data_dir="data/minisabdab/chothia", cdrs_to_generate=["H1"]
    )

    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    for batch in tqdm(loader, desc="testing DiffAbDataset-single batch size"):
        pass


def test_DiffAbDataset_batch_size_more_than_1():
    meta_df = pd.read_csv("data/minisabdab/meta.csv")
    ds = DiffAbDataset(
        meta_df=meta_df, data_dir="data/minisabdab/chothia", cdrs_to_generate=["H1"]
    )

    loader = DataLoader(
        ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    for batch in tqdm(loader, desc="testing DiffAbDataset-single batch size"):
        pass


def test_DiffAb_step():
    meta_df = pd.read_csv("data/minisabdab/meta.csv")
    ds = DiffAbDataset(meta_df, "data/minisabdab/chothia", cdrs_to_generate=["H3"])

    loader = DataLoader(
        ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    model = DiffAb(
        d_emb=32,
        n_ipa_layers=2,
        d_scalar_per_head=4,
        n_query_point_per_head=4,
        n_value_point_per_head=4,
        n_head=8,
    )
    for batch_idx, batch in tqdm(enumerate(loader), desc="testing DiffAb forward"):
        model._shared_step(batch, batch_idx)
        break
