import os
import torch
import pandas as pd
import pytorch_lightning as pl

from typing import List
from torch.utils.data import Dataset
from protstruc import AntibodyStructureBatch


def _always_list(x):
    return x if isinstance(x, list) else [x]


class DiffAbDataset(Dataset):
    CDRS = ["H1", "H2", "H3", "L1", "L2", "L3"]

    def __init__(
        self, meta_df: pd.DataFrame, data_dir: str, cdrs_to_generate: List[str]
    ):
        """Dataset for DiffAb training

        Args:
            meta_df: Pandas dataframe for metadata. Must contain columns
                - pdb_id: PDB identifier of each antibody-antigen complex
                - Hchain: Chain identifier of heavy chain
                - Lchain: Chain identifier of light chain
                - antigen_chain: Chain identifiers of antigen
            data_dir: Directory containing PDB files
            cdrs_to_generate: List of CDRs to generate. Must be a subset of
                ["H1", "H2", "H3", "L1", "L2", "L3"]
        """
        super().__init__()

        self.meta_df = meta_df
        self.records = meta_df.to_records()
        self.data_dir = data_dir

        self.cdrs_to_generate = _always_list(cdrs_to_generate)
        if not set(self.cdrs_to_generate).issubset(self.CDRS):
            raise ValueError(f"cdrs_to_generate must be a subset of {self.CDRS}")

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, i):
        record = self.records[i]

        ret = {
            "pdb_path": os.path.join(self.data_dir, f"{record.pdb_id}.pdb"),
            "heavy_chain_id": record.Hchain,
            "light_chain_id": record.Lchain,
            "antigen_chain_id": record.antigen_chain,
            "cdrs_to_generate": self.cdrs_to_generate,
        }

        return ret


def collate_fn(batch):
    """Collate function for DiffAbDataset"""
    pdb_path = [b["pdb_path"] for b in batch]
    heavy_chain_id = [b["heavy_chain_id"] for b in batch]
    light_chain_id = [b["light_chain_id"] for b in batch]
    antigen_chain_ids = [b["antigen_chain_id"] for b in batch]
    cdrs_to_generate = batch[0]["cdrs_to_generate"]  # must be the same for all

    sb = AntibodyStructureBatch.from_pdb(
        pdb_path,
        heavy_chain_id,
        light_chain_id,
        antigen_chain_ids,
    )

    backbone_dihedrals, backbone_dihedrals_mask = sb.backbone_dihedrals()
    distmat, distmat_mask = sb.pairwise_distance_matrix()

    phi = sb.pairwise_dihedrals(atoms_i=["C"], atoms_j=["N", "CA", "C"])
    psi = sb.pairwise_dihedrals(atoms_i=["N", "CA", "C"], atoms_j=["N"])
    pairwise_dihedrals = torch.stack([phi, psi], dim=-1)  # b n n 2

    ret = {
        "xyz": sb.get_xyz(),
        "orientations": sb.backbone_orientations(),
        "backbone_dihedrals": backbone_dihedrals,
        "backbone_dihedrals_mask": backbone_dihedrals_mask,
        "distmat": distmat,
        "distmat_mask": distmat_mask,
        "pairwise_dihedrals": pairwise_dihedrals,
        "atom_mask": sb.get_atom_mask(),
        "seq_idx": sb.get_seq_idx(),
        "chain_idx": sb.get_chain_idx(),
        "residue_idx": torch.arange(sb.get_max_n_residues()).unsqueeze(0),
        "residue_mask": sb.get_residue_mask(),
        "generation_mask": sb.get_cdr_mask(subset=cdrs_to_generate),
    }

    return ret


class DiffAbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        data_dir: str,
        cdrs_to_generate: List[str],
        batch_size: int,
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.data_dir = data_dir
        self.cdrs_to_generate = cdrs_to_generate

        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = DiffAbDataset(
            self.train_df, self.data_dir, self.cdrs_to_generate
        )
        self.val_dataset = DiffAbDataset(
            self.val_df, self.data_dir, self.cdrs_to_generate
        )
        self.test_dataset = DiffAbDataset(
            self.test_df, self.data_dir, self.cdrs_to_generate
        )

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=True,
        )
        return loader
