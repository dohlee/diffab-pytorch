import os
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from protstruc import AntibodyFvStructureBatch


class DiffAbDataset(Dataset):
    def __init__(self, meta_df, data_dir):
        super().__init__()

        self.meta_df = meta_df
        self.records = meta_df.to_records()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, i):
        record = self.records[i]
        pdb_path = os.path.join(self.data_dir, f"{record.pdb_id}.pdb")

        sb = AntibodyFvStructureBatch.from_pdb(pdb_path)
        print(record.pdb_id, sb.get_max_n_residues())

        backbone_dihedrals, backbone_dihedrals_mask = sb.backbone_dihedrals()
        distmat, distmat_mask = sb.pairwise_distance_matrix()
        pairwise_dihedrals = torch.stack(
            [
                sb.pairwise_dihedrals(atoms_i=["C"], atoms_j=["N", "CA", "C"]),
                sb.pairwise_dihedrals(atoms_i=["N", "CA", "C"], atoms_j=["N"]),
            ],
            dim=-1,
        )  # b n n 2

        ret = {}
        ret["backbone_dihedrals"] = backbone_dihedrals
        ret["backbone_dihedrals_mask"] = backbone_dihedrals_mask
        ret["distmat"] = distmat
        ret["distmat_mask"] = distmat_mask
        ret["pairwise_dihedrals"] = pairwise_dihedrals
        ret["atom_mask"] = sb.get_atom_mask()
        ret["chain_idx"] = sb.get_chain_idx()
        ret["residue_idx"] = torch.arange(sb.get_max_n_residues()).unsqueeze(0)
        ret["residue_mask"] = sb.get_residue_mask()

        # TODO: determine mask for residues that are to be generated
        ret["generation_mask"] = torch.randn(1, sb.get_max_n_residues())

        return ret


class DiffAbDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
