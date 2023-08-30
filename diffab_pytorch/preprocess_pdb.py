import argparse
import torch

from protstruc import AntibodyStructureBatch
from protstruc.general import ATOM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input PDB file.")
    parser.add_argument("--heavy-chain-id", default=None, type=str)
    parser.add_argument("--light-chain-id", default=None, type=str)
    parser.add_argument(
        "-k",
        "--nearest-k",
        default=128,
        type=int,
        help="Make patches with this number of nearest residues around CDR anchor residues.",
    )
    parser.add_argument("-a", "--antigen-chain-ids", default=None, type=str)
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output .pt file."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    antigen_chain_ids = list(args.antigen_chain_ids)
    sb = AntibodyStructureBatch.from_pdb(
        args.input,
        heavy_chain_id=args.heavy_chain_id,
        light_chain_id=args.light_chain_id,
        antigen_chain_ids=antigen_chain_ids,
        keep_fv_only=True,
    )

    seq_idx = sb.get_seq_idx()

    # create patch around anchor
    xyz = sb.get_xyz()[0]

    anchor_mask = sb.get_cdr_anchor_mask()[0]
    anchor_xyz = xyz[anchor_mask, ATOM.CA]

    k_nearest_mask_ab_or_ag = sb.get_topk_nearest_residue_mask(
        anchor_xyz, k=128, mask=None
    )

    ag_mask = sb.get_antigen_mask()[0]
    k_nearest_mask_ag_only = sb.get_topk_nearest_residue_mask(
        anchor_xyz, k=128, mask=ag_mask
    )

    mask = k_nearest_mask_ab_or_ag | k_nearest_mask_ag_only
    sb = sb.residue_masked_select(mask)

    backbone_dihedrals, backbone_dihedrals_mask = sb.backbone_dihedrals()
    distmat, distmat_mask = sb.pairwise_distance_matrix()

    phi = sb.pairwise_dihedrals(atoms_i=["C"], atoms_j=["N", "CA", "C"])
    psi = sb.pairwise_dihedrals(atoms_i=["N", "CA", "C"], atoms_j=["N"])
    pairwise_dihedrals = torch.stack([phi, psi], dim=-1)  # b n n 2

    data = {
        "xyz": sb.get_xyz(),  # 80kb
        "orientations": sb.backbone_orientations(),  # 17kb
        "backbone_dihedrals": backbone_dihedrals,  # 6kb
        "backbone_dihedrals_mask": backbone_dihedrals_mask,  # 2.1kb
        "pairwise_dihedrals": pairwise_dihedrals,  # 1.6M
        "atom_mask": sb.get_atom_mask(),  # 27k
        "seq_idx": seq_idx,
        "chain_idx": sb.get_chain_idx(),  # 4.2k
        "residue_idx": torch.arange(sb.get_max_n_residues()).unsqueeze(0),  # 4.2k
        "residue_mask": sb.get_residue_mask(),  # 1.2k
        # "distmat": distmat, # 171M
        # "distmat_mask": distmat_mask, # 171M
    }

    torch.save(data, args.output)


if __name__ == "__main__":
    main()
