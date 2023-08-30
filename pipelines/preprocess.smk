import pandas as pd

meta = pd.read_csv('data/meta.csv')

pdb_ids = meta['pdb_id'].values
heavy_chain_ids = meta['Hchain'].values
light_chain_ids = meta['Lchain'].values

antigen_chain_ids = meta['antigen_chain'].values
antigen_chain_ids = [''.join([x.strip() for x in chain.split('|')]) for chain in antigen_chain_ids]

rule all:
    input: expand('data/preprocessed/{pdb}_{h}_{l}_{a}.pt', zip, pdb=pdb_ids, h=heavy_chain_ids, l=light_chain_ids, a=antigen_chain_ids)

rule preprocess_pdb:
    input: 'data/all_structures/chothia/{pdb}.pdb'
    output: 'data/preprocessed/{pdb}_{h}_{l}_{a}.pt'
    params:
        heavy_chain = lambda wc: '--heavy-chain-id ' + wc.h if wc.h != 'nan' else '',
        light_chain = lambda wc: '--light-chain-id ' + wc.l if wc.l != 'nan' else '',
        antigen_chain = lambda wc: '--antigen-chain-ids ' + wc.a if wc.a != 'nan' else '',
        k = 128,
    shell:
        'python -m diffab_pytorch.preprocess_pdb '
        '-i {input} '
        '-o {output} '
        '-k {params.k} '
        '{params.heavy_chain} '
        '{params.light_chain} '
        '{params.antigen_chain} '
    