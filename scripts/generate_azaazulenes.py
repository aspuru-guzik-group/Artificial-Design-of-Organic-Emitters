#!/usr/bin/env python

import numpy as np
import itertools as it
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
import pathlib as pl

def sanitize_smiles(smi):
    return mol2smi(smi2mol(smi, sanitize=True), isomericSmiles=True, canonical=True)

mols = []
perm = it.product('NC', repeat=8)

for ni in perm:
    string = ''
    string += ni[0]
    string += '1='
    string += ni[1]
    string += 'C2='
    string += ni[2]
    string += ni[3]
    string += '='
    string += ni[4]
    string += ni[5]
    string += '='
    string += ni[6]
    
    string += 'C2='
    string += ni[7]
    string += '1'
    mols.append(string)

mols_san = []

for smi in mols:
    mols_san.append(sanitize_smiles(smi))

mols_san = np.array(mols_san)

one = ''
for smi in np.unique(mols_san):
    one += smi
    one += '.'
one = one[:-1]

cwd = pl.Path.cwd()
with open(cwd / pl.Path('gen.smi'), 'w') as f:
    f.write(str(len(np.unique(mols_san))))
    f.write('\n')
    f.write(one)
    f.write('\n')

