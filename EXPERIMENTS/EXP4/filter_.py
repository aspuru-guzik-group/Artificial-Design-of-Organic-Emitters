"""
Filtering GDB-13
Authors: Robert Pollice, Akshat Nigam
Date: Sep. 2020
"""
import numpy as np
import pandas as pd
import rdkit as rd
import rdkit.Chem as rdc
import rdkit.Chem.rdMolDescriptors as rdcmd
import rdkit.Chem.Lipinski as rdcl
# import argparse as ap
import pathlib as pl
cwd = pl.Path.cwd() # define current working directory

def smiles_to_mol(smiles):
    """
    Convert SMILES to mol object using RDKit
    """
    try:
        mol = rdc.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def maximum_ring_size(mol):
    """
    Calculate maximum ring size of molecule
    """
    cycles = mol.GetRingInfo().AtomRings()
    if len(cycles) == 0:
        maximum_ring_size = 0
    else:
        maximum_ring_size = max([len(ci) for ci in cycles])
    return maximum_ring_size
    
def minimum_ring_size(mol):
    """
    Calculate minimum ring size of molecule
    """
    cycles = mol.GetRingInfo().AtomRings()
    if len(cycles) == 0:
        minimum_ring_size = 0
    else:
        minimum_ring_size = min([len(ci) for ci in cycles])
    return minimum_ring_size


def substructure_violations(mol):
    """
    Check for substructure violates
    Return True: contains a substructure violation
    Return False: No substructure violation
    """
    violation = False
    # forbidden_fragments = ['[S&X3]', '[S&X4]', '[S&X5]', '[S&X6]', '[r3]', 'P', 'p', '[B,N,O,S]~[F,Cl,Br,I]', '[Cl]', '*=*=*', '*#*', '[O,o,S,s]~[O,o,S,s]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[N,n,O,o,S,s]', '[C,c]~N=,:[O,o,S,s;!R]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[C,c]=,:[O,o,S,s,N,n;!R]', '*=[NH]', '*=N-[*;!R]', '*~[N,n,O,o,S,s]-[N,n,O,o,S,s;!R]', '*-[CH1]-*', '*-[CH2]-*', '*-[CH3]']
    # forbidden_fragments = ['*=[S,s;!R]', '[S&X3]', '[S&X4]', '[S&X5]', '[S&X6]', '[r3]', 'P', 'p', '[B,N,O,S]~[F,Cl,Br,I]', '[Cl,Br,I]', '*=*=*', '*#*', '[O,o,S,s]~[O,o,S,s]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[N,n,O,o,S,s]', '[C,c]~N=,:[O,o,S,s;!R]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[C,c]=,:[O,o,S,s,N,n;!R]', '*=[NH]', '*=N-[*;!R]', '*~[N,n,O,o,S,s]-[N,n,O,o,S,s;!R]', '*-[CH1]-*', '*-[CH2]-*', '*-[CH3]']
    forbidden_fragments = ['[*+]', '[*-]', 'C(=O)-S', '*~C(=O)-F', '[O,o,S,s,N,n;!R]-,:[C,c;!R]=,:[C,c]', '[C;!R]=,:[N;!R]', '[C&H0;!R]=,:[C&H2;!R]','[C;R]=,:[C&H2;!R]', '[*;R]=,:[*;!R]-,:[*;!R]=,:[*;!R]', '[N&X5]', '*=[S,s;!R]', '[S&X3]', '[S&X4]', '[S&X5]', '[S&X6]', '[P,p]', '[B,b,N,n,O,o,S,s]~[F,Cl,Br,I]', '[*;!R]=,:[*;!R]-,:[*;!R]=,:[*;!R]',  '[Cl,Br,I]', '*=*=*', '*#*', '[O,o,S,s]~[O,o,S,s]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[N,n,O,o,S,s]', '[C,c]~N=,:[O,o,S,s;!R]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[C,c]=,:[O,o,S,s,N,n;!R]', '*=[NH]', '*=N-[*;!R]', '*~[N,n,O,o,S,s]-[N,n,O,o,S,s;!R]', '*-[CH1]-*', '*-[CH2]-*', '*-[CH3]']
    for ni in range(len(forbidden_fragments)):
        
        if mol.HasSubstructMatch(rdc.MolFromSmarts(forbidden_fragments[ni])) == True:
            violation = True
            break
        else:
            continue

    return violation

def aromaticity_degree(mol):
    """
    Compute the percentage of non-hydrogen atoms in a molecule that are aromatic
    """
    atoms = mol.GetAtoms()
    atom_number = rdcl.HeavyAtomCount(mol)
    aromaticity_count = 0.
    
    for ai in atoms:
        if ai.GetAtomicNum() != 1:
            if ai.GetIsAromatic() == True:
                aromaticity_count += 1.
        
    degree = aromaticity_count / atom_number
    
    return degree

def conjugation_degree(mol):
    """
    Compute the percentage of bonds between non-hydrogen atoms in a molecule that are conjugated
    """
    bonds = mol.GetBonds()
    bond_number = 0.
    conjugation_count = 0.
    
    for bi in bonds:
        a1 = bi.GetBeginAtom()
        a2 = bi.GetEndAtom()
        if (a1.GetAtomicNum() != 1) and (a2.GetAtomicNum() != 1):
            bond_number += 1.
            if bi.GetIsConjugated() == True:
                conjugation_count += 1.
        
    degree = conjugation_count / bond_number
    
    return degree


from rdkit import Chem
def passes_filter(smi): 
    mol = rdc.MolFromSmiles(smi)
    mol_hydrogen = Chem.AddHs(mol)
    if rdcmd.CalcNumBridgeheadAtoms(mol) == 0 and rdcmd.CalcNumSpiroAtoms(mol) == 0 and aromaticity_degree(mol) >= 0.5 and conjugation_degree(mol) >= 0.7 and (5 <= maximum_ring_size(mol) <=7) and (5 <= minimum_ring_size(mol) <=7) and substructure_violations(mol)==False and mol_hydrogen.GetNumAtoms()<=70: 
        return True
    else: 
        return False 

