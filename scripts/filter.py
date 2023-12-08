"""
Filtering GDB-13
Authors: Robert Pollice, Akshat Nigam
Date: Sep. 2020
"""

# SETTING UP PYTON ENVIRONMENT
# Load modules
import numpy as np
import pandas as pd
import rdkit as rd
import rdkit.Chem as rdc
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd
import rdkit.Chem.Lipinski as rdcl
import rdkit.Chem.rdmolops as rdcmo
import argparse as ap
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
    # filter used for GDB13 filtering
    forbidden_fragments = ['[Cl,Br,I]', '*=*=*', '*#*', '[O,o,S,s]~[O,o,S,s]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[N,n,O,o,S,s]', '[C,c]~N=,:[O,o,S,s;!R]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[C,c]=,:[O,o,S,s,N,n;!R]', '*=[NH]', '*=N-[*;!R]', '*~[N,n,O,o,S,s]-[N,n,O,o,S,s;!R]', '*-[CH1]-*', '*-[CH2]-*', '*-[CH3]']
    
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

def main():
    # parse arguments
    parser = ap.ArgumentParser()
    parser.add_argument('data', help="CSV file of SMILES. Delimiter should be space and it should have no header.", type=str)
    parser.add_argument('--removed', help="Wheter or not to save list of removed SMILES. '0', do not save, or '1', save. Default is '0'.", type=str, choices=['0', '1'], default='0')
    arguments = parser.parse_args()
    
    # Generate output name
    output = arguments.data.split('.')
    output[-2] += '_filtered'
    output = '.'.join(output)
    output_removed = arguments.data.split('.')
    output_removed[-2] += '_removed'
    output_removed = '.'.join(output_removed)
    
    # Load data
    data = pd.read_csv(arguments.data, delimiter=' ', usecols=[0, 1], names=['SMILES','NUMBER'], skiprows=0)
    #data = pd.read_csv(arguments.data, delimiter=' ', usecols=[0, 1, 2], names=['SMILES','S-T','OSC'], skiprows=0)
    print('Original Data: ' + str(len(data.index)))

    # Generate mol objects from smiles and remove compounds that returned erros
    data['MOL'] = data['SMILES'].apply(smiles_to_mol)
    new_data = data[data['MOL'] != None]
    data = new_data.copy()
    data['CSMILES'] = data['MOL'].apply(lambda x: rdc.MolToSmiles(x, isomericSmiles=False, canonical=True))
    print('RDKit Converted: ' + str(len(data.index)))
    
    # Added after GDB-13 was filtered to get rid charged molecules
    data['CHARGE'] = data['MOL'].apply(lambda x: rdcmo.GetFormalCharge(x))
    new_data = data[(data['CHARGE'] == 0)]
    data = new_data.copy()
    print('Filtered by molecular charge: ' + str(len(data.index)))
    
    # Added after GDB-13 was filtered to get rid radicals
    data['RADICALS'] = data['MOL'].apply(lambda x: rdcd.NumRadicalElectrons(x))
    new_data = data[(data['RADICALS'] == 0)]
    data = new_data.copy()
    print('Filtered by number of radicals: ' + str(len(data.index)))
    
    # Filter by bridgehead atoms
    # Note: filters are ordered by increasing timing requirements
    data['BRIDGEHEAD'] = data['MOL'].apply(lambda x: rdcmd.CalcNumBridgeheadAtoms(x))
    new_data = data[data['BRIDGEHEAD'] == 0]
    data = new_data.copy()
    print('Filtered by bridgehead atoms: ' + str(len(data.index)))

    # Filter by spiro atoms
    data['SPIROATOMS'] = data['MOL'].apply(lambda x: rdcmd.CalcNumSpiroAtoms(x))
    new_data = data[data['SPIROATOMS'] == 0]
    data = new_data.copy()
    print('Filtered by spiro atoms: ' + str(len(data.index)))
    
    # Filter by aromaticity
    data['AROMATICITY'] = data['MOL'].apply(lambda x: aromaticity_degree(x))
    new_data = data[data['AROMATICITY'] >= 0.50]
    data = new_data.copy()
    print('Filtered by aromaticity: ' + str(len(data.index)))
    
    # Filter by conjugation
    data['CONJUGATION'] = data['MOL'].apply(lambda x: conjugation_degree(x))
    new_data = data[data['CONJUGATION'] >= 0.70]
    data = new_data.copy()
    print('Filtered by conjugation: ' + str(len(data.index)))
    
    # Filter by ring size
    data['MAXIMUM_RINGSIZE'] = data['MOL'].apply(lambda x: maximum_ring_size(x))
    new_data = data[(data['MAXIMUM_RINGSIZE'] >= 5) & (data['MAXIMUM_RINGSIZE'] <= 7)]
    #new_data = data[(data['MAXIMUM_RINGSIZE'] >= 4) & (data['MAXIMUM_RINGSIZE'] <= 8)] # old version for GDB-13
    data = new_data.copy()
    print('Filtered by maximum ring size: ' + str(len(data.index)))
    
    # Added after GDB-13 was filtered to get rid of 3-membered rings
    data['MINIMUM_RINGSIZE'] = data['MOL'].apply(lambda x: minimum_ring_size(x))
    new_data = data[(data['MINIMUM_RINGSIZE'] >= 5) & (data['MINIMUM_RINGSIZE'] <= 7)]
    #new_data = data[(data['MINIMUM_RINGSIZE'] >= 4) & (data['MINIMUM_RINGSIZE'] <= 8)] # old version for GDB-13
    data = new_data.copy()
    print('Filtered by minimum ring size: ' + str(len(data.index)))

    # Filter by functional groups
    data['VIOLATIONS'] = data['MOL'].apply(lambda x: substructure_violations(x))
    new_data = data[data['VIOLATIONS'] == False]
    removed_data = data[data['VIOLATIONS'] == True]
    data = new_data.copy()
    print('Filtered by functional groups: ' + str(len(data.index)))

    # Save processed data
    #data.to_csv(output, columns = ['SMILES','S-T','OSC'], sep = ' ', index = False, header = True)
    data.to_csv(output, columns = ['SMILES','NUMBER'], sep = ' ', index = False, header = False)
    
    if arguments.removed == '1':
        #removed_data.to_csv(output_removed, columns = ['SMILES','S-T','OSC'], sep = ' ', index = False, header = True)
        removed_data.to_csv(output_removed, columns = ['SMILES','NUMBER'], sep = ' ', index = False, header = False)
    
    return


if __name__ == "__main__":
    main()
