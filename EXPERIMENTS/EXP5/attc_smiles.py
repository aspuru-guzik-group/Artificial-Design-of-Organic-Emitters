#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 04:31:10 2021

@author: akshat
"""
import selfies
import rdkit
import random
import numpy as np
import random
from rdkit import Chem
from selfies import encoder, decoder
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import Mol
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import rdkit.Chem as rdc
import random 

def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    
import rdkit.Chem as rdc
def substructure_preserver(mol):
    """
    Check for substructure violates
    Return True: contains a substructure violation
    Return False: No substructure violation
    """        
    if mol.HasSubstructMatch(rdc.MolFromSmarts('*1***2****-2**1')) == True: 
        return True # The molecule is good! 
    else: 
        return False # Molecule is bad! 
    
def smiles_functionalization(core_smiles, ligand_smiles_list):
	"""
	Generate the SMILES for a functionalized molecule.
	The core SMILES is formatted with tags [R], [R*], [R**], ...
	These indicate different ligands to be attached.
	Each tag will receive the same ligand.
	The ligand SMILES are formatted just with [R] tags.
    
    [R]c1nnc-2cnc([R*])ccc1-2
    
    
	Args:
		core_smiles: Center molecule SMILES, formatted with [R], [R*], [R**], ...
		ligand_smiles_list: Ligand SMILES, formatted with [R].
	Returns:
		Combined molecule SMILES.
	Raises:
		n/a
	"""
	# Specify the dummy atoms
	core_dummy_atoms = ['[La]', '[Ce]', '[Pr]', '[Nd]', '[Pm]', '[Sm]', '[Eu]', '[Gd]', '[Tb]', '[Dy]']
	lig_dummy_atom = '[Ac]'
	# Count the core atom dummy occurrences
	dummy_tags = [f"[R{'*' * i}]" for i in range(len(core_dummy_atoms))]
	dummy_counts = [core_smiles.count(tag) for tag in dummy_tags]
	# Replace tags for dummy atoms
	for tag, dummy in zip(dummy_tags, core_dummy_atoms):
		core_smiles = core_smiles.replace(tag, dummy)
	core_mol = Chem.MolFromSmiles(core_smiles)
	ligand_mol_list = [Chem.MolFromSmiles(smiles.replace('[R]', lig_dummy_atom)) for smiles in ligand_smiles_list]
	# Functionalize!
	for count, dummy, ligand_mol in zip(dummy_counts, core_dummy_atoms, ligand_mol_list):
		for _ in range(count):
			dummy_atoms = [dummy[1:3], lig_dummy_atom[1:3]]
			dummy_indices = [-1, -1]
			neigh_indices = [-1, -1]
			# Combine molecules into one object
			combo = Chem.CombineMols(core_mol, ligand_mol)
			# Determine indices of dummy atoms
			for i, atom in enumerate(combo.GetAtoms()):
				for j in range(2):
					if atom.GetSymbol() == dummy_atoms[j]:
						dummy_indices[j] = i
			# Determine indices of dummy neighbors
			for bond in combo.GetBonds():
				# Track both bond indices
				bond_indices = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
				for j in range(2):
					if dummy_indices[j] in bond_indices:
						# Save the other index
						bond_indices.remove(dummy_indices[j])
						neigh_indices[j] = bond_indices[0]
			# Create the sigma bond on an editable molecule
			edit_combo = Chem.EditableMol(combo)
			edit_combo.AddBond(neigh_indices[0], neigh_indices[1], order=Chem.rdchem.BondType.SINGLE)
			edit_combo.RemoveAtom(dummy_indices[1])
			edit_combo.RemoveAtom(dummy_indices[0])
			# Recreate a proper molecule object
			core_mol = edit_combo.GetMol()
	# Return the SMILES
	return Chem.MolToSmiles(core_mol)


def attach_smiles(filter_pass, core_mutated): 


    frag_points = []
    for item in core_mutated: 
        A = '[R]' + item[0:10] + '([R])' + item[10: ]
        frag_points.append(A)
    val_cores = frag_points.copy()

    
    attached_smi = []
    
    # Variable is important for going onto the next generation! 
    combined_origina_map = {} # Complex -> [Smiles, Smile_with_attc_point, core_attached_to]
    
    for i,smi_ in enumerate(filter_pass): 
        print('Working on smile: {}/{}'.format(i, len(filter_pass)))
        fail_counter_ = 0
        while fail_counter_ <= 5: 
            try: 
                random_idx     = random.randint(0, len(smi_)-1)
                smi_point_attc = smi_[:random_idx] + '[R]' + smi_[random_idx: ]
                ligand_smiles_list = [smi_point_attc, smi_point_attc]
                print('Attaching: {}  {}'.format(val_cores[i], ligand_smiles_list))
                atached = smiles_functionalization(val_cores[i], ligand_smiles_list)
                if '.' in atached: 
                    for item in atached.split('.'): 
                        if substructure_preserver(Chem.MolFromSmiles(item)) == True: 
                            atached = item.copy()
                            break
                attached_smi.append(atached)
                combined_origina_map[atached] = [smi_, smi_point_attc, val_cores[i]] # THe 
                fail_counter_ += 1
            except: 
                fail_counter_ += 1
    
    print('Attachment to core complete!')
    
    canon_smi_ls = []
    for item in attached_smi: 
        mol, smi_canon, did_convert = sanitize_smiles(item)
        if mol == None or smi_canon == '' or did_convert == False: 
            continue
        canon_smi_ls.append(item)
    canon_smi_ls        = list(set(canon_smi_ls))
    
    from filter_ import passes_filter
    filter_pass = []
    for item in canon_smi_ls: 
        try: 
            if passes_filter(item) == True:
                filter_pass.append(item)
        except: 
            print('Filter Failed on: ', item)
            
    
    return combined_origina_map, filter_pass
