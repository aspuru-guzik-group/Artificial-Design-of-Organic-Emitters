#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 03:07:36 2020

@author: akshat
"""
import time
import os
import rdkit
import shutil
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import rdMolDescriptors
from selfies import decoder 
import numpy as np
import inspect
from collections import OrderedDict
manager = multiprocessing.Manager()
lock = multiprocessing.Lock()
from rdkit.Chem.MolStandardize import rdMolStandardize
enumerator = rdMolStandardize.TautomerEnumerator()

from realize_path import obtain_path, get_compr_paths



def get_ECFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2)

def get_fp_scores(smiles_back, target_smi): 
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = get_ECFP4(target)
    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = get_ECFP4(mol)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores


def get_logP(mol):
    '''Calculate logP of a molecule 
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object, for which logP is to calculates
    
    Returns:
    float : logP of molecule (mol)
    '''
    return Descriptors.MolLogP(mol)

def get_best_taut(m):
    m_1 = enumerator.Canonicalize(m)
    return Chem.MolToSmiles(m_1)

def calc_prop_RingP(mol):
    cycle_list = mol.GetRingInfo().AtomRings()
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
        
    if cycle_length <= 7:
        return False
    else:
        return True 
    

def sanitize_smiles(smi):
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    
def has_substr(mol):
    substr_match = False
    
    substrc_str_ls = ['*=*=*'] 
    

    for substrc_str in substrc_str_ls:
        _, substrc_str, _ = sanitize_smiles(substrc_str)
        contains_strct = mol.HasSubstructMatch(Chem.MolFromSmarts(substrc_str))
        if contains_strct == True:
            substr_match = True
            break

def get_median_mols(starting_smile, target_smile, num_tries, num_random_samples, collect_bidirectional, num_top_iter, apply_filter=False): 
    
    smiles_paths_dir1, smiles_paths_dir2 = get_compr_paths(starting_smile, target_smile, num_tries, num_random_samples, collect_bidirectional)
    
    # Find the median molecule & plot: 
    all_smiles_dir_1 = [item for sublist in smiles_paths_dir1 for item in sublist] # all the smile string of dir1
    all_smiles_dir_2 = [item for sublist in smiles_paths_dir2 for item in sublist] # all the smile string of dir2
    
    all_smiles = [] # Collection of valid smile strings 
    for smi in all_smiles_dir_1 + all_smiles_dir_2: 
        if Chem.MolFromSmiles(smi) != None: 
            mol, smi_canon, _ = sanitize_smiles(smi)
            all_smiles.append(smi_canon)
    
    # print('All smiles len: ', len(all_smiles))
       
    if apply_filter == True:
        better_smi = []
        for smi in all_smiles: 
            mol = Chem.MolFromSmiles(smi)
            if rdMolDescriptors.CalcNumBridgeheadAtoms(mol)==0 and rdMolDescriptors.CalcNumSpiroAtoms(mol)==0 and calc_prop_RingP(mol)==False and has_substr(mol)==False:
                # better_smi.append(get_best_taut(mol))
                mol, smi_canon, _ = sanitize_smiles(smi)
                better_smi.append(smi_canon)
                
        better_smi = list(set(better_smi))

        scores_start  = get_fp_scores(better_smi, starting_smile)   # similarity to target
        scores_target = get_fp_scores(better_smi, target_smile)     # similarity to starting structure

        data          = np.array([scores_target, scores_start])
        avg_score     = np.average(data, axis=0)

        better_score  = avg_score - (np.abs(data[0] - data[1]))   
        better_score  = ((1/9) * better_score**3) - ((7/9) * better_score**2) + ((19/12) * better_score)
        
        
        best_idx = better_score.argsort()[-num_top_iter:][::-1]
        best_smi = [better_smi[i] for i in best_idx]
        best_scores = [better_score[i] for i in best_idx]
    
        return best_smi, best_scores    
        
    else:         
        all_smiles = list(set(all_smiles))
    
        scores_start  = get_fp_scores(all_smiles, starting_smile)   # similarity to target
        scores_target = get_fp_scores(all_smiles, target_smile)     # similarity to starting structure
        data          = np.array([scores_target, scores_start])
        avg_score     = np.average(data, axis=0)
        better_score  = avg_score - (np.abs(data[0] - data[1]))   
        better_score  = ((1/9) * better_score**3) - ((7/9) * better_score**2) + ((19/12) * better_score)
        
        best_idx = better_score.argsort()[-num_top_iter:][::-1]
        best_smi = [all_smiles[i] for i in best_idx]
        best_scores = [better_score[i] for i in best_idx]
    
        return best_smi, best_scores    
    
def crossover(smi_1, smi_2):     
    
    num_tries             = 2 
    num_random_samples    = 2 
    collect_bidirectional = True # Doubles the number of paths: source->target & target->source
    apply_filter          = False
    num_top_iter          = 1500  # Number of molecules that are selected after each iteration
    
    best_smi, best_scores = get_median_mols(smi_1, smi_2, num_tries, num_random_samples, collect_bidirectional, num_top_iter, apply_filter=apply_filter)
        
    from filter_ import passes_filter
    filter_pass = []
    for item in best_smi: 
        try: 
            if passes_filter(item) == True:
                filter_pass.append(item)
        except: 
            print('Filter Failed on: ', item)

    if filter_pass == []:
        print('No crossovers found for: {} {}'.format(smi_1, smi_2))
        print('Conducting more thorough crossover search! ')
        
        best_smi, best_scores = get_median_mols(smi_1, smi_2, 10, 10, collect_bidirectional, 5000, apply_filter=apply_filter)
        filter_pass = []
        for item in best_smi: 
            try: 
                if passes_filter(item) == True :
                    filter_pass.append(item)
            except: 
                print('Filter Failed on: ', item)
                
    return filter_pass[:]









