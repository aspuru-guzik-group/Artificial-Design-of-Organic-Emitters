#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 07:28:26 2021

@author: akshat
"""
import random 
import selfies
from selfies import encoder, decoder
from glob import glob
import numpy as np

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
from crossover import crossover
from mutate import get_mutated_smi
from crossover_core import crossover_core


import torch 
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem, DataStructs

class _FingerprintCalculator:
    """
    Calculate the fingerprint while avoiding a series of if-else.
    See recipe 8.21 of the book "Python Cookbook".

    To support a new type of fingerprint, just add a function "get_fpname(self, mol)".
    """

    def get_fingerprint(self, mol: Mol, fp_type: str):
        method_name = 'get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)

    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    def get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)

    def get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)

    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)


def get_fingerprint(mol: Mol, fp_type: str):
    return _FingerprintCalculator().get_fingerprint(mol=mol, fp_type=fp_type)

class Reg(nn.Module):
    def __init__(self, layer_1_dim, layer_2_dim, dropout_rate):
        super(Reg, self).__init__()
        self.fc1 = nn.Linear(1024       , layer_1_dim)
        self.fc2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.fc3 = nn.Linear(layer_2_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer_1_dim)
        self.batchnorm2 = nn.BatchNorm1d(layer_2_dim)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.batchnorm1(x)
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x
    
def get_fp_scores(smiles_back, target_smi, fp_type): 
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = get_fingerprint(target, fp_type)

    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = get_fingerprint(mol, fp_type)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores    

def get_fp_score(smiles_back): 
    
    mol    = Chem.MolFromSmiles(smiles_back)
    if mol == None: 
        raise Exception('Invalid molecule encountered when calculating fingerprint')
    fp_mol = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    
    A = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_mol, A)
    
    return A
    
def eval_smi(test_mol): 
    model_testing = torch.load('./saved_models_st/1/model_checkpoint.pt')
    model_testing.eval()
    
    
    # test_mol  = 'CCCCCC'
    model_inp = get_fp_score(test_mol)
    model_inp = torch.tensor(model_inp)
    model_inp = model_inp.reshape((1, 1024))
    
    pred_      = model_testing(model_inp.float())
    y_pred_tag = torch.round(torch.sigmoid(pred_))
    y_pred_tag = y_pred_tag.detach().numpy().flatten()[0]
    
    return y_pred_tag

def sanitize_smiles(smi):
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    
    
def order_based_on_fitness(fitness_here, smiles_here):
    '''Order elements of a list (smiles_here) based om Decreasing fitness (fitness_here)
    '''
    order = np.argsort(fitness_here)[::-1] # Decreasing order of indices, based on fitness
    fitness_ordered = [fitness_here[idx] for idx in order]
    smiles_ordered = [smiles_here[idx] for idx in order]
    
    return order, fitness_ordered, smiles_ordered

def obtain_next_gen_molecules(order, to_replace, to_keep, fragment_smiles_ord, core_smiles_ordered):
    smiles_mutated = []
    core_mutated   = []

    for idx in range(0,len(order)):
        print('On {}/{}/ {}'.format(idx, len(order), fragment_smiles_ord[idx]))

        if idx in to_replace: # smiles to replace (by better molecules)

            random_index = np.random.choice(to_keep, size=1, replace=True, p=None)[0]  # select a random molecule that survived
            print('Good mol index: ', random_index)
        
            # MUTATE WITH A GOOD MOL: 
            if np.random.random() < 0.5: 
                # print('Performing mutation on: ! ', fragment_smiles[random_index])
                smiles_new = get_mutated_smi(fragment_smiles_ord[random_index])  # do the mutation   
                for smi in smiles_new: smiles_mutated.append(smi)
                
                
                # Crossover the core
                cross_core = crossover_core(core_smiles_ordered[random_index], core_smiles_ordered[idx])
                for _ in smiles_new: core_mutated.append(cross_core)
                
                f = open('./geneology/gen_15.txt', 'a+')
                f.writelines(['{} mut: {}\n'.format(smiles_new, fragment_smiles_ord[random_index])])
                f.close()
                

            # CROSSOVER WITH A GOOD MOL: 
            else: 
                # print('Performing crossover! ')
                crossover_smi = crossover(fragment_smiles_ord[idx], fragment_smiles_ord[random_index])
                # print(crossover_smi)
                for smi in crossover_smi: smiles_mutated.append(smi)

                # Crossover the core
                cross_core = crossover_core(core_smiles_ordered[random_index], core_smiles_ordered[idx])
                for _ in crossover_smi: core_mutated.append(cross_core)
                                
                f = open('./geneology/gen_15.txt', 'a+')
                f.writelines(['{} cross: {} {}\n'.format(crossover_smi, fragment_smiles_ord[idx], fragment_smiles_ord[random_index])])
                f.close()
                

        else: # smiles to be kept
            smiles_mutated.append(fragment_smiles_ord[idx])
            core_mutated.append(core_smiles_ordered[idx])

    return smiles_mutated, core_mutated

# Step 1: Read all the smiles and properties: 
import pickle
with open('global_collect.pickle', 'rb') as handle:
    collected_dict = pickle.load(handle)    
    
    
content = list(collected_dict.keys()) # This is only for generation 1! 


failed = []
smi_to_prop = {}
for item in content: 
    try: 
        smi_to_prop[item] = collected_dict[item]
    except: 
        failed.append(item)
        continue


# Calculate the Fitness Function: 
singlet_trip      = []
oscillator_stngth = []
smiles_           = list(smi_to_prop.keys()) # All the smiles for which there are calculations! 

for key in smiles_: 
    singlet_trip.append(smi_to_prop[key][0])
    oscillator_stngth.append(smi_to_prop[key][1])


prop_1 = []
for x in singlet_trip: 
    if x>= 0.3: prop_1.append(-10**6)
    else: prop_1.append(0.3-x) 
    
prop_2 = []
for x in oscillator_stngth: 
    if x< 0.0: prop_2.append(-10**6)
    else: prop_2.append(x) 
    
fitness = np.array(prop_1) + np.array(prop_2)


# Create an orderings for the smiles: 
order, fitness_ordered, smiles_ordered = order_based_on_fitness(fitness, smiles_)

ratio      = 0.1 # Ratio of population to be preserved  
to_keep    = [i for i in range(int(ratio*len(smiles_ordered)))] # Smiles indices to be kept WITHIN smiles_ordered! 


to_replace = [i for i in range(to_keep[-1]+1, len(smiles_ordered))]

with open('dict_map_14.pickle', 'rb') as handle: 
    fragment_map = pickle.load(handle)    
    

with open('./CORES.smi', 'r') as f: 
    all_cores = f.readlines()
all_cores = [x.strip() for x in all_cores] # List of all cores

frag_smiles_ordered = [] # FRAGMENT ordered based on the fitness ordering in smiles_ordered
core_smiles_ordered = [] # CORE ordered based on the fitness ordering in smiles_ordered
for item in smiles_ordered: 
    if item in fragment_map: 
        frag_smiles_ordered.append(fragment_map[item][0])
        
        frag_ = fragment_map[item][-1]
        frag_ = frag_.replace('([R])', '')
        frag_ = frag_.replace('[R]', '')
        core_smiles_ordered.append(frag_)
    else: 
        frag_smiles_ordered.append('C')
        core_smiles_ordered.append(random.sample(all_cores, 1)[0])




smiles_next_gen, core_mutated = obtain_next_gen_molecules(order, to_replace, to_keep, frag_smiles_ordered, core_smiles_ordered)

from attc_smiles import attach_smiles
combined_origina_map, attached_smiles = attach_smiles(smiles_next_gen, core_mutated)


with open('./dict_map_15.pickle', 'wb') as f: 
    pickle.dump(combined_origina_map, f)

smiles_next_gen = attached_smiles.copy()

# Get all the unique smiles: 
canon_smi_ls = []
for item in smiles_next_gen: 
    mol, smi_canon, did_convert = sanitize_smiles(item)
    if mol == None or smi_canon == '' or did_convert == False: 
        raise Exception('Invalid smile string found')
        

    canon_smi_ls.append(smi_canon)
    
smiles_next_gen = canon_smi_ls.copy() # Now contains canonical smiles! 
canon_smi_ls        = list(set(canon_smi_ls))

############## ADD IN THE NN HERE #####################
NN_pass = []
for item in canon_smi_ls: 
    if eval_smi(item) == 1.0: 
        NN_pass.append(item)    
#######################################################    

for item in smiles_next_gen: 
    if item not in canon_smi_ls: 
        raise Exception('Could not find: ', item)

for item in list(collected_dict.keys()): 
    if item in canon_smi_ls: canon_smi_ls.remove(item)
    
    

f = open('./gen_15_unique.smi', 'a+')
f.writelines(['{} {}\n'.format(item, i+0) for i,item in enumerate(NN_pass)])
f.close()

