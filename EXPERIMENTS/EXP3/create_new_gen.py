#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 02:37:00 2020

@author: akshat
"""
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
from crossover import * 
from mutate import * 

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

def apply_generation_cutoff(order, generation_size):
    ''' Return of a list of indices of molecules that are kept (high fitness)
        and a list of indices of molecules that are replaced   (low fitness)

    The cut-off is imposed using a Fermi-Function

    Parameters:
    order (list)          : list of molecule indices arranged in Decreasing order of fitness
    generation_size (int) : number of molecules in a generation

    Returns:
    to_replace (list): indices of molecules that will be replaced by random mutations of
                       molecules in list 'to_keep'
    to_keep    (list): indices of molecules that will be kept for the following generations
    '''
    # Get the probabilities that a molecule with a given fitness will be replaced
    # a fermi function is used to smoothen the transition
    positions     = np.array(range(0, len(order))) - 0.45*float(len(order))
    probabilities = 1.0 / (1.0 + np.exp(-0.004 * generation_size * positions / float(len(order))))

    to_replace = [] # all molecules that are to be replaced
    to_keep    = [] # all molecules that are to be kept
    for idx in range(0, len(order)):
        if np.random.rand(1) < probabilities[idx]:
            to_replace.append(idx)
        else:
            to_keep.append(idx)

    return to_replace, to_keep



def obtain_next_gen_molecules(order, to_replace, to_keep, smiles_):
    smiles_mutated = []
    # selfies_mutated = []
    for idx in range(0,len(order)):
        print('On {}/{}'.format(idx, len(order)))
        if idx in to_replace: # smiles to replace (by better molecules)
        
            random_index = np.random.choice(to_keep, size=1, replace=True, p=None)[0]  # select a random molecule that survived
        
            # MUTATE WITH A GOOD MOL: 
            if np.random.random() < 0.5: 
                print('Performing mutation on: ! ', smiles_[random_index])
                smiles_new = get_mutated_smi(smiles_[random_index])  # do the mutation   
                # print(smiles_new)
                for smi in smiles_new: smiles_mutated.append(smi)
                
                f = open('./geneology/gen_15.txt', 'a+')
                f.writelines(['{} mut: {}\n'.format(smiles_new, smiles_[random_index])])
                f.close()
            
            # CROSSOVER WITH A GOOD MOL: 
            else: 
                print('Performing crossover! ')
                crossover_smi = crossover(smiles_[idx], smiles_[random_index])
                for smi in crossover_smi: smiles_mutated.append(smi)                
                f = open('./geneology/gen_15.txt', 'a+')
                f.writelines(['{} cross: {} {}\n'.format(crossover_smi, smiles_[idx], smiles_[random_index])])
                f.close()
                

        else: # smiles to be kept
            smiles_mutated.append(smiles_ordered[idx])
            
    return smiles_mutated


# Step 1: Read all the smiles and properties: 
import pickle
with open('global_collect.pickle', 'rb') as handle:
    collected_dict = pickle.load(handle)    



# Read in all of the gen 2 smiles: 
with open('gen_14_all.smi', 'r') as f: # TODO: FOR GENERATION 2! 
    content = f.readlines()
content = [x.split(' ')[0] for x in content]

# content = list(collected_dict.keys()) # TODO: This is only for generation 1! 

failed = []
smi_to_prop = {}
for item in content: 
    try: 
        smi_to_prop[item] = collected_dict[item]
    except: 
        failed.append(item)
        continue

# raise Exception('TEST')
        

#Step 2: Calculae the fitness function: 
# smi_to_prop = collected_dict.copy()
singlet_trip = []
oscillator_stngth = []
smiles_ = list(smi_to_prop.keys()) # All the smiles for which there are calculations! 

for key in smiles_: 
    singlet_trip.append(smi_to_prop[key][0])
    oscillator_stngth.append(smi_to_prop[key][1])


prop_1 = []
for x in singlet_trip: 
    if x>= 0.3: prop_1.append(-10**6)
    else: prop_1.append(0) 
    
prop_2 = []
for x in oscillator_stngth: 
    if x< 0.0: prop_2.append(-10**6)
    else: prop_2.append(x) 
    
fitness = np.array(prop_1) + np.array(prop_2)



# Step 3: Decide which members to_keep & to replace with mutations/ crossovers: 
order, fitness_ordered, smiles_ordered = order_based_on_fitness(fitness, smiles_)
to_replace, to_keep = apply_generation_cutoff(order, generation_size=len(smiles_ordered)) # len(smiles_ordered)

print('Keeping: {} Replacing: {}'.format(len(to_keep), len(to_replace)))

smiles_next_gen = obtain_next_gen_molecules(order, to_replace, to_keep, smiles_)

# Get all the unique smiles: 
canon_smi_ls = []
for item in smiles_next_gen: 
    mol, smi_canon, did_convert = sanitize_smiles(item)
    if mol == None or smi_canon == '' or did_convert == False: 
        raise Exception('Invalid smile string found')
    canon_smi_ls.append(smi_canon)
    
smiles_next_gen = canon_smi_ls.copy() # Now contains canonical smiles! 
canon_smi_ls        = list(set(canon_smi_ls))

for item in smiles_next_gen: 
    if item not in canon_smi_ls: 
        raise Exception('Could not find: ', item)

for item in list(collected_dict.keys()): 
    if item in canon_smi_ls: canon_smi_ls.remove(item)
    
    

f = open('./gen_15_unique.smi', 'a+')
f.writelines(['{} {}\n'.format(item, i+20000) for i,item in enumerate(canon_smi_ls)])
f.close()


# WRITING ALL THE SMILES! 
f = open('./gen_15_all.smi', 'a+')
for i,item in enumerate(smiles_next_gen): 
    f.writelines(['{} {}\n'.format(item, i)])
f.close()
