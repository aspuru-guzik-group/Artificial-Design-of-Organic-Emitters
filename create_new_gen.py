
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
from crossover import crossover
from mutate import get_mutated_smi

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


def obtain_next_gen_molecules(order, to_replace, to_keep, fragment_smiles, fragment_ordered):
    smiles_mutated = []

    for idx in range(0,len(order)):
        print('On {}/{}/ {}'.format(idx, len(order), fragment_smiles[idx]))

        if idx in to_replace: # smiles to replace (by better molecules)

            random_index = np.random.choice(to_keep, size=1, replace=True, p=None)[0]  # select a random molecule that survived
        
            # MUTATE WITH A GOOD MOL: 
            if np.random.random() < 0.5: 
                # print('Performing mutation on: ! ', fragment_smiles[random_index])
                smiles_new = get_mutated_smi(fragment_smiles[random_index])  # do the mutation   
                # print(smiles_new)
                for smi in smiles_new: smiles_mutated.append(smi)
                
                f = open('./geneology/gen_2.txt', 'a+')
                f.writelines(['{} mut: {}\n'.format(smiles_new, fragment_smiles[random_index])])
                f.close()
            
            # CROSSOVER WITH A GOOD MOL: 
            else: 
                crossover_smi = crossover(fragment_smiles[idx], fragment_smiles[random_index])
                for smi in crossover_smi: smiles_mutated.append(smi)
                
                f = open('./geneology/gen_2.txt', 'a+')
                f.writelines(['{} cross: {} {}\n'.format(crossover_smi, fragment_smiles[idx], fragment_smiles[random_index])])
                f.close()

        else: # smiles to be kept
            smiles_mutated.append(fragment_ordered[idx])

    return smiles_mutated


# Step 1: Read all the smiles and properties: 
import pickle
with open('global_collect.pickle', 'rb') as handle:
    collected_dict = pickle.load(handle)    



content = list(collected_dict.keys()) 


failed = []
smi_to_prop = {}
for item in content: 
    try: 
        smi_to_prop[item] = collected_dict[item]
    except: 
        failed.append(item)
        continue

        

#Step 2: Calculae the fitness function: 
singlet_trip = []
oscillator_stngth = []
smiles_ = list(smi_to_prop.keys()) # All the smiles for which there are calculations! 
smiles_ = smiles_[0:5] 

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



# Step 3: Decide which members to_keep & to replace with mutations/ crossovers: 
order, fitness_ordered, smiles_ordered = order_based_on_fitness(fitness, smiles_)
to_replace, to_keep = apply_generation_cutoff(order, generation_size=len(smiles_ordered)) # len(smiles_ordered)

raise Exception('Getting used to the ordering :) ')

with open('dict_map_1.pickle', 'rb') as handle: # TODO: The fragments
    fragment_map = pickle.load(handle)    
    
print('Combined smiles: ', smiles_)
fragment_smiles = []
for item in smiles_: 
    if item in fragment_map: 
        fragment_smiles.append(fragment_map[item][0])
    else: 
        fragment_smiles.append('C')
smiles_ = fragment_smiles.copy() # Contains all the fragment smiles now! 


fragment_ordered = []
for item in smiles_ordered: 
    if item in fragment_map: 
        fragment_smiles.append(fragment_map[item][0])
    else: 
        fragment_smiles.append('C')
        
##################################

print('Keeping: {} Replacing: {}'.format(len(to_keep), len(to_replace)))



smiles_next_gen = obtain_next_gen_molecules(order, to_replace, to_keep, smiles_, fragment_ordered)
gen_smiles = smiles_next_gen.copy()
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

#### MAKE smiles_next_gen unique! 
smiles_next_gen = canon_smi_ls.copy()
f = open('./frags_gen_2.smi', 'a+')
for i,item in enumerate(smiles_next_gen): 
    f.writelines(['{} {}\n'.format(item, i)])
f.close()


from attc_smiles import attach_smiles
combined_origina_map, attached_smiles = attach_smiles(smiles_next_gen)

with open('./dict_map_2.pickle', 'wb') as f: 
    pickle.dump(combined_origina_map, f)

smiles_next_gen = attached_smiles.copy()
# raise Exception('Obtaining next generation of smiles :) ')


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
    
    

f = open('./gen_2_unique.smi', 'a+')
f.writelines(['{} {}\n'.format(item, i+0) for i,item in enumerate(canon_smi_ls)])
f.close()


