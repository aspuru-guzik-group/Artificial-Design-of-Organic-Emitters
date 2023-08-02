#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:44:14 2020

@author: akshat
"""
from glob import glob
from rdkit.Chem import MolFromSmiles as smi2mol

from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')




def sanitize_smiles(smi):
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)

# A = glob("./gen_1_out/*/")
A = glob("./gen_*/*/") # This is for all the output generation files! 
smi_to_prop = {}
failed_files = []

for dummy_dir in A: 

    try: 
        out_dir  = dummy_dir+'results.out'
        
        # Read in the properties: 
        with open(out_dir, 'r') as f: 
            out_ = f.readlines()
        out_ = out_[0].strip()
        out_ = out_.split(' ')
        out_ = [float(x) for x in out_]
        
        
        # Read in the smiles string:
        smi_file = dummy_dir + dummy_dir.split('/')[-2] + '.sh'
        with open(smi_file, 'r') as f: 
            smi_ = f.readlines()
        smi_ = smi_[3].strip()
        smi_ = smi_.split('smi="')[1]
        smi_ = smi_[:-1]
        
        _, smi_, _ = sanitize_smiles(smi_)
        smi_to_prop[smi_] = [out_[-2], out_[-1], out_[0]]
        
    except:
        failed_files.append(dummy_dir)


        
import pickle

# WRITE: 
with open('global_collect.pickle', 'wb') as handle:
    pickle.dump(smi_to_prop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# READ:     
with open('global_collect.pickle', 'rb') as handle:
    collected_dict = pickle.load(handle)    
    
    