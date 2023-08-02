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


import torch 
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem, DataStructs


def randomize_smiles(mol):
    """
    Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.

    """
    if not mol:
        return None

    Chem.Kekulize(mol)
    
    return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True)



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
    

def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie


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

def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    '''Return a mutated selfie string
    
    Mutations are done until a valid molecule is obtained 
    Rules of mutation: With a 50% propbabily, either: 
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
    
    Parameters:
    selfie            (string)  : SELFIE string to be mutated 
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"
    
    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    '''
    valid=False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)
    
    while not valid:
        fail_counter += 1
                
        alphabet = list(selfies.get_semantic_robust_alphabet()) + ['[C][=C][C][=C][C][=C][Ring1][Branch1_2]']*10 # 34 SELFIE characters 

        choice_ls = [1, 2, 3] # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]
        
        # Insert a character in a Random Location
        if random_choice == 1: 
            random_index = np.random.randint(len(chars_selfie)+1)
            random_character = np.random.choice(alphabet, size=1)[0]
            
            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

        # Replace a random character 
        elif random_choice == 2:                         
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
                
        # Delete a random character
        elif random_choice == 3: 
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index+1:]
                
        else: 
            raise Exception('Invalid Operation trying to be performed')

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)
        
        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon=="":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid=False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                f.close()
    
    return (selfie_mutated, smiles_canon)

def get_mutated_SELFIES(selfies_ls, num_mutations): 
    for _ in range(num_mutations): 
        selfie_ls_mut_ls = []
        for u,str_ in enumerate(selfies_ls): 
            # print('Mutate {}/{}'.format(u,len(selfies_ls)))            
            str_chars = get_selfie_chars(str_)
            # max_molecules_len = len(str_chars)*num_mutations
            
            selfie_mutated, _ = mutate_selfie(str_, max_molecules_len=500)
            selfie_ls_mut_ls.append(selfie_mutated)
        
        selfies_ls = selfie_ls_mut_ls.copy()
    return selfies_ls


def get_fp_scores(smiles_back, target_smi, fp_type): 
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    # fp_target = get_ECFP4(target)
    fp_target = get_fingerprint(target, fp_type)

    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        # fp_mol = get_ECFP4(mol)
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


def get_mutated_smi(smi): 

    num_random_samples = 40 # TODO 
    num_mutation_ls    = [1, 2, 3, 4, 5]

    # mol = Chem.MolFromSmiles(smi)
    mol, smi_START, did_convert = sanitize_smiles(smi)

    if mol == None: 
        raise Exception('Invalid starting structure encountered')
    
        
    randomized_smile_orderings  = [randomize_smiles(mol) for _ in range(num_random_samples)]
    # print('Len is : ', len(randomized_smile_orderings))
    
    # Convert all the molecules to SELFIES
    selfies_ls = [encoder(x) for x in randomized_smile_orderings]
    
    all_smiles_collect = []
    all_smiles_collect_broken = []
    
    for num_mutations in num_mutation_ls: 
        # print('On: ', num_mutations)
        # print('On ', num_mutations)
        # Mutate the SELFIE string: 
        selfies_mut = get_mutated_SELFIES(selfies_ls.copy(), num_mutations=num_mutations)
        
        # Convert back to SMILES: 
        smiles_back = [decoder(x) for x in selfies_mut]
        all_smiles_collect = all_smiles_collect + smiles_back
        all_smiles_collect_broken.append(smiles_back)
        

    # Work on:  all_smiles_collect
    canon_smi_ls = []
    for item in all_smiles_collect: 
        mol, smi_canon, did_convert = sanitize_smiles(item)
        if mol == None or smi_canon == '' or did_convert == False: 
            raise Exception('Invalid smile string found')
        canon_smi_ls.append(smi_canon)
    canon_smi_ls        = list(set(canon_smi_ls))
    
    from filter_ import passes_filter
    filter_pass = []
    for item in canon_smi_ls: 
        try: 
            if passes_filter(item) == True:
                filter_pass.append(item)
        except: 
            print('Filter Failed on: ', item)
    
    if smi_START in filter_pass: 
        filter_pass.remove(smi_START)
        
    NN_pass = []
    for item in filter_pass: 
        if eval_smi(item) == 1.0: 
            NN_pass.append(item)
    
    # return filter_pass[0:5]
    print('Mut: ', NN_pass)
    return NN_pass

    
# A = get_mutated_smi('Sc1ccnc2c3cncc-3ccnc12')
# print(A)
