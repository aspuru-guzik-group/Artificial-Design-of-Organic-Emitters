# Artificial Design of Organic Emitters via a Genetic Algorithm Enhanced by a Deep Neural Network
This repository contains code for the paper: [Artificial Design of Organic Emitters via a Genetic Algorithm Enhanced by a Deep Neural Network]([https://chemrxiv.org/engage/chemrxiv/article-details/64ca6585dfabaf06ff958a4f](https://doi.org/10.26434/chemrxiv-2023-nrxtl)). 
By: AkshatKumar Nigam, Robert Pollice, Pascal Friederich and Alán Aspuru-Guzik

## Data Availability

The results of the high-throughput virtual screening of the subset of GDB-13 can be downloaded via this [link](https://drive.google.com/file/d/1RgNKbc6S--fu8bUmAowyvh5teLT44Bbl/view?usp=sharing). For a comprehensive description of the full GDB-13 dataset, please consult the original [publication](https://doi.org/10.1021/ja902302h). For every artificial design experiment conducted, we have made the corresponding data sets available. This includes the SMILES strings, singlet-triplet gaps (STG), oscillator strengths (OS), vertical excitation energies (VEE), and optimized Cartesian coordinates of the molecules. Each of these is provided as a pickle file. Below are the respective links:

1. [Experiment 1](https://drive.google.com/file/d/1XW8vF_RYZMJpgqjWy4UaiugR9ZN1GogG/view?usp=sharing): Methane Seed - Optimization of STG and OS
2. [Experiment 2](https://drive.google.com/file/d/1Bn1YZhMJsMCJkyN-rEpF3DUpFqySlejU/view?usp=sharing): Optimization of STG and OS
3. [Experiment 3](https://drive.google.com/file/d/10-GhyN5qAp1tomuiaD_WB3eqXjWHOJDY/view?usp=sharing): Optimization of OS
4. [Experiment 4](https://drive.google.com/file/d/1BBXp1jSyg4f5Ljm0-eIbfyl03NU_dlIe/view?usp=sharing): Optimization of STG, OS, and VEE
5. [Experiment 5](https://drive.google.com/file/d/19MeNvzUIGAPFxg9qz8Gsgox3ggMStx0M/view?usp=sharing): Optimization of STG and OS
6. [Experiment 6](https://drive.google.com/file/d/1CM05aY-SCCpth3pu9M5j_cZw_mtMBKfZ/view?usp=sharing): Optimization of STG and OS

The pickle files are structured as dictionaries, where the key-value pairs are formatted as: SMILES string -> [singlet_triplet_gap, oscillator_strength, excitation_energy, xyz_file].

For loading the data, the following code snippet can be utilized:

```python
import pickle
with open("./collect_FINAL_EXP2.pickle", "rb") as input_file:
    data = pickle.load(input_file)  # Smiles -> [singlet_triplet_gap, oscillator_strength, excitation_energy, xyz_file]
```

## Prerequisites: 

The following are required for running the scripts: 
- [SELFIES (any version)](https://github.com/aspuru-guzik-group/selfies) - 
  The code was run with v1.0.1.
- [RDKit](https://www.rdkit.org/docs/Install.html)
- [Python 3.0 or up](https://www.python.org/download/releases/3.0/)
- [numpy](https://pypi.org/project/numpy/)
- [Pytorch](https://pytorch.org/)

## File Navigator

For applying our genetic algorithm to other inverse molecular design tasks, we recommend using the most recent version of JANUS, which is available for installation through pip. You can find the code at [JANUS Repository](https://github.com/aspuru-guzik-group/JANUS). For more detailed information, you can refer to our publication: [JANUS Paper](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00003b#!).

The codes in this repository are a development version of JANUS, which we used to generate molecules. Here is a breakdown of key files and their roles:

- `mutate.py`: This script is used to generate a list of mutated (or altered) molecules, starting from a base set.
- `crossover.py`: This script creates a list of molecules that embody characteristics from two parent molecules.
- `create_new_gen.py`: This file facilitates the transition from generation 'x' to 'x+1' by invoking the mutation and crossover functions.
- `EXPERIMENTS/`: This directory contains both the adjusted scripts (incorporating the relevant fitness functions) and the corresponding neural networks, organized based on the specific experiments conducted.
- `classification_models.zip`: This archive contains the code for training classification models using Bayesian optimization, specifically tailored to individual experiments, and also includes the top-performing models observed across these experiments.
- `data/`: This directory contains the results of the high-throughput virtual screening performed and of the validation.
- `inputs/`: This directory contains the input files of the quantum chemical simulations carried out.



## Questions, problems?
Make a github issue 😄. Please be as clear and descriptive as possible. Please feel free to reach
out in person: (akshat[DOT]nigam[AT]mail[DOT]utoronto[DOT]ca,  r[DOT]pollice[AT]rug[DOT]nl)


## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
