# Artificial Design of Organic Emitters via a Genetic Algorithm Enhanced by a Deep Neural Network
This repository contains code for the paper: [Artificial Design of Organic Emitters via a Genetic Algorithm Enhanced by a Deep Neural Network](TODO). 
By: AkshatKumar Nigam, Robert Pollice, and Alán Aspuru-Guzik

## Data Availability

For every experiment conducted, we have made available the corresponding data set. This includes the SMILES string, singlet-triplet gap (STG), oscillator strength (OS), vertical excitation energy (VEE), and optimized xyz coordinates of molecules. Each of these is provided as a pickle file. Below are the respective links:

1. [Experiment 1](https://drive.google.com/file/d/1XW8vF_RYZMJpgqjWy4UaiugR9ZN1GogG/view?usp=sharing): Methane Seed - Optimization of STG and OS
2. [Experiment 2](https://drive.google.com/file/d/1Bn1YZhMJsMCJkyN-rEpF3DUpFqySlejU/view?usp=sharing): Optimization of STG and OS
3. [Experiment 3](https://drive.google.com/file/d/10-GhyN5qAp1tomuiaD_WB3eqXjWHOJDY/view?usp=sharing): Optimization of OS
4. [Experiment 4](https://drive.google.com/file/d/1BBXp1jSyg4f5Ljm0-eIbfyl03NU_dlIe/view?usp=sharing): Optimization of STG, OS, and VEE
5. [Experiment 5](https://drive.google.com/file/d/19MeNvzUIGAPFxg9qz8Gsgox3ggMStx0M/view?usp=sharing): Optimization of STG and OS
6. [Experiment 6](https://drive.google.com/file/d/1CM05aY-SCCpth3pu9M5j_cZw_mtMBKfZ/view?usp=sharing): Optimization of STG and OS

For loading the data, the following code snippet can be utilized:

```python
import pickle
with open("./collect_FINAL_EXP2.pickle", "rb") as input_file:
    data = pickle.load(input_file)  # Smiles -> [singlet_triplet_gap, oscillator_strength, excitation_energy, xyz_file]
```

## File Navigator

We recommend using our genetic algorithm, JANUS, which is available through pip-installation. You can find this at [JANUS Repository](https://github.com/aspuru-guzik-group/JANUS). For more detailed information, you can refer to our publication: [JANUS Paper](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00003b#!).

The codes in this repository are a developmental version of JANUS, used to generate molecules. Here's a breakdown of key files and their roles:

- `mutate.py`: This script is used to generate a list of mutated (or altered) molecules, starting from a base set.
- `crossover.py`: This script creates a list of molecules that embody characteristics from two parent molecules.
- `create_new_gen.py`: This file facilitates the transition from generation 'x' to 'x+1' by invoking the mutation and crossover functions.
- `EXPERIMENTS/`: This directory contains both the adjusted scripts (incorporating the relevant fitness functions) and the corresponding neural networks, organized based on the specific experiments conducted.



## Questions, problems?
Make a github issue 😄. Please be as clear and descriptive as possible. Please feel free to reach
out in person: (akshat[DOT]nigam[AT]mail[DOT]utoronto[DOT]ca,  r[DOT]pollice[AT]rug[DOT]nl)


## License

TODO