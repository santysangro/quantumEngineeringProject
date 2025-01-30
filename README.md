# Simulation of a Hydrogen Molecule Using Logical Qubits

Welcome to **Group 6**'s repository!  
**Our goal** - Simulation of a Hydrogen Molecule using Logical Qubits.

In this repository, we provide all the code we developed while researching the hydrogen molecule and constructing our hamiltonian circuits.  
To observe our circuit and our results, run the Jupyter Notebook file as described in [Running](#running).

Contributors: Yair Chizi, Alexandru Cîrjaliu-Davidescu, Antreas Ioannou, Santiago Sangro  
Institution: Delft University of Technology, Faculty of Applied Sciences, Minor Programme Quantum Science and Quantum Information

## Dependencies and Installation

#### 0. Python
This project is coded in Python. For installation, please consult: https://www.python.org/.  
If Python version is lower than 3.14, pip might need to be installed separately. Try the following command, or visit https://pypi.org/project/pip/.
```
python -m ensurepip --upgrade
```

#### 1. Qiskit
Required for creation and running of quantum circuits.  
For more information, please consult: https://docs.quantum.ibm.com/guides.

Can be installed via pip:
```
pip install qiskit
```

#### 2. Jupyter Notebook
Required for running of the Simulation Pipeline file.  
Can be installed via pip:
```
pip install notebook
```

#### 3. Other Python Libraries
Required for diverse computations throughout the project.  
[library-name]: numpy, matplotlib, qiskit-aer, qiskit-nature, openfermion, openfermionpyscf, scipy, qiskit_algorithms 

Can be installed individually via pip:
```
pip install [library_name]
```
or to install them all at once:
```
pip install numpy matplotlib qiskit-aer qiskit-nature openfermion openfermionpyscf scipy qiskit_algorithms 
```

⚠️ **openfermionpyscf requires PySCF, which is not supported natively on Windows. Use the Windows Subsystem for Linux.**

## Project structure

- Logical Qubits Implementation

| File | Description |
|------|------------|
| `Simulation Pipeline.ipynb` | *Jupyter Notebook for creation & execution of custom circuits* |
| `doubleSpinBuilder.py` | *Builder file for Double-Qubit encoding* |
| `qiskitBuilder.py` | *Builder file for regular Physical qubits* |
| `steaneBuilder.py` | *Builder file for Steane encoding* |
| `vizualize.py` | *Plotting and Vizualization functions for graphs and statistics* |
| `util.py` | *Custom circuit construction file* |

- Quantum Phase Estimation

| File | Description |
|------|------------|
| `hydrogen_simulation_qiskit.py` | *Replication of the hydrogen simulation ground energy* |
| `quantum_phase_estimation.py` | *Qiskit implementation of the Hamiltonian and QPE* |

## Running

For **Quantum Phase Estimation**: You may run each individual Python file in your local environment.

For **Logical Qubits Implementation**, change your terminal's active directory via
```
cd "Logical Qubits Implementation"
```
and run the Jupyter Notebook file:
```
jupyter notebook "Simulation Pipeline.ipynb"
```
