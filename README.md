# Simulation of a Hydrogen Molecule Using Logical Qubits

Welcome to our project's repository.

## Dependencies and Installation

### 0. Python
This project is coded in Python. For installation, please consult: https://www.python.org/.

If Python version is lower than 3.14, pip might need to be installed separately. Try the following command, or visit https://pypi.org/project/pip/.
```
python -m ensurepip --upgrade
```

### 1. Qiskit. 
Required for creation and running of quantum circuits.

For more information, please consult: https://docs.quantum.ibm.com/guides.

Can be installed via pip:
```
pip install qiskit
```

### 2. Jupyter Notebook
Required for running of the Simulation Pipeline file.

Can be installed via pip:
```
pip install qiskit
```

### 3. Other Python Libraries
Required for diverse computations throughout the project.

[library-name]: numpy, matplotlib, qiskit-aer, qiskit-nature, openfermion, openfermionpyscf, scipy, qiskit_algorithms 

Can be installed via pip:
```
pip install [library_name]
```
or to install them all at once:
```
pip install numpy matplotlib qiskit-aer qiskit-nature openfermion openfermionpyscf scipy qiskit_algorithms 
```

⚠️ **openfermionpyscf requires PySCF, which is not supported natively on Windows. Use the Windows Subsystem for Linux.**

## Running

For example usage see the python scripts and Jupyter notebooks in the [docs/examples](docs/examples) directory
when installed from source or the share/doc/quantuminspire/examples/ directory in the
library root (Python’s sys.prefix for system installations; site.USER_BASE for user
installations) when installed from PyPI.

For example, to run the ProjectQ example notebook after installing from source:

```
cd docs/examples
jupyter notebook example_projectq.ipynb
```

Or to perform Grover's with the ProjectQ backend from a Python script:

```
cd docs/examples
python example_projectq_grover.py
```

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuTech-Delft/quantuminspire/master?filepath=docs/examples)

Another way to browse and run the available notebooks is by clicking the 'launch binder' button above.

It is also possible to use the API through the QuantumInspireAPI object
directly. This is for advanced users that really know what they are
doing. The intention of the QuantumInspireAPI class is that it is used
as a thin layer between existing SDK's such as ProjectQ and Qiskit,
and is not primarily meant for general use. You may want to explore this
if you intend to write a new backend for an existing SDK.

A simple example to perform entanglement between two qubits by using the
API wrapper directly:

```python
from getpass import getpass
from coreapi.auth import BasicAuthentication
from quantuminspire.api import QuantumInspireAPI

print('Enter mail address')
email = input()

print('Enter password')
password = getpass()

server_url = r'https://api.quantum-inspire.com'
authentication = BasicAuthentication(email, password)
qi = QuantumInspireAPI(server_url, authentication, 'my-project-name')

qasm = '''version 1.0

qubits 2

H q[0]
CNOT q[0], q[1]
Measure q[0,1]
'''

backend_type = qi.get_backend_type_by_name('QX single-node simulator')
result = qi.execute_qasm(qasm, backend_type=backend_type, number_of_shots=1024)

if result.get('histogram', {}):
    print(result['histogram'])
else:
    reason = result.get('raw_text', 'No reason in result structure.')
    print(f'Result structure does not contain proper histogram data. {reason}')
```

## Configure a project name for Quantum Inspire

As a default, SDK stores the jobs in a Quantum Inspire project with the name "qi-sdk-project-" concatenated with a
unique identifier for each run. Providing a project name yourself makes it easier to find the project in the Quantum
Inspire web-interface and makes it possible to gather related jobs to the same project.

Qiskit users do something like:
```python
from coreapi.auth import BasicAuthentication
from quantuminspire.qiskit import QI

authentication = BasicAuthentication("email", "password")
QI.set_authentication(authentication, project_name='my-project-name')
```
or set the project name separately after setting authentication
```python
from coreapi.auth import BasicAuthentication
from quantuminspire.qiskit import QI

authentication = BasicAuthentication("email", "password")
QI.set_authentication(authentication)
QI.set_project_name('my-project-name')
```
ProjectQ users set the project name while initializing QuantumInspireAPI:
```python
from coreapi.auth import BasicAuthentication
from quantuminspire.api import QuantumInspireAPI

authentication = BasicAuthentication("email", "password")
qi_api = QuantumInspireAPI(authentication=authentication, project_name='my-project-name')
```

## Configure your token credentials for Quantum Inspire

1. Create a Quantum Inspire account if you do not already have one.
2. Get an API token from the Quantum Inspire website.
3. With your API token run:
```python
from quantuminspire.credentials import save_account
save_account('YOUR_API_TOKEN')
```
After calling save_account(), your credentials will be stored on disk.
Those who do not want to save their credentials to disk should use instead:
```python
from quantuminspire.credentials import enable_account
enable_account('YOUR_API_TOKEN')
```
and the token will only be active for the session.

After calling save_account() once or enable_account() within your session, token authentication is done automatically
when creating the Quantum Inspire API object.

For Qiskit users this means:
```python
from quantuminspire.qiskit import QI
QI.set_authentication()
```
ProjectQ users do something like:
```python
from quantuminspire.api import QuantumInspireAPI
qi = QuantumInspireAPI()
```
To create a token authentication object yourself using the stored token you do:
```python
from quantuminspire.credentials import get_authentication
authentication = get_authentication()
```
This `authentication` can then be used to initialize the Quantum Inspire API object.

## Testing

Run all unit tests and collect the code coverage using:

```
coverage run --source="./src/quantuminspire" -m unittest discover -s src/tests -t src -v
coverage report -m
```

## Known issues

* Known issues and common questions regarding the Quantum Inspire platform
  can be found in the [FAQ](https://www.quantum-inspire.com/faq/).

## Bug reports

Please submit bug-reports [on the github issue tracker](https://github.com/QuTech-Delft/quantuminspire/issues).
