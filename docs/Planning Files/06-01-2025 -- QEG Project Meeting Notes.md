# QEG Project -- Meeting Notes 

### General Information
**Time:** 06-Jan-2025
**Location:** Classroom 6 @ TU Delft
**Attendance:** Alexandru, Antreas, Yair;  Santiago online.

### Quantum Inspire Installation
- All group members managed to successfully create a QI account, install the SDK locally, and run the example Jupyter Notebook locally.
### Communication and File Sharing Platforms
**Communication**: via a WhatsApp group.
**File Sharing**: via a git repository hosted on GitHub.
### Backend Platform Overview
- Spin-2:
	- Spin-based semiconductor platform
	- 2 qubits
	- Limitations:
		- Might not be available during the duration of the project due to maintenance.
		- A limited number of qubits may not allow us to investigate problems with high qubit count overhead.
	- Positives:
		- A physical working platform
- Starmon-5:
	- Transmon-based superconductor platform
	- 5 qubits
	- Limitations:
		- No binary bits to use.
		- No debugging capability in case error arises.
		- May be prone to error.
		- May not be able to handle complex circuits due to decoherence.
		- May require blocking to ensure correct operation order.
		- Its service is not on-demand; may be unreliable.
	- Positives:
		- A complete set of universal quantum operators.
		- A physical working platform.
- Simulation:
	- Limitations:
		- Not a physical platform
		- For highly complex problems, may require a very long execution time with high space overhead.
	- Positives:
		- Works with 'any' qubit number.
		- Allows for all quantum operations.
		- Allows for debugging of the system's state.
		- Works on-demand; almost always reliable

Which platforms to use:
- We will use the simulation by default since it's always reliable.
	- Should any physical platform be available during the data collection period of the project, we may also simulate on them.

