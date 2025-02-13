from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister
import numpy as np
from qiskitBuilder import qiskitBuilder

class doubleSpinBuilder(qiskitBuilder):
    def __init__(self, qubit_num : int):
        # States: 0_L = Up Up, 1_L = Down Down
        # Always detects a single X-error.
        self.logical_num = qubit_num
        self.physical_num = qubit_num * 2
        self.pending = []
        self.autoPop = False
        self.qs = QuantumCircuit(QuantumRegister(self.physical_num))


    def appendCircuit(self, qc, min_qubit, max_qubit, control_qubit = None):
        if (control_qubit == None):
            self.qs.append(qc, list(range(2 * min_qubit, 2 * max_qubit + 1)))
        else:
            self.qs.append(qc, [2 * control_qubit, 2 * control_qubit + 1] + list(range(2 * min_qubit, 2 * max_qubit + 1)))
        
    

    """
    Adds a 180-degrees rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "i", "x", "y", and "z" to determine the axis. 
    """
    @qiskitBuilder.autoPopDecorator
    def addPauli(self, axis, qubit, sym = False):

        # Handle multiple callings
        if type(axis) is list or type(qubit) is list:

            # Validate argument types for multiple callings
            if type(axis) is not list or type(qubit) is not list or len(axis) != len(qubit):
                raise ValueError("Both arguments must be lists of equal length for multiple calling")

            for i in range(0, len(axis)):
                self.addPauli(axis[i], qubit[i], sym)
                self.embed()

            return
        
        # Determine the rotation axis
        if axis == "x":
            # X_L = X_1 X_2 (Transversal)
            self.qs.x(2 * qubit)
            self.qs.x(2 * qubit + 1)
            if sym:
                self._pushGate_([lambda: self.qs.x(2 * qubit + 1), lambda: self.qs.x(2 * qubit)])
            
        elif axis == "y":
            # Y_L = Y_1 X_2 (Transversal)
            self.qs.y(2 * qubit)
            self.qs.x(2 * qubit + 1)
            if sym:
                self._pushGate_([lambda: self.qs.x(2 * qubit + 1), lambda: self.qs.y(2 * qubit)])
                
        elif axis == "z":
            # Z_L = Z_1 (Transversal)
            self.qs.z(2 * qubit)
            if sym:
                self._pushGate_([lambda: self.qs.x(2 * qubit + 1), lambda: self.qs.z(2 * qubit)])
    """
    Adds a controlled 180-degrees rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "i", "x", "y", and "z" to determine the axis. 
    """
    @qiskitBuilder.autoPopDecorator
    def addCPauli(self, axis, control, target, sym = False):

        # Handle multiple callings
        if type(axis) is list or type(control) is list or type(target) is list:

            # Validate argument types for multiple callings
            if type(axis) is not list or type(control) is not list or type(target) is not list or len(axis) != len(control) or len(axis) != len(target):
                raise ValueError("Both arguments must be lists of equal length for multiple calling")

            for i in range(0, len(axis)):
                self.addCPauli(axis[i], control[i], target[i], sym)
                self.embed()

            return
        
        # Determine the rotation axis
        if axis == "x":
            # CX_L = C1X_1 C2X_2 (Transversal)
            
            self.qs.cx(2 * control, 2 * target)
            self.qs.cx(2 * control + 1, 2 * target + 1)
            if sym:
                self._pushGate_([lambda: self.qs.cx(2 * control + 1, 2 * target + 1), lambda: self.qs.cx(2 * control, 2 * target)])
            
        elif axis == "y":
            # CY_L = C1Y_1 C2X_2 (Transversal)
            self.qs.cy(control, 2 * target)
            self.qs.cx(2 * control + 1, 2 * target + 1)
            if sym:
                self._pushGate_([lambda: self.qs.cx(2 * control + 1, 2 * target + 1), lambda: self.qs.cy(2 * control, 2 * target)])
                
        elif axis == "z":
            # CZ_L = C1Z_1  (Transversal)
            self.qs.cz(2 * control, 2 * target)
            if sym:
                self._pushGate_([lambda: self.qs.cx(2 * control + 1, 2 * target + 1), lambda: self.qs.cz(2 * control, 2 * target)])
     
    """
    Adds a custom rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "x", "y", and "z" to determine the axis and rotation direction. 
    """
    @qiskitBuilder.autoPopDecorator
    def addRotation(self, axis, qubit, angle, sym = False):

        # Handle multiple callings
        if type(axis) is list or type(qubit) is list or type(angle) is list:

            # Validate argument types for multiple callings
            if type(axis) is not list or type(qubit) is not list or type(angle) is not list or len(axis) != len(qubit) or len(axis) != len(angle):
                raise ValueError("All arguments must be lists of equal length for multiple calling")

            for i in range(0, len(axis)):
                self.addRotation(axis[i], qubit[i], angle[i], sym)
                self.embed()

            return
        
        # Determine the rotation axis
        if axis == "x":
            # Rx(theta)_L = CNOT RX_1 CNOT
            self.qs.cx(qubit * 2, qubit * 2 + 1)
            self.qs.rx(angle, qubit * 2)
            self.qs.cx(qubit * 2, qubit * 2 + 1)
            if sym:
                self._pushGate_([lambda: self.qs.cx(qubit * 2, qubit * 2 + 1), lambda: self.qs.rx(angle, qubit * 2), lambda: self.qs.cx(qubit * 2, qubit * 2 + 1)])
        elif axis == "y":
            # Ry(theta)_L = CNOT RY_1 CNOT
            self.qs.cx(qubit * 2, qubit * 2 + 1)
            self.qs.ry(angle, qubit * 2)
            self.qs.cx(qubit * 2, qubit * 2 + 1)
            if sym:
                self._pushGate_([lambda: self.qs.cx(qubit * 2, qubit * 2 + 1), lambda: self.qs.ry(angle, qubit * 2), lambda: self.qs.cx(qubit * 2, qubit * 2 + 1)])
                
        elif axis == "z":
            # RZ(theta)_L = CNOT RZ_1 CNOT
            self.qs.cx(qubit * 2, qubit * 2 + 1)
            self.qs.rz(angle, qubit * 2)
            self.qs.cx(qubit * 2, qubit * 2 + 1)
            if sym:
                self._pushGate_([lambda: self.qs.cx(qubit * 2, qubit * 2 + 1), lambda: self.qs.rz(angle, qubit * 2), lambda: self.qs.cx(qubit * 2, qubit * 2 + 1)])

    """
    Adds Hadamard gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.
    """
    @qiskitBuilder.autoPopDecorator
    def addH(self, qubit, sym = False):

        # Handle multiple callings
        if type(qubit) is list :

            for i in range(0, len(qubit)):
                self.addH(qubit[i],sym)
                self.embed()

            return

        # Determine the rotation axis
        self.qs.cx(2 * qubit, 2 * qubit + 1)
        self.qs.h(2 * qubit)
        self.qs.cx(2 * qubit, 2 * qubit + 1)
        if sym:
            self._pushGate_([lambda: self.qs.cx(2 * qubit, 2 * qubit + 1), lambda: self.qs.h(2 * qubit), lambda: self.qs.cx(2 * qubit, 2 * qubit + 1)])

    """
    Adds Hadamard gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.
    """
    @qiskitBuilder.autoPopDecorator
    def addP(self, qubit, angle, sym = False):

        # Handle multiple callings
        if type(qubit) is list and type(angle) is list and len(qubit) == len(angle):

            for i in range(0, len(qubit)):
                self.addP(qubit[i], angle[i], sym)
                self.embed()
            return

        
        # Determine the rotation axis
        self.qs.p(angle, 2 * qubit)
        if sym:
            self._pushGate_(lambda: self.qs.p(2 * qubit, angle))

    @qiskitBuilder.autoPopDecorator
    def addCP(self, control, target, angle, sym = False):

        # Handle multiple callings
        if type(control) is list and type(target) is list and type(angle) is list and len(qubit) == len(angle):

            for i in range(0, len(qubit)):
                self.addCP(control[i], target[i], angle[i], sym)
                self.embed()
            return

        
        self.qs.cp(angle, 2 * control, 2 * target)
        
        if sym:
            self._pushGate_(lambda: self.qs.cp(2 * control, 2 * target, angle))


    def initializeToLogicalGround(self, initial_state = None):

        if (initial_state == None):
            # No initialization required
            self.qs.i(0)
        else:
            self.qs.initialize(initial_state, list(range(0, self.physical_num, 2)))
            for log_qubit in range(0, self.logical_num):
                self.qs.cx(2 * log_qubit, 2 * log_qubit + 1)
                
    @qiskitBuilder.autoPopDecorator
    def addGlobalPhaseShift(self, qubit1, angle, sym = False):
        # Handle multiple callings
        if type(qubit1) is list and type(angle) is list and len(qubit1) == len(angle):

            for i in range(0, len(qubit1)):
                self.addGlobalPhaseShift(qubit1[i], angle[i], sym)
                self.embed()
            return

        Ri_matrix = np.array([[np.exp(-1j * angle / 2), 0],
                          [0, np.exp(-1j * angle / 2)]])
        unitary_gate = Operator(Ri_matrix)

        self.qs.append(unitary_gate, [2*qubit1])
        #self.qs.append(unitary_gate, [2*qubit1+1])
        if sym:
            self.pushGate([lambda: self.qs.append(unitary_gate, [2*qubit1])])
