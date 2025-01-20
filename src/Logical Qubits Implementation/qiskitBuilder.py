import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT

# GLOBALS:
# sim = StatevectorSimulator():

class qiskitBuilder():
    def __init__(self, qubit_num : int):
        # States: 0_L = Up Up, 1_L = Down Down
        # Always detects a single X-error.
        self.logical_num = qubit_num
        self.physical_num = qubit_num * 2
        self.pending = []
        self.autoPop = False
        self.qs = QuantumCircuit(QuantumRegister(self.physical_num))

    """
    A decorator for handling symmetric nesting of components
    """
    def handleAutoPop(self):
        if (self.autoPop):
            self.autoPop = False
            
            out = self.pending.pop()
            if (type(out) is list):
                for outt in out:
                    outt()
            else:
                out()

    
    @staticmethod
    def autoPopDecorator(func):
        def wrapper(self, *args, **kwargs):
            self.handleAutoPop()
            return func(self, *args, **kwargs)
        return wrapper

    
    """
    Add a single gate or multiple gate to the pending queue.
    !! DIRECT USAGE OUTSIDE OF THE CLASS IS DISCOURAGED
    """
    def _pushGate_(self, gate):
        self.pending.append(gate)
        self.autoPop = True


    def appendCircuit(qc, min_qubit, max_qubit, control_qubit = None):
        if (control_qubit == None):
            self.qs.append(qc, list(range(2 * min_qubit, 2 * max_qubit + 1))
        else:
            self.qs.append(qc, [2 * control_qubit, 2 * control_qubit + 1] + list(range(2 * min_qubit, 2 * max_qubit + 1)))
        
    """
    Embed further gates between the two symmetric gates; does not change the functionality if there is no
    active symmetric gate.
    """
    def embed(self):
        self.autoPop = False


    """
    Close a symmetric gate
    """
    def pop(self, n = 1):
        while n > 0:
            out = self.pending.pop()
            if (type(out) is list):
                for outt in out:
                    outt()
            else:
                out()
            n -= 1

    """
    Builds the string representing the circuit
    """
    def build(self):
        self.pop(n = len(self.pending))
        return self.qs


    """
    Adds a 180-degrees rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "i", "x", "y", and "z" to determine the axis. 
    """
    @autoPopDecorator
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
            # Z_L = Z_1 X_2 (Transversal)
            self.qs.y(2 * qubit)
            self.qs.x(2 * qubit + 1)
            if sym:
                self._pushGate_([lambda: self.qs.x(2 * qubit + 1), lambda: self.qs.z(2 * qubit)])

    """
    Adds a controlled 180-degrees rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "i", "x", "y", and "z" to determine the axis. 
    """
    @autoPopDecorator
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
            self.qs.cy(2 * control, 2 * target)
            self.qs.cx(2 * control + 1, 2 * target + 1)
            if sym:
                self._pushGate_([lambda: self.qs.cx(2 * control + 1, 2 * target + 1), lambda: self.qs.cy(2 * control, 2 * target)])
                
        elif axis == "z":
            # CZ_L = C1Z_1 C2X_2 (Transversal)
            self.qs.cz(2 * control, 2 * target)
            self.qs.cx(2 * control + 1, 2 * target + 1)
            if sym:
                self._pushGate_([lambda: self.qs.cx(2 * control + 1, 2 * target + 1), lambda: self.qs.cz(2 * control, 2 * target)])
                
    """
    Adds a custom rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "x", "y", and "z" to determine the axis and rotation direction. 
    """
    @autoPopDecorator
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
            # Rx(theta)_L = Rx^1(theta/2) Rx^2(theta/2) (transversal)
            self.qs.rx(angle / 2, qubit * 2)
            self.qs.rx(angle / 2, qubit * 2 + 1)
            if sym:
                self._pushGate_([lambda: self.qs.rx(angle / 2, qubit * 2 + 1), lambda: self.qs.rx(angle / 2, qubit * 2)])
        elif axis == "y":
            # Ry(theta)_L = Ry^1(theta/2) Ry^2(theta/2) (transversal)
            self.qs.ry(angle / 2, qubit * 2)
            self.qs.ry(angle / 2, qubit * 2 + 1)
            if sym:
                self._pushGate_([lambda: self.qs.ry(angle / 2, qubit * 2 + 1), lambda: self.qs.ry(angle / 2, qubit * 2)])
                
        elif axis == "z":
            # Rz(theta)_L = Rz^1(theta/2) Rz^2(theta/2) (transversal)
            self.qs.rz(angle / 2, qubit * 2)
            self.qs.rz(angle / 2, qubit * 2 + 1)
            if sym:
                self._pushGate_([lambda: self.qs.rz(angle / 2, qubit * 2 + 1), lambda: self.qs.rz(angle / 2, qubit * 2)])

    """
    Adds Hadamard gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.
    """
    @autoPopDecorator
    def addH(self, qubit, sym = False):

        # Handle multiple callings
        if type(qubit) is list :

            for i in range(0, len(qubit)):
                self.addH(qubit[i],sym)
                self.embed()

            return

        
        # Determine the rotation axis
        self.qs.h(qubit)
        if sym:
            self._pushGate_(lambda: self.qs.h(qubit))

    
