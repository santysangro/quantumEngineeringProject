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
        self.qs.h(2 * qubit)
        self.qs.cx(2 * qubit, 2 * qubit + 1)
        if sym:
            self._pushGate_([lambda: self.qs.h(2 * qubit), lambda: self.qs.cx(2 * qubit, 2 * qubit + 1)])

    """
    Adds Hadamard gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.
    """
    @autoPopDecorator
    def addP(self, qubit, angle, sym = False):

        # Handle multiple callings
        if type(qubit) is list and type(angle) is list and len(qubit) == len(angle):

            for i in range(0, len(qubit)):
                self.addP(qubit[i], angle[i], sym)
                self.embed()
            return

        
        # Determine the rotation axis
        self.qs.p(2 * qubit, angle)
        if sym:
            self._pushGate_(lambda: self.qs.p(2 * qubit, angle))

    @autoPopDecorator
    def addCP(self, control, target, angle, sym = False):

        # Handle multiple callings
        if type(control) is list and type(target) is list and type(angle) is list and len(qubit) == len(angle):

            for i in range(0, len(qubit)):
                self.addCP(control[i], target[i], angle[i], sym)
                self.embed()
            return

        
        self.qs.cp(2 * control, 2 * target, angle)
        
        if sym:
            self._pushGate_(lambda: self.qs.cp(2 * control, 2 * target, angle))

def generateHamiltonian(theta):
    # Build a hamiltonian
    hamiltonianBuilder = qiskitBuilder(4)

    
    # First Part
    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addP(3, theta[0])
    hamiltonianBuilder.pop(n = 3)

    hamiltonianBuilder.addRotation(["z"] * 3, [0, 1, 2], [theta[1], theta[2], theta[3]])
    
    hamiltonianBuilder.addCPauli("x", 0, 1, sym = True)

    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 1, theta[4])
    hamiltonianBuilder.pop()

    # Second Part
    
    hamiltonianBuilder.addCPauli("x", 0, 2, sym = True)

    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 2, theta[5])
    hamiltonianBuilder.pop()

    hamiltonianBuilder.addCPauli("x", 1, 3, sym = True)

    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[6])
    hamiltonianBuilder.pop()

    hamiltonianBuilder.addH([0,2], sym=True)

    hamiltonianBuilder.embed()

    hamiltonianBuilder.addCPauli(["x"] * 2, [0, 1], [1, 2], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 2, theta[7])
    hamiltonianBuilder.pop(n = 4)

    hamiltonianBuilder.addRotation("x", [0,2], [0.785] * 2)

    hamiltonianBuilder.addCPauli(["x"] * 2, [0, 1], [1, 2], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 2, theta[8])
    hamiltonianBuilder.pop(n = 2)

    hamiltonianBuilder.addRotation("x", [0,2], [-0.785] * 2)

    hamiltonianBuilder.addCPauli(["x"] * 2, [0, 1], [1, 2], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 2, theta[9])
    hamiltonianBuilder.pop(n = 2)


    # Third Part
    hamiltonianBuilder.addCPauli(["x"] * 2, [0, 2], [2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[10])
    hamiltonianBuilder.pop(n = 2)

    hamiltonianBuilder.addCPauli(["x"] * 2, [1, 2], [2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[11])
    hamiltonianBuilder.pop(n = 2)

    hamiltonianBuilder.addH([0,2], sym=True)

    hamiltonianBuilder.embed()

    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[12])
    hamiltonianBuilder.pop(n = 5)

    hamiltonianBuilder.addRotation("x", [0,2], [0.785] * 2)

    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[13])
    hamiltonianBuilder.pop(n = 3)

    hamiltonianBuilder.addRotation("x", [0,2], [-0.785] * 2)

    # Fourth Part
    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[14])
    hamiltonianBuilder.pop(n = 3)

    hamiltonianBuilder.addH([0, 1, 2, 3])
    return hamiltonianBuilder.build()


def generateQFT(num_logical):
    
    qftBuilder = qiskitBuilder(num_logical)
    
    #for (target in range(0, num_logical)):
    #    qftBuilder.addH(target)
    #    for (control in range(target + 1, num_logical)):
            





