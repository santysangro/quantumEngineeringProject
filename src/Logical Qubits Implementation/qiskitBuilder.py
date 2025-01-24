from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister

class qiskitBuilder():
    def __init__(self, qubit_num : int):
        # States: 0_L = Up Up, 1_L = Down Down
        # Always detects a single X-error.
        self.logical_num = qubit_num
        self.physical_num = qubit_num
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


<<<<<<< HEAD
=======
    def appendCircuit(qc, min_qubit, max_qubit, control_qubit = None):
        if (control_qubit == None):
            self.qs.append(qc, list(range(2 * min_qubit, 2 * max_qubit + 1)))
        else:
            self.qs.append(qc, [2 * control_qubit, 2 * control_qubit + 1] + list(range(2 * min_qubit, 2 * max_qubit + 1)))
        
>>>>>>> origin/builders
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

    def appendCircuit(qc, min_qubit, max_qubit, control_qubit = None):
        if (control_qubit == None):
            self.qs.append(qc, list(range(min_qubit, max_qubit)))
        else:
            self.qs.append(qc, [control_qubit] + list(range(min_qubit, max_qubit)))

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
            self.qs.x(qubit)
            if sym:
                self._pushGate_([lambda: self.qs.x(qubit)])
            
        elif axis == "y":
            self.qs.y(qubit)
            if sym:
                self._pushGate_([lambda: self.qs.y(qubit)])
                
        elif axis == "z":
            self.qs.z(qubit)
            if sym:
                self._pushGate_([lambda: self.qs.z(qubit)])

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
            self.qs.cx(control, target)
            if sym:
                self._pushGate_([lambda: self.qs.cx(control, target)])
            
        elif axis == "y":
            self.qs.cy(control, target)
            if sym:
                self._pushGate_([lambda: self.qs.cy(control, target)])
                
        elif axis == "z":
            self.qs.cz(control, target)
            if sym:
                self._pushGate_([lambda: self.qs.cz(control, target)])

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
            self.qs.rx(angle, qubit)
            if sym:
                self._pushGate_([lambda: self.qs.rx(angle, qubit)])
        elif axis == "y":
            self.qs.ry(angle, qubit)
            if sym:
                self._pushGate_([lambda: self.qs.ry(angle, qubit)])
                
        elif axis == "z":
            self.qs.rz(angle, qubit)
            if sym:
                self._pushGate_([lambda: self.qs.rz(angle, qubit)])

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
            self._pushGate_([lambda: self.qs.h(qubit)])

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
        self.qs.p(angle, qubit)
        if sym:
            self._pushGate_(lambda: self.qs.p(qubit, angle))

    @autoPopDecorator
    def addCP(self, control, target, angle, sym = False):

        # Handle multiple callings
        if type(control) is list and type(target) is list and type(angle) is list and len(control) == len(angle) and len(control) == len(target):

            for i in range(0, len(control)):
                self.addCP(control[i], target[i], angle[i], sym)
                self.embed()
            return
 
        self.qs.cp(angle, control, target)
        
        if sym:
            self._pushGate_(lambda: self.qs.cp(angle, 2 * control, 2 * target))
    
    """
    Adds swap gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.
    """
    @autoPopDecorator
    def addSwap(self, qubit1, qubit2, sym = False):
        # Handle multiple callings
        if type(qubit1) is list and type(qubit2) is list and len(qubit1) == len(qubit2):

            for i in range(0, len(qubit1)):
                self.addSwap(qubit1[i], qubit2[i], sym)
                self.embed()
            return

        # Swap(Q1_L, Q2_L) = Swap(Q11_L, Q12_L, Q21_L, Q22_L) => Q21_L Q22_L Q11_L Q12_L     
        self.qs.swap(2 * qubit1, 2 * qubit2)
        self.qs.swap(2 * qubit1 + 1, 2 * qubit2 + 1)
        if sym:
            self._pushGate_([lambda: self.qs.swap(2 * qubit1 + 1, 2 * qubit2 + 1), lambda: self.qs.swap(2 * qubit1, 2 * qubit2)])

    def initializeToLogicalGround(self):

        # No initialization required
        self.qs.i(0)