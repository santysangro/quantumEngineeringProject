from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister

from qiskitBuilder import qiskitBuilder

class heterogenousSurfaceBuilder(qiskitBuilder):
    def __init__(self, qubit_num : int, logical_surface_qubit = 3):
        # 
        # A code distance 3 surface qubit embedded heterogenously with physical qubits.
        if (qubit_num  < 1):
            raise error("Insufficient numebr of logical qubits")
        
        if (logical_surface_qubit >= qubit_num):
            raise error ("Incorrect surface qubit index")

        
        self.logical_num = qubit_num
        self.physical_num = 8 + qubit_num
        self.surface_qubit = logical_surface_qubit
        self.pending = []
        self.autoPop = False
        self.qs = QuantumCircuit(QuantumRegister(self.physical_num)) 

    """
    Helper Method to shift qubits accounting for additional physical qubits for the surface qubit
    """
    def qubitOrienter(self, qubit):
        if (qubit < self.surface_qubit):
            return qubit
        else:
            return qubit + 8

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


        # if qubit is non-surface, defer to superclass
        if (qubit != self.surface_qubit):
            super().addPauli(axis, self.qubitOrienter(qubit), sym)

        else:
            # Determine the rotation axis
            # We assume a cross pathways for the following array:
            #     Z
            #    \/
            # D1 D2 D3
            # D4 D5 D6 < X
            # D7 D8 D9
            if axis == "x":
                # X_L = X_4 X_5 X_6 (Transversal)
    
                for offset in [3, 4, 5]:
                    self.qs.x(qubit + offset)
                if sym:
                    self._pushGate_([lambda offset = offset: self.qs.x(qubit + offset) for offset in [3, 4, 5]])
                
            elif axis == "y":
                # Omitted due to non-usage
                return
            elif axis == "z":
                # Z_L = Z_2 Z_5 Z_8 (Transversal)
                for offset in [1, 4, 7]:
                    self.qs.z(qubit + offset)
                if sym:
                    self._pushGate_([lambda offset = offset: self.qs.z(qubit + offset) for offset in [1, 4, 7]])
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
        
        # if qubit is non-surface, defer to superclass
        if (control != self.surface_qubit and target != self.surface_qubit):
            super().addCPauli(axis, self.qubitOrienter(control), self.qubitOrienter(target), sym)

        elif (target != self.surface_qubit):

            # Perform a flip here with CX = H CX'H, and CZ = CZ', ignore CY
            if axis == "x":
                self.addH([control, target], sym = True)
                self.embed()
                self.addCPauli(axis, target, control, sym = False)
                self.pop()
                if sym:
                   self._pushGate_([lambda: self.addH([control, target], sym = True), lambda: self.embed(), lambda: self.addCPauli(axis, target, control, sym = False), lambda: self.pop()])
            elif axis == "z":
                self.addCPauli(axis, target, control, sym)
        else:
            # Determine the rotation axis
            # We assume a cross pathways for the following array:
            #     Z
            #    \/
            # D1 D2 D3
            # D4 D5 D6 < X
            # D7 D8 D9
            #
            # CNOT is not between surface codes so it is far simpler
            
            if axis == "x":
                # CX_L = CX_4 CX_5 CX_6 (Transversal)
    
                for offset in [3, 4, 5]:
                    self.qs.cx(self.qubitOrienter(control), target + offset)
                if sym:
                    self._pushGate_([lambda offset = offset: self.qs.cx(self.qubitOrienter(control), target + offset) for offset in [3, 4, 5]])
                
            elif axis == "y":
                # Omitted due to non-usage
                return
            elif axis == "z":
                # Z_L = CZ_2 CZ_5 CZ_8 (Transversal)
                for offset in [1, 4, 7]:
                    self.qs.cz(self.qubitOrienter(control), target + offset)
                if sym:
                    self._pushGate_([lambda offset = offset: self.qs.cz(self.qubitOrienter(control), target + offset) for offset in [1, 4, 7]])
    
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
            # TODO Later
            return;
        elif axis == "y":
            # Note: It is possible to implement it indirectly without performing any additional optimizations,
            # but it is really not used for anything in the hamiltonian.
            raise ValueError("Not required")
                
        elif axis == "z":
            # TODO Later
            return;
            

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


        # if qubit is non-surface, defer to superclass
        if (qubit != self.surface_qubit):
            super().addH(self.qubitOrienter(qubit), sym)

        else:
            # H_L = H_1 ... H_8 (Transversal)
            for offset in range(0, 9):
                self.qs.h(qubit + offset)
                
            if sym:
                self._pushGate_([lambda offset = offset: self.qs.h(qubit + offset) for offset in range(0, 9)])

   # Note: P and CP gates omitted due to lack of usage.