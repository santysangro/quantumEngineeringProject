from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskitBuilder import qiskitBuilder

class steaneBuilder(qiskitBuilder):
    def __init__(self, qubit_num : int, fancy_qubits = False):
        # States: 0_L = Up Up, 1_L = Down Down
        # Always detects a single X-error.
        self.logical_num = qubit_num
        self.physical_num = qubit_num * 7 + 1
        self.pending = []
        self.autoPop = False
        if fancy_qubits:
            self.qs = QuantumCircuit(*[QuantumRegister(7, name) for name in ["Q", "P", "R", "S" ][0:self.logical_num]], QuantumRegister(1, "Ancilla"), ClassicalRegister(1 + 7*(self.logical_num)))
        else:
            self.qs = QuantumCircuit(QuantumRegister(self.physical_num), ClassicalRegister(1))


    def appendCircuit(qc, min_qubit, max_qubit, control_qubit = None):
        if (control_qubit == None):
            self.qs.append(qc, list(range(7 * min_qubit, 7 * max_qubit + 6)))
        else:
            self.qs.append(qc, list(range(7 * control_qubit, 7 * control_qubit + 7)) + list(range(7 * min_qubit, 7 * max_qubit + 6)))
        
    

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
            # X_L = X_1 X_2 ... X_7 (Transversal)
            
            for offset in range(0, 7):
                self.qs.x(7 * qubit + offset)
                
            if sym:
                self._pushGate_([lambda offset = offset: self.qs.x(7 * qubit + offset) for offset in range(0, 7)])
            
        elif axis == "y":
            # Y_L = Y_1 Y_2 ... Y_5 X_6 Z_6 X_7 Z_7 (Transversal)
            for offset in range(0, 5):
                self.qs.y(7 * qubit + offset)
            for offset in range(5, 7):
                self.qs.z(7 * qubit + offset)
                self.qs.x(7 * qubit + offset)
            
            if sym:
                self._pushGate_([lambda offset = offset: self.qs.y(7 * qubit + offset) for offset in range(0, 5)] + [lambda: self.qs.z(7 * qubit + 5), lambda: self.qs.x(7 * qubit + 5), lambda: self.qs.z(7 * qubit + 6), lambda: self.qs.x(7 * qubit + 6)])
                
        elif axis == "z":
            # Z_L = Z_1 Z_2 ... Z_7 (Transversal)
            for offset in range(0, 7):
                self.qs.z(7 * qubit + offset)
            
            if sym:
                self._pushGate_([lambda offset = offset: self.qs.z(7 * qubit + offset) for offset in range(0, 7)])
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
            # CX_L = C1X_1 ... C7X_7 (Transversal)
            for offset in range(0, 7):
                self.qs.cx(7 * control + offset, 7 * target + offset)
            
            if sym:
                self._pushGate_([lambda offset = offset: self.qs.cx(7 * control + offset, 7 * target + offset) for offset in range(0, 7)])
            
        elif axis == "y":
            # CY_L = CY_1 CY_2 ... CY_5 CX_6 CZ_6 CX_7 CZ_7 (Transversal)
            for offset in range(0, 5):
                self.qs.cy(7 * control + offset, 7 * target + offset)
            for offset in range(5, 7):
                self.qs.cz(7 * control + offset, 7 * target + offset)
                self.qs.cx(7 * control + offset, 7 * target + offset)
            
            if sym:
                self._pushGate_([lambda offset = offset: self.qs.cy(7 * control + offset, 7 * target + offset) for offset in range(0, 5)] + [lambda: self.qs.cz(7 * control + offset, 7 * target + offset), lambda: self.qs.cx(7 * control + 5, 7 * target + 5), lambda: self.qs.cz(7 * control + 5, 7 * target + 5), lambda: self.qs.cx(7 * control + 6, 7 * target + 6)])
                
        elif axis == "z":
            # CZ_L = C1Z_1 ... C7Z_7 (Transversal)
            for offset in range(0, 7):
                self.qs.cz(7 * control + offset, 7 * target + offset)
            
            if sym:
                self._pushGate_([lambda offset = offset: self.qs.cz(7 * control + offset, 7 * target + offset) for offset in range(0, 7)])

    
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
            # Use Magic Gate Injection
            # NOT SAFE! Measurement-based Ancilla Injection

            self.addH(qubit)
            self.addRotation("z", qubit, angle)
            self.addH(qubit)
            
            if sym:
                raise ValueError("Not implemented")
            
        elif axis == "y":
            # Note: It is possible to implement it indirectly without performing any additional optimizations,
            # but it is really not used for anything in the hamiltonian.
            raise ValueError("Not required")
                
        elif axis == "z":
            # Use Magic Gate Injection
            # NOT SAFE! Measurement-based Ancilla Injection

            self.qs.reset(self.physical_num - 1)
            self.qs.h(self.physical_num - 1)
            self.qs.barrier()
            self.qs.rz(angle, self.physical_num - 1)
            self.qs.barrier()
            
            #self.qs.h(self.physical_num - 1)
            self.qs.barrier()

            for offset in range(0, 7):
                self.qs.cx(7 * qubit + offset, self.physical_num -1)

            
            self.qs.barrier()
            
            
            self.qs.measure(self.physical_num - 1, 0) # Z-basis measurement
            self.qs.barrier()
            #for offset in range(0, 7):
            #    self.qs.cx(self.physical_num -1, 7 * qubit + offset)
            #self.qs.barrier()
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [1,3,5]], [7 * qubit + 6])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [0,3,4]], [7 * qubit + 6])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [2,3,6]], [7 * qubit + 0])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [0,1,2]], [7 * qubit + 6])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [1,4,6]], [7 * qubit + 0])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [0,5,6]], [7 * qubit + 1])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [2,4,5]], [7 * qubit + 6])
            self.qs.mcp(-6 * angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [0,1,2,3,4]], [7 * qubit + 6])
            self.qs.barrier()
            for offset in range(0, 7):
                self.qs.x(7 * qubit + offset)
            self.qs.barrier()
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [1,3,5]], [7 * qubit + 6])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [0,3,4]], [7 * qubit + 6])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [2,3,6]], [7 * qubit + 0])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [0,1,2]], [7 * qubit + 6])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [1,4,6]], [7 * qubit + 0])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [0,5,6]], [7 * qubit + 1])
            self.qs.mcp(angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [2,4,5]], [7 * qubit + 6])
            self.qs.mcp(-6 * angle, [self.physical_num - 1] + [7 * qubit + offset for offset in [0,1,2,3,4]], [7 * qubit + 6])
            self.qs.barrier()
            for offset in range(0, 7):
                self.qs.x(7 * qubit + offset)
                
            #self.qs.reset(self.physical_num - 1)
            
            if sym:
                raise ValueError("Not implemented")
                

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

        
        # H_L = H_1 ... H_7 (Transversal)
        for offset in range(0, 7):
            self.qs.h(7 * qubit + offset)
            
        if sym:
            self._pushGate_([lambda offset = offset: self.qs.h(7 * qubit + offset) for offset in range(0, 7)])

   # Note: P and CP gates omitted due to lack of usage.

    def initializeToLogicalGround(self, initial_state):

        if (initial_state != None):
            self.qs.initialize(initial_state, list(range(0, self.physical_num - 1, 7)))
        
        # Initialization required
        for qubit in range(0, self.logical_num):
            self.qs.h(7 * qubit + 6)
            self.qs.h(7 * qubit + 5)
            self.qs.h(7 * qubit + 4)

            self.qs.cx(7 * qubit, 7 * qubit + 3)
            self.qs.cx(7 * qubit, 7 * qubit + 2)

            self.qs.cx(7 * qubit + 6, 7 * qubit + 3)
            self.qs.cx(7 * qubit + 6, 7 * qubit + 2)
            self.qs.cx(7 * qubit + 6, 7 * qubit + 1)
            self.qs.cx(7 * qubit + 5, 7 * qubit + 3)
            self.qs.cx(7 * qubit + 5, 7 * qubit + 1)
            self.qs.cx(7 * qubit + 5, 7 * qubit)
            self.qs.cx(7 * qubit + 4, 7 * qubit + 2)
            self.qs.cx(7 * qubit + 4, 7 * qubit + 1)
            self.qs.cx(7 * qubit + 4, 7 * qubit + 0)