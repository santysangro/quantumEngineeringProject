class qasmBuilder:
    def __init__(self, qubit_num : int, version="1.0"):
        self.qasm = [f"version {version}", f"qubits {qubit_num}", ""]
        self.pending = []
        self.autoPop = False
        self.physical_num = qubit_num
        self.logical_num = qubit_num

    """
    A decorator for handling symmetric nesting of components
    """
    def handleAutoPop(self):
        if (self.autoPop):
            self.autoPop = False
            self.qasm.extend(self.pending.pop())
            
    @staticmethod
    def autoPopDecorator(func):
        def wrapper(self, *args, **kwargs):
            self.handleAutoPop()
            return func(self, *args, **kwargs)
        return wrapper

    """
    Add a single gate or multiple gates.
    !! DIRECT USAGE OUTSIDE OF THE CLASS IS DISCOURAGED
    """
    def _addGate_(self, gate : str | list):
        if (type(gate) is str):
            self.qasm.append(gate)
        else:
            self.qasm.extend(gate)
        
    """
    Add a single gate or multiple gate to the pending queue.
    !! DIRECT USAGE OUTSIDE OF THE CLASS IS DISCOURAGED
    """
    def _pushGate_(self, gate : str | list):
        self.pending.append(gate)
        self.autoPop = True
    
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
            self._addGate_(self.pending.pop())
            n -= 1
    
    """
    Builds the string representing the circuit
    """
    def build(self):
        self.pop(len(self.pending))
        self._addGate_("measure_all")
        return "\n".join(self.qasm)

    
    """
    Add a single custom gate (use is discouraged when considering complex logical qubits)
    """
    @autoPopDecorator
    def addGate(self, gate : str):
        self._addGate_(gate)



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
        gate = ""
        if axis == "i":
            gate = f"I q[{qubit}]"
        elif axis == "x":
            gate = f"X q[{qubit}]"
        elif axis == "y":
            gate = f"Y q[{qubit}]"
        elif axis == "z":
            gate = f"Z q[{qubit}]"

        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)
    """
    Adds a 90-degrees rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "x", "y", "mx", and "my" to determine the axis and rotation direction. 
    """
    @autoPopDecorator
    def addRootPauli(self, axis, qubit, sym = False):

        # Handle multiple callings
        if type(axis) is list or type(qubit) is list:

            # Validate argument types for multiple callings
            if type(axis) is not list or type(qubit) is not list or len(axis) != len(qubit):
                raise ValueError("Both arguments must be lists of equal length for multiple calling")

            for i in range(0, len(axis)):
                self.addRootPauli(axis[i], qubit[i], sym)
                self.embed()

            return
        
        # Determine the rotation axis
        gate = ""
        if axis == "x":
            gate = f"X90 q[{qubit}]"
        elif axis == "y":
            gate = f"Y90 q[{qubit}]"
        elif axis == "mx":
            gate = f"mX90 q[{qubit}]"
        elif axis == "my":
            gate = f"mY90 q[{qubit}]"

        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)

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
        gate = ""
        if axis == "x":
            gate = f"Rx q[{qubit}], {angle}"
        elif axis == "y":
            gate = f"Ry q[{qubit}], {angle}"
        elif axis == "z":
            gate = f"Rz q[{qubit}], {angle}"

        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)

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
        gate = [f"H q[{qubit}]"]
        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)
    
    """
    Adds a CNOT gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.
    """
    @autoPopDecorator
    def addCNOT(self, control, target, sym = False):

        # Handle multiple callings
        if type(control) is list or type(target) is list:
            
            # Validate argument types for multiple callings
            if type(control) is not list or type(target) is not list or len(control) != len(target):
                raise ValueError("Both arguments must be lists of equal length for multiple calling")

            for i in range(0, len(control)):
                self.addCNOT(control[i], target[i], sym)
                self.embed()
                
            return

        
        # Determine the rotation axis
        gate = [f"CNOT q[{control}], q[{target}]"]
        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)


class twoQubitLogicBuilder(qasmBuilder):
    def __init__(self, qubit_num : int, version="1.0"):
        super().__init__(2 * qubit_num, version)
        self.logical_num = qubit_num

        # Initialize so that the qubits are alternating
        self._addGate_([f"X q[{qubit}]" for qubit in range(1, self.physical_num, 2)])


    """
    Overriden.
    
    Adds a 180-degrees rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "i", "x", "y", and "z" to determine the axis. 
    """
    @qasmBuilder.autoPopDecorator
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
        gate = ""
        if axis == "i":
            gate = [f"I q[{2 * qubit}]", f"I q[{2 * qubit + 1}]"]
        elif axis == "x":
            gate = [f"X q[{2 * qubit}]", f"X q[{2 * qubit + 1}]"]
        elif axis == "y":
            gate = [f"Y q[{ 2 * qubit}]", f"X q[{ 2 * qubit + 1}]"]
        elif axis == "z":
            gate = f"Z q[{2 * qubit}]"

        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)

    """
    Overriden.
    
    Adds a custom rotation gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.

    Use "x", "y", and "z" to determine the axis and rotation direction. 
    """
    @qasmBuilder.autoPopDecorator
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
        gate = ""
        if axis == "x":
            gate = f"Rx q[{qubit}], {angle}"
        elif axis == "y":
            gate = f"Ry q[{qubit}], {angle}"
        elif axis == "z":
            gate = f"Rz q[{qubit}], {angle}"

        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)

    """
    Overriden.
    
    Adds Hadamard gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.
    """
    @qasmBuilder.autoPopDecorator
    def addH(self, qubit, sym = False):

        # Handle multiple callings
        if type(qubit) is list :

            for i in range(0, len(qubit)):
                self.addH(qubit[i],sym)
                self.embed()

            return

        
        # Determine the rotation axis
        gate = [f"CNOT q[{2 * qubit}], q[{2 * qubit + 1}]", f"H q[{2 * qubit}]", f"CNOT q[{2 * qubit}], q[{2 * qubit + 1}]"]
        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)

    """
    Overriden.
    
    Adds a CNOT gate to the specified (logical) qubits. Supports parallel lists
    as arguments for multiple gates.
    """
    @qasmBuilder.autoPopDecorator
    def addCNOT(self, control, target, sym = False):

        # Handle multiple callings
        if type(control) is list or type(target) is list:
            
            # Validate argument types for multiple callings
            if type(control) is not list or type(target) is not list or len(control) != len(target):
                raise ValueError("Both arguments must be lists of equal length for multiple calling")

            for i in range(0, len(control)):
                self.addCNOT(control[i], target[i], sym)
                self.embed()

            
            return

        
        # Determine the rotation axis
        gate = [f"CNOT q[{2 * control}], q[{2 * target}]", f"CNOT q[{2 * control}], q[{2 * target + 1}]"]
        self._addGate_(gate)
        if sym:
            self._pushGate_(gate)


def get_builder_type_by_name(s):
    if s == "physical":
        return qasmBuilder
    elif s == "2 qubit":
        return twoQubitLogicBuilder
    else:
        return None