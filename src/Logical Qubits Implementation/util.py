import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT

from qiskitBuilder import qiskitBuilder
from doubleSpinBuilder import doubleSpinBuilder
from steaneBuilder import steaneBuilder
from heterogenousSurfaceBuilder import heterogenousSurfaceBuilder

def getBuilderByType(builderType):
    if builderType == "Physical":
        return qiskitBuilder
    elif builderType == "Double Spin":
        return doubleSpinBuilder
    elif builderType == "Steane":
        return steaneBuilder
    elif builderType == "Heterogenous Surface":
        return heterogenousSurfaceBuilder
    

def generateHamiltonian(theta, builder):
    # Build a hamiltonian
    hamiltonianBuilder = builder(4)
    # hamiltonianBuilder.addPauli("x", 3) // remove later when implementing
    
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

    hamiltonianBuilder.addRotation(["x"] * 2, [0,2], [0.785] * 2)

    hamiltonianBuilder.addCPauli(["x"] * 2, [0, 1], [1, 2], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 2, theta[8])
    hamiltonianBuilder.pop(n = 2)

    hamiltonianBuilder.addRotation(["x"] * 2, [0,2], [-0.785] * 2)

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

    hamiltonianBuilder.addRotation(["x"] * 2, [0,2], [0.785] * 2)

    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[13])
    hamiltonianBuilder.pop(n = 3)

    hamiltonianBuilder.addRotation(["x"] * 2, [0,2], [-0.785] * 2)

    # Fourth Part
    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[14])
    hamiltonianBuilder.pop(n = 3)

    hamiltonianBuilder.addH([0, 1, 2, 3])
    return hamiltonianBuilder

def generatePhaseEstimation (num_ancila, num_target, Dt, hamiltonian):

    QPEBuilder = qiskitBuilder(num_ancila + num_target, bin_num = num_ancila)

    for q in range(num_ancila):
        QPEBuilder.addH(q)

    # Controlled-U operations using repeated applications of U
    for q in range(num_ancila):
        for _ in range(2 ** q):
            controlled_circuit = hamiltonian.control(num_ctrl_qubits=1)
            QPEBuilder.appendCircuit(controlled_circuit, num_ancila, num_ancila + num_target, q)

    # Apply inverse Quantum Fourier Transform (QFT) to ancilla qubits
    QPEBuilder.appendCircuit(generateInverseQFT(num_ancila), 0, num_ancila)

    return QPEBuilder
    
    
def generateQFT(num_logical):
    """
    Generates a Quantum Fourier Transform (QFT) circuit for the given number of logical qubits.
    """
    qftBuilder = qiskitBuilder(num_logical)

    # Iterate over each target qubit
    for target in range(num_logical):
        # Add a Hadamard gate to the target qubit
        qftBuilder.addH(target)

        # Add controlled phase gates with decreasing angles
        for control in range(target + 1, num_logical):
            angle = np.pi / (2 ** (control - target))
            qftBuilder.addCP(control, target, angle)

    # Add a final step to reverse the qubit order for QFT output
    for i in range(num_logical // 2):
        #qftBuilder.qs.swap(i * 2, (num_logical - 1 - i) * 2)
        #qftBuilder.qs.swap(i * 2 + 1, (num_logical - 1 - i) * 2 + 1)
        qftBuilder.addSwap(i, (num_logical - 1 - i))

    return qftBuilder.build()
            
def generateInverseQFT(num_logical):
    """
    Generates a Quantum Fourier Transform (QFT) circuit for the given number of logical qubits.
    """
    qftBuilder = qiskitBuilder(num_logical)
    # Reverse the qubit order for QFT output
    for i in range(num_logical // 2):
        #qftBuilder.qs.swap(i * 2, (num_logical - 1 - i) * 2)
        #qftBuilder.qs.swap(i * 2 + 1, (num_logical - 1 - i) * 2 + 1)
        qftBuilder.addSwap(i, (num_logical - 1 - i))
    
    # Iterate over each target qubit
    for target in range(num_logical - 1, -1, -1):
        # Add controlled phase gates with decreasing angles
        for control in range(num_logical - 1, target, -1):
            angle = np.pi / (2 ** (control - target))
            qftBuilder.addCP(control, target, -angle)      # This is an Rdagger gate
        # Add a Hadamard gate to the target qubit
        qftBuilder.addH(target)

    return qftBuilder.build()
