import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT
#from openfermionpyscf import generate_molecular_hamiltonian
from openfermion.chem import MolecularData
from qiskitBuilder import qiskitBuilder
from doubleSpinBuilder import doubleSpinBuilder
from steaneBuilder import steaneBuilder
from heterogenousSurfaceBuilder import heterogenousSurfaceBuilder

def getBuilderByType(builderType):
    if builderType == "Single Qubit":
        return qiskitBuilder
    elif builderType == "Double Qubit":
        return doubleSpinBuilder
    elif builderType == "Steane":
        return steaneBuilder
    elif builderType == "Heterogenous Surface":
        return heterogenousSurfaceBuilder
    

#def generateThetas(bond_length, Dt):
#    geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, bond_length))]
#    basis = 'sto-3g'
#    multiplicity = 1
#    charge = 0
#    hamiltonian = generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)
#    molecule = MolecularData(geometry, basis, multiplicity)
#    molecule.load()
#    one_body_coefficients = hamiltonian.one_body_tensor
#    two_body_coefficients = hamiltonian.two_body_tensor
#
#    h00 = one_body_coefficients[0, 0]
#    h11 = one_body_coefficients[1, 1]
#    h22 = one_body_coefficients[2, 2]
#    h33 = one_body_coefficients[3, 3]
#
#    h0110 = 2 * two_body_coefficients[0, 1, 1, 0]
#    h0330 = 2 * two_body_coefficients[0, 3, 3, 0]
#    h1221 = 2 * two_body_coefficients[1, 2, 2, 1]
#    h2332 = 2 * two_body_coefficients[2, 3, 3, 2]
#    h0220 = 2 * two_body_coefficients[0, 2, 2, 0]
#    h2020 = 2 * two_body_coefficients[2, 0, 2, 0]
#    h1313 = 2 * two_body_coefficients[1, 3, 1, 3]
#    h1331 = 2 * two_body_coefficients[1, 3, 3, 1]
#    h0132 = 2 * two_body_coefficients[0, 1, 3, 2]
#    h0312 = 2 * two_body_coefficients[0, 3, 1, 2]
#    h0202 = 2 * two_body_coefficients[0, 2, 0, 2]
#
#    omega_1 = (h00 + h11 + h22 + h33) / 2 + (h0110 + h0330 + h1221 + h2332) / 4 + (h0220 - h0202) / 4 + (
#            h1331 - h1313) / 4
#    omega_2 = - (h00 / 2 + h0110 / 4 + h0330 / 4 + h0220 / 4 - h0202 / 4)
#    omega_3 = h0110 / 4
#    omega_4 = -(h22 / 2 + h1221 / 4 + h2332 / 4 + h0220 / 4 - h0202 / 4)
#    omega_5 = -(h11 / 2 + h0110 / 4 + h1221 / 4 + h1331 / 4 - h1313 / 4)
#    omega_6 = (h0220 - h2020) / 4
#    omega_7 = h2332 / 4
#    omega_8 = h0132 / 4
#    omega_9 = (h0132 + h0312) / 8
#    omega_10 = h1221 / 4
#    omega_11 = (h1331 - h1313) / 4
#    omega_12 = -(h33 / 2 + h0330 / 4 + h2332 / 4 + h1331 / 4 - h1313 / 4)
#    omega_13 = (h0132 + h0312) / 8
#    omega_14 = (h0132 + h0312) / 8
#    omega_15 = h0330 / 4
#
#    omegas = [omega_1, omega_2, omega_3, omega_4, omega_5, omega_6, omega_7, omega_8, omega_9, omega_10, omega_11,
#              omega_12, omega_13, omega_14, omega_15]
#    theta = [2 * omega * Dt for omega in omegas]
#    return theta, molecule.nuclear_repulsion



def generateHamiltonian(theta, builder, initial_state = None):

    quarter_pi = -0.785 # np.pi / 4
    # Build a hamiltonian
    hamiltonianBuilder = builder(4)

    if initial_state != None:
        hamiltonianBuilder.initializeToLogicalGround(initial_state)
    
    # First Part

    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addGlobalPhaseShift(3, theta[0])
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
    #return hamiltonianBuilder
    hamiltonianBuilder.addRotation(["x"] * 2, [0,2], [np.pi/2] * 2)

    hamiltonianBuilder.addCPauli(["x"] * 2, [0, 1], [1, 2], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 2, theta[8])
    hamiltonianBuilder.pop(n = 2)

    hamiltonianBuilder.addRotation(["x"] * 2, [0,2], [-np.pi/2] * 2)
    
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

    hamiltonianBuilder.addRotation(["x"] * 2, [0,2], [np.pi/2] * 2)

    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[13])
    hamiltonianBuilder.pop(n = 3)

    hamiltonianBuilder.addRotation(["x"] * 2, [0,2], [-np.pi/2] * 2)

    # Fourth Part
    hamiltonianBuilder.addCPauli(["x"] * 3, [0, 1, 2], [1, 2, 3], sym=True)
    
    hamiltonianBuilder.embed()
    hamiltonianBuilder.addRotation("z", 3, theta[14])
    hamiltonianBuilder.pop(n = 3)
    return hamiltonianBuilder

def generatePhaseEstimation (num_ancila, num_target, Dt, hamiltonian, initial_state = None):

    QPEBuilder = qiskitBuilder(num_ancila + num_target, bin_num = num_ancila)

    if initial_state != None:
        initializerBuilder = hamiltonian.__class__(hamiltonian.getLogicalNumber())
        initializerBuilder.initializeToLogicalGround(initial_state)
        QPEBuilder.appendCircuit(initializerBuilder.build(), num_ancila, num_ancila + num_target)

    for q in range(num_ancila):
        QPEBuilder.addH(q)

    # Controlled-U operations using repeated applications of U
    for q in range(num_ancila):
        for _ in range(2 ** q):
            controlled_circuit = hamiltonian.build().control(1)
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
    for target in range(num_logical - 1, -1, -1):
        # Add controlled phase gates with decreasing angles
        for control in range(num_logical - 1, target, -1):
            angle = np.pi / (2 ** (control - target))
            qftBuilder.addCP(control, target, -angle)      # This is an Rdagger gate
        # Add a Hadamard gate to the target qubit
        qftBuilder.addH(target)
    
    for i in range(num_logical // 2):
        #qftBuilder.qs.swap(i * 2, (num_logical - 1 - i) * 2)
        #qftBuilder.qs.swap(i * 2 + 1, (num_logical - 1 - i) * 2 + 1)
        qftBuilder.addSwap(i, (num_logical - 1 - i))

    return qftBuilder.build()
            
def generateInverseQFT(num_logical):
    qftBuilder = qiskitBuilder(num_logical)
    
    for i in range(num_logical // 2):
        #qftBuilder.qs.swap(i * 2, (num_logical - 1 - i) * 2)
        #qftBuilder.qs.swap(i * 2 + 1, (num_logical - 1 - i) * 2 + 1)
        qftBuilder.addSwap(i, (num_logical - 1 - i))

    # Iterate over each target qubit
    for target in range(num_logical):
        # Add a Hadamard gate to the target qubit
        qftBuilder.addH(target)

        # Add controlled phase gates with decreasing angles
        for control in range(target + 1, num_logical):
            angle = np.pi / (2 ** (control - target))
            qftBuilder.addCP(control, target, -angle)

    return qftBuilder.build()

