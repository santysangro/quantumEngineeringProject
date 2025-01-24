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
    return hamiltonianBuilder.build()


def generateQFT(num_logical):
    
    qftBuilder = qiskitBuilder(num_logical)
    
    #for (target in range(0, num_logical)):
    #    qftBuilder.addH(target)
    #    for (control in range(target + 1, num_logical)):
            
