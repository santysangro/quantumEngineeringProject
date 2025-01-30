import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.algorithms import ExcitedStatesEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms import NumPyEigensolver
from qiskit_aer import StatevectorSimulator
from qiskit_nature.second_q.circuit.library import HartreeFock
import matplotlib.pyplot as plt
bond_lengths = np.array([0.3])#np.array([1.0, 1.2, 1.3, 1.39, 1.4, 1.4011, 1.41, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2])/2
energy = []

#REPLICATION OF THE HYDROGEN SIMULATION GROUND ENERGY AT DIFFERENT BOND LENGTHS FOLLOWING THE FINDINGS OF Yili Zhang (2022).
sim = StatevectorSimulator()

# Press the green button in the gutter to run the script.
for bond_length in bond_lengths:
    geometry = f"H 0 0 0; H 0 0 {bond_length}"
    basis = 'sto3g'
    spin = 0
    charge = 0
    unit = DistanceUnit.ANGSTROM

    driver = PySCFDriver(atom=geometry, basis=basis, charge=charge, spin=spin, unit=unit)
    problem = driver.run()

    hamiltonian = problem.hamiltonian

    coefficients = hamiltonian.electronic_integrals

    second_q_op = hamiltonian.second_q_op()
    mapper = BravyiKitaevMapper()
    qubit_op = mapper.map(second_q_op)

    eigensolver = NumPyEigensolver(k=2)  # Find the ground state and the first excited state
    excited_states_solver = ExcitedStatesEigensolver(mapper, eigensolver)
    # Solve for the excited states
    result = excited_states_solver.solve(problem)
    state_prep = result.eigenstates[0][0] #1 = excited, 0 = ground

    compiled = transpile(state_prep, sim)
    job = sim.run(compiled)
    state_vector_result = job.result()
    # Get statevector
    state_vector = state_vector_result.get_statevector(compiled)
    #print(state_vector)
    # Store the ground state energy for each bond length

    energy.append(result.eigenvalues[0] + hamiltonian.nuclear_repulsion_energy)
    print(hamiltonian.nuclear_repulsion_energy)
    print(f"Energy at bond length {bond_length}: {result.eigenvalues[0]+ hamiltonian.nuclear_repulsion_energy}")

plt.scatter(bond_lengths, energy, marker="o", label="This Study")
plt.xlabel('Bond length [$\mathrm{\AA}$]')
plt.ylabel('Ground state energy [Hartree]')
plt.title('Ground State Energy Levels of Hydrogen Molecule System \nat Different Bond Lengths')
plt.show()
