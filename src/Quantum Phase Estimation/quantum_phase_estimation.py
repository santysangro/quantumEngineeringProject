import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator
from openfermionpyscf import generate_molecular_hamiltonian
from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
import openfermion as of
from openfermion.chem import MolecularData
import openfermionpyscf as ofpyscf
from scipy.sparse import linalg
from scipy.sparse.linalg import expm


# GLOBALS:
sim = StatevectorSimulator()
geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))]
basis = 'sto-3g'
multiplicity = 1
charge = 0


def get_H_paper():
    n_qubits = 4

    hamiltonian = ofpyscf.generate_molecular_hamiltonian(
        geometry, basis, multiplicity, charge)

    # Convert to a FermionOperator
    molecule = MolecularData(geometry, basis, multiplicity)
    molecule.load()
    hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)
    nuclear_repulsion_energy = molecule.nuclear_repulsion
    # Remove the nuclear repulsion term (constant term)
    hamiltonian_ferm_op = hamiltonian_ferm_op - nuclear_repulsion_energy
    hamiltonian_bk = of.bravyi_kitaev(hamiltonian_ferm_op)
    hamiltonian_bk_sparse = of.get_sparse_operator(hamiltonian_bk)
    eigs, state = linalg.eigsh(hamiltonian_bk_sparse, k=1, which='SA')
    # print(state, eigs)
    # Convert to Scipy sparse matrix
    terms = hamiltonian_bk.terms  # Returns a dictionary of Pauli strings and coefficients
    # Convert the OpenFermion QubitOperator to Qiskit Pauli sum format
    converted_list = []
    for term, coeff in terms.items():
        # Start with a list of 'I's representing the identity on each qubit
        pauli_str = ['I'] * 4

        # Apply Pauli operators at the appropriate positions
        for qubit, op in term:
            pauli_str[(3 - qubit)] = op

        # Convert the list into a string (like 'IIII', 'ZZZI', etc.)
        pauli_str = ''.join(pauli_str)
        # Append the Pauli string and its corresponding coefficient as a tuple
        converted_list.append((pauli_str, coeff))

    pauli_op = SparsePauliOp.from_list(converted_list)
    print(pauli_op)
    hamiltonian_matrix = pauli_op.to_matrix()
    U = expm(-1j * hamiltonian_matrix)
    print("Hamiltonian matrix:")
    print(hamiltonian_matrix)
    print("DONE")
    qc = QuantumCircuit(n_qubits)
    qc.unitary(U, range(n_qubits))
    e, v = np.linalg.eig(hamiltonian_matrix)  # to match with classical result
    ground_state_energy = np.min(e)
    ground_state_index = np.argmin(e)
    ground_state_eigenstate = v[:, ground_state_index]
    print(f"Ground state energy: {ground_state_energy}")
    ground_state_bitstring = format(ground_state_index,
                                    '0{}b'.format(int(np.log2(len(v)))))  # Format the index as a bitstring
    print(f"Ground state eigenstate (bitstring): {ground_state_bitstring}")
    return qc
    # Compute ground energy
    time = 1.0  # Set the time for evolution (arbitrary units)
    # Create the time evolution circuit
    evolution_gate = PauliEvolutionGate(pauli_op, time)
    evolution_circuit = QuantumCircuit(n_qubits)
    evolution_circuit.append(U, range(n_qubits))
    return evolution_circuit


def get_H_qiskit():
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.741237",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM, )
    problem = driver.run()
    hamiltonian = problem.hamiltonian
    second_q_op = hamiltonian.second_q_op()
    mapper = BravyiKitaevMapper()
    qubit_op = mapper.map(second_q_op)

    # Convert the Hamiltonian to a SparsePauliOp for evolution
    pauli_op = SparsePauliOp.from_list(qubit_op.to_list())
    time = 1.0  # Set the time for evolution (arbitrary units)
    # Create the time evolution circuit
    suzuki = SuzukiTrotter(order=2, reps=1)  # 3 repetitions of second-order Suzuki
    evolution_gate = PauliEvolutionGate(pauli_op, time, synthesis=suzuki)
    # evolution_gate = PauliEvolutionGate(pauli_op, time)
    n_qubits = qubit_op.num_qubits
    evolution_circuit = QuantumCircuit(n_qubits)
    evolution_circuit.append(evolution_gate, range(n_qubits))
    return evolution_circuit


def calculate_overlap_integrals():
    hamiltonian = generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)

    one_body_coefficients = hamiltonian.one_body_tensor
    two_body_coefficients = hamiltonian.two_body_tensor

    h00 = one_body_coefficients[0, 0]
    h11 = one_body_coefficients[1, 1]
    h22 = one_body_coefficients[2, 2]
    h33 = one_body_coefficients[3, 3]

    h0110 = 2 * two_body_coefficients[0, 1, 1, 0]
    h0330 = 2 * two_body_coefficients[0, 3, 3, 0]
    h1221 = 2 * two_body_coefficients[1, 2, 2, 1]
    h2332 = 2 * two_body_coefficients[2, 3, 3, 2]
    h0220 = 2 * two_body_coefficients[0, 2, 2, 0]
    h2020 = 2 * two_body_coefficients[2, 0, 2, 0]
    h1313 = 2 * two_body_coefficients[1, 3, 1, 3]
    h1331 = 2 * two_body_coefficients[1, 3, 3, 1]
    h0132 = 2 * two_body_coefficients[0, 1, 3, 2]
    h0312 = 2 * two_body_coefficients[0, 3, 1, 2]
    h0202 = 2 * two_body_coefficients[0, 2, 0, 2]

    print(f"""
    h00 = {h00}, h11 = {h11}, h22 = {h22}, h33 = {h33}
    h0110 = {h0110}, h0330 = {h0330}, h1221 = {h1221}, h2332 = {h2332}
    h0220 = {h0220}, h2020 = {h2020}, h1313 = {h1313}, h1331 = {h1331}
    h0132 = {h0132}, h0312 = {h0312}, h0202 = {h0202}
    """)

    omega_1 = (h00 + h11 + h22 + h33) / 2 + (h0110 + h0330 + h1221 + h2332) / 4 + (h0220 - h0202) / 4 + (
            h1331 - h1313) / 4
    omega_2 = - (h00 / 2 + h0110 / 4 + h0330 / 4 + h0220 / 4 - h0202 / 4)
    omega_3 = h0110 / 4
    omega_4 = -(h22 / 2 + h1221 / 4 + h2332 / 4 + h0220 / 4 - h0202 / 4)
    omega_5 = -(h11 / 2 + h0110 / 4 + h1221 / 4 + h1331 / 4 - h1313 / 4)
    omega_6 = (h0220 - h2020) / 4
    omega_7 = h2332 / 4
    omega_8 = h0132 / 4
    omega_9 = (h0132 + h0312) / 8
    omega_10 = h1221 / 4
    omega_11 = (h1331 - h1313) / 4
    omega_12 = -(h33 / 2 + h0330 / 4 + h2332 / 4 + h1331 / 4 - h1313 / 4)
    omega_13 = (h0132 + h0312) / 8
    omega_14 = (h0132 + h0312) / 8
    omega_15 = h0330 / 4

    omegas = [omega_1, omega_2, omega_3, omega_4, omega_5, omega_6, omega_7, omega_8, omega_9, omega_10, omega_11,
              omega_12, omega_13, omega_14, omega_15]
    return omegas


# Example circuit from lecture slides Quantum Communication & Computation Week 3
def example_circuit():
    R_matrix = np.array([[0, -1j],
                         [1j, 0]])

    U = Operator(R_matrix)
    qc = QuantumCircuit(1)
    qc.append(U, [0])
    return qc


def generate_Hamiltonian_circuit(n, theta):
    q = QuantumRegister(n)

    qc = QuantumCircuit(q)
    # Forward iteration
    """
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    Ri_matrix = np.array([[np.exp(-1j * theta[0] / 2), 0],
                          [0, np.exp(-1j * theta[0] / 2)]])
    unitary_gate = Operator(Ri_matrix)

    qc.append(unitary_gate, [3])
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    """
    qc.rz(theta[1], 0)
    qc.rz(theta[2], 1)
    qc.rz(theta[3], 2)

    qc.cx(0, 1)
    qc.rz(theta[4], 1)
    qc.cx(0, 1)
   
    # SECOND PART
    qc.cx(0, 2)
    qc.rz(theta[5], 2)
    qc.cx(0, 2)

    qc.cx(1, 3)
    qc.rz(theta[6], 3)
    qc.cx(1, 3)

    qc.h(0)
    qc.h(2)

    qc.cx(0, 1)
    qc.cx(1, 2)

    qc.rz(theta[7], 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.h(0)
    qc.h(2)
    #qc.draw(output="mpl")
    #plt.show()
    #return qc
    qc.rx(np.pi / 2, 0)
    qc.rx(np.pi / 2, 2)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(theta[8], 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.rx(-np.pi / 2, 0)
    qc.rx(-np.pi / 2, 2)
    
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(theta[9], 2)
    qc.cx(1, 2)
    qc.cx(0, 1)
    # THIRD PART
    qc.cx(0, 2)
    qc.cx(2, 3)
    qc.rz(theta[10], 3)
    qc.cx(2, 3)
    qc.cx(0, 2)

    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[11], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)

    qc.h(0)
    qc.h(2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[12], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(2)

    qc.rx(np.pi / 2, 0)
    qc.rx(np.pi / 2, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[13], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.rx(-np.pi / 2, 0)
    qc.rx(-np.pi / 2, 2)

    # FOURTH SECTION
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[14], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    """
    # Backward iteration
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[14], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.rx(np.pi/2, 0)
    qc.rx(np.pi/2, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[13], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.rx(-np.pi/2, 0)
    qc.rx(-np.pi/2, 2)

    qc.h(0)
    qc.h(2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[12], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(2)

    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[11], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)

    qc.cx(0, 2)
    qc.cx(2, 3)
    qc.rz(theta[10], 3)
    qc.cx(2, 3)
    qc.cx(0, 2)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(theta[9], 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.rx(np.pi/2, 0)
    qc.rx(np.pi/2, 2)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(theta[8], 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.rx(-np.pi/2, 0)
    qc.rx(-np.pi/2, 2)

    qc.h(0)
    qc.h(2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(theta[7], 2)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(2)

    qc.cx(1, 3)
    qc.rz(theta[6], 3)
    qc.cx(1, 3)

    qc.cx(0, 2)
    qc.rz(theta[5], 2)
    qc.cx(0, 2)

    qc.cx(0, 1)
    qc.rz(theta[4], 1)
    qc.cx(0, 1)

    qc.rz(theta[3], 2)
    qc.rz(theta[2], 1)
    qc.rz(theta[1], 0)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    Ri_matrix = np.array([[np.exp(-1j * theta[0] / 2), 0],
                          [0, np.exp(-1j * theta[0] / 2)]])
    unitary_gate = Operator(Ri_matrix)

    qc.append(unitary_gate, [3])
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    """
    qc.draw(output="mpl")
    plt.show()
    return qc


def suzuki_approx(k, n, theta):
    uk = 1 / (4 - 4 ** (1 / (2 * k - 1)))
    theta1 = [2 * uk * value for value in theta]
    first_term = generate_Hamiltonian_circuit(n, theta1)
    theta2 = [(1 - 4 * uk) * value for value in theta]
    second_term = generate_Hamiltonian_circuit(n, theta2)
    third_term = generate_Hamiltonian_circuit(n, theta1)
    q = QuantumRegister(n)
    quantum_circuit = QuantumCircuit(q)
    quantum_circuit.compose(first_term, inplace=True)
    quantum_circuit.compose(second_term, inplace=True)
    quantum_circuit.compose(third_term, inplace=True)
    # quantum_circuit.draw(output="mpl")
    # plt.show()
    return quantum_circuit


def quantum_phase_estimation(initial_state, Dt, qc, n_ancilla=3, n_target=4):
    qpe_circuit = QuantumCircuit(n_ancilla + n_target, n_ancilla)

    for q in range(n_ancilla):
        #qpe_circuit.reset(q)
        qpe_circuit.h(q)

    qpe_circuit.initialize(initial_state, qubits=list(range(n_ancilla, n_ancilla + n_target)))
    # Controlled-U operations using repeated applications of U
    for q in range(n_ancilla):
        for _ in range(2 ** q):
            controlled_circuit = qc.to_gate().control(1)
            qpe_circuit.append(controlled_circuit, [q] + list(range(n_ancilla, n_ancilla + n_target)))

    # Apply inverse Quantum Fourier Transform (QFT) to ancilla qubits
    qpe_circuit.append(QFT(num_qubits=n_ancilla, inverse=True), range(n_ancilla))
    # Measure the ancilla qubits
   
    qpe_circuit.measure(range(n_ancilla), range(n_ancilla))

    compiled = transpile(qpe_circuit, sim)
    job = sim.run(compiled, shots=1024)
    result = job.result()
    # Interpret QPE results
    counts = result.get_counts()
    print(counts)
    most_frequent_result = max(counts.items(), key=lambda x: x[1])[0]
    phase_decimal = int(most_frequent_result, 2) / (2 ** n_ancilla)

    # Calculate energy eigenvalue
    energy = - phase_decimal * 2 * np.pi / Dt
    # nuclear_repulsion_energy = 0.7559674441714287
    print(f"Most Frequent Result: {most_frequent_result}")
    print(f"Phase (Decimal): {phase_decimal}")
    print(f"Energy Eigenvalue: {energy}")
    # print(f"Energy Eigenvalue: {energy + nuclear_repulsion_energy}")


def main():
    test_case = False
    Dt = 1
    n = 4
    n_ancilla = 6
    if test_case:
        initial_state = np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])
        qc = example_circuit()

        n = 1
    else:
        initial_state = Statevector([0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j],
                                    dims=(2, 2, 2, 2))

        omegas = calculate_overlap_integrals()
        theta = [2 * omega * Dt for omega in omegas]
        qc = generate_Hamiltonian_circuit(n, theta)
        #qc = get_H_paper()
        #qc = suzuki_approx(4, n, theta)
        """
        q = QuantumRegister(n)
        quantum_circuit = QuantumCircuit(q)
  
        for _ in range(10):
            quantum_circuit.compose(qc, inplace=True)

        initial_state = initial_state.data
        #qc = generate_Hamiltonian_circuit(n, theta)
        #qc = get_H_qiskit()
        # qc = get_H_paper()
        """
    quantum_phase_estimation(initial_state, Dt, qc, n_ancilla, n)


if __name__ == "__main__":
    main()

"""
GROUND STATE: 
initial_state = Statevector([0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                                     0. + 0.j, 0. + 0.j],
                                    dims=(2, 2, 2, 2))
E bond length 0.7 ground state = -1.892 (not including nuclear repulsion energy)
                            
EXCITED STATE: 
        initial_state = Statevector([-1.42965093e-17 - 4.74869138e-18j,
                                     6.92830477e-15 + 4.77406071e-15j,
                                     -1.98489648e-17 + 4.14077216e-17j,
                                     -1.82153156e-15 - 6.56312244e-16j,
                                     5.70005001e-16 - 1.50948627e-16j,
                                     -1.42734596e-16 + 1.13579168e-16j,
                                     1.85069225e-15 + 6.93802076e-16j,
                                     1.09559587e-18 + 6.24897272e-18j,
                                     1.19127514e-16 - 1.80123607e-17j,
                                     -1.79595055e-17 - 1.96672094e-16j,
                                     2.56420140e-18 + 2.89721510e-17j,
                                     1.19184607e-02 + 8.16412167e-01j,
                                     4.47687952e-01 - 3.64560861e-01j,
                                     4.90838621e-15 - 7.21192017e-16j,
                                     -3.07276076e-17 - 5.13694653e-17j,
                                     -6.03166265e-16 + 6.61965784e-15j])


E bond length 0.7 excited state = -1.2779 (not including nuclear repulsion energy)
"""