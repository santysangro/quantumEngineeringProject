import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator
from openfermionpyscf import generate_molecular_hamiltonian
from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT

# GLOBALS:
sim = StatevectorSimulator()


def calculate_overlap_integrals():
    geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7))]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0

    hamiltonian = generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)

    one_body_coefficients = hamiltonian.one_body_tensor
    two_body_coefficients = hamiltonian.two_body_tensor

    h00 = one_body_coefficients[0, 0]
    h11 = one_body_coefficients[1, 1]
    h22 = one_body_coefficients[2, 2]
    h33 = one_body_coefficients[3, 3]

    h0110 = two_body_coefficients[0, 1, 1, 0]
    h0330 = two_body_coefficients[0, 3, 3, 0]
    h1221 = two_body_coefficients[1, 2, 2, 1]
    h2332 = two_body_coefficients[2, 3, 3, 2]
    h0220 = two_body_coefficients[0, 2, 2, 0]
    h2020 = two_body_coefficients[2, 0, 2, 0]
    h1313 = two_body_coefficients[1, 3, 1, 3]
    h3113 = two_body_coefficients[3, 1, 1, 3]
    h0132 = two_body_coefficients[0, 1, 3, 2]
    h0312 = two_body_coefficients[0, 3, 1, 2]

    omega_1 = sum([h00, h11, h22, h33]) / 2 + (h0110 + h0330 + h1221 + h2332) / 4 + (h0220 - h2020) / 4 + (
            h1313 - h3113) / 4
    omega_2 = h00 / 2 + h0110 / 4 + h0330 / 4 + h0220 / 4 - h2020 / 4
    omega_3 = h0110 / 4
    omega_4 = -(h22 / 2 + h1221 / 4 + h2332 / 4 + h0220 / 4 - h2020 / 4)
    omega_5 = -(h11 / 2 + h0110 / 4 + h1221 / 4 + h1313 / 4 - h3113 / 4)
    omega_6 = (h0220 - h2020) / 4
    omega_7 = h2332 / 4
    omega_8 = h0132 / 4
    omega_9 = (h0132 - h0312) / 8
    omega_10 = h1221 / 4
    omega_11 = (h1313 - h3113) / 4
    omega_12 = -(h33 / 2 + h0330 / 4 + h2332 / 4 + h1313 / 4 - h3113 / 4)
    omega_13 = (h0132 + h0312) / 8
    omega_14 = (h0132 + h0312) / 8
    omega_15 = h0330 / 4

    omegas = [omega_1, omega_2, omega_3, omega_4, omega_5, omega_6, omega_7, omega_8, omega_9, omega_10, omega_11,
              omega_12, omega_13, omega_14, omega_15]
    return omegas


#Example circuit from lecture slides Quantum Comunication& Computation Week 3
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

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    Ri_matrix = np.array([[np.exp(-1j * theta[0] / 2), 0],
                          [0, np.exp(1j * theta[0] / 2)]])
    unitary_gate = Operator(Ri_matrix)

    qc.append(unitary_gate, [3])
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
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

    qc.rx(0.785, 0)
    qc.rx(0.785, 2)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(theta[8], 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.rx(-0.785, 0)
    qc.rx(-0.785, 2)

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

    qc.rx(0.785, 0)
    qc.rx(0.785, 2)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[13], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.rx(-0.785, 0)
    qc.rx(-0.785, 2)
    # FOURTH SECTION
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta[14], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    return qc


def quantum_phase_estimation(initial_state, Dt, qc, n_ancilla=3, n_target=4):
    qpe_circuit = QuantumCircuit(n_ancilla + n_target, n_ancilla)

    for q in range(n_ancilla):
        qpe_circuit.reset(q)
        qpe_circuit.h(q)
    qpe_circuit.initialize(initial_state, qubits=list(range(n_ancilla, n_ancilla + n_target)))
    # Controlled-U operations using repeated applications of U
    for q in range(n_ancilla):
        for _ in range(2 ** q):
            controlled_circuit = qc.control(num_ctrl_qubits=1)
            qpe_circuit.append(controlled_circuit, [q] + list(range(n_ancilla, n_ancilla + n_target)))

    # Apply inverse Quantum Fourier Transform (QFT) to ancilla qubits
    qpe_circuit.append(QFT(num_qubits=n_ancilla, inverse=True).to_gate(), range(n_ancilla))

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
    t = Dt  # Evolution time used in your unitary
    energy = phase_decimal * 2 * np.pi / t

    print(f"Most Frequent Result: {most_frequent_result}")
    print(f"Phase (Decimal): {phase_decimal}")
    print(f"Energy Eigenvalue: {energy}")


def main():
    test_case = True
    Dt = 1
    n_ancilla = 7
    if test_case:
        initial_state = np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])
        qc = example_circuit()

        n_target = 1
    else:
        initial_state = Statevector([-1.43242713e-16 - 3.42061922e-19j,
                                     2.69077742e-16 + 1.31011834e-16j,
                                     -6.13036438e-02 + 8.48454935e-02j,
                                     -6.66979993e-17 - 2.45946145e-16j,
                                     -5.83722787e-17 + 1.64383932e-17j,
                                     -2.50090981e-16 - 3.34441110e-16j,
                                     2.15239159e-16 - 7.78175101e-17j,
                                     5.82438613e-01 - 8.06106921e-01j,
                                     1.25880380e-16 + 7.29404981e-18j,
                                     -1.06842552e-17 - 1.77982080e-16j,
                                     9.95321351e-18 - 2.35111180e-17j,
                                     1.03461020e-16 - 2.90948381e-16j,
                                     3.28184906e-16 - 1.94749371e-16j,
                                     9.02125371e-18 - 4.80773953e-17j,
                                     2.45417902e-16 - 3.46055651e-16j,
                                     -4.98914730e-17 + 3.15834335e-17j])

        omegas = [-1.38503261201572, -0.4083196888649184, 0.08529869164400526, -0.009296478540293005,
                  0.5312527394267565,
                  0.061466525280919046, 0.08813820402159804, 0.022375072007675825, 0.0, 0.08384159728859487,
                  -0.06146652528091903, 0.11363657202154508, 0.022375072007675825, 0.022375072007675825,
                  0.08384159728859487]

        theta = [2 * omega * Dt for omega in omegas]
        n = 4
        initial_state = initial_state.data
        qc = generate_Hamiltonian_circuit(n, theta)
        n_target = 4

    quantum_phase_estimation(initial_state, Dt, qc, n_ancilla, n_target)


if __name__ == "__main__":
    main()
