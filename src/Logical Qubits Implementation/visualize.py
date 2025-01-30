# Simulation Imports
import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer import QasmSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer.noise import NoiseModel, pauli_error

import time

# Internal Imports
from util import generateHamiltonian, getBuilderByType #generateThetas





def vizualizeFidelity(theta : list, initial_state : Statevector, logical_encodings : list = ["Single Qubit", "Double Qubit"] , prob_samples : int = 50, model_samples : int = 100, include_errors : bool = False ):
    """
    Visualize the fidelity of the Hydrogen Molecule Hamiltonian under Noise.

    This function produces a plot of the average fidelity computed for the 
    H2 Molecule Hamiltonian for varying noise rates. 100 noisy circuits are produced,
    and simulated to produce noisy statevectors that are compared to an idealstatevector.

    Parameters:
        theta (List): the list of parameters for the Hamiltonian.
        initial_state (Statevector): the initial state of the H2 molecule.
        logicalEncodings (List): a list containing all the logical codes to simulate.
        prob_samples (int): the total number of gate error probability samples
        model_samples (int): number of noisy circuits produced.
        include_errors (bool): include error bars in the plot.

    """

    def _compute_fidelity(probabilities, repetitions, encoding_type):
        """A hidden helper function to compute the fidelity for a single logical code"""
        
        fidelities_mean = []
        fidelities_std = []

        sim = QasmSimulator()

        # Derive the ideal statevector
        circuit = generateHamiltonian(theta, getBuilderByType(encoding_type), initial_state).build()
        circuit.save_statevector()
        compiled = transpile(circuit, sim)
        job = sim.run(compiled)
        ideal_state = job.result().get_statevector()

        
        # Iterate over the gate error probabilities
        for p in probabilities:

            fidelities = []


            # Construct the noise model with Pauli errors for the specific error probability.
            noise_model = NoiseModel()
            pauli_error_model = pauli_error([('X', p/3), ('Y', p/3), ('Z', p/3), ('I', 1 - p)])
            noise_model.add_all_qubit_quantum_error(pauli_error_model, 'unitary')


            for _ in range(repetitions):

                # Simulate the circuit with the noise model
                sim = QasmSimulator(noise_model=noise_model)
                compiled = transpile(circuit, sim)
                job = sim.run(compiled, shots = 1)
                noisy_result = job.result()
                noisy_state = Statevector(noisy_result.get_statevector())

                # Compute fidelity
                fidelity = state_fidelity(ideal_state, noisy_state)
                fidelities.append(fidelity)

            # Compute the average and the standard deviation.
            fidelities_mean.append(np.mean(fidelities))
            fidelities_std.append(np.std(fidelities))
        return fidelities_mean, fidelities_std


    # Compute the fidelities for each encoding
    probabilities = np.linspace(0, 0.1, prob_samples)  
    fidelities_mean = []
    fidelities_std = []

    

    print("----Starting Fidelty Test----")
    for encoding_type in logical_encodings:

        # Simulate using the logical code.
        mean, std = _compute_fidelity(probabilities, repetitions=model_samples, encoding_type=encoding_type)
        
        fidelities_mean.append(mean)
        fidelities_std.append(std)
    print("----Ending Fidelity Test----")

    # Plot the results
    colors = ['blue', 'green', 'orange'] 
    plt.figure(figsize=(10, 6))

    for i, encodingType in enumerate(logical_encodings):
        
        if not include_errors:
            plt.plot(
                probabilities,
                fidelities_mean[i],
                color=colors[i], 
                label=f'{encodingType}'
            )
            plt.scatter(
                probabilities,
                fidelities_mean[i],
                color=colors[i], 
            )
        else:
            plt.errorbar(
                probabilities,
                fidelities_mean[i],
                yerr=fidelities_std[i],
                fmt='o-', 
                ecolor=colors[i], 
                capsize=3, 
                label=f'{encodingType}'
            )

    # Generate the graph
    plt.title("The Variation of State Fidelity With Error Probabiliy in the Hydrogen Molecule Hamiltonian")
    plt.xlabel(r"Error Probability per Gate, $p$")
    plt.ylabel(r"Average Fidelity, $F$")
    plt.grid(True)
    plt.legend()
    plt.show()


def vizualizeErrorRate(theta : list, initial_state : Statevector, logical_encoding : str = "Double Qubit" ,  prob_samples : int = 50, model_samples : int = 100, shots : int = 100):
    """
    Visualize the Error of the Hydrogen Molecule Hamiltonian under Noise.

    This function produces a plot of the error computed for the 
    H2 Molecule Hamiltonian for varying noise rates. 100 noisy circuits are produced, of which 100
    shots are simulated. The error categorization occurs following measurement.

    Parameters:
        theta (List): the list of parameters for the Hamiltonian.
        initial_state (Statevector): the initial state of the H2 molecule.
        logicalEncodings (str): the logical encoding to simulate.
        prob_samples (int): the total number of gate error probability samples.
        model_samples (int): number of noisy circuits produced.
        shots (int): the number of shots simulated per noisy model

    """
    if (logical_encoding != "Double Qubit"):
        raise ValueError("Only 2-qubit code is supported. Steane code is too impractical to simulate")
    

    def check(kv):
        """A helper method to help decode 2-qubit logical code Z-basis measurements and count the total number of errors."""
        s, v = kv

        if all(s[i] == s[i+1] for i in range(0, 8, 2)):
            return 0
        else:
            return v
    
    def compute_error(probabilities, repetitions, logical_encodings = "Double Qubit"):
        """A hidden helper function to compute the error rate for a single logical code"""
        

        circuit = generateHamiltonian(theta, getBuilderByType(logical_encodings), initial_state).build()
        circuit.measure_all()



        error_rates = []

        for p in probabilities:

            
            # Construct the noise model with Pauli errors for the specific error probability.
            noise_model = NoiseModel()
            pauli_error_model = pauli_error([('X', p/3),('Y', p/3),('Z', p/3), ('I', 1 - p)])
            noise_model.add_all_qubit_quantum_error(pauli_error_model, 'unitary')
            
            errors = 0

            # Simulate the circuit with the noise model
            for _ in range(repetitions):
                simulator = QasmSimulator(noise_model=noise_model)
                compiled = transpile(circuit, simulator)
                job = simulator.run(compiled, shots = shots)
                result = job.result()
                errors = errors + np.sum(list(map(lambda item: check(item), result.get_counts().items())))

            # Store the error rate
            error_rates.append(errors / (repetitions * shots))

        
        return error_rates, np.sum(list(circuit.count_ops().values()))

    probabilities = np.linspace(0, 0.1, prob_samples)  


    # Simulate to produce the error rates
    print("----Starting Error Detection Test----")
    errors, gate_count = compute_error(probabilities, repetitions=model_samples, logical_encodings=logical_encodings)
    print("----Ending Error Detection Test----")

    # Create bar chart
    plt.figure(figsize=(8, 5))  # Set figure size
    plt.scatter(probabilities, errors, color='blue', marker='o', edgecolors='black', label="Measured Error Rate")
    plt.plot(probabilities, [ 1 - float(1.0 -p ) ** np.sqrt(gate_count) for p in probabilities], color='red', label="Expected Error Rate")

    plt.legend()

    plt.xlabel(r"Error Probability per Gate, $p$") 
    plt.ylabel(r"Actual Error Rate, $r_{\text{actual}}$")
    plt.title("The Effect of Gate Error Probability on Observed Error Rate ")

    plt.grid(True)
    plt.show()


def vizualizeCircuit(circuit):
    """Vizualize the quantum circuit in an MPL format."""
    return circuit.draw(output="mpl", plot_barriers=False, idle_wires=False, scale=2)


def computeTime(theta : list, initial_state : Statevector, logical_encodings : list = ["Single Qubit", "Double Qubit"], trials : int = 10):
    """Compute the execution time for an Hamiltonian circuit. Steane takes a long time to compute, and likely won't be able to if compute is lacking."""
    
    for logical_encoding in logical_encodings:

        # Set up the circuit and simulator
        sim = QasmSimulator()

        circuit = generateHamiltonian(theta, getBuilderByType(logical_encoding), initial_state).build()
        compiled = transpile(circuit, sim)

        
        times = []
        for _ in range(trials):
            # Avoid scheduling conflicts
            time.sleep(1)

            start_time = time.perf_counter()

            job = sim.run(compiled)
            job.result()

            end_time = time.perf_counter()

            # Compute the elapsed time
            times.append(end_time - start_time)

        print(f"Total Time Elapsed for {logical_encoding} Encoding: {np.mean(times):.6f} seconds.")


def computeGateCount(theta : list, initial_state : Statevector, logical_encodings : list = ["Single Qubit", "Double Qubit", "Steane"]):
    """Computes the total number of gates in a given circuit."""
    for logical_encoding in logical_encodings:

        # Create the circuit
        circuit = generateHamiltonian(theta, getBuilderByType(logical_encoding), initial_state).build()

        print(f"Total Gates for the {logical_encoding} Encoding: {sum(circuit.count_ops().values())}")