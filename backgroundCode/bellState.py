import numpy as np
import matplotlib.pyplot as plt

# --- Single Qubit States and Gates (from previous steps) ---
def create_qubit(alpha, beta):
    """Creates a single qubit state vector [alpha, beta] and normalizes it."""
    qubit = np.array([alpha, beta], dtype=complex)
    norm = np.linalg.norm(qubit)
    if np.isclose(norm, 0):
        raise ValueError("Cannot create a zero vector qubit.")
    return qubit / norm

# Define single qubit states
q_zero = np.array([1, 0], dtype=complex)
q_one = np.array([0, 1], dtype=complex)

# Pauli Gates
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
# Hadamard Gate
H_GATE = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
# Identity Gate
I_GATE = np.array([[1, 0], [0, 1]], dtype=complex)

# --- Multi-Qubit Functions ---

def apply_gate_multiqubit(state_vector, gate_matrix):
    """Applies a quantum gate (matrix) to a multi-qubit state (vector)."""
    n_qubits = int(np.log2(len(state_vector)))
    expected_dim = 2**n_qubits
    if state_vector.shape != (expected_dim,) or gate_matrix.shape != (expected_dim, expected_dim):
        raise ValueError(f"State vector length ({len(state_vector)}) or gate dimensions ({gate_matrix.shape}) are incorrect for {n_qubits} qubits.")
        
    new_state = np.dot(gate_matrix, state_vector)
    # Optional: Re-normalize for numerical stability
    # norm = np.linalg.norm(new_state)
    # if not np.isclose(norm, 0): new_state = new_state / norm
    return new_state

def get_measurement_probabilities(state_vector):
    """Calculates the probability of measuring each basis state."""
    probabilities = np.abs(state_vector)**2
    return probabilities

def plot_probabilities(probabilities):
    """Plots measurement probabilities as a bar chart."""
    n_qubits = int(np.log2(len(probabilities)))
    basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    
    plt.figure(figsize=(8, 4))
    plt.bar(basis_states, probabilities, color='skyblue')
    plt.xlabel("Basis State")
    plt.ylabel("Probability")
    plt.title(f"Measurement Probabilities for {n_qubits}-Qubit State")
    plt.ylim(0, 1) # Probabilities are between 0 and 1
    plt.show()

# --- Define 2-Qubit Gates ---
CNOT_GATE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# Gate for H on qubit 0, I on qubit 1
H_on_0 = np.kron(H_GATE, I_GATE)

# Gate for I on qubit 0, H on qubit 1
H_on_1 = np.kron(I_GATE, H_GATE)

# --- Example: Creating a Bell State (|Φ+>) ---
if __name__ == "__main__":
    print("--- Creating Bell State |Φ+> ---")
    
    # 1. Start with |00> state
    # |00> = |0> kron |0>
    initial_state_00 = np.kron(q_zero, q_zero)
    print(f"Initial state |00>:\n{initial_state_00}")
    probs_initial = get_measurement_probabilities(initial_state_00)
    print(f"Initial probabilities: {probs_initial}")
    # plot_probabilities(probs_initial) # Should be 100% |00>

    # 2. Apply Hadamard gate to the first qubit (qubit 0)
    # We need the operator H⊗I
    state_after_H = apply_gate_multiqubit(initial_state_00, H_on_0)
    # This should result in (1/sqrt(2))(|00> + |10>)
    print(f"\nState after H on qubit 0:\n{state_after_H}") 
    # Amplitudes should be [1/sqrt(2), 0, 1/sqrt(2), 0]
    probs_after_H = get_measurement_probabilities(state_after_H)
    print(f"Probabilities after H on qubit 0: {probs_after_H}")
    # plot_probabilities(probs_after_H) # Should be 50% |00>, 50% |10>

    # 3. Apply CNOT gate (control=qubit 0, target=qubit 1)
    final_state_bell = apply_gate_multiqubit(state_after_H, CNOT_GATE)
    # This transforms (1/sqrt(2))(|00> + |10>) into (1/sqrt(2))(|00> + |11>)
    print(f"\nFinal state after CNOT (Bell state |Φ+>):\n{final_state_bell}")
    # Amplitudes should be [1/sqrt(2), 0, 0, 1/sqrt(2)]

    # 4. Check measurement probabilities for the Bell state
    probs_bell = get_measurement_probabilities(final_state_bell)
    print(f"\nProbabilities for Bell state |Φ+>:")
    n_qubits = int(np.log2(len(probs_bell)))
    basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    for state, prob in zip(basis_states, probs_bell):
        print(f"  P(|{state}>) = {prob:.4f}")

    # 5. Visualize the probabilities
    plot_probabilities(probs_bell) # Should show 50% |00>, 50% |11>

    print("\n--- Interpretation of Bell State Probabilities ---")
    print("The Bell state |Φ+> = (1/sqrt(2))(|00> + |11>) is entangled.")
    print("If you measure the qubits, you will only ever find them BOTH 0 or BOTH 1.")
    print("You will never find |01> or |10>.")
    print("Furthermore, if you measure just the first qubit and find it's 0, you instantly know the second qubit MUST be 0 (and vice-versa).")