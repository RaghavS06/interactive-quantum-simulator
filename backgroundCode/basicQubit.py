import numpy as np

def create_qubit(alpha, beta):
    """Creates a qubit state using NumPy arrays.

    Args:
        alpha: Complex amplitude for the |0> state.
        beta: Complex amplitude for the |1> state.

    Returns:
        A NumPy array representing the qubit state.
    """
    qubit = np.array([alpha, beta], dtype=complex)
    # Normalize the qubit to ensure |alpha|^2 + |beta|^2 = 1
    norm = np.linalg.norm(qubit)
    if norm != 0:
      qubit = qubit / norm
    return qubit

# Example usage:
# This creates a qubit in the state (|0> + |1>)/sqrt(2)
alpha_example = (1 / np.sqrt(2))
beta_example = (1 / np.sqrt(2))
my_qubit = create_qubit(alpha_example, beta_example)
print(f"Qubit state for alpha={alpha_example}, beta={beta_example}:")
print(my_qubit)

# Example for |0> state
qubit_0 = create_qubit(1, 0)
print(f"\nQubit state for |0>:")
print(qubit_0)

# Example for |1> state
qubit_1 = create_qubit(0, 1)
print(f"\nQubit state for |1>:")
print(qubit_1)

# Example for a different superposition: (sqrt(0.3)|0> - sqrt(0.7)|1>)
alpha_superposition = np.sqrt(0.3)
beta_superposition = -np.sqrt(0.7) # Note the negative sign
my_superposition_qubit = create_qubit(alpha_superposition, beta_superposition)
print(f"\nQubit state for alpha={alpha_superposition}, beta={beta_superposition}:")
print(my_superposition_qubit)

# Example with complex amplitudes: ( (1/sqrt(2))|0> + (i/sqrt(2))|1> )
alpha_complex = (1 / np.sqrt(2))
beta_complex = (1j / np.sqrt(2)) # Note '1j' for the imaginary unit
my_complex_qubit = create_qubit(alpha_complex, beta_complex)
print(f"\nQubit state for alpha={alpha_complex}, beta={beta_complex}:")
print(my_complex_qubit)


user_alpha = float(input("Enter the value for alpha (real part): "))
user_beta = float(input("Enter the value for beta (real part): "))
user_qubit = create_qubit(user_alpha, user_beta)
print(f"\nUser-defined qubit state for alpha={user_alpha}, beta={user_beta}:")
print(user_qubit)