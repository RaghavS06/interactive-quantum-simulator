import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plots

# --- 1. Qubit Creation (from previous step) ---
def create_qubit(alpha, beta):
    """Creates a qubit state vector [alpha, beta] and normalizes it."""
    if not (isinstance(alpha, (int, float, complex)) and isinstance(beta, (int, float, complex))):
        raise TypeError("Alpha and beta must be numeric values (int, float, or complex).")
    
    qubit = np.array([alpha, beta], dtype=complex)
    norm = np.linalg.norm(qubit)
    if norm == 0:
        # Or raise an error, as a zero vector isn't a valid qubit state by strict definition
        # but for some intermediate calculations it might occur before normalization.
        # For now, let's return a default state or raise error.
        raise ValueError("Alpha and beta cannot both be zero, resulting in a zero vector.")
    qubit = qubit / norm
    return qubit

# --- 2. Define Common Quantum Gates ---
# Pauli Gates
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard Gate
H_GATE = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

# Phase Gates
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)  # Phase shift of pi/2
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=complex) # Phase shift of pi/4

# Rotation Gates (as per your outline)
def Rx_GATE(theta):
    return np.array([
        [np.cos(theta/2), -1j * np.sin(theta/2)],
        [-1j * np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Ry_GATE(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Rz_GATE(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

# --- 3. Function to Apply a Gate ---
def apply_gate(qubit_state, gate_matrix):
    """Applies a quantum gate (matrix) to a qubit state (vector)."""
    if not isinstance(qubit_state, np.ndarray) or qubit_state.shape != (2,):
        raise ValueError("Qubit state must be a 2-element NumPy array.")
    if not isinstance(gate_matrix, np.ndarray) or gate_matrix.shape != (2,2):
        raise ValueError("Gate matrix must be a 2x2 NumPy array.")
        
    new_state = np.dot(gate_matrix, qubit_state)
    # Normalization might be needed here if gates are not perfectly unitary due to float precision
    # but theoretically, unitary gates preserve the norm.
    # For safety, one could re-normalize:
    # norm = np.linalg.norm(new_state)
    # if norm != 0: new_state = new_state / norm
    return new_state

# --- 4. Convert Qubit Amplitudes to Bloch Sphere Coordinates ---
def qubit_to_bloch_coords(qubit_state):
    """Converts qubit state [alpha, beta] to Bloch sphere (x, y, z) coordinates."""
    alpha = qubit_state[0]
    beta = qubit_state[1]
    
    # Ensure normalization for accurate coordinate calculation
    # (already done in create_qubit, but good for standalone use)
    # norm_squared = np.abs(alpha)**2 + np.abs(beta)**2
    # if not np.isclose(norm_squared, 1.0):
    #     raise ValueError(f"Qubit state must be normalized. Sum of squares is {norm_squared}")

    # x = 2 * np.real(np.conj(alpha) * beta)
    # y = 2 * np.imag(np.conj(alpha) * beta)
    # z = np.abs(alpha)**2 - np.abs(beta)**2
    
    # Alternative calculation from alpha = cos(theta/2), beta = exp(i*phi)sin(theta/2)
    # This avoids issues if alpha is 0 for phi calculation
    
    # Global phase doesn't affect Bloch sphere representation.
    # We can factor out global phase to make alpha real for simplicity in deriving theta, phi
    # However, direct calculation of x,y,z is more robust
    
    # Using theta and phi explicitly for clarity with standard definitions:
    # alpha = cos(theta/2)
    # beta = exp(i*phi)sin(theta/2)
    
    # Handle edge case for |1> state where alpha is 0
    if np.isclose(np.abs(alpha), 0): # Qubit is |1> or very close
        theta = np.pi
        phi = np.angle(beta) # phase of beta
    elif np.isclose(np.abs(beta),0): # Qubit is |0> or very close
        theta = 0
        phi = 0 # phi is undefined, conventionally 0
    else:
        # theta = 2 * np.arccos(np.abs(alpha)) -> can be problematic if |alpha| > 1 due to precision
        # A more robust way for theta:
        theta = 2 * np.arctan2(np.abs(beta), np.abs(alpha))
        # phi is the phase of beta relative to alpha
        phi = np.angle(beta) - np.angle(alpha)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return x, y, z

# --- 5. Function to Plot the Bloch Sphere ---
def plot_bloch_sphere(coords, title="Bloch Sphere", ax=None):
    """Plots the Bloch sphere and a qubit state vector on it."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        set_ax_props = True
    else:
        set_ax_props = False # Assume axis is already set up if provided

    # Draw the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='lightblue', alpha=0.2, rstride=4, cstride=4, edgecolor='gray', linewidth=0.5)

    # Draw the qubit state vector
    if coords: # coords can be a list of tuples (x,y,z,color,label) or a single tuple
        if not isinstance(coords, list):
            coords = [(coords[0],coords[1],coords[2], 'r', 'State')]

        for x,y,z,color,label in coords:
            ax.quiver(0, 0, 0, x, y, z, length=1.0, color=color, arrow_length_ratio=0.1, label=label)
            ax.scatter(x,y,z, color=color, s=50) # Mark the point on the sphere

    if set_ax_props:
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1,1,1]) # Aspect ratio is 1:1:1

        # Add labels for |0> and |1>
        ax.text(0, 0, 1.1, r'$|0\rangle$', fontsize=12, ha='center')
        ax.text(0, 0, -1.2, r'$|1\rangle$', fontsize=12, ha='center')
        # Add labels for axes basis states if desired
        ax.text(1.1, 0, 0, r'$|+\rangle, |R_Y(-\pi/2)0\rangle$', fontsize=10, ha='center') # Positive X
        ax.text(-1.2, 0, 0, r'$|-\rangle, |R_Y(\pi/2)0\rangle$', fontsize=10, ha='center') # Negative X
        ax.text(0, 1.1, 0, r'$|i\rangle, |R_X(\pi/2)0\rangle$', fontsize=10, ha='center')  # Positive Y
        ax.text(0, -1.2, 0, r'$|-i\rangle, |R_X(-\pi/2)0\rangle$', fontsize=10, ha='center')# Negative Y

    ax.set_title(title)
    if coords and any(c[4] for c in coords): # if there's any label
        ax.legend()
    return ax


# --- 6. Demonstration ---
if __name__ == "__main__":
    # --- Example 1: Start with |0>, apply Hadamard ---
    q0 = create_qubit(1, 0) # |0> state
    coords_q0 = qubit_to_bloch_coords(q0)
    print(f"Initial state |0>: {q0}, Bloch coords: {coords_q0}")

    q_after_H = apply_gate(q0, H_GATE)
    coords_after_H = qubit_to_bloch_coords(q_after_H)
    print(f"State after H on |0> (|H0>): {q_after_H}, Bloch coords: {coords_after_H}")
    
    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(111, projection='3d')
    plot_bloch_sphere([(coords_q0[0], coords_q0[1], coords_q0[2], 'blue', '|0⟩'),
                       (coords_after_H[0], coords_after_H[1], coords_after_H[2], 'red', 'H|0⟩')], 
                      title="Hadamard on |0⟩", ax=ax1)
    plt.show()

    # --- Example 2: Start with |0>, apply X gate ---
    q1 = create_qubit(1, 0) # |0> state
    coords_q1_initial = qubit_to_bloch_coords(q1)
    
    q_after_X = apply_gate(q1, X_GATE) # Should be |1>
    coords_after_X = qubit_to_bloch_coords(q_after_X)
    print(f"\nInitial state |0>: {q1}, Bloch coords: {coords_q1_initial}")
    print(f"State after X on |0> (|X0>): {q_after_X}, Bloch coords: {coords_after_X}")

    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111, projection='3d')
    plot_bloch_sphere([(coords_q1_initial[0], coords_q1_initial[1], coords_q1_initial[2], 'blue', '|0⟩'),
                       (coords_after_X[0], coords_after_X[1], coords_after_X[2], 'green', 'X|0⟩ = |1⟩')], 
                      title="X Gate on |0⟩", ax=ax2)
    plt.show()

    # --- Example 3: Start with |+> (H|0>), apply Z gate ---
    q_plus = create_qubit(1/np.sqrt(2), 1/np.sqrt(2)) # |+> state
    coords_q_plus = qubit_to_bloch_coords(q_plus)
    print(f"\nInitial state |+>: {q_plus}, Bloch coords: {coords_q_plus}")

    q_after_Z_on_plus = apply_gate(q_plus, Z_GATE) # Should be |-> state
    coords_after_Z_on_plus = qubit_to_bloch_coords(q_after_Z_on_plus)
    print(f"State after Z on |+> (|Z+>): {q_after_Z_on_plus}, Bloch coords: {coords_after_Z_on_plus}")
    
    fig3 = plt.figure(figsize=(8,8))
    ax3 = fig3.add_subplot(111, projection='3d')
    plot_bloch_sphere([(coords_q_plus[0], coords_q_plus[1], coords_q_plus[2], 'red', '|+⟩'),
                       (coords_after_Z_on_plus[0], coords_after_Z_on_plus[1], coords_after_Z_on_plus[2], 'purple', 'Z|+⟩ = |−⟩')], 
                      title="Z Gate on |+⟩", ax=ax3)
    plt.show()
    
    # --- Example 4: Rx Gate ---
    q_initial_rx = create_qubit(1,0) # |0> state
    coords_initial_rx = qubit_to_bloch_coords(q_initial_rx)
    
    # Rotate by pi/2 around X axis
    q_after_Rx = apply_gate(q_initial_rx, Rx_GATE(np.pi/2))
    coords_after_Rx = qubit_to_bloch_coords(q_after_Rx)
    print(f"\nInitial state |0>: {q_initial_rx}, Bloch coords: {coords_initial_rx}")
    print(f"State after Rx(pi/2) on |0>: {q_after_Rx}, Bloch coords: {coords_after_Rx}")

    fig4 = plt.figure(figsize=(8,8))
    ax4 = fig4.add_subplot(111, projection='3d')
    plot_bloch_sphere([(coords_initial_rx[0], coords_initial_rx[1], coords_initial_rx[2], 'blue', '|0⟩'),
                       (coords_after_Rx[0], coords_after_Rx[1], coords_after_Rx[2], 'orange', 'Rx(π/2)|0⟩')],
                      title="Rx(π/2) Gate on |0⟩", ax=ax4)
    plt.show()