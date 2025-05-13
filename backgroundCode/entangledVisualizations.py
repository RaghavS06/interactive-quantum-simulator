import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plots

# --- Qubit Creation & Single Qubit Gates (from previous steps) ---
def create_qubit(alpha, beta):
    qubit = np.array([alpha, beta], dtype=complex)
    norm = np.linalg.norm(qubit)
    if np.isclose(norm, 0): raise ValueError("Qubit norm cannot be zero.")
    return qubit / norm

q_zero = np.array([1, 0], dtype=complex)
q_one = np.array([0, 1], dtype=complex)

X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
I_GATE = np.array([[1, 0], [0, 1]], dtype=complex)

# --- Multi-Qubit Functions (from previous step) ---
def apply_gate_multiqubit(state_vector, gate_matrix):
    return np.dot(gate_matrix, state_vector)

CNOT_GATE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

H_on_0_I_on_1 = np.kron(H_GATE, I_GATE) # H on qubit 0, I on qubit 1

# --- New Functions for Density Matrices & Partial Trace ---

def state_vector_to_density_matrix(state_vector):
    """Converts a state vector |psi> to a density matrix rho = |psi><psi|."""
    # Ensure state_vector is a column vector for outer product, or handle 1D array
    if state_vector.ndim == 1:
        # For 1D array psi, np.outer(psi, psi.conj()) works directly.
        return np.outer(state_vector, state_vector.conj())
    elif state_vector.ndim == 2 and state_vector.shape[1] == 1: # Column vector
        return state_vector @ state_vector.conj().T
    else:
        raise ValueError("Input state_vector must be a 1D array or a column vector.")

def get_reduced_density_matrix_A(rho_AB_4x4):
    """Calculates the reduced density matrix for qubit A from a 2-qubit density matrix rho_AB."""
    # rho_AB indices: (qubitA_row, qubitB_row; qubitA_col, qubitB_col)
    # Basis order: |00>, |01>, |10>, |11>
    # rho_A[0,0] = rho_AB[0,0] (00,00) + rho_AB[1,1] (01,01)
    # rho_A[0,1] = rho_AB[0,2] (00,10) + rho_AB[1,3] (01,11)
    # rho_A[1,0] = rho_AB[2,0] (10,00) + rho_AB[3,1] (11,01)
    # rho_A[1,1] = rho_AB[2,2] (10,10) + rho_AB[3,3] (11,11)
    rho_A = np.zeros((2,2), dtype=complex)
    rho_A[0,0] = rho_AB_4x4[0,0] + rho_AB_4x4[1,1]
    rho_A[0,1] = rho_AB_4x4[0,2] + rho_AB_4x4[1,3]
    rho_A[1,0] = rho_AB_4x4[2,0] + rho_AB_4x4[3,1]
    rho_A[1,1] = rho_AB_4x4[2,2] + rho_AB_4x4[3,3]
    return rho_A

def get_reduced_density_matrix_B(rho_AB_4x4):
    """Calculates the reduced density matrix for qubit B from a 2-qubit density matrix rho_AB."""
    # rho_B[0,0] = rho_AB[0,0] (00,00) + rho_AB[2,2] (10,10)
    # rho_B[0,1] = rho_AB[0,1] (00,01) + rho_AB[2,3] (10,11)
    # rho_B[1,0] = rho_AB[1,0] (01,00) + rho_AB[3,2] (11,10)
    # rho_B[1,1] = rho_AB[1,1] (01,01) + rho_AB[3,3] (11,11)
    rho_B = np.zeros((2,2), dtype=complex)
    rho_B[0,0] = rho_AB_4x4[0,0] + rho_AB_4x4[2,2]
    rho_B[0,1] = rho_AB_4x4[0,1] + rho_AB_4x4[2,3]
    rho_B[1,0] = rho_AB_4x4[1,0] + rho_AB_4x4[3,2]
    rho_B[1,1] = rho_AB_4x4[1,1] + rho_AB_4x4[3,3]
    return rho_B

def density_matrix_to_bloch_vector(rho_single_qubit):
    """Converts a single-qubit density matrix to its Bloch vector (x,y,z)."""
    # Ensure rho_single_qubit is a 2x2 matrix
    if rho_single_qubit.shape != (2,2):
        raise ValueError("Input density matrix must be 2x2.")
    
    # Pauli matrices (already defined as X_GATE, Y_GATE, Z_GATE)
    # x = Tr(sigma_x * rho)
    # y = Tr(sigma_y * rho)
    # z = Tr(sigma_z * rho)
    
    # Direct calculation from rho elements for speed/clarity:
    # rho = [[rho_00, rho_01], [rho_10, rho_11]]
    # x = rho_01 + rho_10
    # y = i * (rho_10 - rho_01)
    # z = rho_00 - rho_11
    
    x = np.real(np.trace(X_GATE @ rho_single_qubit))
    y = np.real(np.trace(Y_GATE @ rho_single_qubit))
    z = np.real(np.trace(Z_GATE @ rho_single_qubit))
    
    return x, y, z

# --- Bloch Sphere Plotting Function (Slightly Modified) ---
def plot_bloch_sphere(coords_list, title="Bloch Sphere", ax=None, subplot_title=""):
    """Plots the Bloch sphere and qubit state vectors on it.
    coords_list is a list of (x,y,z,color,label,linestyle) tuples.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        set_ax_props = True
    else:
        set_ax_props = False # Assume axis is already set up

    # Draw the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='lightblue', alpha=0.2, rstride=5, cstride=5, edgecolor='gray', linewidth=0.2)

    # Draw the qubit state vectors
    for x,y,z,color,label,linestyle in coords_list:
        vector_length = np.sqrt(x**2 + y**2 + z**2)
        ax.quiver(0, 0, 0, x, y, z, length=vector_length, color=color, 
                  arrow_length_ratio=0.1 if vector_length > 0.01 else 0, # No arrow for zero vector
                  label=label, linestyle=linestyle, linewidth=1.5)
        if vector_length > 0.01: # Don't plot a big dot for the origin
            ax.scatter(x,y,z, color=color, s=30)

    if set_ax_props or not ax.get_title(): # Set properties if new axis or if subplot title is empty
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.view_init(elev=20., azim=30) # Set a nice viewing angle
        ax.set_box_aspect([1,1,1]) 

        ax.text(0, 0, 1.05, r'$|0\rangle$', fontsize=10, ha='center')
        ax.text(0, 0, -1.15, r'$|1\rangle$', fontsize=10, ha='center')
    
    if subplot_title:
        ax.set_title(subplot_title, fontsize=10)
    elif set_ax_props:
         ax.set_title(title, fontsize=12)

    # Create legend if labels are present
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Filter out duplicate labels for clarity if multiple steps are on same plot
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(0.7, 0.95))
    return ax

# --- Main Demonstration ---
if __name__ == "__main__":
    # Initial state vectors for |0> and |1> for qubit A and B
    psi_A_0 = q_zero
    psi_B_0 = q_zero

    # --- Figure Setup ---
    fig = plt.figure(figsize=(18, 6 * 3)) # Adjusted for 3 rows
    fig.suptitle("Visualizing Entanglement: Evolution of Individual Qubit States", fontsize=16, y=0.95)


    # --- Step 0: Initial State |00> ---
    current_A_coords = []
    current_B_coords = []

    psi_AB_step0 = np.kron(psi_A_0, psi_B_0) # |00>
    rho_AB_step0 = state_vector_to_density_matrix(psi_AB_step0)
    
    rho_A_step0 = get_reduced_density_matrix_A(rho_AB_step0)
    rho_B_step0 = get_reduced_density_matrix_B(rho_AB_step0)
    
    bloch_A_step0 = density_matrix_to_bloch_vector(rho_A_step0)
    bloch_B_step0 = density_matrix_to_bloch_vector(rho_B_step0)
    
    current_A_coords.append((*bloch_A_step0, 'blue', 'Step 0: |0⟩_A', '-'))
    current_B_coords.append((*bloch_B_step0, 'green', 'Step 0: |0⟩_B', '-'))

    ax_A_s0 = fig.add_subplot(3, 2, 1, projection='3d')
    plot_bloch_sphere([current_A_coords[-1]], ax=ax_A_s0, subplot_title="Qubit A: Initial |00⟩")
    ax_B_s0 = fig.add_subplot(3, 2, 2, projection='3d')
    plot_bloch_sphere([current_B_coords[-1]], ax=ax_B_s0, subplot_title="Qubit B: Initial |00⟩")
    
    print("Step 0: State = |00>")
    print(f"  Bloch A: {bloch_A_step0}")
    print(f"  Bloch B: {bloch_B_step0}\n")

    # --- Step 1: Apply H to Qubit A ---
    # State is (H|0>)_A |0>_B = |+>_A |0>_B
    psi_AB_step1 = apply_gate_multiqubit(psi_AB_step0, H_on_0_I_on_1)
    rho_AB_step1 = state_vector_to_density_matrix(psi_AB_step1)

    rho_A_step1 = get_reduced_density_matrix_A(rho_AB_step1)
    rho_B_step1 = get_reduced_density_matrix_B(rho_AB_step1)

    bloch_A_step1 = density_matrix_to_bloch_vector(rho_A_step1)
    bloch_B_step1 = density_matrix_to_bloch_vector(rho_B_step1)
    
    current_A_coords.append((*bloch_A_step1, 'red', 'Step 1: H_A|0⟩_A', '--'))
    current_B_coords.append((*bloch_B_step1, 'orange', 'Step 1: |0⟩_B', '--'))
    
    ax_A_s1 = fig.add_subplot(3, 2, 3, projection='3d')
    plot_bloch_sphere(current_A_coords, ax=ax_A_s1, subplot_title="Qubit A: After H on A") # Show history
    ax_B_s1 = fig.add_subplot(3, 2, 4, projection='3d')
    plot_bloch_sphere(current_B_coords, ax=ax_B_s1, subplot_title="Qubit B: After H on A") # Show history
    
    print("Step 1: State = (H⊗I)|00> = |+>|0>")
    print(f"  Bloch A: {bloch_A_step1}")
    print(f"  Bloch B: {bloch_B_step1}\n")

    # --- Step 2: Apply CNOT (A as control, B as target) ---
    # State is CNOT_AB (|+)A |0>B) = Bell state (1/sqrt(2))(|00> + |11>)
    psi_AB_step2_bell_state = apply_gate_multiqubit(psi_AB_step1, CNOT_GATE)
    rho_AB_step2_bell_state = state_vector_to_density_matrix(psi_AB_step2_bell_state)

    rho_A_step2 = get_reduced_density_matrix_A(rho_AB_step2_bell_state)
    rho_B_step2 = get_reduced_density_matrix_B(rho_AB_step2_bell_state)

    bloch_A_step2 = density_matrix_to_bloch_vector(rho_A_step2) # Should be (0,0,0)
    bloch_B_step2 = density_matrix_to_bloch_vector(rho_B_step2) # Should be (0,0,0)

    current_A_coords.append((*bloch_A_step2, 'purple', 'Step 2: Entangled A', ':'))
    current_B_coords.append((*bloch_B_step2, 'brown', 'Step 2: Entangled B', ':'))

    ax_A_s2 = fig.add_subplot(3, 2, 5, projection='3d')
    plot_bloch_sphere(current_A_coords, ax=ax_A_s2, subplot_title="Qubit A: After CNOT (Entangled)")
    ax_B_s2 = fig.add_subplot(3, 2, 6, projection='3d')
    plot_bloch_sphere(current_B_coords, ax=ax_B_s2, subplot_title="Qubit B: After CNOT (Entangled)")

    print("Step 2: State = CNOT(H⊗I)|00> = Bell State (1/sqrt(2))[|00> + |11>]")
    print(f"  Density Matrix for Qubit A (rho_A):\n{rho_A_step2}")
    print(f"  Bloch A: {bloch_A_step2}")
    print(f"  Density Matrix for Qubit B (rho_B):\n{rho_B_step2}")
    print(f"  Bloch B: {bloch_B_step2}\n")
    
    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout to make space for suptitle
    plt.show()