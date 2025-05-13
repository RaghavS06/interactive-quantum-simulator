import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plots

# --- Quantum Code (Copied and adapted from previous steps) ---
# Qubit States
q_zero = np.array([1, 0], dtype=complex)
q_one = np.array([0, 1], dtype=complex)

# Single Qubit Gates
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
I_GATE = np.array([[1, 0], [0, 1]], dtype=complex)

# 2-Qubit Gates
CNOT_GATE = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
H_on_0_I_on_1 = np.kron(H_GATE, I_GATE) # H on qubit 0, I on qubit 1

def apply_gate_multiqubit(state_vector, gate_matrix):
    return np.dot(gate_matrix, state_vector)

def state_vector_to_density_matrix(state_vector):
    if state_vector.ndim == 1:
        # For 1D array psi, np.outer(psi, psi.conj()) works directly.
        return np.outer(state_vector, state_vector.conj())
    elif state_vector.ndim == 2 and state_vector.shape[1] == 1: # Column vector
        return state_vector @ state_vector.conj().T
    else:
        raise ValueError("Input state_vector must be a 1D array or a column vector.")

def get_reduced_density_matrix_A(rho_AB_4x4):
    rho_A = np.zeros((2,2), dtype=complex)
    rho_A[0,0] = rho_AB_4x4[0,0] + rho_AB_4x4[1,1]
    rho_A[0,1] = rho_AB_4x4[0,2] + rho_AB_4x4[1,3]
    rho_A[1,0] = rho_AB_4x4[2,0] + rho_AB_4x4[3,1]
    rho_A[1,1] = rho_AB_4x4[2,2] + rho_AB_4x4[3,3]
    return rho_A

def get_reduced_density_matrix_B(rho_AB_4x4):
    rho_B = np.zeros((2,2), dtype=complex)
    rho_B[0,0] = rho_AB_4x4[0,0] + rho_AB_4x4[2,2]
    rho_B[0,1] = rho_AB_4x4[0,1] + rho_AB_4x4[2,3]
    rho_B[1,0] = rho_AB_4x4[1,0] + rho_AB_4x4[3,2]
    rho_B[1,1] = rho_AB_4x4[1,1] + rho_AB_4x4[3,3]
    return rho_B

def density_matrix_to_bloch_vector(rho_single_qubit):
    if rho_single_qubit.shape != (2,2): return (0,0,0) # Should not happen with valid rho
    x = np.real(np.trace(X_GATE @ rho_single_qubit))
    y = np.real(np.trace(Y_GATE @ rho_single_qubit))
    z = np.real(np.trace(Z_GATE @ rho_single_qubit))
    return x, y, z

def get_measurement_probabilities(state_vector):
    return np.abs(state_vector)**2

# --- Plotting Functions for Streamlit ---
def plot_bloch_sphere_streamlit(ax, bloch_vector, title_str): # Renamed title to title_str to avoid conflict
    """Plots a single Bloch sphere on a given Matplotlib axis."""
    ax.clear() # Clear previous drawing on this axis
    # Draw the sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='lightblue', alpha=0.2, rstride=4, cstride=4, edgecolor='gray', linewidth=0.2)

    x, y, z = bloch_vector
    vector_length = np.sqrt(x**2 + y**2 + z**2)
    # Only draw arrow if vector has significant length
    ax.quiver(0, 0, 0, x, y, z, length=vector_length, color='r',
              arrow_length_ratio=0.1 if vector_length > 0.01 else 0, linewidth=1.5)
    if vector_length > 0.01: # Don't plot a big dot for the origin
        ax.scatter(x,y,z, color='r', s=30)

    ax.set_xlabel('X', labelpad=-10)
    ax.set_ylabel('Y', labelpad=-10)
    ax.set_zlabel('Z', labelpad=-10)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(axis='both', which='major', pad=-3) # Adjust padding to bring ticks closer
    ax.set_box_aspect([1,1,1]) # Ensure aspect ratio is 1:1:1 for a sphere
    ax.text(0, 0, 1.05, r'$|0\rangle$', fontsize=9, ha='center')
    ax.text(0, 0, -1.15, r'$|1\rangle$', fontsize=9, ha='center')
    ax.set_title(title_str, fontsize=10)

def plot_probabilities_streamlit(ax, probabilities, n_qubits):
    """Plots measurement probabilities on a given Matplotlib axis."""
    ax.clear()
    basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    ax.bar(basis_states, probabilities, color='skyblue')
    ax.set_xlabel("Basis State")
    ax.set_ylabel("Probability")
    ax.set_title(f"Measurement Probabilities ({n_qubits}-Qubit)", fontsize=10)
    ax.set_ylim(0, 1.05) # Set y-limit to make 1.0 clear
    ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability


# --- Streamlit App Logic ---
st.set_page_config(layout="wide") # Use wide layout
st.title("Interactive 2-Qubit Entanglement Explorer")

# Initialize session state variables if they don't exist
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0 # 0: initial, 1: after H, 2: after CNOT (entangled)
    st.session_state.psi_AB = np.kron(q_zero, q_zero) # Default initial |00>
    st.session_state.history = ["Initial state: |00>"]

def reset_state():
    st.session_state.current_step = 0
    st.session_state.psi_AB = np.kron(q_zero, q_zero)
    st.session_state.history = ["State reset to |00>"]

def step_forward():
    if st.session_state.current_step == 0: # Initial state |00> -> Apply H to Qubit A
        psi_initial = np.kron(q_zero, q_zero) # Ensure we always start from |00> for this step
        st.session_state.psi_AB = apply_gate_multiqubit(psi_initial, H_on_0_I_on_1)
        st.session_state.history.append("Applied H to Qubit A. State is (H⊗I)|00⟩")
        st.session_state.current_step = 1
    elif st.session_state.current_step == 1: # After H on Qubit A -> Apply CNOT
        # psi_AB at this point is (H⊗I)|00>
        st.session_state.psi_AB = apply_gate_multiqubit(st.session_state.psi_AB, CNOT_GATE)
        st.session_state.history.append("Applied CNOT(A,B). State is CNOT(H⊗I)|00⟩ (Bell State)")
        st.session_state.current_step = 2
    # No action if current_step is already 2 (or higher)

# --- UI Controls ---
col_controls1, col_controls2 = st.columns(2)
with col_controls1:
    if st.button("▶️ Next Step / Apply Gate", use_container_width=True, disabled=(st.session_state.current_step >= 2)):
        step_forward()
with col_controls2:
    if st.button("🔄 Reset to |00>", use_container_width=True):
        reset_state()
        # Need to re-run to reflect reset immediately in display
        # st.experimental_rerun() # Use st.rerun() for newer Streamlit versions

# --- Calculations based on current state ---
psi_AB_current = st.session_state.psi_AB
rho_AB_current = state_vector_to_density_matrix(psi_AB_current)

rho_A_current = get_reduced_density_matrix_A(rho_AB_current)
rho_B_current = get_reduced_density_matrix_B(rho_AB_current)

bloch_A_current = density_matrix_to_bloch_vector(rho_A_current)
bloch_B_current = density_matrix_to_bloch_vector(rho_B_current)

probabilities_current = get_measurement_probabilities(psi_AB_current)

# --- Displaying Visualizations and Data ---
step_description = "Initial |00>"
if st.session_state.current_step == 1:
    step_description = "After H on Qubit A: (H⊗I)|00⟩"
elif st.session_state.current_step == 2:
    step_description = "Entangled Bell State: CNOT(H⊗I)|00⟩"

st.header(f"Current State: {step_description}")

col1, col2, col3 = st.columns([2,2,3]) # Define column widths

with col1:
    st.subheader("Qubit A")
    fig_A = plt.figure(figsize=(4,4)) # Create a new figure
    ax_A = fig_A.add_subplot(111, projection='3d') # Add a 3D subplot
    plot_bloch_sphere_streamlit(ax_A, bloch_A_current, "Qubit A")
    st.pyplot(fig_A)

with col2:
    st.subheader("Qubit B")
    fig_B = plt.figure(figsize=(4,4)) # Create a new figure
    ax_B = fig_B.add_subplot(111, projection='3d') # Add a 3D subplot
    plot_bloch_sphere_streamlit(ax_B, bloch_B_current, "Qubit B")
    st.pyplot(fig_B)

with col3:
    st.subheader("System Properties")
    fig_P, ax_P = plt.subplots(figsize=(5,4))
    plot_probabilities_streamlit(ax_P, probabilities_current, n_qubits=2)
    st.pyplot(fig_P)
    
    st.markdown("##### Current 2-Qubit State Vector ($|\psi_{AB}\rangle$):")
    # Improved formatting for state vector
    state_vector_terms = []
    basis_labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    for i, amp in enumerate(psi_AB_current):
        if not np.isclose(amp, 0): # Only show non-zero terms
            real_part = f"{amp.real:.2f}"
            imag_part = ""
            if not np.isclose(amp.imag, 0):
                if amp.imag > 0:
                    imag_part = f"+{amp.imag:.2f}i" if amp.real != 0 or (np.isclose(amp.real,0) and amp.imag >=0) else f"{amp.imag:.2f}i"
                else: # amp.imag < 0
                    imag_part = f"{amp.imag:.2f}i"

            # Handle cases where real or imag part is zero
            if np.isclose(amp.real, 0) and not np.isclose(amp.imag, 0):
                term_amp = imag_part
            elif not np.isclose(amp.real, 0) and np.isclose(amp.imag, 0):
                term_amp = real_part
            else: # Both non-zero, or both zero (filtered out)
                term_amp = f"({real_part}{imag_part})" if not (np.isclose(amp.real,0) and np.isclose(amp.imag,0)) else ""
            
            if term_amp:
                state_vector_terms.append(f"{term_amp} {basis_labels[i]}")
    
    state_vector_str = "  +  ".join(state_vector_terms) if state_vector_terms else "0 (Zero Vector)"
    st.text(state_vector_str)


# --- Optional: Display Density Matrices (can be verbose) ---
with st.expander("Show Operation History and Density Matrices"):
    st.markdown("##### Operation History:")
    # Display history in order of application
    for item in st.session_state.history:
        st.text(f"- {item}")
        
    st.markdown("##### Qubit A Reduced Density Matrix ($\rho_A$):")
    st.text(np.array2string(rho_A_current, formatter={'complex_kind': lambda x: "%.2f+%.2fi" % (x.real, x.imag) if abs(x.imag) > 1e-9 else "%.2f" % x.real}))
    st.markdown("##### Qubit B Reduced Density Matrix ($\rho_B$):")
    st.text(np.array2string(rho_B_current, formatter={'complex_kind': lambda x: "%.2f+%.2fi" % (x.real, x.imag) if abs(x.imag) > 1e-9 else "%.2f" % x.real}))
    st.markdown("##### Full 2-Qubit Density Matrix ($\rho_{AB}$):")
    st.text(np.array2string(rho_AB_current, formatter={'complex_kind': lambda x: "%.2f+%.2fi" % (x.real, x.imag) if abs(x.imag) > 1e-9 else "%.2f" % x.real}))

st.caption("Bloch sphere vectors for individual qubits are derived from their reduced density matrices. When entangled, these vectors go to the center (length ~0), indicating a maximally mixed state for the individual qubit when considered alone.")