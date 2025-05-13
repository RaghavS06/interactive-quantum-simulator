import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math # For pi and trigonometric functions

# --- Quantum Core Functions (Adapted and Extended) ---
# Qubit States (basis vectors)
q_zero_sv = np.array([1, 0], dtype=complex) # sv for state vector
q_one_sv = np.array([0, 1], dtype=complex)

# Single Qubit Gate Matrices
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=complex)
I_GATE = np.array([[1, 0], [0, 1]], dtype=complex)

def Rx_GATE(theta_rad):
    c = np.cos(theta_rad / 2)
    s = np.sin(theta_rad / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def Ry_GATE(theta_rad):
    c = np.cos(theta_rad / 2)
    s = np.sin(theta_rad / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rz_GATE(theta_rad):
    return np.array([[np.exp(-1j * theta_rad / 2), 0],
                     [0, np.exp(1j * theta_rad / 2)]], dtype=complex)

def P_GATE(phi_rad):
    return np.array([[1, 0], [0, np.exp(1j * phi_rad)]], dtype=complex)

# 2-Qubit Gate Matrices
CNOT_01_GATE = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex) # Q0 ctrl, Q1 target
CNOT_10_GATE = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex) # Q1 ctrl, Q0 target
SWAP_GATE = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)


def qubit_from_bloch_angles(theta_rad, phi_rad):
    alpha = np.cos(theta_rad / 2)
    beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
    if np.isclose(theta_rad, 0.0): alpha, beta = 1.0, 0.0
    if np.isclose(theta_rad, np.pi): alpha = 0.0 
    if np.isclose(np.abs(beta), 0.0) and not (np.isclose(theta_rad, 0.0) or np.isclose(theta_rad, np.pi)):
        beta = 0.0
    return np.array([alpha, beta], dtype=complex)

def get_bloch_angles_from_state_vector(psi_sv):
    if psi_sv is None or len(psi_sv) != 2: return 0.0, 0.0
    alpha, beta = psi_sv[0], psi_sv[1]
    norm_sq = np.abs(alpha)**2 + np.abs(beta)**2
    if not np.isclose(norm_sq, 1.0):
        if np.isclose(norm_sq, 0.0): return (0.0, 0.0)
        norm = np.sqrt(norm_sq)
        if norm > 1e-9: alpha, beta = alpha/norm, beta/norm
        else: return (0.0,0.0)

    abs_alpha = np.clip(np.abs(alpha), 0.0, 1.0)
    theta = 2 * np.arccos(abs_alpha)
    phi = np.angle(beta) if not np.isclose(np.abs(beta), 0) else 0.0
    return theta, phi


def construct_full_operator(gate_name_val, base_matrix_or_func_val, target_qubits_list_val, angle_rad_val, num_total_qubits_val):
    actual_base_gate_matrix = base_matrix_or_func_val(angle_rad_val) if callable(base_matrix_or_func_val) and "angle" in GATE_OPTIONS.get(gate_name_val,{}).get("params",[]) else base_matrix_or_func_val

    if num_total_qubits_val == 1:
        return actual_base_gate_matrix

    if num_total_qubits_val == 2:
        gate_n_qubits_def = GATE_OPTIONS.get(gate_name_val, {}).get("n_qubits_gate")
        if gate_n_qubits_def == 1:
            if not target_qubits_list_val: # Should not happen if UI is correct
                st.error(f"Target qubit not specified for {gate_name_val}"); return None
            q_idx = target_qubits_list_val[0]
            if q_idx == 0: return np.kron(actual_base_gate_matrix, I_GATE)
            if q_idx == 1: return np.kron(I_GATE, actual_base_gate_matrix)
            st.error(f"Invalid target qubit index {q_idx} for 1-qubit gate on 2-qubit system."); return None
        elif gate_n_qubits_def == 2:
            return actual_base_gate_matrix # Assumes CNOT, SWAP matrix is already correctly specified (4x4)
    st.error(f"Operator construction for {gate_name_val} on {num_total_qubits_val} qubits failed or is not supported."); return None


def apply_circuit_ops(initial_psi_val, circuit_operations_list_val):
    current_psi_val = initial_psi_val.copy()
    num_total_qubits_session = st.session_state.get('n_qubits', 1)

    if initial_psi_val is None or len(initial_psi_val) != (2**num_total_qubits_session):
        st.error(f"Initial state dimension mismatch. Expected {2**num_total_qubits_session}, got {len(initial_psi_val) if initial_psi_val is not None else 'None'}.")
        return initial_psi_val 

    for op_idx, op_info in enumerate(circuit_operations_list_val):
        gate_name = op_info['name']
        base_matrix_or_func = op_info['matrix_or_func']
        target_qubits = op_info['targets']
        angle_rad = op_info.get('angle_rad')

        full_op_matrix = construct_full_operator(gate_name, base_matrix_or_func, target_qubits, angle_rad, num_total_qubits_session)

        if full_op_matrix is None:
            st.warning(f"Circuit step {op_idx+1} ({gate_name}): Failed to construct operator. Circuit halted for this run.")
            return current_psi_val 

        try:
            current_psi_val = np.dot(full_op_matrix, current_psi_val)
            norm = np.linalg.norm(current_psi_val)
            if not np.isclose(norm, 0.0) and not np.isclose(norm,1.0) :
                current_psi_val /= norm
            elif np.isclose(norm, 0.0) and not np.allclose(current_psi_val,0.0, atol=1e-8):
                 st.warning(f"Circuit step {op_idx+1} ({gate_name}): State vector norm became zero. Check gate unitarity.")
                 non_zero_mask = ~np.isclose(current_psi_val, 0.0, atol=1e-8)
                 if np.any(non_zero_mask):
                     current_psi_val = current_psi_val / np.linalg.norm(current_psi_val[non_zero_mask])
                 else: 
                     st.error(f"Circuit step {op_idx+1} ({gate_name}): State collapsed to zero vector. Circuit halted."); return initial_psi_val
        except Exception as e_apply_dot:
            st.error(f"Error applying gate {gate_name} (Step {op_idx+1}): type={type(e_apply_dot)}, error='{e_apply_dot}'. Circuit halted.")
            return initial_psi_val
    return current_psi_val


# --- Density Matrix and Plotting Functions ---
def state_vector_to_density_matrix(state_vector_val):
    if state_vector_val is None or state_vector_val.ndim != 1:
        default_dim = 2**st.session_state.get('n_qubits', 1)
        return np.zeros((default_dim, default_dim), dtype=complex)
    return np.outer(state_vector_val, state_vector_val.conj())

def get_reduced_density_matrix_A(rho_AB_4x4_val):
    if rho_AB_4x4_val is None or rho_AB_4x4_val.shape != (4,4): return np.array([[0.5,0],[0,0.5]],dtype=complex)
    rho_A = np.zeros((2,2), dtype=complex)
    rho_A[0,0] = rho_AB_4x4_val[0,0] + rho_AB_4x4_val[1,1]; rho_A[0,1] = rho_AB_4x4_val[0,2] + rho_AB_4x4_val[1,3]
    rho_A[1,0] = rho_AB_4x4_val[2,0] + rho_AB_4x4_val[3,1]; rho_A[1,1] = rho_AB_4x4_val[2,2] + rho_AB_4x4_val[3,3]
    return rho_A

def get_reduced_density_matrix_B(rho_AB_4x4_val):
    if rho_AB_4x4_val is None or rho_AB_4x4_val.shape != (4,4): return np.array([[0.5,0],[0,0.5]],dtype=complex)
    rho_B = np.zeros((2,2), dtype=complex)
    rho_B[0,0] = rho_AB_4x4_val[0,0] + rho_AB_4x4_val[2,2]; rho_B[0,1] = rho_AB_4x4_val[0,1] + rho_AB_4x4_val[2,3]
    rho_B[1,0] = rho_AB_4x4_val[1,0] + rho_AB_4x4_val[3,2]; rho_B[1,1] = rho_AB_4x4_val[1,1] + rho_AB_4x4_val[3,3]
    return rho_B

def density_matrix_to_bloch_vector(rho_single_qubit_val):
    if rho_single_qubit_val is None or rho_single_qubit_val.shape != (2,2): return (0.0,0.0,0.0)
    if not np.allclose(rho_single_qubit_val, rho_single_qubit_val.conj().T, atol=1e-7):
        st.debug("Density matrix for Bloch vector slightly non-Hermitian.")
    x = np.real(np.trace(X_GATE @ rho_single_qubit_val))
    y = np.real(np.trace(Y_GATE @ rho_single_qubit_val))
    z = np.real(np.trace(Z_GATE @ rho_single_qubit_val))
    return np.clip(x,-1,1), np.clip(y,-1,1), np.clip(z,-1,1)


def get_measurement_probabilities(state_vector_val):
    if state_vector_val is None: return np.zeros(2**st.session_state.get('n_qubits',1))
    probs = np.abs(state_vector_val)**2
    sum_probs = np.sum(probs)
    if not np.isclose(sum_probs, 1.0) and sum_probs > 1e-9 : probs /= sum_probs
    return probs

def plot_bloch_sphere_streamlit(ax_val, bloch_vector_val, title_str_val):
    ax_val.clear()
    u, v = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
    sphere_x, sphere_y, sphere_z = [np.outer(np.cos(u), np.sin(v)),
                                   np.outer(np.sin(u), np.sin(v)),
                                   np.outer(np.ones(np.size(u)), np.cos(v))]
    try:
        ax_val.plot_surface(sphere_x, sphere_y, sphere_z, color='lightblue', alpha=0.15, rstride=2, cstride=2, edgecolor=(0.5,0.5,0.5,0.1), linewidth=0.1)
    except AttributeError: st.error("Bloch sphere plot failed: Axis not 3D."); return

    x_bv, y_bv, z_bv = bloch_vector_val
    vec_len = np.sqrt(x_bv**2 + y_bv**2 + z_bv**2)
    ax_val.quiver(0,0,0, x_bv,y_bv,z_bv, length=vec_len, color='r', arrow_length_ratio=0.1 if vec_len > 0.05 else 0, linewidth=2, pivot='tail')
    if vec_len > 0.01: ax_val.scatter(x_bv,y_bv,z_bv, color='darkred', s=25)
    
    ax_val.set_xlabel('X',labelpad=-10, fontsize=8); ax_val.set_ylabel('Y',labelpad=-10, fontsize=8); ax_val.set_zlabel('Z',labelpad=-10, fontsize=8)
    ax_val.set_xlim([-1.1,1.1]); ax_val.set_ylim([-1.1,1.1]); ax_val.set_zlim([-1.1,1.1])
    ax_val.set_xticks([-1,0,1]); ax_val.set_yticks([-1,0,1]); ax_val.set_zticks([-1,0,1])
    ax_val.tick_params(axis='both', pad=-4, labelsize=7)
    ax_val.set_box_aspect((1,1,1))
    ax_val.text(0,0,1.1,r'$|0\rangle$',fontsize=9,ha='center',va='center'); ax_val.text(0,0,-1.1,r'$|1\rangle$',fontsize=9,ha='center',va='center')
    ax_val.set_title(title_str_val, fontsize=10)

def plot_probabilities_streamlit(ax_val, probabilities_val, n_qubits_val):
    ax_val.clear()
    if probabilities_val is None or len(probabilities_val) != 2**n_qubits_val:
        probabilities_val = np.zeros(2**n_qubits_val)
        if n_qubits_val > 0 and len(probabilities_val) > 0: probabilities_val[0]=1.0

    basis_states = [format(i, f'0{n_qubits_val}b') for i in range(2**n_qubits_val)]
    ax_val.bar(basis_states, probabilities_val, color='skyblue', edgecolor='black', linewidth=0.7)
    ax_val.set_xlabel("Basis State", fontsize=9); ax_val.set_ylabel("Probability", fontsize=9)
    ax_val.set_title(f"Probabilities ({n_qubits_val}-Qubit)", fontsize=10)
    ax_val.set_ylim(0, 1.05)
    ax_val.tick_params(axis='x', rotation=45 if n_qubits_val > 2 else 0, labelsize=8)
    ax_val.tick_params(axis='y', labelsize=8)
    ax_val.grid(axis='y', linestyle='--', alpha=0.7)


# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide", page_title="Interactive Quantum Simulator")
st.title("⚛️ Interactive Quantum Circuit Simulator")

# --- Gate Definitions for UI ---
GATE_OPTIONS = {
    "I": {"matrix_or_func": I_GATE, "n_qubits_gate": 1, "params": []},
    "H": {"matrix_or_func": H_GATE, "n_qubits_gate": 1, "params": []},
    "X": {"matrix_or_func": X_GATE, "n_qubits_gate": 1, "params": []},
    "Y": {"matrix_or_func": Y_GATE, "n_qubits_gate": 1, "params": []},
    "Z": {"matrix_or_func": Z_GATE, "n_qubits_gate": 1, "params": []},
    "S": {"matrix_or_func": S_GATE, "n_qubits_gate": 1, "params": []},
    "T": {"matrix_or_func": T_GATE, "n_qubits_gate": 1, "params": []},
    "P(φ)": {"matrix_or_func": P_GATE, "n_qubits_gate": 1, "params": ["angle"]},
    "Rx(θ)": {"matrix_or_func": Rx_GATE, "n_qubits_gate": 1, "params": ["angle"]},
    "Ry(θ)": {"matrix_or_func": Ry_GATE, "n_qubits_gate": 1, "params": ["angle"]},
    "Rz(θ)": {"matrix_or_func": Rz_GATE, "n_qubits_gate": 1, "params": ["angle"]},
    # CNOT: matrix_or_func is for CNOT_01, matrix_10 for CNOT_10
    "CNOT": {"matrix_or_func": CNOT_01_GATE, "matrix_10": CNOT_10_GATE, "n_qubits_gate": 2, "params": ["control", "target"]},
    "SWAP": {"matrix_or_func": SWAP_GATE, "n_qubits_gate": 2, "params": []},
}

# --- Session State Management Functions (Define BEFORE init_session_state_vars) ---
def reset_quantum_state_and_circuit():
    st.session_state.circuit_ops = []
    current_n_qubits_local = st.session_state.get('n_qubits', 1)
    try:
        if current_n_qubits_local == 1:
            st.session_state.current_psi = qubit_from_bloch_angles(
                np.deg2rad(st.session_state.get('theta_q0_deg', 0.0)), 
                np.deg2rad(st.session_state.get('phi_q0_deg', 0.0))
            )
        elif current_n_qubits_local == 2:
            psi_A = qubit_from_bloch_angles(np.deg2rad(st.session_state.get('theta_qA_deg', 0.0)), np.deg2rad(st.session_state.get('phi_qA_deg', 0.0)))
            psi_B = qubit_from_bloch_angles(np.deg2rad(st.session_state.get('theta_qB_deg', 0.0)), np.deg2rad(st.session_state.get('phi_qB_deg', 0.0)))
            if psi_A is not None and psi_B is not None:
                st.session_state.current_psi = np.kron(psi_A, psi_B)
            else: 
                 st.session_state.current_psi = np.kron(q_zero_sv, q_zero_sv) # Fallback
                 st.warning("Error initializing 2-qubit state components during reset; used default |00>.")
        else:
            st.error(f"Unsupported n_qubits for reset: {current_n_qubits_local}"); 
            st.session_state.current_psi = q_zero_sv # Minimal safe default for 1 qubit
        st.session_state.last_n_qubits_init = current_n_qubits_local
    except Exception as e_reset:
        st.error(f"Reset Error: type={type(e_reset)}, val='{e_reset}', args={e_reset.args}"); 
        fallback_n_q = st.session_state.get('n_qubits', 1)
        st.session_state.current_psi = q_zero_sv if fallback_n_q == 1 else np.kron(q_zero_sv, q_zero_sv)
        st.session_state.last_n_qubits_init = fallback_n_q


def init_session_state_vars():
    defaults = {
        'n_qubits': 1, 'theta_q0_deg': 0.0, 'phi_q0_deg': 0.0,
        'theta_qA_deg': 0.0, 'phi_qA_deg': 0.0, 'theta_qB_deg': 0.0, 'phi_qB_deg': 0.0,
        'circuit_ops': [], 'current_psi': None, 'gate_angle_default_deg': 0.0, 'last_n_qubits_init': 0
    }
    # Ensure n_qubits is int before using it in conditions
    if 'n_qubits' in st.session_state and not isinstance(st.session_state.n_qubits, int):
        st.session_state.n_qubits = defaults['n_qubits'] # Force to default int if type is wrong

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    n_q_sess = st.session_state.get('n_qubits', defaults['n_qubits'])
    last_n_q_init_sess = st.session_state.get('last_n_qubits_init', defaults['last_n_qubits_init'])

    if st.session_state.get('current_psi') is None or last_n_q_init_sess != n_q_sess:
        reset_quantum_state_and_circuit()

# --- Initialize Session State ---
init_session_state_vars()


# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ System Setup")
    
    n_qubits_choice_sidebar = st.radio("Number of Qubits:", (1, 2), 
                                       index=(st.session_state.get('n_qubits', 1) - 1), 
                                       key="n_qubits_radio_sidebar")
    if n_qubits_choice_sidebar != st.session_state.get('n_qubits'):
        st.session_state.n_qubits = n_qubits_choice_sidebar
        reset_quantum_state_and_circuit()
        st.rerun()

    st.subheader("Initial State Definition")
    current_n_qubits_for_ui = st.session_state.get('n_qubits',1) # Safe access for UI conditional
    if current_n_qubits_for_ui == 1:
        st.markdown("**Qubit 0 ($|\psi_0\rangle$):**")
        st.session_state.theta_q0_deg = st.slider("Theta (θ) (degrees)", 0.0, 180.0, float(st.session_state.get('theta_q0_deg',0.0)), 1.0, key="theta_q0_slider_sidebar")
        st.session_state.phi_q0_deg = st.slider("Phi (φ) (degrees)", 0.0, 359.9, float(st.session_state.get('phi_q0_deg',0.0)), 1.0, key="phi_q0_slider_sidebar")
    else: 
        st.markdown("**Qubit A (Q0 - $|\psi_A\rangle$):**")
        st.session_state.theta_qA_deg = st.slider("Theta (θ) Q_A", 0.0, 180.0, float(st.session_state.get('theta_qA_deg',0.0)), 1.0, key="theta_qA_slider_sidebar")
        st.session_state.phi_qA_deg = st.slider("Phi (φ) Q_A", 0.0, 359.9, float(st.session_state.get('phi_qA_deg',0.0)), 1.0, key="phi_qA_slider_sidebar")
        st.markdown("**Qubit B (Q1 - $|\psi_B\rangle$):**")
        st.session_state.theta_qB_deg = st.slider("Theta (θ) Q_B", 0.0, 180.0, float(st.session_state.get('theta_qB_deg',0.0)), 1.0, key="theta_qB_slider_sidebar")
        st.session_state.phi_qB_deg = st.slider("Phi (φ) Q_B", 0.0, 359.9, float(st.session_state.get('phi_qB_deg',0.0)), 1.0, key="phi_qB_slider_sidebar")

    if st.button("🔄 Set & Reset State", use_container_width=True, key="reset_button_sidebar"):
        reset_quantum_state_and_circuit()
        st.rerun()

    st.divider()
    st.subheader("🛠️ Add Gate to Circuit")
    
    gate_filter_sidebar = lambda name, info: info["n_qubits_gate"] <= st.session_state.get('n_qubits',1)
    if st.session_state.get('n_qubits') == 1: gate_filter_sidebar = lambda name, info: info["n_qubits_gate"] == 1
    
    available_gate_names_sidebar = [name for name, info in GATE_OPTIONS.items() if gate_filter_sidebar(name, info)]
    selected_gate_name_sidebar = st.selectbox("Select Gate:", available_gate_names_sidebar, key="gate_selector_sidebar")
    
    gate_info_sidebar = GATE_OPTIONS.get(selected_gate_name_sidebar)
    target_qubit_indices_for_op_sidebar = []
    gate_angle_rad_for_op_sidebar = None
    base_matrix_or_func_for_op_sidebar = None 
    display_gate_name_for_circuit_sidebar = selected_gate_name_sidebar

    if gate_info_sidebar:
        if gate_info_sidebar["n_qubits_gate"] == 1:
            if st.session_state.get('n_qubits') == 1: target_qubit_indices_for_op_sidebar = [0]
            else: 
                target_idx_single_sidebar = st.radio("Target Qubit:", (0, 1), format_func=lambda x: f"Q{x}", key=f"target_q_radio_sidebar_{selected_gate_name_sidebar}_{st.session_state.get('n_qubits')}")
                target_qubit_indices_for_op_sidebar = [target_idx_single_sidebar]
                display_gate_name_for_circuit_sidebar += f"(Q{target_idx_single_sidebar})"
            
            if "angle" in gate_info_sidebar["params"]:
                angle_deg_input_sidebar = st.slider(f"Angle {selected_gate_name_sidebar} (deg)", -360.0, 360.0, float(st.session_state.get('gate_angle_default_deg',0.0)), 1.0, key=f"angle_slider_sidebar_{selected_gate_name_sidebar}")
                gate_angle_rad_for_op_sidebar = np.deg2rad(angle_deg_input_sidebar)
                base_matrix_or_func_for_op_sidebar = gate_info_sidebar["matrix_or_func"] 
                display_gate_name_for_circuit_sidebar += f"({angle_deg_input_sidebar:.0f}°)"
            else: 
                base_matrix_or_func_for_op_sidebar = gate_info_sidebar["matrix_or_func"]

        elif gate_info_sidebar["n_qubits_gate"] == 2: 
            if selected_gate_name_sidebar == "CNOT":
                control_idx_cnot_sidebar = st.radio("Control Qubit:", (0,1), format_func=lambda x: f"Q{x}", key="cnot_control_sidebar")
                target_idx_cnot_sidebar = 1 - control_idx_cnot_sidebar
                st.markdown(f"Target Qubit (auto): Q{target_idx_cnot_sidebar}")
                target_qubit_indices_for_op_sidebar = [control_idx_cnot_sidebar, target_idx_cnot_sidebar]
                # CORRECTED CNOT MATRIX SELECTION:
                if control_idx_cnot_sidebar == 0: base_matrix_or_func_for_op_sidebar = GATE_OPTIONS["CNOT"]["matrix_or_func"] # This is CNOT_01
                else: base_matrix_or_func_for_op_sidebar = GATE_OPTIONS["CNOT"]["matrix_10"] # This is CNOT_10
                display_gate_name_for_circuit_sidebar = f"CNOT(Q{control_idx_cnot_sidebar}→Q{target_idx_cnot_sidebar})"
            elif selected_gate_name_sidebar == "SWAP":
                target_qubit_indices_for_op_sidebar = [0,1] 
                base_matrix_or_func_for_op_sidebar = GATE_OPTIONS["SWAP"]["matrix_or_func"]
                display_gate_name_for_circuit_sidebar = "SWAP(Q0,Q1)"

    if st.button("➕ Add Gate to Circuit", use_container_width=True, disabled=(not gate_info_sidebar or base_matrix_or_func_for_op_sidebar is None), key="add_gate_button_sidebar"):
        if base_matrix_or_func_for_op_sidebar is not None and (target_qubit_indices_for_op_sidebar is not None and len(target_qubit_indices_for_op_sidebar) > 0):
            st.session_state.circuit_ops.append({
                'name': selected_gate_name_sidebar, 'matrix_or_func': base_matrix_or_func_for_op_sidebar, 
                'targets': target_qubit_indices_for_op_sidebar, 'angle_rad': gate_angle_rad_for_op_sidebar, 
                'display_text': display_gate_name_for_circuit_sidebar
            })
            if "angle" in gate_info_sidebar.get("params", []) and angle_deg_input_sidebar is not None: 
                 st.session_state.gate_angle_default_deg = angle_deg_input_sidebar # Store last used angle
            st.success(f"Added: {display_gate_name_for_circuit_sidebar}"); st.rerun()
        else: st.error("Cannot add gate. Parameters (e.g., target qubits) may be incomplete or gate matrix is undefined.")

    if st.session_state.get('circuit_ops', []):
        st.divider()
        if st.button("🗑️ Clear Full Circuit", use_container_width=True, type="primary", key="clear_circuit_button_sidebar"):
            reset_quantum_state_and_circuit(); st.rerun()
        if st.button("⬅️ Remove Last Gate", use_container_width=True, key="remove_last_gate_button_sidebar"):
            if st.session_state.circuit_ops: st.session_state.circuit_ops.pop(); st.rerun()


# --- Main Display Area ---
st.header("📊 Quantum State & Visualizations")
main_calc_error_flag = False
current_display_psi_main = None
initial_psi_for_display_circuit_main = None

try:
    n_qubits_main = st.session_state.get('n_qubits', 1)
    if n_qubits_main == 1:
        initial_psi_for_display_circuit_main = qubit_from_bloch_angles(
            np.deg2rad(st.session_state.get('theta_q0_deg',0.0)), 
            np.deg2rad(st.session_state.get('phi_q0_deg',0.0)))
    elif n_qubits_main == 2:
        psi_A_init_main = qubit_from_bloch_angles(np.deg2rad(st.session_state.get('theta_qA_deg',0.0)), np.deg2rad(st.session_state.get('phi_qA_deg',0.0)))
        psi_B_init_main = qubit_from_bloch_angles(np.deg2rad(st.session_state.get('theta_qB_deg',0.0)), np.deg2rad(st.session_state.get('phi_qB_deg',0.0)))
        if psi_A_init_main is None or psi_B_init_main is None: raise ValueError("Initial component qubit state is None.")
        initial_psi_for_display_circuit_main = np.kron(psi_A_init_main, psi_B_init_main)
    else: raise ValueError(f"Unsupported n_qubits in main display: {n_qubits_main}")
    
    current_display_psi_main = apply_circuit_ops(initial_psi_for_display_circuit_main, st.session_state.get('circuit_ops', []))

except Exception as e_main_calc:
    main_calc_error_flag = True
    st.error(f"State Calculation Error: type={type(e_main_calc)}, val='{e_main_calc}', args={e_main_calc.args}")
    current_display_psi_main = st.session_state.get('current_psi') 
    if current_display_psi_main is None: 
        current_display_psi_main = q_zero_sv if st.session_state.get('n_qubits',1) == 1 else np.kron(q_zero_sv,q_zero_sv)

# Update central session state if calculation was successful (or use fallback)
st.session_state.current_psi = current_display_psi_main


# Visualizations
vis_cols_main = st.columns([2, 3]) 
with vis_cols_main[0]:
    n_q_vis = st.session_state.get('n_qubits',1)
    psi_vis = st.session_state.get('current_psi') 

    if n_q_vis == 1:
        if psi_vis is not None and len(psi_vis) == 2:
            st.subheader("Qubit 0 Bloch Sphere")
            theta_vis, phi_vis = get_bloch_angles_from_state_vector(psi_vis)
            x_vis, y_vis, z_vis = (np.sin(theta_vis)*np.cos(phi_vis), np.sin(theta_vis)*np.sin(phi_vis), np.cos(theta_vis))
            fig_q0_vis = plt.figure(figsize=(4,4)) # Create new figure
            ax_q0_vis = fig_q0_vis.add_subplot(111, projection='3d') # Explicitly 3D
            plot_bloch_sphere_streamlit(ax_q0_vis, (x_vis,y_vis,z_vis), "Qubit 0")
            st.pyplot(fig_q0_vis, clear_figure=True)
    elif n_q_vis == 2:
        if psi_vis is not None and len(psi_vis) == 4:
            rho_AB_vis = state_vector_to_density_matrix(psi_vis)
            bloch_A_vis = density_matrix_to_bloch_vector(get_reduced_density_matrix_A(rho_AB_vis))
            bloch_B_vis = density_matrix_to_bloch_vector(get_reduced_density_matrix_B(rho_AB_vis))
            
            sc_A, sc_B = st.columns(2)
            with sc_A: 
                st.subheader("Qubit A (Q0)")
                fig_A_vis = plt.figure(figsize=(3.5,3.5)) # Create new figure
                ax_A_vis = fig_A_vis.add_subplot(111, projection='3d') # Explicitly 3D
                plot_bloch_sphere_streamlit(ax_A_vis,bloch_A_vis,"Reduced State")
                st.pyplot(fig_A_vis,clear_figure=True)
            with sc_B: 
                st.subheader("Qubit B (Q1)")
                fig_B_vis = plt.figure(figsize=(3.5,3.5)) # Create new figure
                ax_B_vis = fig_B_vis.add_subplot(111, projection='3d') # Explicitly 3D
                plot_bloch_sphere_streamlit(ax_B_vis,bloch_B_vis,"Reduced State")
                st.pyplot(fig_B_vis,clear_figure=True)

with vis_cols_main[1]:
    n_q_prob_vis = st.session_state.get('n_qubits',1)
    psi_prob_vis = st.session_state.get('current_psi')
    st.subheader("Measurement Probabilities")
    if psi_prob_vis is not None:
        probs_vis = get_measurement_probabilities(psi_prob_vis)
        fig_pr_vis, ax_pr_vis = plt.subplots(figsize=(5,4)); 
        plot_probabilities_streamlit(ax_pr_vis, probs_vis, n_q_prob_vis)
        st.pyplot(fig_pr_vis, clear_figure=True)

    st.subheader(f"State Vector $|\psi\rangle$ ({n_q_prob_vis}-Qubit)")
    if psi_prob_vis is not None:
        state_vector_terms_disp = []
        actual_n_qubits_for_psi = 0
        if len(psi_prob_vis) > 0 : actual_n_qubits_for_psi = int(np.log2(len(psi_prob_vis)))

        basis_labels_disp = []
        if actual_n_qubits_for_psi > 0:
            basis_labels_disp = ["|"+format(i,f'0{actual_n_qubits_for_psi}b')+"⟩" for i in range(2**actual_n_qubits_for_psi)]
        
        if len(basis_labels_disp) == len(psi_prob_vis):
            for i, amp_val in enumerate(psi_prob_vis):
                if not np.isclose(amp_val, 0.0, atol=1e-7):
                    real_part, imag_part = amp_val.real, amp_val.imag
                    term_str = ""
                    if not np.isclose(real_part, 0.0, atol=1e-7) or \
                       (np.isclose(real_part, 0.0, atol=1e-7) and np.isclose(imag_part, 0.0, atol=1e-7)):
                        term_str += f"{real_part:.3f}"
                    if not np.isclose(imag_part, 0.0, atol=1e-7):
                        if term_str and imag_part > 0 and not (np.isclose(real_part, 0.0, atol=1e-7) and term_str == f"{real_part:.3f}"): term_str += "+" 
                        term_str += f"{imag_part:.3f}i"
                    if not term_str: term_str = "0.000"
                    state_vector_terms_disp.append(f"({term_str}) {basis_labels_disp[i]}")
            final_state_str_disp = " + ".join(state_vector_terms_disp) if state_vector_terms_disp else "(Zero Vector)"
            final_state_str_disp = final_state_str_disp.replace(" + (-", " - (").replace("(+", "(").replace("()","(0.000)").replace("(0.000)", "0.000")
            st.code(final_state_str_disp, language=None)
        else: st.code(str(psi_prob_vis)) 
    else: st.warning("State vector not available.")


st.divider()
st.subheader("📜 Current Circuit Definition")
if not st.session_state.get('circuit_ops', []): st.markdown("_Circuit is empty._")
else:
    for i, op_item_vis in enumerate(st.session_state.get('circuit_ops',[])):
        st.markdown(f"**{i+1}.** {op_item_vis['display_text']}")

st.caption("Amplitudes rounded. Reduced states use density matrices. Pure 1-qubit Bloch vector from angles.")
st.caption("Made by Raghav Sharma (2025) - [GitHub](https://github.com/RaghavS06)")