# ⚛ Interactive Quantum Circuit Simulator ⚛

This is an interactive Streamlit application designed for hands-on learning of fundamental quantum mechanics and quantum computing concepts. You can define initial qubit states, build quantum circuits by applying various gates, and visualize the resulting quantum states and measurement probabilities!

<p align="center">
  <img src="resources/app screen.png" width="700">
</p>

## Features

* **Qubit System:** Simulate 1-qubit or 2-qubit systems.
* **Initial State Definition:** Interactively set initial qubit states using Bloch sphere angles ($\theta, \phi$).
* **Interactive Circuit Building:**
    * Select and apply a variety of common quantum gates (H, X, Y, Z, S, T, P(φ), Rx(θ), Ry(θ), Rz(θ)).
    * For 2-qubit systems, apply CNOT and SWAP gates with selectable control/target qubits.
    * Specify angles for rotation gates.
    * Build circuits step-by-step.
* **Visualizations:**
    * **Bloch Sphere:** For single qubit states, and for the reduced density matrix of individual qubits in a 2-qubit system (demonstrates mixed states for entangled qubits).
    * **Probability Bar Chart:** Shows the measurement probabilities for the basis states of the system.
* **State Display:** View the complex state vector of the quantum system.
* **Error Handling:** Basic error checks for invalid operations.

# Prerequisites (IMPORTANT ‼️)

* Proper installation of: **Python 3.13+**
* **Git** (for cloning the repository)
* Proper installation of: **pip 25.0+**

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RaghavS06/interactive-quantum-simulator.git
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd interactive-quantum-simulator
    ```

3.  **Create and activate a virtual environment (recommended):**
    * To create:
        ```bash
        python -m venv venv
        ```
    * To activate:
        * On Windows:
            ```bash
            venv\Scripts\activate
            ```
        * On macOS/Linux:
            ```bash
            source venv/bin/activate
            ```

4.  **Install dependencies:**
    Make sure you have a `requirements.txt` file in your project directory with the following content:
    ```txt
    streamlit==1.45.1
    numpy
    matplotlib
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  Ensure your virtual environment is activated (if you created one).
2.  Run the Streamlit application using the command:
    ```bash
    streamlit run interactiveQuantumSimulation.py
    ```

3.  The application will automatically open in your default web browser.

## How to Use

1.  **System Setup (Sidebar):**
    * Choose the number of qubits (1 or 2).
    * Define the initial state(s) using the Theta (θ) and Phi (φ) sliders. Click "Set & Reset State" to apply these initial conditions.
2.  **Add Gate to Circuit (Sidebar):**
    * Select a gate from the dropdown.
    * If applicable, choose the target qubit(s) and set any required angles.
    * Click "Add Gate to Circuit."
3.  **Circuit Management (Sidebar):**
    * "Clear Full Circuit" will remove all gates and reset the state to the defined initial state.
    * "Remove Last Gate" will undo the last gate addition.
4.  **Visualizations (Main Area):**
    * The Bloch sphere(s) and probability plots will update automatically as you define the initial state and add gates to the circuit.
    * The current state vector and the defined circuit are also displayed.

## Future Enhancements

* Support for more than 2 qubits (visualization will be a challenge).
* More advanced gates and multi-qubit controlled gates.
* Quantum circuit diagram visualization.
* Simulation of measurement and state collapse.
* Saving and loading circuit definitions.
