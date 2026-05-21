# GHZ Preparation Protocol Simulations

This repository contains the simulation and plotting code used for comparing different GHZ-state preparation protocols under several isolated noise models. The protocols studied are non-adaptive, semi-adaptive, and 
fully adaptive GHZ preparation circuits.

The simulations are used to compare how the protocols perform under different error sources, with the final GHZ fidelity used as the main performance measure.

## Repository contents

The repository contains six Python files:

### Protocol-specific simulation files

- `Non_Adaptive_Sweeps.py`  
  Simulates the non-adaptive GHZ preparation protocol and plots the results separately, combining T1 with Tphi, and measurement- and CX error for all isolated error regimes.

- `Semi_Adaptive_Sweeps.py`  
  Simulates the semi-adaptive GHZ preparation protocol and plots the results separately, combining T1 with Tphi, and measurement- and CX error for all isolated error regimes.

- `Full_Adaptive_Sweeps.py`  
  Simulates the fully adaptive GHZ preparation protocol and plots the results separately, combining T1 with Tphi, and measurement- and CX error for all isolated error regimes.

### Combined plotting files

- `Compare_All_Protocols_All_Sweeps.py`  
  Combines the results from all three protocols and plots them together for each of the four isolated error regimes.

- `Compare_All_Protocols_Tphi_Only.py`  
  Similar to the combined plotting script above, but only plots the pure dephasing case, corresponding to the variation of \(T_\phi\), for all protocols.

### IBM hardware information

- `measurementtime.py`  
  Extracts the readout times of qubits from IBM Quantum hardware backends. These values can be used when choosing realistic measurement-delay parameters for the simulations.

## Error regimes

The simulations consider four main error sources. These are implemented separately, not simultaneously. In each simulation, only one error source is active while the others are set to zero.

The four isolated error regimes are:

1. Relaxation during idle time, controlled by T_1
2. Pure dephasing during idle time, controlled by T_\phi
3. Two-qubit gate errors, applied to CX gates
4. Measurement errors, applied to the readout of ancilla qubits

This separation is used to study how strongly each individual error source affects the different GHZ preparation protocols.


## Output

The scripts generate plots showing the final GHZ fidelity as a function of the relevant noise parameter. Depending on the script, the plots either show one protocol at a time or compare all protocols in the same figure.

## Requirements

The code requires Python and common scientific-computing packages such as:

```bash
numpy
matplotlib
qiskit
qiskit-ibm-runtime
