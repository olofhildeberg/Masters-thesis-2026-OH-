from qiskit_ibm_runtime import QiskitRuntimeService
#Make sure you are logged into IBM and have your API key!!

service = QiskitRuntimeService()

backend = service.backend("ibm_fez") #Can also use ibm_kingstong or ibm_marrakesh or ibm_fez
props = backend.properties()

print(f"Backend: {backend.name}")

for q in range(backend.num_qubits):
    try:
        ro_len = props.readout_length(q)
        print(f"Qubit {q}: {ro_len * 1e6:.3f} µs")
    except Exception:
        pass