from typing import List

from sklearn.utils import shuffle
from cliffford import sample_clifford
import cirq
import numpy as np
import quaff


class CliffordShadow:

    def __init__(self, state: cirq.Circuit, cliffords: List[cirq.Circuit],
                 result_obj) -> None:
        """
        :param state: can be vector or circuit
        :param cliffords: List of circuits
        :param result_obj: List of result obj from sampling the clifford channel
        """
        self.state = state
        self.cliffords = cliffords
        self.result_obj = result_obj
        self.n_qubits = len(self.state.all_qubits())

    def reconstruct_shadows(self,) -> np.ndarray:
        """
        reconstruct clifford state
        """
        rhos = []
        zero_state = np.zeros((2**self.n_qubits, 1), )
        eye = np.eye(2**self.n_qubits)
        for clifford_circuit, measurements in zip(self.cliffords, self.result_obj):
            u = cirq.unitary(clifford_circuit)
            df = measurements[0].data
            for bitstring_as_int in df['all_qubits']:
                b_ket = zero_state.copy()
                b_ket[bitstring_as_int, 0] = 1
                rho_i = (2**self.n_qubits + 1) * \
                    (u @ b_ket @ b_ket.T @ u.conj().T) - eye
                rhos.append(rho_i)
        self.rhos = rhos

    def median_of_means(self, operator, partitions=30) -> float:
        """
        Construct median-of-means estimator
        """
        self.rhos = shuffle(self.rhos)
        rhos_per_partition = len(self.rhos) // partitions
        expectations_to_take_median = []
        for kk in range(0, len(self.rhos), rhos_per_partition):
            rhos = self.rhos[kk:kk + rhos_per_partition]
            exp_rho_k = [np.trace(rho_xx @ operator) for rho_xx in rhos]
            expectations_to_take_median.append(exp_rho_k)
        return np.median(expectations_to_take_median)


def shadow_tomo_on_state_sim(state: cirq.Circuit,
                             num_cliffords: int,
                             num_measurement_per_clifford: int,
                             rng=None,
                             sampler=None) -> CliffordShadow:
    if rng is None:
        rng = np.random.default_rng()

    if sampler is None:
        sampler = cirq.Simulator(dtype=np.complex128)

    qubits = sorted(state.all_qubits())

    cliffords = [sample_clifford(qubits, rng) for _ in range(num_cliffords)]

    qubit_partition = [qubits]  # List of list where each sublist is a subset of the qubits

    base_circuit = cirq.Circuit()
    parameterized_clifford_circuit = cirq.Circuit(
        quaff.get_parameterized_truncated_cliffords_ops(qubit_partition))

    compiled_circuit = cirq.Circuit(
        [base_circuit, parameterized_clifford_circuit])
        
    truncated_cliffords = [[
        quaff.TruncatedCliffordGate.random(len(qubits_part), rng)
        for qubits_part in qubit_partition 
    ]
        for _ in range(num_cliffords)]

    resolvers = [
        quaff.get_truncated_cliffords_resolver(gates)
        for gates in truncated_cliffords
    ]

    truncated_cliffords = [[
        cirq.Circuit(clifford.on(*part))
        for clifford, part in zip(clifford_set, qubit_partition)
    ]
        for clifford_set in truncated_cliffords]

    for tc in truncated_cliffords:
        print(tc.to_text_diagram(transpose=True))
        print()
    exit()

    circuits_to_measure = []
    for clifford_circuit in cliffords:
        new_trial = state.copy()
        new_trial.append(clifford_circuit)
        new_trial.append(cirq.measure(*qubits, key='all_qubits'))
        circuits_to_measure.append(new_trial)

    # make this a trully multicore distributed simulation
    results = sampler.run_batch(circuits_to_measure, repetitions=[
                                num_measurement_per_clifford] * num_cliffords)

    shadow_inst = CliffordShadow(state, cliffords, results)
    return shadow_inst


if __name__ == "__main__":
    from cliffford import sample_clifford
    from ising_model import OneDimensionalIsingModel, CarleoTrialWF
    import cirq
    import openfermion as of
    ising_inst = OneDimensionalIsingModel(4, -1, -1)
    qop_ham = ising_inst.get_qubit_operator()
    qop_ham_sparse = of.get_sparse_operator(qop_ham)

    psi_t_inst = CarleoTrialWF(ising_inst)
    psi_t_vec = psi_t_inst.get_psi_t_vec()
    psi_t_vec /= np.linalg.norm(psi_t_vec)

    qubits = cirq.LineQubit.range(ising_inst.num_sites)
    state_prep_gate = cirq.ops.StatePreparationChannel(
        psi_t_vec.flatten(),
        name='stateprep'
    )
    trial_wf = cirq.Circuit(
        [state_prep_gate.on(*qubits)]
    )

    shadow_inst = shadow_tomo_on_state_sim(
        trial_wf, num_cliffords=15_000, num_measurement_per_clifford=1)

    shadow_inst.reconstruct_shadows()

    print("Exact calc trial wf")
    print(psi_t_vec.T @ qop_ham_sparse @ psi_t_vec)
    print("Shadow density matrix calculation")
    expectations = [np.trace(xx @ qop_ham_sparse) for xx in shadow_inst.rhos]
    print(np.mean(expectations))
