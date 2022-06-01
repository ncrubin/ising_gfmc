from typing import List
import openfermion as of
import numpy as np
import cirq


class OneDimensionalIsingModel:
    
    def __init__(self, num_sites, j_zz_interaction=-1, gamma_x_interaction=-1) -> None:
        """
        H = -J\sum_{k=0}^{L-1}sigma_{k}^{Z}sigma_{(k+1)%L}^{Z} + -Gamma \sum_{k=0}^{L-1}sigma_{k}^{X}
        """
        self.num_sites = num_sites
        self.j = j_zz_interaction
        self.gamma = gamma_x_interaction

        self.qop_hamiltonian = None
        self.fermion_ham = None

    def get_qubit_operator(self,) -> of.QubitOperator:
        if self.qop_hamiltonian is not None:
            return self.qop_hamiltonian
        else:
            self.qop_hamiltonian = of.QubitOperator()
            for k in range(self.num_sites):
                self.qop_hamiltonian += self.j * of.QubitOperator(((k, 'Z'), ((k + 1) % self.num_sites, 'Z')))
                self.qop_hamiltonian += self.gamma * of.QubitOperator(((k, 'X')))
            return self.qop_hamiltonian
    
    def get_cirq_operator(self, qubits: List[cirq.Qid], ov_basis=False) -> cirq.PauliSum:
        """
        Construct the Hamiltonian as a PauliSum object

        :param qubits: list of qubits         
        :return: cirq.PauliSum representing the Hamiltonian
        """
        n_qubits = len(qubits)
        qubit_operator = self.get_qop_hamiltonian()
        qubit_map = dict(zip(range(n_qubits), qubits))
        cirq_pauli_terms = []
        for term, val in qubit_operator.terms.items():
            pauli_term_dict = dict([(qubit_map[xx], yy) for xx, yy in term])
            pauli_term = cirq.PauliString(pauli_term_dict, coefficient=val)
            cirq_pauli_terms.append(pauli_term)
        return cirq.PauliSum().from_pauli_strings(cirq_pauli_terms)


    def elocal_exact(self, x: int, psi_t_vec) -> float:
        """Compute local energy"""
        x_vec = np.zeros((2 ** self.num_sites, 1))
        eloc = x_vec.T @ self.qop_hamiltonian @ psi_t_vec  / x_vec.T @ self.qop_hamiltonian @ psi_t_vec 
        return eloc


class CarleoTrialWF:
    def __init__(self, ham_inst: OneDimensionalIsingModel, l_param=0.233, s_param=0.083) -> None:
        self.ham_inst = ham_inst
        self.l_param = l_param
        self.s_param = s_param
        self.L = self.ham_inst.num_sites
        self.psi_t_vec = None
    
    def psi_t(self, x: int) -> np.complex128:
        """
        psi_t construction input is a computational basis state and output is an amplitude

        s^z indicates the eigenvalue 1 or -1 for the bit value 0 or 1.

        psi_{T}(x) = exp(l1 sum_{k}s^z_{k}s^z_{k+1} + s1 sum_{k}s^z_{k}s^z_{k+2})
        """
        L = self.L
        ket = [(-1)**int(xx) for xx in np.binary_repr(x, width=L)]
        l1_term = self.l_param * sum([ket[xx] * ket[(xx + 1) % L] for xx in range(L)])
        s1_term = self.s_param * sum([ket[xx] * ket[(xx + 2) % L] for xx in range(L)])
        return np.exp(l1_term + s1_term)
    
    def get_psi_t_vec(self,) -> np.ndarray:
        """
        Get a vector representation of psi_t
        """
        if self.psi_t_vec is not None:
            return self.psi_t_vec
        L = self.L
        psi_t_vec = np.zeros((2**L, 1), dtype=np.complex128)
        for xx in range(2**L):
            psi_t_vec[xx, 0] = self.psi_t(xx)
        self.psi_t_vec = psi_t_vec
        return psi_t_vec

    def get_local_energy(self, x, amplitudes, sparse_hamiltonian):
        """
        construct the Clifford shadow representation of the state.
        """
        x_amp = amplitudes[x]
        ham_row = sparse_hamiltonian.getrow(x)
        numerator = np.sum(ham_row.data * amplitudes[ham_row.indices])
        return numerator / x_amp.real


if __name__ == "__main__":
    from collections import Counter
    import matplotlib.pyplot as plt
    ising_inst = OneDimensionalIsingModel(6, -1, -1)
    qop_ham = ising_inst.get_qubit_operator()
    print(qop_ham)
    sparse_qop_ham = of.get_sparse_operator(qop_ham)

    psi_t_inst = CarleoTrialWF(ising_inst)
    psi_t_vec = psi_t_inst.get_psi_t_vec()
    psi_t_vec /= np.linalg.norm(psi_t_vec)

    gs_e = (psi_t_vec.conj().T @ sparse_qop_ham @ psi_t_vec)[0, 0].real

    probs = abs(psi_t_vec)**2

    num_trials = 50
    num_samples = 10_000
    local_energy_values = []
    for _ in range(num_trials):
        bitstring_samples = np.random.choice(range(2**ising_inst.num_sites),
                                             size=num_samples, p=probs.flatten())
        freq_dist = Counter(bitstring_samples)
        sampled_amplitudes = np.zeros(2**ising_inst.num_sites)
        for key, value in freq_dist.items():
            print(key, value)
            sampled_amplitudes[key] = value / num_samples

        sampled_amplitudes = abs(np.sqrt(sampled_amplitudes))
        elocal = 0
        for xx in bitstring_samples:
            eloc_val = psi_t_inst.get_local_energy(xx, sampled_amplitudes,
                                                   sparse_qop_ham)
            elocal += eloc_val
        elocal /= num_samples

        local_energy_values.append(elocal / ising_inst.num_sites)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(local_energy_values, bins=20)
    plt.show()