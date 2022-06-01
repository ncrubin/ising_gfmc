from ising_model import OneDimensionalIsingModel, CarleoTrialWF
import openfermion as of
import numpy as np


def test_ising_ham_and_trail_wf():
    ising_inst = OneDimensionalIsingModel(6, -1, -1)
    qop_ham = ising_inst.get_qubit_operator()
    print(qop_ham)

    qop_ham_sparse = of.get_sparse_operator(qop_ham)
    w, v = np.linalg.eigh(qop_ham_sparse.toarray())
    print(w)
    for xx in range(2**ising_inst.num_sites):
        if not np.isclose(v[xx, 0], 0):
            print(np.binary_repr(xx, width=ising_inst.num_sites), v[xx, 0])

    psi_t_inst = CarleoTrialWF(ising_inst)
    psi_t_vec = psi_t_inst.get_psi_t_vec()

    psi_t_vec /= np.linalg.norm(psi_t_vec)
    psi_t_energy = (psi_t_vec.conj().T @ qop_ham_sparse @ psi_t_vec)[0, 0].real
    # In the Carleo paper they assert that the trial wavefunction is within 1% error of the true ground state
    print((np.abs((psi_t_energy - w[0]) / w[0])) * 100)

