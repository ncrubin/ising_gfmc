import numpy as np

import openfermion as of

import matplotlib.pyplot as plt

from collections import Counter

from ising_model import OneDimensionalIsingModel, CarleoTrialWF


if __name__ == "__main__":

    ising_inst = OneDimensionalIsingModel(14, -1, -1)
    qop_ham = ising_inst.get_qubit_operator()
    print(qop_ham)
    sparse_qop_ham = of.get_sparse_operator(qop_ham)

    psi_t_inst = CarleoTrialWF(ising_inst)
    psi_t_vec = psi_t_inst.get_psi_t_vec()
    psi_t_vec /= np.linalg.norm(psi_t_vec)

    gs_e = (psi_t_vec.conj().T @ sparse_qop_ham @ psi_t_vec)[0, 0].real

    probs = abs(psi_t_vec)**2
    bitstrings_as_ints = list(range(2**ising_inst.num_sites))
    bitstring_idx_sorted_by_prob = np.argsort(probs.flatten())[::-1]
    print(bitstring_idx_sorted_by_prob)
    elocal_by_canonical_bitstring = []
    for xx in range(2**ising_inst.num_sites):
        eloc_val = psi_t_inst.get_local_energy(xx, abs(psi_t_vec.flatten()), sparse_qop_ham)
        elocal_by_canonical_bitstring.append(eloc_val)

    elocal_by_canonical_bitstring = np.array(elocal_by_canonical_bitstring) / ising_inst.num_sites
    elocal_by_prob_order = elocal_by_canonical_bitstring.flatten()[bitstring_idx_sorted_by_prob]


    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].set_title("L = {}".format(ising_inst.num_sites))
    ax[0].plot(range(2**ising_inst.num_sites), elocal_by_prob_order, color='k', linestyle='--', label='exact')
    # ax[0].set_ylim([-2.5, 0])
    ax[0].axhline(gs_e / ising_inst.num_sites, color='green')
    ax[0].set_ylabel("local energy per site", fontsize=12)

    ax[1].plot(range(2**ising_inst.num_sites), psi_t_vec.flatten()[bitstring_idx_sorted_by_prob],
               color='k', marker='None', linestyle='--', markersize=5)
    # ax[1].set_ylim([0, 0.6])
    ax[1].set_ylabel("Overlap", fontsize=12)


    # get M=640 for L=8
    num_samples = 10 * 2**ising_inst.num_sites
    for trial_num in range(16):
        bitstrings = np.random.choice(range(2**ising_inst.num_sites), size=num_samples, p=probs.flatten())
        freq_dist = Counter(bitstrings)
        sampled_amplitudes = np.zeros(2**ising_inst.num_sites)
        for key, value in freq_dist.items():
            print(key, value)
            sampled_amplitudes[key] = value / num_samples

        sampled_amplitudes = abs(np.sqrt(sampled_amplitudes))
        sampled_eloc = []
        sampled_ovlp = []
        for xx in range(2**ising_inst.num_sites):
            eloc_val = psi_t_inst.get_local_energy(xx, sampled_amplitudes, sparse_qop_ham)
            sampled_eloc.append(eloc_val)
            sampled_ovlp.append(sampled_amplitudes[xx])
        sampled_eloc = np.array(sampled_eloc)[bitstring_idx_sorted_by_prob] / ising_inst.num_sites
        sampled_ovlp = np.array(sampled_ovlp)[bitstring_idx_sorted_by_prob]

        if trial_num == 0:
            ax[0].plot(range(2**ising_inst.num_sites), sampled_eloc, marker='o', markersize=3, color='g', alpha=0.6,
                       linestyle='None', label='M={}'.format(num_samples))

        else:
            ax[0].plot(range(2**ising_inst.num_sites), sampled_eloc, marker='o', markersize=3, color='g', alpha=0.6,
                       linestyle='None', )
        ax[1].plot(range(2 ** ising_inst.num_sites), sampled_ovlp, marker='o',
                   markersize=3, color='g', alpha=0.6,
                   linestyle='None')

    num_samples = 1000 * 2**ising_inst.num_sites
    for trial_num in range(16):
        bitstrings = np.random.choice(range(2 ** ising_inst.num_sites),
                                      size=num_samples, p=probs.flatten())
        freq_dist = Counter(bitstrings)
        sampled_amplitudes = np.zeros(2 ** ising_inst.num_sites)
        for key, value in freq_dist.items():
            print(key, value)
            sampled_amplitudes[key] = value / num_samples

        sampled_amplitudes = abs(np.sqrt(sampled_amplitudes))
        sampled_eloc = []
        sampled_ovlp = []
        for xx in range(2 ** ising_inst.num_sites):
            eloc_val = psi_t_inst.get_local_energy(xx, sampled_amplitudes,
                                                   sparse_qop_ham)
            sampled_eloc.append(eloc_val)
            sampled_ovlp.append(sampled_amplitudes[xx])
        sampled_eloc = np.array(sampled_eloc)[
                           bitstring_idx_sorted_by_prob] / ising_inst.num_sites
        sampled_ovlp = np.array(sampled_ovlp)[bitstring_idx_sorted_by_prob]

        if trial_num == 0:
            ax[0].plot(range(2 ** ising_inst.num_sites), sampled_eloc,
                       marker='o', markersize=3, color='b', alpha=0.6,
                       linestyle='None', label='M={}'.format(num_samples))

        else:
            ax[0].plot(range(2 ** ising_inst.num_sites), sampled_eloc,
                       marker='o', markersize=3, color='b', alpha=0.6,
                       linestyle='None', )
        ax[1].plot(range(2 ** ising_inst.num_sites), sampled_ovlp, marker='o',
                   markersize=3, color='b', alpha=0.6,
                   linestyle='None')

    ax[0].legend(loc='upper left', frameon=False)
    plt.savefig("carleo_fig1_l{}.png".format(ising_inst.num_sites), format='PNG', dpi=300)
    plt.savefig("carleo_fig1_l{}.pdf".format(ising_inst.num_sites), format='PDF', dpi=300)
    plt.show()
