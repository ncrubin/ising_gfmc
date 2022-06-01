from typing import List
import numpy as np
import cirq


def fix_anticommute_tableaux(tableaux, n_qubits, j):
    tableaux = np.copy(tableaux)
    anti_commute_count = 0

    for i in range(n_qubits):
        top_bits = tableaux[0, 2 * i:2 * i + 2]
        bottom_bits = tableaux[1, 2 * i:2 * i + 2]

        if (np.array_equal(top_bits, bottom_bits) |
                np.array_equal([0, 0], bottom_bits) | np.array_equal(top_bits, [0, 0])):
            continue

        anti_commute_count += 1

    # We want them to anti-commute. If they do already, we're done.
    if anti_commute_count % 2 == 1:
        return tableaux

    ind = (j // 2 * 2)
    top_bits = tableaux[0, ind:ind + 2]
    bottom_bits = tableaux[1, ind:ind + 2]

    if (np.array_equal(top_bits, [1, 0]) & np.array_equal(bottom_bits, [0, 0])):
        new_bottom_bits = np.asarray([1, 1], dtype=np.int8)
    if (np.array_equal(top_bits, [1, 0]) & np.array_equal(bottom_bits, [0, 1])):
        new_bottom_bits = np.asarray([1, 0], dtype=np.int8)

    if (np.array_equal(top_bits, [1, 1]) & np.array_equal(bottom_bits, [0, 0])):
        new_bottom_bits = np.asarray([1, 0], dtype=np.int8)
    if (np.array_equal(top_bits, [1, 1]) & np.array_equal(bottom_bits, [0, 1])):
        new_bottom_bits = np.asarray([1, 1], dtype=np.int8)

    if (np.array_equal(top_bits, [0, 1]) & np.array_equal(bottom_bits, [0, 0])):
        new_bottom_bits = np.asarray([1, 1], dtype=np.int8)
    if (np.array_equal(top_bits, [0, 1]) & np.array_equal(bottom_bits, [1, 0])):
        new_bottom_bits = np.asarray([0, 1], dtype=np.int8)

    tableaux[1, ind:ind + 2] = new_bottom_bits

    return tableaux


def binarize(integer, n_bits):
    return np.asarray(list(np.binary_repr(integer).zfill(n_bits))).astype(np.int8)


def x_ind(i):
    return 2 * i


def z_ind(i):
    return 2 * i + 1


def cnot_on(a, b, tableaux, qubits):
    tableaux = np.copy(tableaux)
    n_row = tableaux.shape[0]

    for i in range(n_row):
        x_ia = tableaux[i, x_ind(a)]
        x_ib = tableaux[i, x_ind(b)]
        z_ia = tableaux[i, z_ind(a)]
        z_ib = tableaux[i, z_ind(b)]
        r_i = tableaux[i, -1]

        new_r_i = (r_i + x_ia * z_ib * ((x_ib + z_ia + 1) % 2)) % 2
        new_x_ib = (x_ia + x_ib) % 2
        new_z_ia = (z_ia + z_ib) % 2

        tableaux[i, -1] = new_r_i
        tableaux[i, x_ind(b)] = new_x_ib
        tableaux[i, z_ind(a)] = new_z_ia

    return cirq.CNOT(qubits[a], qubits[b]), tableaux


def H_on(a, tableaux, qubits):
    tableaux = np.copy(tableaux)
    n_row = tableaux.shape[0]

    for i in range(n_row):
        x_ia = tableaux[i, x_ind(a)]
        z_ia = tableaux[i, z_ind(a)]
        r_i = tableaux[i, -1]

        new_r_i = (r_i + x_ia * z_ia) % 2
        new_x_ia = z_ia
        new_z_ia = x_ia

        tableaux[i, -1] = new_r_i
        tableaux[i, x_ind(a)] = new_x_ia
        tableaux[i, z_ind(a)] = new_z_ia

    return cirq.H(qubits[a]), tableaux


def S_on(a, tableaux, qubits):
    tableaux = np.copy(tableaux)
    n_row = tableaux.shape[0]

    for i in range(n_row):
        x_ia = tableaux[i, x_ind(a)]
        z_ia = tableaux[i, z_ind(a)]
        r_i = tableaux[i, -1]

        new_r_i = (r_i + x_ia * z_ia) % 2
        new_z_ia = (x_ia + z_ia) % 2

        tableaux[i, -1] = new_r_i
        tableaux[i, z_ind(a)] = new_z_ia

    return cirq.S(qubits[a]), tableaux


def sweep_tableaux_1(tableaux, n_qubits, qubits, row=0):
    tableaux = np.copy(tableaux)
    ops = []

    for j in range(n_qubits):
        z_aj = tableaux[row, z_ind(j)]
        x_aj = tableaux[row, x_ind(j)]

        if z_aj == 1:
            if x_aj == 0:
                op, tableaux = H_on(j, tableaux, qubits)
            else:
                op, tableaux = S_on(j, tableaux, qubits)

            ops.append(op)

    return ops, tableaux


def sweep_tableaux_2(tableaux, n_qubits, qubits, row=0):
    tableaux = np.copy(tableaux)
    ops = []

    J_set = []
    for j in range(n_qubits):
        x_aj = tableaux[row, x_ind(j)]
        if x_aj == 1:
            J_set.append(j)

    while len(J_set) > 1:

        for num, j in enumerate(J_set):
            if num % 2 == 1:
                target = j
                control = J_set[num - 1]

                op, tableaux = cnot_on(control, target, tableaux, qubits)

                ops.append(op)

        J_set = []
        for j in range(n_qubits):
            x_aj = tableaux[row, x_ind(j)]
            if x_aj == 1:
                J_set.append(j)

    return ops, tableaux


def sweep_tableaux_3(tableaux, n_qubits, qubits):
    tableaux = np.copy(tableaux)
    ops = []

    for j in range(n_qubits):
        x_aj = tableaux[0, x_ind(j)]
        if x_aj == 1:
            break

    for i in reversed(range(0, j)):
        # Swap
        op, tableaux = cnot_on(i + 1, i, tableaux, qubits)
        ops.append(op)
        op, tableaux = cnot_on(i, i + 1, tableaux, qubits)
        ops.append(op)
        op, tableaux = cnot_on(i + 1, i, tableaux, qubits)
        ops.append(op)

    return ops, tableaux


def sweep_tableaux_4(tableaux, n_qubits, qubits):
    tableaux = np.copy(tableaux)
    ops = []

    op, tableaux = H_on(0, tableaux, qubits)
    ops.append(op)

    ops_1, tableaux = sweep_tableaux_1(tableaux, n_qubits, qubits, row=1)

    ops_2, tableaux = sweep_tableaux_2(tableaux, n_qubits, qubits, row=1)

    ops += ops_1 + ops_2

    op, tableaux = H_on(0, tableaux, qubits)
    ops.append(op)

    return ops, tableaux


def sweep_tableaux_5(tableaux, n_qubits, qubits):
    tableaux = np.copy(tableaux)
    ops = []

    s_0 = tableaux[0, -1]
    s_1 = tableaux[1, -1]

    if s_0 == 1 and s_1 == 0:
        ops.append(cirq.Z(qubits[0]))
        tableaux[0, -1] = 0

    if s_1 == 1 and s_0 == 0:
        ops.append(cirq.X(qubits[0]))
        tableaux[1, -1] = 0

    if s_0 == 1 and s_1 == 1:
        ops.append(cirq.Y(qubits[0]))
        tableaux[0, -1] = 0
        tableaux[1, -1] = 0

    return ops, tableaux


def sweep_tableaux(tableaux, num_qubits, qubits):
    tableaux = np.copy(tableaux)

    ops_1, tableaux = sweep_tableaux_1(tableaux, num_qubits, qubits)
    ops_2, tableaux = sweep_tableaux_2(tableaux, num_qubits, qubits)
    ops_3, tableaux = sweep_tableaux_3(tableaux, num_qubits, qubits)
    ops_4, tableaux = sweep_tableaux_4(tableaux, num_qubits, qubits)
    ops_5, tableaux = sweep_tableaux_5(tableaux, num_qubits, qubits)

    return [ops_1, ops_2, ops_3, ops_4, ops_5], tableaux


def test_tableaux_ops(full_initial_tableaux, full_final_tableaux, ops, qubits):
    pass


def sample_clifford(qubits: List[cirq.Qid], rng: np.random.Generator) -> cirq.Circuit:
    """
    Sample a clifford circuit

    :param qubits List[cirq.Qid]: list of qubits
    :param rng np.random.Generator: Generator object with `integers' function
    """
    n_qubits = len(qubits)

    full_initial_tableaux = np.zeros((2 * n_qubits, 2 * n_qubits + 1),
                                     dtype=np.int8)
    full_final_tableaux = np.zeros((2 * n_qubits, 2 * n_qubits + 1),
                                   dtype=np.int8)
    full_ops = []

    for i in range(n_qubits, 0, -1):
        # Generating the first row.
        tableaux = np.zeros((2, 2 * i + 1), dtype=np.int8)
        rand_1 = rng.integers(2, 2**(2 * i + 1))
        rand_row_1 = binarize(rand_1, 2 * i + 1)
        tableaux[0, :] = rand_row_1

        # Generating the second row.
        for j in range(2 * i + 1):
            if tableaux[0, j]:
                break

        rand_row_2 = rng.integers(0, 2, size=2 * i + 1)
        rand_row_2[j] = 0
        tableaux[1, :] = rand_row_2

        # Making sure they anti-commute.
        tableaux = fix_anticommute_tableaux(tableaux, i, j)

        idx = (n_qubits - i) * 2
        full_initial_tableaux[idx:idx + 2, idx:] = tableaux
        ops, tableaux = sweep_tableaux(tableaux, i, qubits[n_qubits - i:])
        full_ops.append(ops)
        full_final_tableaux[idx:idx + 2, idx:] = tableaux

    test_tableaux_ops(full_initial_tableaux, full_final_tableaux, ops, qubits)

    circuit = cirq.Circuit(full_ops)

    return circuit
