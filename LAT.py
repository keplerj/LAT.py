# "LAT.py" --  Linear Arrangement Toolbox module in Python
#
# Copyright (C) 2019 Georgios N Printezis
#
# This module is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.%
#
# Contact me on johakepl@gmail.com.

import numpy as np
import scipy
import scipy.sparse.linalg
from itertools import permutations
import networkx as nx

def array_to_graph_mtx(G, filename):
    n, nnz = len(G), int(np.count_nonzero(G) / 2)

    with open(filename, 'w') as f:
        f.write(str(n) + ' ' + str(n) + ' ' + str(nnz) + '\n')

        for i in range(n - 1):
            for j in range(i + 1, n):
                Gij = G[i, j]

                if Gij > 0
                    f.write(str(j + 1) + ' ' + str(i + 1) + ' ' + str(Gij) + '\n')

def array_to_sequence_mtx(sequence, cost, filename):
    n = len(sequence)

    with open(filename, 'w') as f:
        f.write(str(n) + ' ' + str(cost) + '\n')

        for i in range(n):
            f.write(str(sequence[i] + 1) + '\n')

def NOIs_sequence_to_mtx(nois_sequence, filename):
    n = len(nois_sequence)

    with open(filename, 'w') as f:
        f.write(str(n) + '\n')

        for i in range(n):
            f.write(str(nois_sequence[i]) + '\n')

def graph_mtx_to_array(graph_filename):
    with open(graph_filename) as f:
        m, n, nnz = [int(x) for x in next(f).split()]

        G = np.zeros((m, n))

        for line in f:
            ss, tt, ww = line.split()

            s, t, w = int(ss) - 1, int(tt) - 1, float(ww)

            G[s, t] = G[t, s] = w

    return G

def NOIs_mtx_to_array(filename):
    names_list = []

    with open(filename) as f:
        n = int(next(f))

        for i in range(n):
            names_list.append(next(f).rstrip())

    return np.array(names_list)

def sequence_mtx_to_array(sequence_filename):
    with open(sequence_filename) as f:
        mm, cc = next(f).split()

        m, c = int(mm), float(cc)

        sequence = np.zeros(m, dtype = int)

        for i in range(m):
            sequence[i] = int(next(f)) - 1

    return sequence

def LA_stable(G, s):
    n = len(s)

    ncosts = int((n ** 2 - n) / 2)

    costs = np.empty(ncosts, dtype = float)

    for i in range(n - 1):
        src = s[i]
        for j in range(i + 1, n):
            trgt = s[j]

            costs[i * n + j] = G[src, trgt] * (j - i)

    la = np.sum(np.sort(costs))

    return la

def LA(G, s, n = None):
    if n == None:
        n = len(s)

    la = 0

    for i in range(n - 1):
        src = s[i]
        for j in range(i + 1, n):
            trgt = s[j]

            la += G[src, trgt] * (j - i)

    return la

def normal_layout(G):
    n = len(G)

    sequence = np.arange(n)

    cost = LA(G, sequence)

    return sequence, cost

def successive_augmentation(G, initialSequence):
    n = len(G)

    sequence =  - np.ones(n, dtype = np.int)

    if n % 2 == 0:
        cend = 2

        mid1, mid2 = int(n / 2 - 1), int(n / 2)

        sequence[0], sequence[1] = initialSequence[mid1], initialSequence[mid2]
    else:
        cend = 1

        mid1 = mid2 = int(n / 2 + 1)

        sequence[0] = initialSequence[mid1]

    for i in range(mid1):
        cend = cend + 1

        print(cend)

        sequence[cend - 1] = initialSequence[mid1 - (i + 1)]

        minCost = LA(G, sequence, cend)

        pos = cend - 1

        for j in  range(cend - 1, 0, - 1):
            sequence[j], sequence[j - 1] = sequence[j - 1], sequence[j]

            cost = LA(G, sequence, cend)

            if cost < minCost:
                minCost, pos = cost, j - 1

        elem = sequence[0]

        sequence[0 : pos], sequence[pos] = sequence[1 : (pos + 1)], elem

        cend = cend + 1

        sequence[cend - 1] = initialSequence[mid2 + (i + 1)]

        minCost = LA(G, sequence, cend)

        pos = cend - 1;

        for j in range(cend - 1, 0, - 1):
            sequence[j], sequence[j - 1] = sequence[j - 1], sequence[j]

            cost = LA(G, sequence, cend)

            if cost < minCost:
                minCost, pos = cost, j - 1

        elem = sequence[0]

        sequence[0 : pos], sequence[pos] = sequence[1 : (pos + 1)], elem

    return sequence, minCost

def full_search(G, sequence):
    n = len(G)

    minCost = LA(G, sequence)

    z = cnt = 0

    while z < 1:
        z = z + 1

        print(cnt, ' ', minCost)

        sequence, newCost = SelectBestNeighbor(G, sequence)

        if newCost < minCost:
            z = 0

            cnt += 1

            minCost = newCost

    return sequence, minCost, cnt

def SelectBestNeighbor(G, sequence):
    n = len(G)

    minCost = LA(G, sequence)

    ii = jj = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            sequence[[i, j]] = sequence[[j, i]]

            newCost = LA(G, sequence)

            if newCost < minCost:
                minCost, ii, jj = newCost, i, j

            sequence[[i, j]] = sequence[[j, i]]

    sequence[ii], sequence[jj] = sequence[jj], sequence[ii]

    return sequence, minCost

def spectral_sequencing(G):
    laplacian = - G + np.diag(sum(G))

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(laplacian, k = 2, which = 'SM')

    sequence = np.argsort(eigenvectors[:, 1])

    cost = LA(G, sequence)

    return sequence, cost

def spectral_algorithm(G):
    n = len(G)

    laplacian = - G + np.diag(sum(G))

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(laplacian, k = 3, which = 'SM')

    x2, x3 = eigenvectors[:, 1], eigenvectors[:, 2]

    u = (x2[0] / (x2[0] ** 2 + x3[0] ** 2) ** .5) * x2 + (x3[0] / (x2[0] ** 2 + x3[0] ** 2) ** .5) * x3

    s = np.argsort(u)

    minCost = LA(G, s)

    best = 0

    for i in range(1, n):
        u = (x2[i] / (x2[i] ** 2 + x3[i] ** 2) ** .5) * x2 + (x3[i] / (x2[i] ** 2 + x3[i] ** 2) ** .5) * x3

        s = np.argsort(u)

        cost = LA(G, s)

        if cost < minCost:
            best, minCost = i, cost

    u_best = (x2[best] / (x2[best] ** 2 + x3[best] ** 2) ** .5) * x2 + (x3[best] / (x2[best] ** 2 + x3[best] ** 2) ** .5) * x3

    sequence = np.argsort(u_best)

    return sequence, minCost

def recursive_weighted_matching_sequencing(G):
    n = len(G)

    print(n)

    if n <= 8:
        sequence, minCost = minla_exact(G)

        return sequence, minCost

    outmate = maximum_matching(G)

    C, coarseNodes = coarsening(G, outmate)

    seqCoarse, minCost = recursive_weighted_matching_sequencing(C)

    print(n)

    sequence = uncoarsening(coarseNodes, seqCoarse, outmate)

    sequence, minCost = adjacent_refinement(G, sequence, outmate)

    return sequence, minCost

def minla_exact(G):
    n = len(G)

    perms = np.array(list(permutations(range(n))), dtype = int)

    best_perm = 0

    s = np.array(perms[0])

    minCost = LA(G, s)

    for i in range(1, len(perms)):
        s = perms[i]

        cost = LA(G, s)

        if cost < minCost:
            best_perm, minCost = i, cost

    sequence = perms[best_perm]

    return sequence, minCost

def maximum_matching(G):
    n = len(G)

    matchings = nx.max_weight_matching(nx.from_numpy_array(G))

    outmate = - np.ones(n, dtype = int)

    for src, trgt in matchings:
        outmate[src], outmate[trgt] = trgt, src

    return outmate

def coarsening(G, outmate):
    n = len(G)

    unmatchedNodes = (outmate == - 1)

    nodes = np.copy(outmate)

    nodes[unmatchedNodes] = find(unmatchedNodes)

    part = matching_partitions(nodes)

    m = max(part) + 1

    S = np.zeros((m, n))

    S[part, range(n)] = 1

    C = S @ G @ S.T

    for i in range(n):
        if nodes[i] == - 1:
            continue

        if nodes[nodes[i]] != nodes[i]:
            nodes[outmate[i]] = - 1

    coarseNodes = find(nodes > - 1)

    return C, coarseNodes

def find(bool_vec):
    return np.where(bool_vec)[0]

def matching_partitions(nodes):
    n = len(nodes)

    part =  - np.ones(n, dtype = int)

    j = 0

    for i in range(n):
        if part[i] == - 1:
            part[i] = part[nodes[i]] = j

            j += 1

    return part

def uncoarsening(coarseNodes, seqCoarse, outmate):
    n = len(outmate)

    sequence = np.empty(n, dtype = int)

    i = - 1

    for j in range(len(seqCoarse)):
        i += 1

        node = coarseNodes[seqCoarse[j]]

        sequence[i] = node

        match = outmate[node]

        if match > - 1:
            i += 1

            sequence[i] = match

    return sequence

def adjacent_refinement(G, sequence, outmate):
    n = len(G)

    minCost = LA(G, sequence)

    i = - 1

    while i < (n - 1):
        i += 1

        match = outmate[sequence[i]]

        if match > - 1:
            i += 1

            sequence[i], sequence[i - 1] = sequence[i - 1], sequence[i]

            cost = LA(G, sequence)

            if cost < minCost:
                minCost = cost
            else:
                sequence[i], sequence[i - 1] = sequence[i - 1], sequence[i]

    return sequence, minCost

def sequence_distance(G, s1, s2, r):
    assert(r >= 0)

    n = len(s1)

    normal_layout = np.arange(n)

    assert(np.all(np.sort(s1) == normal_layout))

    assert(np.all(np.sort(s2) == normal_layout))

    if n == 0:
        return .0, .0, .0

    J = np.empty(n, dtype = int)

    for i in range(n):
        J[i] = find(s1[i] == s2)

    diff = np.empty(n, dtype = int)

    for i in range(n):
        diff[i] = abs(i - J[i])

    dm1 = sum((1 / (diff[diff <= r] + 1)) ** (1 / r))

    flips2 = np.flip(s2)

    for i in range(n):
        J[i] = find(s1[i] == flips2)

    for i in range(n):
        diff[i] = abs(i - J[i])

    dm2 = sum((1 / (diff[diff <= r] + 1)) ** (1 / r))

    m = max(dm1, dm2)

    matching_distance = m / n

    la_s1, la_s2 = LA(G, s1), LA(G, s2)

    cost_distance = min(la_s1, la_s2) / max(la_s1, la_s2)

    distance = np.sqrt(matching_distance * cost_distance)

    flipped = dm1 < dm2

    if flipped:
        sequence = flips2
    else:
        sequence = s2

    return sequence, distance, matching_distance, cost_distance
