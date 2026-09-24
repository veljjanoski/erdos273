"""Heuristic gauge: how close can distinct-moduli classes from a pool get to covering Z_L?
Greedy (best new coverage over unused moduli and residues), then local search: for each used modulus, move its class
to the residue minimising uncovered count (given the others), repeated until no improvement; several restarts with
random tie-breaking / random initial order.  Reports min uncovered fraction.  NOT exact: a positive value proves nothing.
Usage: python gauge273.py L mode [restarts]   modes as in cover_dfs.py (direct, direct3, Q, Q2)"""
import sys, time
import numpy as np
from numba import njit
from cover_dfs import pool

@njit
def hist_uncovered(cov, m, L, out):
    for a in range(m): out[a] = 0
    for x in range(L):
        if cov[x] == 0: out[x % m] += 1

@njit
def add_class(cov, m, a, L, sign):
    for x in range(a, L, m): cov[x] += sign

@njit
def greedy(cov, ms, L, order):
    res = -np.ones(len(ms), np.int64); tmp = np.zeros(ms.max() + 1, np.int64)
    for t in range(len(ms)):
        j = order[t]; m = ms[j]
        hist_uncovered(cov, m, L, tmp)
        best = 0; ba = 0
        for a in range(m):
            if tmp[a] > best: best = tmp[a]; ba = a
        if best > 0:
            res[j] = ba; add_class(cov, m, ba, L, 1)
    return res

@njit
def local_search(cov, ms, res, L, max_pass):
    tmp = np.zeros(ms.max() + 1, np.int64)
    for p in range(max_pass):
        improved = False
        for j in range(len(ms)):
            m = ms[j]
            if res[j] >= 0: add_class(cov, m, res[j], L, -1)
            hist_uncovered(cov, m, L, tmp)
            best = 0; ba = -1
            for a in range(m):
                if tmp[a] > best: best = tmp[a]; ba = a
            if ba >= 0: add_class(cov, m, ba, L, 1)
            if ba != res[j]: improved = True
            res[j] = ba
        if not improved: break
    unc = 0
    for x in range(L):
        if cov[x] == 0: unc += 1
    return unc

def gauge(L, ms, restarts=5, seed=0):
    ms = np.array(sorted(ms), np.int64); rng = np.random.default_rng(seed); best = L; best_res = None
    for r in range(restarts):
        order = np.arange(len(ms)) if r == 0 else rng.permutation(len(ms))
        cov = np.zeros(L, np.int32); res = greedy(cov, ms, L, order)
        unc = local_search(cov, ms, res, L, 50)
        if unc < best: best = unc; best_res = res.copy()
    return best, best_res, ms

if __name__ == "__main__":
    L = int(sys.argv[1]); mode = sys.argv[2]; restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    ms = pool(L, mode); t = time.time(); unc, res, msa = gauge(L, ms, restarts)
    # independent recount
    cov = np.zeros(L, np.int32)
    for j in range(len(msa)):
        if res[j] >= 0:
            for x in range(int(res[j]), L, int(msa[j])): cov[x] += 1
    assert int((cov == 0).sum()) == unc
    print(f"{mode} L={L} pool {len(ms)} sum={sum(1/m for m in ms):.4f}: best uncovered {unc} of {L} = {unc/L:.5f} ({time.time()-t:.0f}s)", flush=True)
