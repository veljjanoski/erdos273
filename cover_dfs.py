"""Exact depth-first search for a covering system with DISTINCT moduli from a given pool (all dividing L).
State: uncovered residues in Z_L (uint8 array).  Moduli processed in increasing order; at each modulus either skip it
or pick one residue a (all a tried, ordered by gain).  Prune: if the remaining reciprocal budget is below the
uncovered density the branch cannot cover (each class mod m covers at most L/m residues).  numba, exact.
Usage: python cover_dfs.py L mode   mode in {direct (pool p-1, p>=5), direct3 (p>=3), Q2 (halved pool with 2), Q (halved, no 2)}"""
import sys, time
import numpy as np
from numba import njit
from sympy import isprime, divisors

def solve(L, ms, node_limit=2_000_000):
    """Wrapper that keeps correctness: we use an explicit copy-based recursion in Python-free numba variant below."""
    ms = np.array(sorted(ms), np.int64)
    unc = np.ones(L, np.uint8); chosen = -np.ones(len(ms), np.int64); nodes = np.zeros(1, np.int64)
    ok = dfs_copy(unc, L, ms, 0, float(sum(1.0 / m for m in ms)), chosen, nodes, node_limit)
    return ok, [(int(ms[i]), int(chosen[i])) for i in range(len(ms)) if chosen[i] >= 0], int(nodes[0])

@njit(cache=True)
def dfs_copy(unc, L, ms, idx, remaining_budget, chosen, nodes, node_limit):
    nodes[0] += 1
    if nodes[0] > node_limit: return False
    total_unc = 0
    for x in range(L): total_unc += unc[x]
    if total_unc == 0: return True
    if idx == len(ms): return False
    if remaining_budget * L < total_unc - 1e-9: return False
    m = ms[idx]
    gain = np.zeros(m, np.int64)
    for x in range(L):
        if unc[x]: gain[x % m] += 1
    order = np.argsort(-gain)
    for t in range(m):
        a = order[t]
        if gain[a] == 0: break
        unc2 = unc.copy()
        for x in range(a, L, m): unc2[x] = 0
        chosen[idx] = a
        if dfs_copy(unc2, L, ms, idx + 1, remaining_budget - 1.0 / m, chosen, nodes, node_limit): return True
        chosen[idx] = -1
    # skip this modulus
    return dfs_copy(unc, L, ms, idx + 1, remaining_budget - 1.0 / m, chosen, nodes, node_limit)

def pool(L, mode):
    if mode == 'direct': return [m for m in divisors(L) if m >= 4 and isprime(m + 1)]
    if mode == 'direct3': return [m for m in divisors(L) if m >= 2 and isprime(m + 1)]
    if mode == 'Q2': return [q for q in divisors(L) if q >= 2 and isprime(2 * q + 1)]
    if mode == 'Q': return [q for q in divisors(L) if q >= 3 and isprime(2 * q + 1)]
    raise ValueError(mode)

if __name__ == "__main__":
    L = int(sys.argv[1]); mode = sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5_000_000
    ms = pool(L, mode); t = time.time()
    ok, sol, nodes = solve(L, ms, limit)
    if ok:
        assert all(any(x % m == a for m, a in sol) for x in range(L)) and len({m for m, _ in sol}) == len(sol)
    print(f"{mode} L={L}: pool {len(ms)} sum={sum(1/m for m in ms):.4f}: {'COVERING '+str(sol) if ok else ('no covering (exhaustive)' if nodes <= limit else 'UNDECIDED (node limit)')} nodes={nodes} ({time.time()-t:.1f}s)", flush=True)
