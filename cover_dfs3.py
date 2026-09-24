"""Exact covering search, element branching: pick the smallest uncovered x; every covering must cover x, so branch over
the unused moduli m (class x mod m).  No skip branches.  Bound: uncovered count <= sum over unused moduli of their
best class.  Translation symmetry: the first modulus used gets residue 0 (i.e. we cover x=0 first: automatic, since
the smallest uncovered integer at the root is 0 and its class is 0 mod m).  Usage: python cover_dfs3.py L mode [limit]"""
import sys, time
import numpy as np
from numba import njit
from cover_dfs import pool
from cover_dfs2 import apply_class, unapply

@njit
def rec3(unc, cnt, off, ms, used, total_unc, nodes, limit, chosen, start):
    nodes[0] += 1
    if nodes[0] > limit: return False
    if total_unc == 0: return True
    bound = 0
    for j in range(len(ms)):
        if used[j]: continue
        o = off[j]; best = 0
        for a in range(ms[j]):
            if cnt[o + a] > best: best = cnt[o + a]
        bound += best
    if bound < total_unc: return False
    x = start
    while unc[x] == 0: x += 1                      # smallest uncovered integer
    # branch: which unused modulus covers x (order: largest class count first)
    L = len(unc); cand = np.empty(len(ms), np.int64); gains = np.empty(len(ms), np.int64); k = 0
    for j in range(len(ms)):
        if used[j]: continue
        cand[k] = j; gains[k] = cnt[off[j] + x % ms[j]]; k += 1
    order = np.argsort(-gains[:k])
    for t in range(k):
        j = cand[order[t]]; m = ms[j]; a = x % m; c = cnt[off[j] + a]
        used[j] = 1; chosen[j] = a
        newly = apply_class(unc, cnt, off, ms, j, a, L)
        if rec3(unc, cnt, off, ms, used, total_unc - c, nodes, limit, chosen, x): return True
        unapply(unc, cnt, off, ms, newly, L)
        used[j] = 0; chosen[j] = -1
    return False

def solve(L, ms, limit=10**9):
    ms = np.array(sorted(ms), np.int64); off = np.zeros(len(ms), np.int64)
    for j in range(1, len(ms)): off[j] = off[j - 1] + ms[j - 1]
    cnt = np.zeros(int(off[-1] + ms[-1]), np.int64)
    for j in range(len(ms)):
        for x in range(L): cnt[off[j] + x % ms[j]] += 1
    unc = np.ones(L, np.uint8); used = np.zeros(len(ms), np.uint8); nodes = np.zeros(1, np.int64)
    chosen = -np.ones(len(ms), np.int64)
    ok = rec3(unc, cnt, off, ms, used, L, nodes, limit, chosen, 0)
    sol = [(int(ms[j]), int(chosen[j])) for j in range(len(ms)) if chosen[j] >= 0]
    if ok: assert all(any(x % m == a for m, a in sol) for x in range(L)) and len({m for m, _ in sol}) == len(sol)
    return ok, sol, int(nodes[0])

if __name__ == "__main__":
    L = int(sys.argv[1]); mode = sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    ms = pool(L, mode); t = time.time(); ok, sol, nodes = solve(L, ms, limit)
    print(f"{mode} L={L} pool {len(ms)} sum={sum(1/m for m in ms):.4f}: {'COVERING '+str(sol) if ok else ('no covering (exhaustive)' if nodes <= limit else 'UNDECIDED')} nodes={nodes} ({time.time()-t:.1f}s)", flush=True)
