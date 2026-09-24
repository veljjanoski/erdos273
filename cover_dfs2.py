"""Exact covering search with the 'sum of best classes' bound.
State: unc[x] (uncovered), and for every modulus m in the pool cnt[m][a] = number of uncovered x with x = a mod m.
Prune: uncovered total > sum over unused moduli of max_a cnt[m][a]  -> impossible (each unused modulus, used at most
once, covers at most its best class).  Branch on the unused modulus with the fewest classes that could still matter
(smallest m), residues in decreasing count order, plus 'skip'.  Translation symmetry: first used modulus gets residue 0.
Exhaustive within a node limit.  Usage: python cover_dfs2.py L mode [node_limit]"""
import sys, time
import numpy as np
from numba import njit
from cover_dfs import pool

@njit
def apply_class(unc, cnt, off, ms, m_idx, a, L):
    """cover class a mod ms[m_idx]; update unc and all cnt tables; return list of newly covered x (as array)"""
    m = ms[m_idx]; newly = np.empty(L // m + 1, np.int64); k = 0
    for x in range(a, L, m):
        if unc[x]:
            unc[x] = 0; newly[k] = x; k += 1
    for j in range(len(ms)):
        mj = ms[j]; o = off[j]
        for t in range(k): cnt[o + (newly[t] % mj)] -= 1
    return newly[:k]

@njit
def unapply(unc, cnt, off, ms, newly, L):
    for t in range(len(newly)): unc[newly[t]] = 1
    for j in range(len(ms)):
        mj = ms[j]; o = off[j]
        for t in range(len(newly)): cnt[o + (newly[t] % mj)] += 1

@njit
def rec(unc, cnt, off, ms, used, total_unc, nodes, limit, first_used, chosen):
    nodes[0] += 1
    if nodes[0] > limit: return False
    if total_unc == 0: return True
    # bound: sum of best classes over unused moduli
    bound = 0; pick = -1
    for j in range(len(ms)):
        if used[j]: continue
        o = off[j]; best = 0
        for a in range(ms[j]):
            if cnt[o + a] > best: best = cnt[o + a]
        bound += best
        if pick < 0: pick = j                       # smallest unused modulus
    if bound < total_unc: return False
    if pick < 0: return False
    j = pick; m = ms[j]; o = off[j]; used[j] = 1
    # residues in decreasing count order
    order = np.argsort(-cnt[o:o + m])
    for t in range(m):
        a = order[t]; c = cnt[o + a]
        if c == 0: break
        if first_used and a != 0: continue      # translation symmetry
        newly = apply_class(unc, cnt, off, ms, j, a, len(unc))
        chosen[j] = a
        if rec(unc, cnt, off, ms, used, total_unc - c, nodes, limit, False, chosen): return True
        chosen[j] = -1
        unapply(unc, cnt, off, ms, newly, len(unc))
    # skip this modulus
    r = rec(unc, cnt, off, ms, used, total_unc, nodes, limit, first_used, chosen)
    used[j] = 0
    return r

def solve(L, ms, limit=10**9):
    ms = np.array(sorted(ms), np.int64); off = np.zeros(len(ms), np.int64)
    for j in range(1, len(ms)): off[j] = off[j - 1] + ms[j - 1]
    cnt = np.zeros(int(off[-1] + ms[-1]), np.int64)
    for j in range(len(ms)):
        for x in range(L): cnt[off[j] + x % ms[j]] += 1
    unc = np.ones(L, np.uint8); used = np.zeros(len(ms), np.uint8); nodes = np.zeros(1, np.int64)
    chosen = -np.ones(len(ms), np.int64)
    ok = rec(unc, cnt, off, ms, used, L, nodes, limit, True, chosen)
    sol = [(int(ms[j]), int(chosen[j])) for j in range(len(ms)) if chosen[j] >= 0]
    if ok: assert all(any(x % m == a for m, a in sol) for x in range(L)) and len({m for m, _ in sol}) == len(sol)
    return ok, sol, int(nodes[0])

if __name__ == "__main__":
    L = int(sys.argv[1]); mode = sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    ms = pool(L, mode); t = time.time(); ok, sol, nodes = solve(L, ms, limit)
    print(f"{mode} L={L} pool {len(ms)} sum={sum(1/m for m in ms):.4f}: {'COVERING '+str(sol) if ok else ('no covering (exhaustive)' if nodes <= limit else 'UNDECIDED')} nodes={nodes} ({time.time()-t:.1f}s)", flush=True)
