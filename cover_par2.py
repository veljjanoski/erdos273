"""Parallel version of cover_dfs2: enumerate prefixes over the first `depth` moduli (skip or residue; first used
modulus gets residue 0), solve each prefix with the bound-based recursion in a worker pool (limit per prefix).
Exact: 'no covering' only if every prefix finished under the limit.  Usage: python cover_par2.py L mode depth limit procs"""
import sys, time, itertools
import numpy as np
from multiprocessing import Pool
from cover_dfs import pool

def run_prefix(args):
    L, ms, prefix, limit = args
    from cover_dfs2 import rec, apply_class          # import inside worker (numba compile per process)
    ms = np.array(ms, np.int64); off = np.zeros(len(ms), np.int64)
    for j in range(1, len(ms)): off[j] = off[j - 1] + ms[j - 1]
    cnt = np.zeros(int(off[-1] + ms[-1]), np.int64)
    for j in range(len(ms)):
        for x in range(L): cnt[off[j] + x % ms[j]] += 1
    unc = np.ones(L, np.uint8); used = np.zeros(len(ms), np.uint8); chosen = -np.ones(len(ms), np.int64)
    total = L; first_used = True
    for i, a in enumerate(prefix):
        used[i] = 1
        if a >= 0:
            c = cnt[off[i] + a]; apply_class(unc, cnt, off, ms, i, a, L); chosen[i] = a; total -= c; first_used = False
    nodes = np.zeros(1, np.int64)
    ok = rec(unc, cnt, off, ms, used, total, nodes, limit, first_used, chosen)
    sol = [(int(ms[j]), int(chosen[j])) for j in range(len(ms)) if chosen[j] >= 0]
    return ok, sol, int(nodes[0]), prefix

def prefixes(ms, depth):
    out = []
    for combo in itertools.product(*[[-1] + list(range(m)) for m in ms[:depth]]):
        used = [a for a in combo if a >= 0]
        if used and used[0] != 0: continue
        out.append(combo)
    return out

if __name__ == "__main__":
    L = int(sys.argv[1]); mode = sys.argv[2]; depth = int(sys.argv[3]); limit = int(sys.argv[4]); procs = int(sys.argv[5])
    ms = sorted(pool(L, mode)); pre = prefixes(ms, depth); t = time.time()
    with Pool(procs) as P: results = P.map(run_prefix, [(L, ms, p, limit) for p in pre], chunksize=1)
    found = [r for r in results if r[0]]; undecided = [r[3] for r in results if not r[0] and r[2] > limit]; nodes = sum(r[2] for r in results)
    if found:
        sol = found[0][1]; assert all(any(x % m == a for m, a in sol) for x in range(L)) and len({m for m, _ in sol}) == len(sol)
        print(f"{mode} L={L} pool {len(ms)} depth={depth}: COVERING {sol} nodes={nodes} ({time.time()-t:.0f}s)", flush=True)
    else:
        print(f"{mode} L={L} pool {len(ms)} depth={depth}: {'UNDECIDED (' + str(len(undecided)) + ' of ' + str(len(pre)) + ' prefixes hit the limit)' if undecided else 'no covering (exhaustive over ' + str(len(pre)) + ' prefixes)'} nodes={nodes} ({time.time()-t:.0f}s)", flush=True)
