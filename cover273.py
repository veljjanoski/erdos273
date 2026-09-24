"""Erdős #273: covering systems with all moduli of the form p-1 (p prime, p >= 5), distinct moduli.
Exact search inside a fixed lcm L: candidate moduli = {m in pool : m | L}; SAT variables r[m][a] = 'use residue a mod m'
(at most one residue per modulus, modulus may be unused); every x in Z_L must be covered.  Solver: CaDiCaL.
Usage: python cover273.py L [pmin]   (pmin=5 default; pmin=3 allows modulus 2 -> Selfridge's example for validation)"""
import sys, itertools, time
from sympy import isprime, divisors
from pysat.solvers import Cadical153
from pysat.card import CardEnc, EncType

def pool(L, pmin=5):
    return [m for m in divisors(L) if m >= 2 and m + 1 >= pmin and isprime(m + 1)]

def search(L, pmin=5, verbose=True):
    ms = pool(L, pmin); budget = sum(1.0 / m for m in ms)
    var = {}; nv = 0
    for m in ms:
        for a in range(m):
            nv += 1; var[(m, a)] = nv
    s = Cadical153(); top = nv
    for m in ms:                                          # at most one residue per modulus
        cnf = CardEnc.atmost(lits=[var[(m, a)] for a in range(m)], bound=1, top_id=top, encoding=EncType.seqcounter)
        for cl in cnf.clauses: s.add_clause(cl)
        top = max(top, cnf.nv)
    for x in range(L):                                    # every x covered
        s.add_clause([var[(m, x % m)] for m in ms])
    t0 = time.time(); ok = s.solve(); dt = time.time() - t0
    sol = None
    if ok:
        model = set(l for l in s.get_model() if l > 0)
        sol = sorted((m, a) for (m, a), v in var.items() if v in model)
        # independent check: every x in Z_L covered, moduli distinct, all m+1 prime >= pmin
        assert all(any(x % m == a for m, a in sol) for x in range(L))
        assert len({m for m, _ in sol}) == len(sol) and all(isprime(m + 1) and m + 1 >= pmin for m, _ in sol)
    if verbose:
        print(f"L={L} pmin={pmin}: pool {len(ms)} moduli {ms[:12]}{'...' if len(ms) > 12 else ''}, sum 1/m = {budget:.4f}: "
              f"{'COVERING FOUND' if ok else 'no covering'} ({dt:.1f}s)" + (f"  {sol}" if ok else ""), flush=True)
    return ok, sol

if __name__ == "__main__":
    L = int(sys.argv[1]); pmin = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    search(L, pmin)
