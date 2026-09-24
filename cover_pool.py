"""Generic version: covering system with distinct moduli from an arbitrary pool of divisors of L (SAT, CaDiCaL).
Q-diagnostic for #273: since every modulus p-1 is even, a covering with moduli p-1 splits (residues even / odd) into two
disjoint coverings with moduli (p-1)/2, i.e. moduli in Q = {q : 2q+1 prime}; the family without modulus 2 (p=5) is a
covering with min modulus >= 3 using only moduli in Q minus {2}.  Usage: python cover_pool.py L [with2]"""
import sys, time
from sympy import isprime, divisors
from pysat.solvers import Cadical153
from pysat.card import CardEnc, EncType

def search_pool(L, ms, label=""):
    budget = sum(1.0 / m for m in ms); var = {}; nv = 0
    for m in ms:
        for a in range(m):
            nv += 1; var[(m, a)] = nv
    s = Cadical153(); top = nv
    for m in ms:
        cnf = CardEnc.atmost(lits=[var[(m, a)] for a in range(m)], bound=1, top_id=top, encoding=EncType.seqcounter)
        for cl in cnf.clauses: s.add_clause(cl)
        top = max(top, cnf.nv)
    for x in range(L): s.add_clause([var[(m, x % m)] for m in ms])
    t0 = time.time(); ok = s.solve(); dt = time.time() - t0; sol = None
    if ok:
        model = set(l for l in s.get_model() if l > 0)
        sol = sorted((m, a) for (m, a), v in var.items() if v in model)
        assert all(any(x % m == a for m, a in sol) for x in range(L)) and len({m for m, _ in sol}) == len(sol)
    print(f"{label} L={L}: pool {len(ms)} {ms[:14]}{'...' if len(ms) > 14 else ''} sum 1/m={budget:.4f}: {'COVERING ' + str(sol) if ok else 'no covering'} ({dt:.1f}s)", flush=True)
    return ok, sol

if __name__ == "__main__":
    L = int(sys.argv[1]); with2 = len(sys.argv) > 2 and sys.argv[2] == 'with2'
    Q = [q for q in divisors(L) if q >= (2 if with2 else 3) and isprime(2 * q + 1)]
    search_pool(L, Q, f"Q{'(with 2)' if with2 else '(no 2)'}")
