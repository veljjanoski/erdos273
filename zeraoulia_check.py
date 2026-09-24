# Independent re-implementation of the certificate in R. Zeraoulia, "A computer-assisted lower bound for covering systems
# with prime-minus-one moduli", Zenodo, July 2026, doi:10.5281/zenodo.21613011 (lcm >= 393120 for Erdős #273).
from fractions import Fraction as Fr
from sympy import isprime, divisors
from math import gcd, lcm
from itertools import product
import sys

def M(L): return [m for m in divisors(L) if m >= 4 and isprime(m + 1)]

B = 393120
cands = [L for L in range(2, B, 2) if sum(Fr(1, m) for m in M(L)) > 1]
print("candidates:", len(cands))
assert len(cands) == 28

parity_killed, residual = [], []
for L in cands:
    Ms = M(L); S = sum(Fr(1, m) for m in Ms); d = S - 1
    F = [m for m in Ms if Fr(1, m) > d]
    if 4 in F and 6 in F and d < Fr(1, 12):
        tot = Fr(0)
        for m in Ms:
            if m in (4, 6): continue
            A = Fr(1, m) if m % 4 == 0 else Fr(1, 2 * m)
            Bm = Fr(1, m) if m % 6 == 0 else Fr(2, 3 * m)
            tot += max(A, Bm)
        if tot < Fr(7, 12):
            parity_killed.append(L); continue
    residual.append((L, F))
print("parity filter kills", len(parity_killed), "; residual:", [L for L, F in residual])

for L, F in residual:
    Ms = M(L); Q = 1
    for m in F: Q = lcm(Q, m)
    rest = [m for m in Ms if m not in F]
    others = [m for m in F if m != 4]
    best = None
    for cfg in product(*[range(m) for m in others]):
        cov = bytearray(Q)
        for x in range(0, Q, 4): cov[x] = 1
        for m, a in zip(others, cfg):
            for x in range(a % m, Q, m): cov[x] = 1
        U = [u for u in range(Q) if not cov[u]]
        if not U: print("config covers everything?!", L, cfg); sys.exit(1)
        tot = Fr(0)
        for m in rest:
            g = gcd(Q, m); cnt = [0] * g
            for u in U: cnt[u % g] += 1
            tot += Fr(g, Q * m) * max(cnt)
        margin = tot - Fr(len(U), Q)
        if best is None or margin > best: best = margin
    print(f"L={L} F={F} Q={Q} configs={len(list(product(*[range(m) for m in others])))} largest margin={best} ({float(best):.5f})")
    assert best < 0, L
print("ALL 28 CANDIDATES EXCLUDED: lcm >= 393120 confirmed")
