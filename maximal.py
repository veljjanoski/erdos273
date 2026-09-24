"""All halved-lcm values L' <= X whose pool Q(no 2) = {q | L' : q>=3, 2q+1 prime} has reciprocal sum >= 1, reduced to
divisibility-maximal ones (a covering for L' gives one for every multiple, so testing the maximal L' suffices).
A 'no covering' verdict for every listed L' proves: no modulus-2-free family with lcm <= X, hence no covering with
moduli p-1 (p >= 5) whose lcm is <= 2X."""
import sys
from sympy import isprime, divisors
X = int(sys.argv[1])
cand = []
for L in range(2, X + 1):
    Q = [q for q in divisors(L) if q >= 3 and isprime(2 * q + 1)]
    s = sum(1.0 / q for q in Q)
    if s >= 1: cand.append((L, s, len(Q)))
cs = {L for L, _, _ in cand}
maximal = [(L, s, n) for L, s, n in cand if not any(M != L and M % L == 0 for M in cs)]
print(f"X={X}: {len(cand)} lcms with budget >= 1, {len(maximal)} divisibility-maximal:")
for L, s, n in sorted(maximal): print(f"  L'={L} budget={s:.3f} pool={n}")
