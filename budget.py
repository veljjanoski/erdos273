"""Reciprocal-sum budgets: for smooth L list the pool {m | L : m+1 prime, m+1 >= 5} (direct problem) and the halved pool
Q(no 2) = {q | L : 2q+1 prime, q >= 3}.  A covering needs sum 1/m >= 1 (with slack in practice)."""
import itertools
from sympy import isprime, divisors
def pools(L):
    P = [m for m in divisors(L) if m >= 4 and isprime(m + 1)]
    Q = [q for q in divisors(L) if q >= 3 and isprime(2 * q + 1)]
    return P, Q
rows = []
for a in range(1, 9):
    for b in range(0, 6):
        for c in range(0, 3):
            for d in range(0, 3):
                for e in (0, 1):
                    L = 2**a * 3**b * 5**c * 7**d * 11**e
                    if L > 3_000_000: continue
                    P, Q = pools(L)
                    rows.append((sum(1/m for m in P), sum(1/q for q in Q), L, len(P), len(Q)))
rows.sort(reverse=True)
print("top budgets, direct pool (p>=5):"); 
for sP, sQ, L, nP, nQ in rows[:12]: print(f"  L={L:8d} (={L//1})  sumP={sP:.4f} ({nP} moduli)   sumQ(no2)={sQ:.4f} ({nQ})")
rows.sort(key=lambda r: -r[1]); print("top budgets, halved pool without 2:")
for sP, sQ, L, nP, nQ in rows[:12]: print(f"  L={L:8d}  sumQ(no2)={sQ:.4f} ({nQ} moduli)   sumP={sP:.4f}")
import sympy
print("first L with sumP >= 1:", [ (L, round(sP,3)) for sP, sQ, L, nP, nQ in sorted(rows, key=lambda r: r[2]) if sP >= 1][:8])
print("first L with sumQ(no2) >= 1:", [ (L, round(sQ,3)) for sP, sQ, L, nP, nQ in sorted(rows, key=lambda r: r[2]) if sQ >= 1][:8])
