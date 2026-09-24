# Erdős problem #273: covering systems with moduli p − 1

**Problem.** Is there a covering system all of whose moduli are of the form p − 1 for primes p ≥ 5?
(https://www.erdosproblems.com/273; covering system = finitely many residue classes a_i mod m_i with distinct
moduli m_i > 1 whose union is all integers, as in the Lean formalisation `StrictCoveringSystem`.) Selfridge found
such a system when p = 3 is allowed (modulus 2), using divisors of 360; `cover273.py 360 3` reproduces it.

**Result.** Every covering system with all moduli of the form p − 1, p ≥ 5, has lcm of its moduli greater than
30240. Equivalently (see the reduction below) its modulus-2-free half has lcm greater than 15120. Nothing is claimed
about existence at larger lcm; the reciprocal-sum budget of the pool only reaches values typical of known
coverings at lcm around 10^5 and above, where the search trees grow too fast for this method. As a non-rigorous
gauge, greedy placement followed by local search (`gauge273.py`, `gauge.log`) leaves at best 10–14% of the integers
uncovered with the full pool at lcm 55440 to 1663200, and 0.5–2% for the modulus-2-free family alone at lcm 27720
to 831600.

## Reduction

Every modulus p − 1 (p ≥ 3) is even, so a class a mod (p − 1) contains only integers of the parity of a. Writing
x = 2y + a₀ (a₀ ∈ {0,1}), the classes with a ≡ a₀ (mod 2) cover the integers of parity a₀ iff the classes
(a − a₀)/2 mod (p − 1)/2 cover all of ℤ. Hence a covering with moduli p − 1 is the same as two coverings with
disjoint sets of moduli of the form q = (p − 1)/2 (q with 2q + 1 prime), one for each parity. At most one of the two
can use q = 2 (p = 5), so one of them is a covering with minimum modulus at least 3 whose moduli lie in
Q₃ = {q ≥ 3 : 2q + 1 prime} = {3, 5, 6, 8, 9, 11, 14, 15, 18, 20, 21, 23, 26, 29, 30, …}, a pool without 4, 10, 12,
24, 40. If that family has lcm L′, the whole system has lcm 2·lcm(all q) ≥ 2L′.

## Computation

`cover_dfs2.py L Q` decides exactly whether a covering with distinct moduli from Q₃ ∩ {divisors of L} exists
(depth-first search over the moduli in increasing order, each either skipped or assigned one residue class, with the
bound "uncovered count ≤ sum over unused moduli of the size of their largest uncovered class"; translation symmetry
fixes the first used residue to 0). A covering for L is also one for every multiple of L, so it suffices to test the
divisibility-maximal L′ ≤ X among those whose pool has reciprocal sum ≥ 1 (all others are impossible by counting).
`maximal.py 15120` lists these 129 values (`maximal_15120.txt`); all were found to admit no covering
(`maximal_run.log`, `qladder3.log`, `par15120.log`; L′ = 15120 needed 2.56·10⁹ nodes, run by `cover_par2.py` on
8 workers; the other 128 at most 2.4·10⁶ nodes, none reaching its node cap). Hence no modulus-2-free family has
lcm ≤ 15120, and no covering with moduli p − 1 has lcm ≤ 30240.

Validation: the same code, with the pool of all divisors m of L with m + 1 prime, finds Selfridge's covering for
L = 360 (`cover_dfs.py 360 direct3`), and with the halved pool including 2 finds a covering for L = 360
(`cover_dfs.py 360 Q2`); both were also found by an independent SAT encoding (`cover273.py`, `cover_pool.py`). Every
covering reported by any program is re-checked by direct enumeration of ℤ_L. The two exact programs (`cover_dfs.py`
with the density bound, `cover_dfs2.py` with the best-class bound) give the same negative verdict on every case both
decided: the 22 lcms 180, 240, 270, 360, 420, 450, 480, 540, 630, 720, 840, 900, 1080, 1260, 1440, 1680, 2160, 3360,
3780, 6480 with the pool Q₃, and 6, 12, 720 with the direct pools (`qladder2.log`, `crosscheck.log`).

## Files

`cover273.py`, `cover_pool.py` (SAT versions), `cover_dfs.py` (density bound), `cover_dfs2.py` (best-class bound),
`cover_par2.py` (prefix-parallel driver), `cover_dfs3.py` (element branching, slower, not used for results),
`budget.py`, `maximal.py`, job lists and logs. Requirements: numpy, numba, sympy, python-sat.
