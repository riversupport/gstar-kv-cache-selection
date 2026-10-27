# -*- coding: utf-8 -*-
"""GStar Day 1: CPU-only, finite-candidate KV selection demo.

Python 3.8+; standard library only. No models, downloads, or GPU calls.

Scope: unprotected positions for a single layer and a single KV head.
The example uses synthetic a and g scores instead of measured attention.
Protected positions P, score extraction, and attention masking are outside this demo.
Passing this demo checks selection logic, not model accuracy or memory savings.
"""

import math

import logging

logger = logging.getLogger("gstar")


# Synthetic example configuration.
A_SCORES = [0.42, 0.26, 0.12, 0.10, 0.06, 0.04]
G_SCORES = [0.12, 0.08, 0.10, 0.32, 0.23, 0.15]
BUDGET = 2       # Number b of unprotected positions to select
RHO = 0.90       # Required retained g-score fraction

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TOL = 1e-12      # Tolerance for floating-point rounding only


def normalize(scores):
    """Normalize finite, nonnegative scores to sum to one."""
    values = [float(x) for x in scores]
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Scores must be a nonempty list of finite, nonnegative numbers.")
    largest = max(values)
    if largest == 0:
        raise ValueError("At least one score must be positive; all-zero scores need an explicit policy.")
    # Avoid overflow when summing large scores.
    scaled = [x / largest for x in values]
    total = math.fsum(scaled)
    return [x / total for x in scaled]


def select_candidates(a_scores, g_scores, b, rho):
    """Compare five candidates; this heuristic does not optimize over all subsets.

    Returns: (selected candidate, all candidates, G(R_1), constraint floor).
    Break candidate ties by the order in LAMBDAS.
    Break token-score ties by the smaller position index.
    """
    a, g = normalize(a_scores), normalize(g_scores)
    if len(a) != len(g):
        raise ValueError("a and g must have the same length.")
    if isinstance(b, bool) or not isinstance(b, int) or not 0 <= b <= len(a):
        raise ValueError("b must be an integer between 0 and the number of positions.")
    rho = float(rho)
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must be between 0 and 1.")

    candidates = []
    for lam in LAMBDAS:
        mixed = [(1 - lam) * aj + lam * gj for aj, gj in zip(a, g)]
        # Select the top b positions, then sort their indices for deterministic comparisons.
        positions = tuple(sorted(sorted(range(len(a)), key=lambda j: (-mixed[j], j))[:b]))
        candidates.append({
            "lambda": lam,
            "positions": positions,
            "A": math.fsum(a[j] for j in positions),
            "G": math.fsum(g[j] for j in positions),
        })

    g_reference = candidates[-1]["G"]  # lambda=1: g-only candidate
    floor = rho * g_reference
    for candidate in candidates:
        candidate["feasible"] = candidate["G"] + TOL >= floor

    feasible = [c for c in candidates if c["feasible"]]
    # The lambda=1 candidate is always feasible for valid inputs.
    selected = max(feasible, key=lambda c: c["A"])
    return selected, candidates, g_reference, floor


def self_test():
    """Check deterministic examples and edge cases, not model integration."""
    a = [0.42, 0.26, 0.12, 0.10, 0.06, 0.04]
    g = [0.12, 0.08, 0.10, 0.32, 0.23, 0.15]
    chosen, rows, reference, floor = select_candidates(a, g, 2, 0.8)
    assert chosen["positions"] == (0, 3)
    assert math.isclose(reference, 0.55)
    assert math.isclose(floor, 0.44)
    assert all(len(row["positions"]) == 2 for row in rows)
    assert rows[-1]["feasible"]
    assert chosen["G"] + TOL >= floor

    tighter, _, _, _ = select_candidates(a, g, 2, 0.9)
    assert tighter["positions"] == (3, 4)
    strict, _, ref, _ = select_candidates(a, g, 2, 1.0)
    assert math.isclose(strict["G"], ref)
    all_positions, _, _, _ = select_candidates(a, g, 6, 1.0)
    assert all_positions["positions"] == tuple(range(6))
    empty, _, _, _ = select_candidates(a, g, 0, 1.0)
    assert empty["positions"] == ()
    tied, _, _, _ = select_candidates([1, 1, 1], [1, 1, 1], 2, 1.0)
    assert tied["positions"] == (0, 1)
    scaled, _, _, _ = select_candidates([x * 7 for x in a], [x * 3 for x in g], 2, 0.8)
    assert scaled["positions"] == chosen["positions"]
    for bad_a, bad_g, bad_b, bad_rho in [
        ([0, 0], [1, 2], 1, 0.8), ([1, -1], [1, 2], 1, 0.8),
        ([1, 2], [1], 1, 0.8), ([1, 2], [1, 2], 3, 0.8),
        ([1, 2], [1, 2], 1, 1.2),
    ]:
        try:
            select_candidates(bad_a, bad_g, bad_b, bad_rho)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was not rejected.")


def main():
    self_test()
    logger.info('%s', "SELF-TEST: PASS")
    logger.debug('%s', "Synthetic scores only. No LLM / no GPU / no downloads.\n")
    selected, candidates, reference, floor = select_candidates(A_SCORES, G_SCORES, BUDGET, RHO)
    logger.debug('%s', "Positions are numbered from 0.")
    logger.debug('%s', f"b={BUDGET}, rho={RHO:.2f}, G(R_1)={reference:.4f}, floor={floor:.4f}\n")
    logger.debug('%s', "lambda   positions        A        G       feasible")
    for row in candidates:
        logger.debug('%s', f"{row['lambda']:5.2f}    {str(row['positions']):12s}   "
              f"{row['A']:.4f}   {row['G']:.4f}   {row['feasible']}")
    logger.debug('%s', f"\nSelected positions: {list(selected['positions'])}")
    logger.debug('%s', f"Selected lambda: {selected['lambda']:.2f}")
    logger.debug('%s', "\nNext: change RHO from 0.80 to 0.90, save, and run again.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()