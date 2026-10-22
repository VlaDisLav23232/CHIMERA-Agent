from math import comb

import numpy as np

N_RESAMPLES = 2000
SEED = 0


def bootstrap_ci(cases: list[dict], metric) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(cases)
    values = [metric([cases[i] for i in rng.integers(0, n, n)]) for _ in range(N_RESAMPLES)]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap(cases_a: list[dict], cases_b: list[dict], metric) -> dict:
    b_by_id = {c["case_id"]: c for c in cases_b}
    if set(b_by_id) != {c["case_id"] for c in cases_a}:
        raise ValueError("paired comparison needs both runs scored on the same cases")
    pairs = [(c, b_by_id[c["case_id"]]) for c in cases_a]
    rng = np.random.default_rng(SEED)
    n = len(pairs)
    diffs = []
    for _ in range(N_RESAMPLES):
        sample = [pairs[i] for i in rng.integers(0, n, n)]
        diffs.append(metric([a for a, _ in sample]) - metric([b for _, b in sample]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": metric(cases_a) - metric(cases_b), "ci": (float(low), float(high))}


def mcnemar_exact(cases_a: list[dict], cases_b: list[dict]) -> float:
    b_by_id = {c["case_id"]: c for c in cases_b}
    only_a = sum(c["correct"] and not b_by_id[c["case_id"]]["correct"] for c in cases_a)
    only_b = sum(not c["correct"] and b_by_id[c["case_id"]]["correct"] for c in cases_a)
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)
