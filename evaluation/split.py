import json
import random
from collections import defaultdict

from evaluation.official import load_references
from evaluation.runs import SPLIT_FILE, prior_biopsy

SEED = 42
N_TEST = 31


def make_split() -> dict:
    references = load_references()
    strata = defaultdict(list)
    for case_id, ref in sorted(references.items()):
        strata[(ref["biopsy_decision"], prior_biopsy(case_id))].append(case_id)

    rng = random.Random(SEED)
    test = []
    for key in sorted(strata):
        cases = strata[key]
        rng.shuffle(cases)
        test += cases[: round(len(cases) * N_TEST / len(references))]
    dev = sorted(set(references) - set(test))
    return {"seed": SEED, "n_dev": len(dev), "n_test": len(test), "dev": dev, "test": sorted(test)}


def main() -> None:
    split = make_split()
    if SPLIT_FILE.exists() and json.loads(SPLIT_FILE.read_text()) != split:
        raise SystemExit(f"{SPLIT_FILE} exists and differs from the regenerated split; not overwriting")
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=1) + "\n")
    print(f"dev {split['n_dev']}, test {split['n_test']} -> {SPLIT_FILE}")


if __name__ == "__main__":
    main()
