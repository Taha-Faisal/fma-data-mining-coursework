"""Replay the candidate models distributed in the team handoff.

AI-generated code (OpenAI Codex). Full user prompt: "next task"
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_task3 import load_inputs
from run_task7 import replay, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/raw/fma_metadata/features.csv"))
    args = parser.parse_args()
    lock_path = Path("results/task7/validation_3ceb9a571a73d6b3/lock.json")
    lock = json.loads(lock_path.read_text())
    lock["features"] = str(args.features)
    metrics, predictions = score(lock, "validation")
    replay(lock, predictions)
    saved = pd.read_csv(lock_path.parent / "metrics.csv").set_index("model")
    pd.testing.assert_frame_equal(metrics.set_index("model").sort_index(), saved.sort_index(),
                                  check_exact=False, atol=1e-9, rtol=1e-7)
    clusters = Path("results/task6_clustering/candidate_20261112T105612271049Z")
    run = json.loads((clusters / "run.json").read_text())
    bundle = Path(lock["manifest"]).parent
    if run["manifest_id"] != lock["candidate_id"]:
        raise ValueError("Clustering and classification use different cohorts")
    _, _, _, val, _ = load_inputs(bundle, args.features)
    with threadpool_limits(limits=4):
        for family in ("k-means", "GMM"):
            pipe = joblib.load(clusters / (family + ".joblib"))
            observed = pd.read_csv(clusters / (family + "_validation.csv")).set_index("track_id")
            pd.testing.assert_index_equal(observed.index.sort_values(), val.index.sort_values())
            np.testing.assert_array_equal(pipe.predict(val), observed.loc[val.index, "cluster"])
    print(f"Verified {len(predictions):,} classifier predictions, all validation metrics, "
          f"and {2 * len(val):,} clustering assignments.")


if __name__ == "__main__":
    main()
