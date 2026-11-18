"""Verify the archived coursework scores using only Python's standard library.

AI-generated code (OpenAI Codex). Initiating request:
"Complete the five pending tasks. And how are we supposed to fix a to reported task and historical issues? Without going outside of our specification and scope?"
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent / "04_Current_Project_Snapshot"
PHASE = PROJECT / "results/phase4"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hash(path, expected):
    check(hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path))


def main():
    receipt = read_json(PHASE / "TEST_RECEIPT.json")
    verify_hash(PHASE / "test_predictions.csv", receipt["predictions_sha256"])
    verify_hash(PHASE / "EVALUATION_LOCK.json", receipt["lock_sha256"])
    lock = read_json(PHASE / "EVALUATION_LOCK.json")
    base = PROJECT / "data/processed/task2"
    bundle = base / read_json(base / "LATEST_FROZEN.json")["bundle"]
    ledger = read_json(bundle / "bundle.json")
    check(ledger["candidate_id"] == lock["cohort_id"], "Incorrect cohort")
    for name, digest in ledger["files"].items():
        verify_hash(bundle / name, digest)
    manifest = rows(bundle / "manifest.csv")
    expected = {r["track_id"]: int(r["hit"]) for r in manifest if r["set_split"] == "test"}
    predictions = defaultdict(dict)
    for r in rows(PHASE / "test_predictions.csv"):
        model, track = r["model"], r["track_id"]
        check(track not in predictions[model] and int(r["hit"]) == expected[track], track)
        score = float(r["score"])
        check(0 <= score <= 1, track)
        predictions[model][track] = score
    check(len(expected) == receipt["tracks_per_model"], "Test track count")
    check(set(predictions) == set(lock["thresholds"]), "Model names")
    comparison = rows(PHASE / "test_comparison.csv")
    wanted = {(name, protocol) for name in predictions for protocol in ("fixed_0.5", "validation_selected")}
    check(len(comparison) == len(wanted) and {(r["model"], r["protocol"]) for r in comparison} == wanted, "Comparison rows")
    for r in comparison:
        model = r["model"]
        scores = predictions[model]
        check(scores.keys() == expected.keys(), model)
        threshold = 0.5 if r["protocol"] == "fixed_0.5" else lock["thresholds"][model]
        check(float(r["threshold"]) == threshold, model)
        counts = [0, 0, 0, 0]
        for track, score in scores.items():
            counts[2 * expected[track] + int(score >= threshold)] += 1
        tn, fp, fn, tp = counts
        f1 = 2 * tp / (2 * tp + fp + fn)
        check(counts == [int(r[k]) for k in ("tn", "fp", "fn", "tp")], model)
        check(math.isclose(f1, float(r["hit_f1"]), abs_tol=1e-12), model)
        print(f"{model:15s} {r['protocol']:20s} hit F1 {f1:.6f}")
    print(f"Verified {len(comparison)} comparisons on {len(expected)} held-out tracks.")


if __name__ == "__main__":
    main()
