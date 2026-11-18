"""Lock tabular comparisons, replay validation, and evaluate a frozen cohort once.

AI-generated code (OpenAI Codex). Full user prompt: "next task"
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_task3 import digest, evaluate


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(task4):
    neural = read(task4 / "run.json")
    task3 = Path(neural["task3_run"])
    classic = read(task3 / "run.json")
    bundle = Path(classic["manifest_bundle"])
    config = read(bundle / "config.json")
    for key in ("candidate_id", "status"):
        if not config[key] == classic[key] == neural[key]:
            raise ValueError("Training runs must match the same task 2 cohort")
    if classic["feature_columns"] != neural["feature_columns"] or classic["score_threshold"] != .5 or neural["settings"]["threshold"] != .5:
        raise ValueError("Feature schema or decision thresholds differ")
    manifest = bundle / ("manifest.csv" if config["status"] == "frozen" else "manifest_candidate.csv")
    ledger = read(bundle / "bundle.json")["files"]
    if digest(manifest) != ledger[manifest.name] or digest(bundle / "config.json") != ledger["config.json"]:
        raise ValueError("Task 2 bundle changed")
    settings = pd.read_csv(task3 / "validation_results.csv")
    models = {r.model: dict(path=str(task3 / (r.model.replace(" ", "_") + ".joblib")),
                            settings=json.loads(r.settings)) for r in settings.itertuples()}
    best_epoch = pd.read_csv(task4 / "experiments.csv").set_index("trial").loc[neural["selected"], "best_epoch"]
    models["MLP"] = dict(path=str(task4 / "mlp.joblib"), settings=dict(trial=neural["selected"],
                         epoch=int(best_epoch), **neural["settings"]))
    for details in models.values():
        details["sha256"] = digest(Path(details["path"]))
    return dict(status=config["status"], candidate_id=config["candidate_id"], target=config["target"],
                manifest=str(manifest), manifest_sha256=digest(manifest),
                features=config["inputs"]["features"]["path"], features_sha256=classic["feature_sha256"],
                columns=classic["feature_columns"], threshold=.5, primary_metric="hit_f1", models=models,
                validation_files=[str(task3 / "validation_predictions.csv"), str(task4 / "validation_predictions.csv")])


def score(lock, split):
    manifest = Path(lock["manifest"])
    features = Path(lock["features"])
    if digest(manifest) != lock["manifest_sha256"] or digest(features) != lock["features_sha256"]:
        raise ValueError("Locked data changed")
    records = pd.read_csv(manifest).set_index("track_id")
    if not records.index.is_unique or records.groupby("group_id").set_split.nunique().max() != 1:
        raise ValueError("Invalid split membership")
    truth = records.loc[records.set_split.eq(split), "hit"]
    parts = []
    with pd.read_csv(features, header=[0, 1, 2], index_col=0, chunksize=4000) as reader:
        for chunk in reader:
            parts.append(chunk.loc[chunk.index.isin(truth.index)])
    x = pd.concat(parts, verify_integrity=True).loc[truth.index]
    x.columns = ["audio__" + "__".join(c) for c in x.columns]
    x = x[lock["columns"]].replace([np.inf, -np.inf], np.nan)
    rows, predictions = [], []
    with threadpool_limits(limits=4):
        for name, model in lock["models"].items():
            path = Path(model["path"])
            if digest(path) != model["sha256"]:
                raise ValueError(f"Changed checkpoint: {name}")
            pipe = joblib.load(path)
            scores = pipe.predict_proba(x)[:, list(pipe.classes_).index(1)]
            rows.append(dict(model=name, **evaluate(truth, scores)))
            predictions.append(pd.DataFrame(dict(model=name, track_id=truth.index, hit=truth.values,
                               score=scores, prediction=(scores >= lock["threshold"]).astype(int))))
    return pd.DataFrame(rows).sort_values("hit_f1", ascending=False), pd.concat(predictions, ignore_index=True)


def replay(lock, predictions):
    classical, neural = map(pd.read_csv, lock["validation_files"])
    reference = pd.concat([classical, neural.assign(model="MLP")])
    keys = ["model", "track_id"]
    reference = reference.set_index(keys, verify_integrity=True).sort_index()
    actual = predictions.set_index(keys, verify_integrity=True).sort_index()
    pd.testing.assert_index_equal(actual.index, reference.index)
    np.testing.assert_array_equal(actual[["hit", "prediction"]], reference[["hit", "prediction"]])
    np.testing.assert_allclose(actual.score, reference.score, atol=1e-7, rtol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--task4-run", type=Path, help="Prepare lock and replay validation")
    actions.add_argument("--test-lock", type=Path, help="Score test using a previously prepared frozen lock")
    args = parser.parse_args()
    lock = prepare(args.task4_run) if args.task4_run else read(args.test_lock)
    version = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()[:16]
    root = Path("results/task7")
    split = "test" if args.test_lock else "validation"
    if args.test_lock and (lock["status"] != "frozen" or args.test_lock.resolve() !=
                           (root / f"validation_{version}" / "lock.json").resolve()):
        parser.error("Final test requires the saved validation lock for a frozen training cohort")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{split}_{version}"
    if dest.exists():
        print(f"Existing results: {dest}")
        return
    metrics, predictions = score(lock, split)
    if split == "validation":
        replay(lock, predictions)
        substantive = metrics[metrics.model.isin(["Naive Bayes", "Decision Tree", "kNN", "MLP"])]
        preferred = substantive.iloc[0].model
    else:
        preferred = read(args.test_lock.parent / "selection.json")["preferred"]
    with TemporaryDirectory(dir=root) as temp:
        folder = Path(temp)
        metrics.to_csv(folder / "metrics.csv", index=False)
        predictions.to_csv(folder / "predictions.csv", index=False)
        (folder / "lock.json").write_text(json.dumps(lock, indent=2))
        (folder / "selection.json").write_text(json.dumps(dict(preferred=preferred, criterion="highest validation hit F1 among fitted non-dummy models")))
        lines = ["# Task 7 tabular evaluation", "", f"Split **{split}**. Cohort **{lock['status']}**. "
                 f"Tracks scored per model **{predictions.track_id.nunique():,}**. Preferred fitted model from validation **{preferred}**.",
                 "", "| Model | Hit F1 | Precision | Recall |", "|---|---:|---:|---:|"]
        for r in metrics.itertuples():
            lines.append(f"| {r.model} | {r.hit_f1:.3f} | {r.precision:.3f} | {r.recall:.3f} |")
        if split == "validation":
            lines += ["", "All six checkpoints reproduced their saved per-track validation labels and scores. "
                      "Final test evaluation uses a lock prepared from models trained on the reviewed cohort. "
                      "Ahmed’s spectrogram handoff is pending for the CNN comparison."]
        (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        folder.rename(dest)
    print(dest, preferred, len(predictions), flush=True)


if __name__ == "__main__":
    main()
