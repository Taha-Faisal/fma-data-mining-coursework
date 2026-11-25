"""Train and compare tabular classifiers on the task 2 validation split."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_inputs(bundle, features):
    ledger = json.loads((bundle / "bundle.json").read_text())
    status = ledger["status"]
    name = {"candidate": "manifest_candidate.csv", "frozen": "manifest.csv"}[status]
    for file in (name, "config.json"):
        if digest(bundle / file) != ledger["files"][file]:
            raise ValueError(f"Changed task 2 input: {file}")
    config = json.loads((bundle / "config.json").read_text())
    if digest(features) != config["inputs"]["features"]["sha256"]:
        raise ValueError("Feature table differs from the one used for task 2")
    manifest = pd.read_csv(bundle / name).set_index("track_id", verify_integrity=True)
    if manifest.groupby("group_id")["set_split"].nunique().max() != 1:
        raise ValueError("A group crosses splits")
    cohort = manifest.loc[manifest.set_split.isin(["training", "validation"])]
    parts = []
    with pd.read_csv(features, header=[0, 1, 2], index_col=0, chunksize=4000) as reader:
        for chunk in reader:
            parts.append(chunk.loc[chunk.index.isin(cohort.index)])
    matrix = pd.concat(parts, verify_integrity=True).loc[cohort.index]
    matrix.columns = ["audio__" + "__".join(col) for col in matrix.columns]
    if matrix.shape[1] != 518 or not matrix.columns.is_unique:
        raise ValueError("Unexpected audio feature schema")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    train = cohort.set_split.eq("training")
    matrix = matrix.loc[:, matrix.loc[train].notna().any()]
    return config, matrix.loc[train], cohort.loc[train, "hit"], matrix.loc[~train], cohort.loc[~train, "hit"]


def experiments():
    yield "Majority", {}, DummyClassifier(strategy="most_frequent")
    yield "Always hit", {}, DummyClassifier(strategy="constant", constant=1)
    for smoothing in (1e-9, 1e-8, 1e-7):
        yield "Naive Bayes", {"var_smoothing": smoothing}, GaussianNB(var_smoothing=smoothing)
    for depth in (5, 10, 20):
        settings = {"max_depth": depth, "min_samples_leaf": 20, "random_state": 42}
        yield "Decision Tree", settings, DecisionTreeClassifier(**settings)
    for k in (5, 15, 31):
        settings = {"n_neighbors": k, "weights": "distance", "algorithm": "brute"}
        yield "kNN", settings, KNeighborsClassifier(**settings)


def evaluate(labels, scores):
    predicted = scores >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return dict(hit_f1=f1_score(labels, predicted, zero_division=0),
                macro_f1=f1_score(labels, predicted, average="macro", zero_division=0),
                precision=precision_score(labels, predicted, zero_division=0),
                recall=recall_score(labels, predicted, zero_division=0),
                roc_auc=roc_auc_score(labels, scores), average_precision=average_precision_score(labels, scores),
                tn=tn, fp=fp, fn=fn, tp=tp)


def run(bundle, features, output):
    config, x, y, val, truth = load_inputs(bundle, features)
    output.mkdir(parents=True, exist_ok=True)
    status = config["status"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output / f"{status}_{stamp}"
    with TemporaryDirectory(dir=output) as temp, threadpool_limits(limits=4):
        folder = Path(temp)
        rows, winners, predictions = [], {}, {}
        for family, settings, model in experiments():
            print(f"{family} {settings}", flush=True)
            steps = [SimpleImputer(strategy="median")]
            if family in ("kNN", "Naive Bayes"):
                steps.append(StandardScaler())
            pipeline = make_pipeline(*steps, model)
            start = perf_counter()
            pipeline.fit(x, y)
            fit_seconds = perf_counter() - start
            scores = pipeline.predict_proba(val)[:, list(pipeline.classes_).index(1)]
            row = dict(model=family, settings=json.dumps(settings), fit_seconds=fit_seconds,
                       **evaluate(truth, scores))
            rows.append(row)
            if family in winners and row["hit_f1"] <= winners[family]["hit_f1"]:
                continue
            winners[family] = row
            predictions[family] = pd.DataFrame(dict(track_id=truth.index, model=family,
                                                     hit=truth.values, score=scores, prediction=(scores >= .5).astype(int)))
            joblib.dump(pipeline, folder / (family.replace(" ", "_") + ".joblib"))
        for family, table in predictions.items():
            fitted = joblib.load(folder / (family.replace(" ", "_") + ".joblib"))
            replay = fitted.predict_proba(val.iloc[:10])[:, list(fitted.classes_).index(1)]
            np.testing.assert_allclose(replay, table.score.iloc[:10], atol=1e-9)
        pd.DataFrame(rows).to_csv(folder / "validation_experiments.csv", index=False)
        selected = pd.DataFrame(winners.values()).sort_values("hit_f1", ascending=False)
        selected.to_csv(folder / "validation_results.csv", index=False)
        pd.concat(predictions.values()).to_csv(folder / "validation_predictions.csv", index=False)
        metadata = dict(status=status, candidate_id=config["candidate_id"], manifest_bundle=str(bundle),
                        feature_sha256=config["inputs"]["features"]["sha256"], target=config["target"],
                        score_threshold=0.5, selection="Highest validation hit F1 per family, first setting on ties",
                        training_tracks=len(x), validation_tracks=len(val), feature_columns=list(x.columns),
                        sklearn_version=sklearn.__version__, script_sha256=digest(Path(__file__)))
        (folder / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        lines = ["# Task 3 validation results", "",
                 f"Manifest status **{status}**. Training used {len(x):,} tracks and {x.shape[1]} audio descriptors. "
                 f"Validation used {len(val):,} tracks, including {truth.sum():,} hits. "
                 f"The task 2 listens cutoff is {config['target']['listens_cutoff']:g}.", "",
                 "| Model | Hit F1 | Precision | Recall | Average precision |", "|---|---:|---:|---:|---:|"]
        for r in selected.itertuples():
            lines.append(f"| {r.model} | {r.hit_f1:.3f} | {r.precision:.3f} | {r.recall:.3f} | {r.average_precision:.3f} |")
        lines += ["", "Selection used validation hit F1 at a score threshold of 0.5. "
                  "The search varied Bayes smoothing, tree depth and neighbour count. "
                  "Each pipeline fitted preprocessing on training tracks. Test tracks were held out.",
                  "", "The experiment table records all settings. The results table includes confusion counts, "
                  "macro F1 and ROC AUC. Saved pipelines use the feature order in run.json."]
        if status == "candidate":
            lines += ["", "Provisional results. Repeat on the reviewed cohort before reporting final performance."]
        (folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        folder.rename(destination)
    print(destination)
    print(selected[["model", "hit_f1", "precision", "recall", "average_precision"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Task 2 bundle, defaulting to the latest frozen or candidate")
    parser.add_argument("--features", type=Path, default=Path("data/raw/fma_metadata/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/task3"))
    args = parser.parse_args()
    bundle = args.bundle
    if bundle is None:
        root = Path("data/processed/task2")
        pointer = root / "LATEST_FROZEN.json"
        if not pointer.exists():
            pointer = root / "LATEST_CANDIDATE.json"
        bundle = root / json.loads(pointer.read_text())["bundle"]
    run(bundle, args.features, args.output)
