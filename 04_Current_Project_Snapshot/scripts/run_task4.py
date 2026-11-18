"""Train the task 4 MLP and interpret its validation predictions.

AI-generated code (OpenAI Codex). Full user prompt: "implement task 4"
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_task3 import digest, evaluate, load_inputs


def train(x, y, val, truth, layers, k):
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         SelectKBest(f_classif, k=k))
    a = prep.fit_transform(x, y).astype("float32")
    b = prep.transform(val).astype("float32")
    model = MLPClassifier(hidden_layer_sizes=layers, solver="adam", activation="relu",
                          batch_size=128, alpha=.001, learning_rate_init=.001,
                          random_state=np.random.RandomState(42))
    history = []
    best_f1 = -1.
    for epoch in range(1, 41):
        model.partial_fit(a, y, classes=[0, 1])
        scores = model.predict_proba(b)[:, 1]
        f1 = f1_score(truth, scores >= .5, zero_division=0)
        history.append(dict(epoch=epoch, train_log_loss=log_loss(y, model.predict_proba(a)[:, 1]),
                            validation_log_loss=log_loss(truth, scores), validation_hit_f1=f1))
        if f1 > best_f1:
            best_f1, best_epoch, best_model = f1, epoch, deepcopy(model)
        if epoch - best_epoch >= 5:
            break
    return make_pipeline(prep, best_model), best_epoch, history


def family_importance(pipe, val, truth, baseline):
    # One permutation per family preserves relationships between its descriptors.
    families = pd.Series([col.split("__")[1] for col in val.columns], index=val.columns)
    rng = np.random.default_rng(42)
    permutations = [rng.permutation(len(val)) for _ in range(5)]
    rows = []
    for family in sorted(families.unique()):
        cols = families.index[families.eq(family)]
        drops = []
        for order in permutations:
            shuffled = val.copy()
            shuffled[cols] = val.iloc[order][cols].to_numpy()
            score = pipe.predict_proba(shuffled)[:, 1]
            drops.append(baseline - f1_score(truth, score >= .5, zero_division=0))
        rows.append(dict(family=family, descriptors=len(cols), mean_f1_drop=np.mean(drops),
                         shuffle_sd=np.std(drops, ddof=1)))
    return pd.DataFrame(rows).sort_values("mean_f1_drop", ascending=False)


def run(previous, features, output):
    prior = json.loads((previous / "run.json").read_text())
    config, x, y, val, truth = load_inputs(Path(prior["manifest_bundle"]), features)
    if (config["candidate_id"] != prior["candidate_id"] or list(x.columns) != prior["feature_columns"]
            or prior["score_threshold"] != .5):
        raise ValueError("Task 3 comparison inputs differ")
    comparison = pd.read_csv(previous / "validation_results.csv")
    trials = [("64_all", (64,), "all"), ("128_64_all", (128, 64), "all"),
              ("128_64_top100", (128, 64), 100)]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / (config["status"] + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    with TemporaryDirectory(dir=output) as temp, threadpool_limits(limits=4):
        folder = Path(temp)
        rows, histories = [], []
        best = -1.
        for name, layers, k in trials:
            pipe, epoch, history = train(x, y, val, truth, layers, k)
            scores = pipe.predict_proba(val)[:, 1]
            metrics = evaluate(truth, scores)
            rows.append(dict(trial=name, best_epoch=epoch, epochs_run=len(history), **metrics))
            histories.append(pd.DataFrame(history).assign(trial=name))
            print(name, "validation F1", round(metrics["hit_f1"], 4), flush=True)
            if metrics["hit_f1"] > best:
                best, winner, chosen, predictions = metrics["hit_f1"], name, pipe, scores
        joblib.dump(chosen, folder / "mlp.joblib")
        np.testing.assert_allclose(joblib.load(folder / "mlp.joblib").predict_proba(val)[:, 1], predictions)
        pd.DataFrame(rows).to_csv(folder / "experiments.csv", index=False)
        pd.DataFrame(dict(track_id=truth.index, hit=truth.values, score=predictions,
                          prediction=(predictions >= .5).astype(int))).to_csv(folder / "validation_predictions.csv", index=False)
        metrics = evaluate(truth, predictions)
        comparison = pd.concat([comparison, pd.DataFrame([dict(model="MLP", settings=winner, **metrics)])],
                               ignore_index=True).sort_values("hit_f1", ascending=False)
        comparison.to_csv(folder / "comparison.csv", index=False)
        history = pd.concat(histories, ignore_index=True)
        history.to_csv(folder / "learning_curves.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for name, data in history.groupby("trial"):
            line, = axes[0].plot(data.epoch, data.validation_log_loss, label=name)
            axes[0].plot(data.epoch, data.train_log_loss, linestyle="--", color=line.get_color())
            axes[1].plot(data.epoch, data.validation_hit_f1, label=name)
        axes[0].set(xlabel="Epoch", ylabel="Log loss", title="Training (dashed) and validation (solid)")
        axes[1].set(xlabel="Epoch", ylabel="Hit F1", title="Validation at threshold 0.5")
        axes[1].legend()
        fig.savefig(folder / "learning_curves.png", dpi=150)
        plt.close(fig)
        importance = family_importance(chosen, val, truth, best)
        importance.to_csv(folder / "feature_family_importance.csv", index=False)
        metadata = dict(task3_run=str(previous), candidate_id=config["candidate_id"], status=config["status"],
                        selected=winner, feature_columns=list(x.columns), settings=dict(seed=42, threshold=.5,
                        max_epochs=40, patience=5, alpha=.001, learning_rate=.001, batch_size=128,
                        optimizer="Adam", activation="ReLU", loss="binary log loss", selection="highest validation hit F1, first on ties"),
                        script_sha256=digest(Path(__file__)))
        (folder / "run.json").write_text(json.dumps(metadata, indent=2))
        lines = ["# Task 4 MLP validation", "", f"Manifest status **{config['status']}**. Training used {len(y):,} tracks and validation used {len(truth):,}.",
                 "", "| Trial | Selected epoch | Hit F1 | Precision | Recall | Average precision |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append("| {trial} | {best_epoch} | {hit_f1:.3f} | {precision:.3f} | {recall:.3f} | {average_precision:.3f} |".format(**row))
        lines.extend(["", f"Selected **{winner}**. All trials used training-fitted median imputation and standardization. "
                      "The top-100 trial ranked individual descriptors using training-only ANOVA F scores. "
                      "Width and feature selection were compared at fixed regularization and learning rate.",
                      "", "![Training and validation learning curves](learning_curves.png)", "",
                      "The largest mean validation hit-F1 drops after shuffling descriptor families were " +
                      ", ".join(f"{r.family} ({r.mean_f1_drop:+.3f})" for r in importance.head(3).itertuples()) +
                      ". Correlations across families can mask or redistribute importance.",
                      "", "The same validation split selected checkpoints and settings. These scores describe that selection run. "
                      "The task 3 comparisons are in comparison.csv. Final evaluation requires the reviewed cohort and held-out test split."])
        (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        folder.rename(destination)
    print(destination, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task3-run", type=Path)
    parser.add_argument("--features", type=Path, default=Path("data/raw/fma_metadata/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/task4"))
    args = parser.parse_args()
    previous = args.task3_run
    if previous is None:
        runs = list(Path("results/task3").glob("*/run.json"))
        if not runs:
            parser.error("Run task 3 first or specify --task3-run")
        previous = max(runs, key=lambda path: path.stat().st_mtime_ns).parent
    run(previous, args.features, args.output)
