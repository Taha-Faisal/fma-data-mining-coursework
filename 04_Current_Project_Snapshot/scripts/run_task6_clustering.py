"""Compare FMA audio descriptor clusters on the shared validation split.

AI-generated code (OpenAI Codex). Full user prompt:
"that is ahmeds task to give me the spectograms can we move forward without it for now?"
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_task3 import digest, load_inputs


def run(bundle, output):
    info = json.loads((bundle / "config.json").read_text())
    info, x, _, val, _ = load_inputs(bundle, Path(info["inputs"]["features"]["path"]))
    cleaned = Path(info["inputs"]["cleaned_csv"]["path"])
    if digest(cleaned) != info["inputs"]["cleaned_csv"]["sha256"]:
        raise ValueError("Cleaned metadata differs from the task 2 input")
    genres = pd.read_csv(cleaned, usecols=["track_id", "track_genre_top"]).set_index("track_id").loc[val.index]
    sample = np.sort(np.random.default_rng(42).choice(len(val), size=min(2000, len(val)), replace=False))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / (info["status"] + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    with TemporaryDirectory(dir=output) as temp, threadpool_limits(limits=4):
        folder = Path(temp)
        prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             PCA(n_components=30, svd_solver="randomized", random_state=42))
        train = prep.fit_transform(x)
        validation = prep.transform(val)
        rows, selected = [], {}
        for family in ("k-means", "GMM"):
            best = -np.inf
            for k in (4, 8, 12, 16):
                model = (KMeans(n_clusters=k, n_init=10, random_state=42) if family == "k-means"
                         else GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                                              max_iter=200, random_state=42))
                model.fit(train)
                labels = model.predict(validation)
                occupied = len(np.unique(labels[sample]))
                score = (silhouette_score(validation[sample], labels[sample])
                         if 1 < occupied < len(sample) else np.nan)
                converged = model.converged_ if family == "GMM" else model.n_iter_ < model.max_iter
                rows.append(dict(model=family, k=k, validation_silhouette=score,
                                 occupied_sample_clusters=occupied, iterations=model.n_iter_,
                                 converged=bool(converged),
                                 training_bic=model.bic(train) if family == "GMM" else np.nan))
                if converged and score > best:
                    best = score
                    selected[family] = (k, score, model, labels)
            print(f"{family} comparison finished", flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(folder / "scores.csv", index=False)
        pd.Series(val.index[sample], name="track_id").to_csv(folder / "validation_sample.csv", index=False)
        lines = ["# Audio descriptor clustering", "", f"Manifest status **{info['status']}**. "
                 f"Fitted on {len(x):,} training tracks. Scores use the same {len(sample):,} validation tracks.",
                 "", "| Model | Selected clusters | Validation silhouette |", "|---|---:|---:|"]
        for family, (k, score, model, labels) in selected.items():
            pipe = make_pipeline(prep, model)
            path = folder / f"{family}.joblib"
            joblib.dump(pipe, path)
            np.testing.assert_array_equal(joblib.load(path).predict(val), labels)
            assignments = genres.assign(cluster=labels)
            assignments.to_csv(folder / f"{family}_validation.csv")
            pd.crosstab(assignments.cluster, assignments.track_genre_top.fillna("Unknown"),
                        margins=True).to_csv(folder / f"{family}_cluster_genres.csv")
            lines.append(f"| {family} | {k} | {score:.3f} |")
        lines += ["", "The models use 30 PCA components of standardized audio descriptors. "
                  "The highest validation silhouette among converged settings selects each model. "
                  "Higher silhouette indicates tighter clusters with greater separation. Genre counts describe "
                  "the resulting clusters. Scores and BIC for all eight fits are in scores.csv.", "",
                  "These are descriptive results on the identified cohort. Low silhouette values indicate weak separation; genre is used only for interpretation, never as a clustering input."]
        (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        (folder / "run.json").write_text(json.dumps(dict(manifest_id=info["candidate_id"], bundle=str(bundle),
            seed=42, pca_variance_explained=prep[-1].explained_variance_ratio_.sum(),
            feature_names=list(x.columns), script_sha256=digest(Path(__file__)))), encoding="utf-8")
        folder.rename(destination)
    print(destination)
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/task6_clustering"))
    args = parser.parse_args()
    if args.bundle is None:
        root = Path("data/processed/task2")
        pointer = root / "LATEST_FROZEN.json"
        if not pointer.exists():
            pointer = root / "LATEST_CANDIDATE.json"
        args.bundle = root / json.loads(pointer.read_text())["bundle"]
    run(args.bundle, args.output)
