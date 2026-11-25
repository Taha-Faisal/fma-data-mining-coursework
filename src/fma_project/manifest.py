"""Task 2: deterministic grouped splits and a shared, versioned label manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .audit import (AuditError, SPLITS, input_record, json_safe, load_cleaned,
                    normalized_names, provisional_labels, sha256, source_audit, track_index)


DECISION_COLUMNS = ["track_id", "eligibility", "audio_status", "reason"]
REVIEW_AREAS = ("cleaning", "eligibility", "audio", "related_recordings")
LINK_COLUMNS = ["track_id_a", "track_id_b", "link_type", "evidence"]


def load_decisions(path, index):
    if path is None:
        return pd.DataFrame({"eligibility": "pending", "audio_status": "pending", "reason": ""}, index=index)
    data = pd.read_csv(path, dtype="string", keep_default_na=False)
    if list(data.columns) != DECISION_COLUMNS:
        raise AuditError(f"Track decisions require these columns in order: {DECISION_COLUMNS}")
    data.index = track_index(data.pop("track_id"), "track decisions")
    if set(data.index) != set(index):
        raise AuditError("Track decisions must cover every cleaned track exactly once, with no extra IDs")
    for column in data:
        data[column] = data[column].str.strip()
    if not data["eligibility"].isin(["pending", "eligible", "excluded"]).all():
        raise AuditError("Eligibility must be pending, eligible or excluded")
    if not data["audio_status"].isin(["pending", "ok", "excluded"]).all():
        raise AuditError("Audio status must be pending, ok or excluded")
    excluded = data[["eligibility", "audio_status"]].eq("excluded").any(axis=1)
    if data.loc[excluded, "reason"].eq("").any():
        raise AuditError("Every explicit exclusion needs a reason")
    return data.reindex(index)


def feature_inventory(path, index, chunksize=5000):
    """Check coverage and hash fully finite descriptor rows for conservative grouping."""
    seen, parts, count = set(), [], None
    with pd.read_csv(path, header=[0, 1, 2], index_col=0, chunksize=chunksize) as reader:
        for chunk in reader:
            chunk.index = track_index(chunk.index, "features.csv")
            if seen.intersection(chunk.index):
                raise AuditError("features.csv contains duplicate IDs across chunks")
            seen.update(chunk.index)
            count = len(chunk.columns)
            selected = chunk.loc[chunk.index.isin(index)]
            values = selected.apply(pd.to_numeric, errors="coerce")
            if (selected.notna() & values.isna()).any().any():
                raise AuditError("Nonnumeric audio descriptors must be resolved before task 2")
            array = values.to_numpy(dtype="<f8", copy=True)
            finite = np.isfinite(array)
            # Make +0 and -0 equivalent before hashing. Do not hash incomplete rows.
            array[array == 0] = 0
            fingerprints = [hashlib.sha256(row.tobytes()).hexdigest() if valid.all() else pd.NA
                            for row, valid in zip(array, finite)]
            parts.append(pd.DataFrame({"finite_features": finite.sum(axis=1),
                                       "feature_fingerprint": fingerprints}, index=selected.index))
    if not parts or not count:
        raise AuditError("The audio feature table is empty")
    result = pd.concat(parts).reindex(index)
    result["feature_row_present"] = result.index.isin(seen)
    result["finite_features"] = result["finite_features"].fillna(0).astype(int)
    return result, count


def load_related_links(path, allowed_ids):
    if path is None:
        return pd.DataFrame(columns=["track_id_a", "track_id_b", "reason"])
    data = pd.read_csv(path, dtype="string", keep_default_na=False)
    if set(data.columns) != {"track_id_a", "track_id_b", "reason"}:
        raise AuditError("Related links require track_id_a, track_id_b and reason")
    for column in ("track_id_a", "track_id_b"):
        # IDs may appear in several different edges, so validate the unique values.
        track_index(data[column].unique(), f"related links {column}")
        data[column] = pd.to_numeric(data[column]).astype("int64")
        if not set(data[column]).issubset(allowed_ids):
            raise AuditError("Related links reference IDs outside the cleaned dataset")
    if data["reason"].str.strip().eq("").any():
        raise AuditError("Related-record links require evidence/reasons")
    return data


def connected_groups(frame, related):
    """Keep all tracks connected by any known dependency in the same group."""
    parent = {int(i): int(i) for i in frame.index}
    evidence = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def link(a, b, kind, value):
        a, b = int(a), int(b)
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
        evidence.append((a, b, kind, str(value)))

    for column in ("artist_id", "album_id", "artist_name_normalized", "feature_fingerprint"):
        for value, indexes in frame.groupby(column, dropna=True, sort=True).groups.items():
            indexes = sorted(indexes)
            for tid in indexes[1:]:
                link(indexes[0], tid, column, value)
    for row in related.itertuples(index=False):
        if row.track_id_a in parent and row.track_id_b in parent:
            link(row.track_id_a, row.track_id_b, "reviewed_related_record", row.reason)
    groups = pd.Series({i: find(i) for i in sorted(parent)}, name="group_id")
    groups.index.name = "track_id"
    return groups, pd.DataFrame(evidence, columns=LINK_COLUMNS)


def assign_splits(groups, original, seed=42):
    """Preserve a passing split; otherwise use two predeclared group shuffles."""
    groups = groups.sort_index()
    original = original.reindex(groups.index)
    if groups.nunique() < 3:
        raise AuditError("At least three independent groups are needed for three splits")
    passing = set(original.dropna()) == set(SPLITS) and original.notna().all()
    passing = passing and pd.DataFrame({"group": groups, "split": original}).groupby("group")["split"].nunique().max() == 1
    if passing:
        return original.rename("set_split"), "preserved_existing_split"
    positions = np.zeros((len(groups), 1))
    train, held = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(positions, groups=groups))
    held_groups = groups.iloc[held]
    if held_groups.nunique() < 2:
        raise AuditError("The 20% group holdout contains fewer than two groups; cohort is too small")
    validation, test = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1).split(
        positions[held], groups=held_groups))
    splits = pd.Series(index=groups.index, dtype="string", name="set_split")
    splits.iloc[train] = "training"
    splits.iloc[held[validation]] = "validation"
    splits.iloc[held[test]] = "test"
    return splits, "group_shuffle_80_10_10"


def verify_manifest(frame, related):
    counts = {}
    for column in ("group_id", "artist_id", "album_id", "artist_name_normalized", "feature_fingerprint"):
        counts[column] = int(frame.groupby(column)["set_split"].nunique().gt(1).sum())
    counts["related_links"] = sum(
        frame.loc[r.track_id_a, "set_split"] != frame.loc[r.track_id_b, "set_split"]
        for r in related.itertuples(index=False)
        if r.track_id_a in frame.index and r.track_id_b in frame.index)
    if any(counts.values()):
        raise AuditError(f"A dependency crosses splits: {counts}")
    if not frame.index.is_unique or frame["set_split"].isna().any() or frame["hit"].isna().any():
        raise AuditError("Manifest has duplicate IDs or missing assignments/labels")
    if set(frame["set_split"]) != set(SPLITS):
        raise AuditError("All three splits must be nonempty")
    for split in SPLITS:
        if frame.loc[frame["set_split"].eq(split), "hit"].nunique() != 2:
            raise AuditError(f"{split} must contain both target classes; no alternative seed is selected automatically")
    return counts


def review_template(candidate_id):
    return {"candidate_id": candidate_id, "reviews": {
        area: {"status": "pending", "reviewed_by": "", "notes": "", "evidence": []}
        for area in REVIEW_AREAS}}


def validate_review(path, candidate_id):
    if path is None:
        raise AuditError("Freezing requires --review with completed evidence; a candidate can be generated now")
    path = Path(path)
    review = json.loads(path.read_text(encoding="utf-8-sig"))
    if review.get("candidate_id") != candidate_id:
        raise AuditError("Review belongs to a different candidate; regenerate after input/settings changes")
    if not isinstance(review.get("reviews"), dict):
        raise AuditError("Review must contain a reviews object")
    for area in REVIEW_AREAS:
        record = review["reviews"].get(area, {})
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise AuditError(f"The {area} review is not complete")
        if not str(record.get("reviewed_by", "")).strip() or not str(record.get("notes", "")).strip():
            raise AuditError(f"The {area} review needs reviewer and notes")
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise AuditError(f"The {area} review needs file evidence and SHA-256 hashes")
        for item in evidence:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise AuditError(f"Malformed evidence for {area}")
            file = path.parent / item["path"]
            if not file.is_file() or sha256(file) != item["sha256"]:
                raise AuditError(f"Missing or changed evidence for {area}: {item['path']}")
    return review


def _write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_bundle(root, name, candidate_id, tables, config, report, template):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    if destination.exists():
        commit = destination / "bundle.json"
        if not commit.is_file():
            raise AuditError(f"Incomplete existing output; refusing to overwrite {destination}")
        saved = json.loads(commit.read_text(encoding="utf-8"))
        if saved.get("candidate_id") != candidate_id or saved.get("status") != config["status"]:
            raise AuditError("Existing output identity does not match this run")
        for filename, digest in saved["files"].items():
            file = (destination / filename).resolve()
            if not file.is_relative_to(destination) or not file.is_file() or sha256(file) != digest:
                raise AuditError(f"Existing artifact has changed: {filename}; refusing to overwrite it")
        return destination
    # Stage beneath the requested output root and publish only after every file is written.
    with tempfile.TemporaryDirectory(prefix=".task2-", dir=root) as tmp:
        staging = Path(tmp)
        for filename, table in tables.items():
            table.to_csv(staging / filename, index=False)
        _write_json(staging / "config.json", config)
        _write_json(staging / "review_template.json", template)
        (staging / "TASK2_REPORT.md").write_text(report, encoding="utf-8")
        files = {p.name: sha256(p) for p in sorted(staging.iterdir())}
        _write_json(staging / "bundle.json", {"candidate_id": candidate_id, "status": config["status"], "files": files})
        staging.rename(destination)
    return destination


def build_task2(cleaned, tracks, features, output, decisions=None, related_links=None,
                seed=42, freeze=False, review=None):
    if not 0 <= seed < 2**32 - 1:
        raise AuditError("Seed must be between 0 and 4294967294")
    frame = load_cleaned(cleaned).sort_index()
    provisional_labels(frame)  # Validate original splits and all target values before selecting a cohort.
    decision_table = load_decisions(decisions, frame.index)
    related = load_related_links(related_links, set(frame.index))
    ids, _, source_changes, source_info = source_audit(frame, tracks)
    if source_info["status"] != "checked":
        raise AuditError("Original tracks.csv is required for task 2")
    if source_info["cleaned_ids_missing_from_original"] or source_info["cleaned_ids_outside_medium"]:
        raise AuditError("Resolve cleaned IDs absent from the original medium cohort")
    if not source_changes.empty:
        raise AuditError("Resolve changes to source listens/duration/split/subset before task 2")
    inventory, feature_count = feature_inventory(features, frame.index)
    if feature_count != 518:
        raise AuditError(f"Expected 518 audio descriptors; found {feature_count}")
    if "artist_name" not in frame:
        raise AuditError("artist_name is required to keep repeated names together")
    cohort = frame.drop(columns=["artist_id", "album_id"], errors="ignore").join(ids[["artist_id", "album_id"]]).join(inventory).join(decision_table)
    reasons = []
    for row in cohort.itertuples():
        reason = []
        if row.eligibility == "excluded": reason.append("eligibility: " + row.reason)
        if row.audio_status == "excluded": reason.append("audio: " + row.reason)
        if not row.feature_row_present: reason.append("missing_feature_row")
        elif row.finite_features == 0: reason.append("no_finite_audio_features")
        reasons.append("; ".join(reason))
    cohort["exclusion_reason"] = reasons
    included = cohort[cohort["exclusion_reason"].eq("")].copy()
    if included.empty:
        raise AuditError("No usable tracks remain after recorded exclusions")
    if included[["artist_id", "album_id"]].isna().any().any():
        raise AuditError("Resolve missing/sentinel artist and album IDs for retained tracks")
    included["artist_name_normalized"] = normalized_names(included["artist_name"])
    included["source_split"] = included["set_split"]
    groups, links = connected_groups(included, related)
    included["group_id"] = groups
    included["set_split"], method = assign_splits(groups, included["source_split"], seed)
    cutoff, labels = provisional_labels(included)
    included["hit"] = labels
    included["split_changed"] = included["set_split"].ne(included["source_split"])
    overlaps = verify_manifest(included, related)
    inputs = {name: input_record(path) for name, path in {
        "cleaned_csv": cleaned, "original_tracks": tracks, "features": features,
        "track_decisions": decisions, "related_links": related_links}.items()}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")}
    identity = {"schema_version": 1, "source_sha256": sha256(Path(__file__)),
                "audit_source_sha256": sha256(Path(__file__).with_name("audit.py")),
                "input_hashes": {key: value.get("sha256") for key, value in inputs.items()},
                "grouping": ["artist_id", "album_id", "normalized_artist_name", "exact_finite_feature_vector", "related_links"],
                "split_policy": "preserve_if_group_disjoint_else_group_shuffle", "seed": seed,
                "holdout_seed": seed + 1, "target_group_fractions": [0.8, 0.1, 0.1],
                "quantile": 0.75, "interpolation": "linear", "versions": versions}
    candidate_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    pending_eligibility = int(included["eligibility"].ne("eligible").sum())
    pending_audio = int(included["audio_status"].ne("ok").sum())
    status = "frozen" if freeze else "candidate"
    completed_review = None
    if freeze:
        if pending_eligibility or pending_audio:
            raise AuditError(f"Cannot freeze: {pending_eligibility} retained eligibility decisions and "
                             f"{pending_audio} retained audio statuses are still pending")
        completed_review = validate_review(review, candidate_id)
    included["manifest_status"] = status
    stats = []
    for split in SPLITS:
        part = included[included["set_split"].eq(split)]
        stats.append({"split": split, "tracks": len(part), "track_pct": 100 * len(part) / len(included),
                      "groups": part["group_id"].nunique(), "artists": part["artist_id"].nunique(),
                      "albums": part["album_id"].nunique(), "hits": int(part["hit"].sum()),
                      "hit_pct": 100 * part["hit"].mean()})
    split_summary = pd.DataFrame(stats)
    group_summary = included.groupby("group_id").agg(
        tracks=("group_id", "size"), artists=("artist_id", "nunique"), albums=("album_id", "nunique"),
        source_splits=("source_split", lambda x: ";".join(sorted(set(x)))), set_split=("set_split", "first")).reset_index()
    manifest_columns = ["artist_id", "album_id", "group_id", "set_split", "hit", "track_listens",
                        "source_split", "split_changed", "manifest_status"]
    manifest = included[manifest_columns].reset_index()
    manifest_name = "manifest.csv" if freeze else "manifest_candidate.csv"
    blockers = [] if freeze else [
        f"Retained tracks with eligibility pending: {pending_eligibility}.",
        f"Retained tracks with audio verification pending: {pending_audio}.",
        "Attach the cleaning review for this cohort when freezing.",
        "Record eligibility, audio and related-recording reviews with file evidence before freezing.",
    ]
    config = {"candidate_id": candidate_id, "status": status, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": method, "identity": identity, "inputs": inputs,
              "cohort": {"input_tracks": len(frame), "included_tracks": len(included),
                         "excluded_tracks": len(frame) - len(included), "groups": int(groups.nunique()),
                         "largest_group": int(groups.value_counts().max()),
                         "changed_split_tracks": int(included["split_changed"].sum())},
              "target": {"listens_cutoff": cutoff, "minimum_integer_listens": int(np.ceil(cutoff)),
                         "rule": "track_listens >= training_quantile_0.75", "fitted_on": "training"},
              "overlap_counts_after": overlaps, "split_summary": stats,
              "pending_eligibility": pending_eligibility, "pending_audio": pending_audio,
              "open_items": blockers, "review": completed_review,
              "review_file": input_record(review) if freeze else None}
    report = ["# Task 2 shared split and label manifest", "",
              f"Status: **{status}**. Candidate ID: `{candidate_id}`.", "",
              f"Retained {len(included):,} of {len(frame):,} cleaned tracks in {groups.nunique():,} connected groups. "
              f"Recorded exclusions: {len(frame) - len(included):,}. Split method: `{method}`.", "",
              "| Split | Tracks | Track share | Groups | Hits | Hit share |", "|---|---:|---:|---:|---:|---:|"]
    for row in stats:
        report.append(f"| {row['split']} | {row['tracks']:,} | {row['track_pct']:.2f}% | {row['groups']:,} | "
                      f"{row['hits']:,} | {row['hit_pct']:.2f}% |")
    report += ["", f"The training-only 75th percentile is **{cutoff:g} listens** "
               f"(at least **{int(np.ceil(cutoff)):,}** for integer counts). This one cutoff labels every split.", "",
               "## Grouping and verification", "",
               "Tracks sharing an artist ID, album ID, normalized artist name, identical fully finite descriptor "
               "vector, or supplied related-record link stay together, including transitive connections. "
               "Name/feature matches are conservative grouping evidence; they do not prove identical artists/audio.", "",
               f"Largest group: {groups.value_counts().max():,} tracks. "
               f"Tracks moved from their original split: {int(included['split_changed'].sum()):,}. "
               "All implemented group/artist/album/name/feature/link overlap checks pass, and each split has both classes.", "",
               "The 80/10/10 targets apply to groups, so track and class proportions can differ. "
               "Seeds 42/43 are the default; the recorded seeds are in config.json. "
               "Assignment does not use listens or hit labels. No seeds were searched for favourable class proportions or model scores.", "",
               "## Use and remaining work", ""]
    report += [f"- {item}" for item in blockers] if blockers else ["Recorded review evidence was validated before freezing."]
    report += ["", f"Use `{manifest_name}` with `config.json`. Preserve `group_id` in later group-aware validation. "
               "Identifiers, split fields, listens and the hit label are not model inputs. "
               "The link, exclusion and split-change tables retain the audit trail.", "",
               "Audio files have not been inspected by this script. Near-duplicate recordings with different descriptors "
               "may remain unidentified. Candidate outputs are for review/integration checks, not final reported evaluation. "
               "No predictive model was trained or evaluated.", "",
               "Grouping/split reference: [scikit-learn GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).", ""]
    tables = {manifest_name: manifest, "split_summary.csv": split_summary,
              "group_summary.csv": group_summary, "grouping_links.csv": links,
              "split_changes.csv": manifest.loc[manifest["split_changed"]],
              "exclusions.csv": cohort.loc[cohort["exclusion_reason"].ne(""), ["exclusion_reason"]].reset_index(),
              "cohort_decisions.csv": cohort[["eligibility", "audio_status", "reason", "feature_row_present",
                                               "finite_features", "exclusion_reason"]].reset_index(),
              "track_decisions_template.csv": decision_table.reset_index()}
    suffix = ""
    if freeze:
        suffix = "_" + sha256(Path(review))[:8]
    output = Path(output).resolve()
    for record in inputs.values():
        if record.get("path") and Path(record["path"]).resolve().is_relative_to(output):
            raise AuditError("Keep source/review input files outside the generated output folder")
    if review and Path(review).resolve().is_relative_to(output):
        raise AuditError("Copy completed review files outside the generated output folder")
    destination = save_bundle(output, f"{status}_{candidate_id[:12]}{suffix}", candidate_id,
                              tables, config, "\n".join(report), review_template(candidate_id))
    _write_json(output / f"LATEST_{status.upper()}.json", {
        "candidate_id": candidate_id, "status": status, "bundle": destination.name,
        "manifest": f"{destination.name}/{manifest_name}", "report": f"{destination.name}/TASK2_REPORT.md"})
    return destination, config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build task 2 candidate or freeze an evidence-reviewed shared manifest.")
    parser.add_argument("--cleaned", type=Path, default=Path("data/raw/fma_medium_clean2.csv"))
    parser.add_argument("--tracks", type=Path, default=Path("data/raw/fma_metadata/tracks.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/raw/fma_metadata/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/task2"))
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--related-links", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    try:
        print("Building shared groups, splits and training-only labels...", flush=True)
        path, config = build_task2(args.cleaned, args.tracks, args.features, args.output,
                                  args.decisions, args.related_links, args.seed, args.freeze, args.review)
    except (AuditError, OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Task 2 failed: {error}", file=sys.stderr)
        return 2
    print(f"Status: {config['status']}; included tracks: {config['cohort']['included_tracks']:,}")
    print(f"Report: {path / 'TASK2_REPORT.md'}")
    print(f"Training listens cutoff: {config['target']['listens_cutoff']}")
    return 0
