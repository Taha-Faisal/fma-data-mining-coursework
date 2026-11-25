"""Task 1: inventory and audit the data without changing the modelling cohort."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import unicodedata

import numpy as np
import pandas as pd


SPLITS = ("training", "validation", "test")
EXPECTED_COLUMNS = (
    "track_id", "track_title", "artist_name", "album_title", "track_genre_top",
    "track_genres", "track_duration", "track_bit_rate", "track_listens",
    "track_favorites", "track_interest", "track_date_created", "album_date_released",
    "track_language_code", "track_license", "track_tags", "set_subset", "set_split",
)
NUMERIC_COLUMNS = (
    "track_duration", "track_bit_rate", "track_listens", "track_favorites", "track_interest",
)
OVERLAP_COLUMNS = ["group_type", "group_value", "track_id", "set_split"]
DIFFERENCE_COLUMNS = ["track_id", "field", "cleaned_value", "source_value"]
REMOVED_COLUMNS = [
    "track_id", "set_subset", "set_split", "track_listens", "track_duration",
    "below_observed_min_listens", "outside_observed_duration_range", "removal_reason",
]
FEATURE_COLUMNS = [
    "feature", "missing_or_nonfinite", "nonfinite", "nonnumeric",
    "training_missing", "training_missing_pct", "all_training_missing",
]


class AuditError(ValueError):
    """An input cannot be interpreted safely for this audit."""


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_record(path: Path | None) -> dict:
    if path is None:
        return {"status": "not_supplied", "path": None}
    path = Path(path)
    if not path.is_file():
        return {"status": "missing", "path": path.as_posix()}
    return {"status": "present", "path": path.as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def track_index(values, source: str) -> pd.Index:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.gt(0) & numeric.mod(1).eq(0)
    valid &= numeric.le(np.iinfo(np.int64).max)
    if not valid.all():
        raise AuditError(f"{source}: track IDs must be positive integers; invalid IDs found")
    index = pd.Index(numeric.astype("int64"), name="track_id")
    if not index.is_unique:
        duplicate_ids = index[index.duplicated()].unique().tolist()[:10]
        raise AuditError(f"{source}: duplicate track IDs (examples: {duplicate_ids})")
    return index


def load_cleaned(path: Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)) or any(not c.strip() for c in header):
        raise AuditError("Cleaned CSV has an empty or duplicate column header")
    required = {"track_id", "track_listens", "set_split"}
    if not required.issubset(header):
        raise AuditError(f"Cleaned CSV is missing columns: {sorted(required - set(header))}")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""])
    if frame.empty:
        raise AuditError("Cleaned CSV contains no tracks")
    # Whitespace-only cells count as missing; text such as an artist named 'NA' is retained.
    for column in frame:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    frame.index = track_index(frame.pop("track_id"), str(path))
    return frame


def numeric_values(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("float64")


def provisional_labels(frame: pd.DataFrame) -> tuple[float, pd.Series]:
    """Use one linear training quartile, never a separate evaluation-set quartile."""
    if frame["set_split"].isna().any() or set(frame["set_split"]) != set(SPLITS):
        raise AuditError("Expected all three splits: training, validation, test, with no unknown splits")
    listens = numeric_values(frame["track_listens"])
    valid = np.isfinite(listens) & listens.ge(0) & listens.mod(1).eq(0)
    if not valid.all():
        raise AuditError("Listen counts must be finite nonnegative integers with no missing values")
    cutoff = float(listens[frame["set_split"].eq("training")].quantile(0.75, interpolation="linear"))
    return cutoff, listens.ge(cutoff).astype("int8").rename("provisional_hit")


def normalized_names(values: pd.Series) -> pd.Series:
    def normalize(value):
        if pd.isna(value):
            return pd.NA
        return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split()) or pd.NA
    return values.map(normalize).astype("string")


def group_overlaps(groups: pd.Series, splits: pd.Series, kind: str) -> pd.DataFrame:
    table = pd.DataFrame({"group_value": groups, "set_split": splits}).dropna()
    counts = table.groupby("group_value")["set_split"].nunique()
    table = table[table["group_value"].isin(counts[counts.gt(1)].index)].copy()
    table["group_type"] = kind
    return table.reset_index()[OVERLAP_COLUMNS].sort_values(["group_value", "track_id"])


def source_audit(frame: pd.DataFrame, path: Path | None):
    ids = pd.DataFrame(index=frame.index, columns=["artist_id", "album_id"], dtype="Int64")
    removed = pd.DataFrame(columns=REMOVED_COLUMNS)
    differences = pd.DataFrame(columns=DIFFERENCE_COLUMNS)
    info = {"status": "not_available"}
    if path is None or not Path(path).is_file():
        return ids, removed, differences, info
    tracks = pd.read_csv(path, header=[0, 1], index_col=0, dtype="string",
                         keep_default_na=False, na_values=[""])
    tracks.index = track_index(tracks.index, str(path))
    needed = [("artist", "id"), ("album", "id"), ("set", "subset"), ("set", "split"),
              ("track", "listens"), ("track", "duration")]
    if any(column not in tracks for column in needed):
        raise AuditError("Original tracks.csv is missing required two-level metadata columns")
    aligned = tracks.reindex(frame.index)
    ids["source_subset"] = aligned[("set", "subset")]
    for name in ("artist", "album"):
        values = numeric_values(aligned[(name, "id")])
        valid = np.isfinite(values) & values.gt(0) & values.mod(1).eq(0)
        ids[name + "_id"] = values.where(valid).astype("Int64")
    comparisons = {
        "track_listens": ("track", "listens"), "track_duration": ("track", "duration"),
        "set_split": ("set", "split"), "set_subset": ("set", "subset"),
        "artist_id": ("artist", "id"), "album_id": ("album", "id"),
    }
    parts = []
    matched = frame.index.isin(tracks.index)
    for clean_name, original_name in comparisons.items():
        if clean_name not in frame:
            continue
        left, right = frame[clean_name], aligned[original_name]
        if clean_name in NUMERIC_COLUMNS:
            left, right = numeric_values(left), numeric_values(right)
        equal = left.eq(right).fillna(False) | (left.isna() & right.isna())
        mask = matched & ~equal
        parts.append(pd.DataFrame({"track_id": frame.index[mask], "field": clean_name,
                                   "cleaned_value": frame.loc[mask, clean_name].to_numpy(),
                                   "source_value": aligned.loc[mask, original_name].to_numpy()}))
    differences = pd.concat(parts, ignore_index=True) if parts else differences
    # FMA subsets are nested: small AND medium labels comprise the medium cohort.
    medium = tracks[tracks[("set", "subset")].isin(["small", "medium"])]
    omitted = medium.loc[~medium.index.isin(frame.index)]
    removed = pd.DataFrame({
        "track_id": omitted.index,
        "set_subset": omitted[("set", "subset")].to_numpy(),
        "set_split": omitted[("set", "split")].to_numpy(),
        "track_listens": numeric_values(omitted[("track", "listens")]).to_numpy(),
        "track_duration": numeric_values(omitted[("track", "duration")]).to_numpy(),
        "removal_reason": "unknown_without_cleaning_history",
    })
    clean_listens = numeric_values(frame["track_listens"]).replace([np.inf, -np.inf], np.nan)
    removed["below_observed_min_listens"] = removed["track_listens"].lt(clean_listens.min())
    if "track_duration" in frame:
        durations = numeric_values(frame["track_duration"]).replace([np.inf, -np.inf], np.nan)
        removed["outside_observed_duration_range"] = ~removed["track_duration"].between(
            durations.min(), durations.max())
    else:
        removed["outside_observed_duration_range"] = pd.NA
    info = {
        "status": "checked", "original_rows": len(tracks), "original_medium_rows": len(medium),
        "removed_medium_rows": len(removed),
        "cleaned_ids_missing_from_original": frame.index.difference(tracks.index).tolist(),
        "cleaned_ids_outside_medium": frame.index.difference(medium.index).tolist(),
        "differing_field_values": len(differences),
        "missing_or_invalid_artist_ids": int(ids["artist_id"].isna().sum()),
        "missing_or_invalid_album_ids": int(ids["album_id"].isna().sum()),
    }
    return ids, removed[REMOVED_COLUMNS], differences, info


def feature_audit(frame: pd.DataFrame, path: Path | None, chunksize: int = 5000):
    quality = pd.DataFrame(columns=FEATURE_COLUMNS)
    availability = pd.DataFrame(index=frame.index)
    availability["feature_row_present"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    availability["finite_feature_count"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if path is None or not Path(path).is_file():
        return availability, quality, {"status": "not_available"}
    seen, parts, feature_names = set(), [], None
    with pd.read_csv(path, header=[0, 1, 2], index_col=0, chunksize=chunksize) as reader:
        for chunk in reader:
            chunk.index = track_index(chunk.index, str(path))
            if seen.intersection(chunk.index):
                raise AuditError("features.csv contains duplicate track IDs across chunks")
            seen.update(chunk.index)
            columns = ["audio__" + "__".join(map(str, c)) for c in chunk.columns]
            if len(columns) != len(set(columns)):
                raise AuditError("features.csv contains ambiguous flattened feature names")
            feature_names = columns
            chunk.columns = columns
            parts.append(chunk.loc[chunk.index.isin(frame.index)])
    if feature_names is None:
        raise AuditError("features.csv contains no data rows")
    raw = pd.concat(parts).reindex(frame.index)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    nonnumeric = raw.notna() & numeric.isna()
    nonfinite = numeric.isin([np.inf, -np.inf])
    usable = numeric.replace([np.inf, -np.inf], np.nan)
    training = usable.loc[frame["set_split"].eq("training")]
    quality = pd.DataFrame({
        "feature": feature_names,
        "missing_or_nonfinite": usable.isna().sum().to_numpy(),
        "nonfinite": nonfinite.sum().to_numpy(),
        "nonnumeric": nonnumeric.sum().to_numpy(),
        "training_missing": training.isna().sum().to_numpy(),
        "training_missing_pct": (100 * training.isna().mean()).to_numpy(),
        "all_training_missing": training.isna().all().to_numpy(),
    })
    availability["feature_row_present"] = frame.index.isin(seen)
    availability["finite_feature_count"] = usable.notna().sum(axis=1)
    missing = availability.index[~availability["feature_row_present"]].tolist()
    empty = availability.index[availability["feature_row_present"] &
                               availability["finite_feature_count"].eq(0)].tolist()
    info = {
        "status": "checked", "original_rows": len(seen), "feature_count": len(feature_names),
        "missing_feature_track_ids": missing, "present_but_empty_track_ids": empty,
        "nonfinite_values": int(nonfinite.sum().sum()),
        "nonnumeric_values": int(nonnumeric.sum().sum()),
        "all_training_missing_columns": quality.loc[quality["all_training_missing"], "feature"].tolist(),
    }
    return availability, quality, info


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def write_figures(frame, missingness, split_table, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    train = numeric_values(frame.loc[frame["set_split"].eq("training"), "track_listens"])
    train = train[np.isfinite(train) & train.ge(0)]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.hist(np.log1p(train), bins=45, color="#167D9A", edgecolor="white", linewidth=0.4)
    ax.set(title="Recorded listens in the existing training split",
           xlabel="log(1 + recorded listens)", ylabel="Number of tracks")
    fig.savefig(destination / "training_listens.png", dpi=160)
    plt.close(fig)

    selected = missingness[missingness["missing_count"].gt(0)].sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    if not selected.empty:
        ax.barh(selected["column"], selected["missing_pct"], color="#167D9A")
    else:
        ax.text(0.5, 0.5, "No missing metadata cells", ha="center", transform=ax.transAxes)
    ax.set(title="Missing metadata in the cleaned dataset", xlabel="Missing values (%)", xlim=(0, 100))
    fig.savefig(destination / "metadata_missingness.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    if split_table["provisional_hits"].notna().all():
        hits = split_table["provisional_hits"].astype(int)
        ax.bar(split_table["split"], split_table["tracks"] - hits, label="Provisional non-hit", color="#C8D8DE")
        ax.bar(split_table["split"], hits, bottom=split_table["tracks"] - hits,
               label="Provisional hit", color="#167D9A")
        ax.legend(frameon=False)
    else:
        ax.bar(split_table["split"], split_table["tracks"], color="#167D9A")
    ax.set(title="Existing split sizes" + (" with provisional labels" if split_table["provisional_hits"].notna().all() else ""),
           ylabel="Number of tracks")
    fig.savefig(destination / "split_counts.png", dpi=160)
    plt.close(fig)


def report_markdown(summary, missingness, split_table):
    target = summary["target_proposal"]
    lines = ["# Task 1 data audit", "",
             "Machine-generated operational audit. Labels and splits are not frozen. "
             "No predictive model has been trained or evaluated.", "",
             f"Generated (UTC): {summary['generated_at_utc']}", "",
             f"Cleaned dataset: **{summary['rows']:,} tracks and {summary['columns']} columns**.", "",
             f"Unique track IDs: {summary['unique_track_ids']:,}. Duplicate IDs: 0. "
             f"Exact duplicate rows: {summary['duplicate_rows']}.", "",
             "## Existing splits and provisional target", "",
             "| Split | Tracks | Provisional hits |", "|---|---:|---:|"]
    for row in split_table.to_dict("records"):
        hits = row["provisional_hits"]
        lines.append(f"| {row['split']} | {row['tracks']:,} | {int(hits) if pd.notna(hits) else 'unavailable'} |")
    if target["status"] == "provisional":
        lines += ["", f"Training-only 75th percentile: **{target['listens_cutoff']:g} listens**. "
                  f"With integer counts, hit means at least **{target['minimum_integer_listens']:,} listens**. "
                  "The same cutoff is applied to every split. Recompute after any eligibility or split changes."]
    else:
        lines += ["", f"Target unavailable: {target['reason']}"]
    lines += ["", "## Missing metadata", "", "| Column | Missing | Percent |", "|---|---:|---:|"]
    for row in missingness[missingness["missing_count"].gt(0)].to_dict("records"):
        lines.append(f"| {row['column']} | {row['missing_count']:,} | {row['missing_pct']:.2f}% |")
    source = summary["original_metadata"]
    audio = summary["audio_features"]
    lines += ["", "## Source reconciliation", ""]
    if source["status"] == "checked":
        lines += [f"Original medium cohort (small + medium): {source['original_medium_rows']:,} tracks. "
                  f"Absent from the cleaned CSV: {source['removed_medium_rows']:,}. "
                  f"Differences in the compared source fields: {source['differing_field_values']:,}.", "",
                  "See removed_tracks.csv for the omitted IDs and available removal evidence."]
    else:
        lines += ["Original metadata was not available for this run."]
    lines += ["", "## Audio features", ""]
    if audio["status"] == "checked":
        lines += [f"Descriptors per track: {audio['feature_count']}. "
                  f"Missing feature rows: {len(audio['missing_feature_track_ids'])}. "
                  f"Present rows without finite descriptors: {len(audio['present_but_empty_track_ids'])}. "
                  f"Infinite values: {audio['nonfinite_values']}. "
                  f"Nonnumeric values: {audio['nonnumeric_values']}.", "",
                  "Feature availability is not evidence that the corresponding audio file exists or is eligible."]
    else:
        lines += ["The feature table was not available for this run."]
    lines += ["", "## Split overlap checks", ""]
    for kind, count in summary["overlap_group_counts"].items():
        lines.append(f"- {kind}: {count} distinct groups cross splits.")
    lines += ["", "Name matches are review clues. Known positive IDs provide the stronger grouping check. "
              "Shared compilation albums still need review even when artist IDs are separated.", "",
              "## Open items before freezing the cohort", ""]
    lines += [f"- {item}" for item in summary["open_items"]]
    lines += ["", "## Generated files", "",
              "See `summary.json` for machine-readable results and input SHA-256 hashes. "
              "The CSVs contain per-column quality checks, source differences, removed tracks, "
              "overlap evidence and a provisional per-track audit. They do not replace a frozen manifest.", "",
              "![Training listens](figures/training_listens.png)", "",
              "![Metadata missingness](figures/metadata_missingness.png)", "",
              "![Existing splits](figures/split_counts.png)", "",
              "Original source and download checksums: [official FMA repository](https://github.com/mdeff/fma#data).", ""]
    return "\n".join(lines)


def run_audit(cleaned, output, tracks=None, features=None, cleaning_log=None, eligibility=None, removed_duplicates=None):
    frame = load_cleaned(cleaned)
    columns = ["track_id", *frame.columns]
    missingness = pd.DataFrame({
        "column": columns, "missing_count": [0, *frame.isna().sum().tolist()],
    })
    missingness["missing_pct"] = missingness["missing_count"] * 100 / len(frame)
    missingness = missingness.sort_values(["missing_count", "column"], ascending=[False, True])
    numeric = pd.DataFrame({c: numeric_values(frame[c]) for c in NUMERIC_COLUMNS if c in frame})
    numeric_summary = numeric.replace([np.inf, -np.inf], np.nan).describe(
        percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).T
    numeric_summary["missing_or_nonnumeric"] = numeric.isna().sum()
    numeric_summary["infinite"] = numeric.isin([np.inf, -np.inf]).sum()
    numeric_summary.index.name = "column"
    issues = []
    labels = pd.Series(pd.NA, index=frame.index, dtype="Int8", name="provisional_hit")
    try:
        cutoff, labels = provisional_labels(frame)
        target = {"status": "provisional", "training_quantile": 0.75, "interpolation": "linear",
                  "operator": ">=", "listens_cutoff": cutoff,
                  "minimum_integer_listens": math.ceil(cutoff), "definition_fitted_on": "training"}
        for split in SPLITS:
            if labels.loc[frame["set_split"].eq(split)].nunique() < 2:
                issues.append(f"The {split} split contains only one provisional target class.")
    except AuditError as error:
        target = {"status": "unavailable", "reason": str(error)}
        issues.append(str(error))
    ids, removed, differences, source_info = source_audit(frame, tracks)
    if "set_subset" not in frame:
        frame["set_subset"] = ids.get("source_subset", pd.NA)
    if removed_duplicates is not None:
        logged = load_cleaned(removed_duplicates)
        key = ["track_title", "artist_name", "track_duration"]
        kept = pd.MultiIndex.from_frame(frame[key])
        if (source_info["status"] != "checked" or set(logged.index) != set(removed.track_id)
                or not kept.is_unique or not pd.MultiIndex.from_frame(logged[key]).isin(kept).all()):
            raise AuditError("Duplicate log must cover every omitted ID and match retained duplicate keys")
        removed["removal_reason"] = "duplicate_title_artist_duration"
    availability, feature_quality, feature_info = feature_audit(frame, features)
    overlaps = []
    if "artist_name" in frame:
        overlaps.append(group_overlaps(normalized_names(frame["artist_name"]), frame["set_split"], "artist_name"))
    for group in ("artist_id", "album_id"):
        overlaps.append(group_overlaps(ids[group], frame["set_split"], group))
    overlap_table = pd.concat(overlaps, ignore_index=True)
    overlap_counts = {kind: int(overlap_table.loc[overlap_table["group_type"].eq(kind), "group_value"].nunique())
                      for kind in ("artist_name", "artist_id", "album_id")}
    if source_info["status"] != "checked":
        issues.append("Obtain original tracks.csv to recover IDs and reconcile removed tracks.")
    else:
        for key in ("cleaned_ids_missing_from_original", "cleaned_ids_outside_medium"):
            if source_info[key]:
                issues.append(f"Resolve {len(source_info[key])} tracks flagged as {key}.")
        for group in ("artist", "album"):
            count = source_info[f"missing_or_invalid_{group}_ids"]
            if count:
                issues.append(f"Resolve {count} missing or sentinel {group} IDs.")
        if source_info["differing_field_values"]:
            issues.append("Explain source mismatches in source_differences.csv before using the cleaned values.")
    for kind, count in overlap_counts.items():
        if count:
            issues.append(f"Review {count} {kind} groups crossing splits; evidence is in split_overlaps.csv.")
    if feature_info["status"] != "checked":
        issues.append("Obtain features.csv for the audio-feature models.")
    else:
        if feature_info["feature_count"] != 518:
            issues.append(f"Confirm the feature schema: expected 518 descriptors, found {feature_info['feature_count']}.")
        for key in ("missing_feature_track_ids", "present_but_empty_track_ids", "all_training_missing_columns"):
            if feature_info[key]:
                issues.append(f"Resolve {key}: {len(feature_info[key])} entries (see summary.json).")
        if feature_info["nonnumeric_values"]:
            issues.append("Investigate nonnumeric audio descriptors before imputation.")
        if feature_info["nonfinite_values"]:
            issues.append("Replace infinite descriptors with missing values inside the future training pipeline.")
    inputs = {name: input_record(path) for name, path in {
        "cleaned_csv": cleaned, "original_tracks": tracks, "audio_features": features,
        "cleaning_log": cleaning_log, "eligibility_record": eligibility,
        "removed_duplicates": removed_duplicates,
    }.items()}
    for name, action in (("cleaning_log", "Obtain the original cleaning code/removal log and source version; explain removed rows."),
                         ("eligibility_record", "Record the team's per-track eligibility decisions before freezing the cohort.")):
        if name == "cleaning_log" and removed_duplicates is not None and inputs[name]["status"] == "present":
            continue
        issues.append(action if inputs[name]["status"] != "present" else
                      f"Review the supplied {name}; file presence alone is not a completed audit.")
    issues += ["Confirm the audio location and availability with the spectrogram teammate.",
               "Review duplicate/related recordings; unique track IDs do not prove unique audio.",
               "Freeze the eligible cohort and resolve grouping before creating the shared split/label manifest."]
    split_rows = []
    # Preserve unexpected split values in counts even when target creation was refused.
    observed = frame["set_split"].fillna("<missing>")
    for split in [*SPLITS, *sorted(set(observed) - set(SPLITS))]:
        mask = observed.eq(split)
        split_rows.append({"split": split, "tracks": int(mask.sum()),
                           "provisional_hits": int(labels[mask].sum()) if labels[mask].notna().all() else None})
    split_table = pd.DataFrame(split_rows)
    correlation_rows = []
    for column in ("track_interest", "track_favorites"):
        if column in numeric:
            pair = numeric[["track_listens", column]].replace([np.inf, -np.inf], np.nan).dropna()
            # Pearson correlation of average ranks is Spearman's rho; no scipy dependency.
            correlation_rows.append({"field": column, "valid_pairs": len(pair),
                                     "spearman_with_listens": pair.rank(method="average").corr().iloc[0, 1],
                                     "predictor_policy": "exclude_engagement_outcome"})
    expected = set(EXPECTED_COLUMNS)
    if {"artist_id", "album_id"}.issubset(columns):
        expected = (expected - {"set_subset"}) | {"artist_id", "album_id"}
    absent_columns = sorted(expected - set(columns))
    if absent_columns:
        issues.append(f"Confirm cleaned schema; expected columns are missing: {absent_columns}.")
    summary = {
        "audit_version": 1, "status": "audit_completed_with_open_items", "ready_for_frozen_manifest": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame), "columns": len(columns),
        "unique_track_ids": frame.index.nunique(),
        "duplicate_rows": int(frame.reset_index().duplicated().sum()),
        "expected_columns_missing": absent_columns,
        "unexpected_columns": sorted(set(columns) - expected),
        "subset_counts": frame["set_subset"].fillna("<missing>").value_counts().to_dict(),
        "split_counts": split_rows, "target_proposal": target,
        "nonpositive_listens": int(numeric["track_listens"].le(0).sum()),
        "original_metadata": source_info, "audio_features": feature_info,
        "overlap_group_counts": overlap_counts, "engagement_correlations": correlation_rows,
        "open_items": issues, "inputs": inputs,
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")}},
    }
    output = Path(output)
    # Prevent accidental output collisions with source files.
    resolved_output = output.resolve()
    for path in (cleaned, tracks, features, cleaning_log, eligibility, removed_duplicates):
        if path is not None and Path(path).resolve().is_relative_to(resolved_output):
            raise AuditError("Output folder must not contain any input file")
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "missingness.csv": missingness,
        "numeric_summary.csv": numeric_summary.reset_index(),
        "split_summary.csv": split_table,
        "removed_tracks.csv": removed,
        "source_differences.csv": differences,
        "split_overlaps.csv": overlap_table,
        "feature_quality.csv": feature_quality,
        "engagement_correlations.csv": pd.DataFrame(correlation_rows, columns=[
            "field", "valid_pairs", "spearman_with_listens", "predictor_policy"]),
    }
    track_audit = frame[["set_split", "set_subset", "track_listens"]].join(ids[["artist_id", "album_id"]]).join(availability).join(labels)
    track_audit["label_status"] = target["status"]
    tables["track_audit_provisional.csv"] = track_audit.reset_index()
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    write_figures(frame, missingness, split_table, output / "figures")
    summary = json_safe(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "AUDIT_REPORT.md").write_text(report_markdown(summary, missingness, split_table), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task 1: audit the FMA project inputs and proposed target.")
    final_export = Path("data/raw/fma_medium_clean2.csv")
    parser.add_argument("--cleaned", type=Path, default=final_export)
    parser.add_argument("--tracks", type=Path, default=Path("data/raw/fma_metadata/tracks.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/raw/fma_metadata/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/task1"))
    parser.add_argument("--cleaning-log", type=Path)
    parser.add_argument("--removed-duplicates", type=Path)
    parser.add_argument("--eligibility", type=Path)
    args = parser.parse_args(argv)
    if args.cleaned.resolve() == final_export.resolve():
        evidence = Path("data/review/cleaning")
        args.cleaning_log = args.cleaning_log or evidence / "Cleaning.ipynb"
        args.removed_duplicates = args.removed_duplicates or evidence / "fma_medium_removed_duplicates.csv"
    try:
        print("Auditing cleaned metadata and original FMA inputs...", flush=True)
        summary = run_audit(args.cleaned, args.output, args.tracks, args.features,
                            args.cleaning_log, args.eligibility, args.removed_duplicates)
    except (AuditError, OSError, pd.errors.ParserError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 2
    print(f"Audited {summary['rows']:,} tracks. Report: {args.output / 'AUDIT_REPORT.md'}")
    print(f"Open items: {len(summary['open_items'])}. No final manifest or model was produced.")
    return 0
