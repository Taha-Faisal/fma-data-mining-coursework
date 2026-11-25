"""Independently inspect every saved array and shared-manifest association."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import sys
import time
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def check_one(task):
    folder, row = task
    tid = int(row["track_id"])
    path = folder / row["array_path"]
    errors = []
    try:
        if path.name != f"{tid:06d}.npy" or path.parent != folder / "arrays":
            errors.append("Incorrect ID filename/path")
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum != row["array_sha256"]:
            errors.append("Array checksum differs from manifest")
        array = np.load(io.BytesIO(raw), allow_pickle=False)
        if array.shape != (128, 1292):
            errors.append(f"Unexpected shape {array.shape}")
        if array.dtype != np.float32:
            errors.append(f"Unexpected dtype {array.dtype}")
        if not np.isfinite(array).all():
            errors.append("Nonfinite data")
        low, high = float(array.min()), float(array.max())
        if low < -80.001 or high > 0.001 or abs(high) > 0.001:
            errors.append(f"Unexpected relative-dB limits {low}, {high}")
        if not np.isclose(low, float(row["array_min_db"]), atol=1e-6) or not np.isclose(high, float(row["array_max_db"]), atol=1e-6):
            errors.append("Array extrema differ from manifest")
        if row["source_checksum_verified"] != "True":
            errors.append("Audio checksum was not verified")
        return dict(track_id=tid, passed=not errors, errors=errors, bytes=len(raw))
    except Exception as exc:
        return dict(track_id=tid, passed=False, errors=[f"{type(exc).__name__}: {exc}"], bytes=0)


def main():
    base = PROJECT / "data/processed/spectrograms"
    folder = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else base / json.loads((base / "LATEST_FULL_CONVERSION.json").read_text())["folder"]
    summary = json.loads((folder / "summary.json").read_text())
    with (folder / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with (PROJECT / "data/processed/task2/candidate_e630aee72cc4/manifest_candidate.csv").open(encoding="utf-8-sig", newline="") as f:
        candidate = {int(x["track_id"]): x for x in csv.DictReader(f)}
    with (PROJECT / "data/raw/fma_medium_clean2.csv").open(encoding="utf-8-sig", newline="") as f:
        cleaned = {int(x["track_id"]) for x in csv.DictReader(f)}
    ids = [int(row["track_id"]) for row in rows]
    errors = []
    if len(ids) != len(set(ids)) or set(ids) != cleaned or len(ids) != 24896:
        errors.append("Manifest IDs do not cover the unique final cleaned cohort")
    if hashlib.sha256((folder / "manifest.csv").read_bytes()).hexdigest() != summary["manifest_sha256"]:
        errors.append("Manifest checksum differs from conversion summary")
    for row in rows:
        tid = int(row["track_id"])
        expected = candidate[tid]
        if any(row[k] != expected[k] for k in ("group_id", "set_split", "hit")):
            errors.append(f"Candidate grouping/split/label mismatch for {tid}")
        if row["eligibility"] != "eligible":
            errors.append(f"Missing user-confirmed eligibility for {tid}")
        if row["status"] == "ok":
            expected_quality = "review_required" if abs(float(row["duration_seconds"])-30) > 1 else "passed"
            if expected_quality != row["quality_status"]:
                errors.append(f"Duration warning mismatch for {tid}")
    successful = [row for row in rows if row["status"] == "ok"]
    expected_files = {f"{int(row['track_id']):06d}.npy" for row in successful}
    actual_files = {p.name for p in (folder / "arrays").glob("*.npy")}
    if actual_files != expected_files:
        errors.append("Saved array filenames differ from successful manifest IDs")
    if list((folder / "arrays").glob("*.partial")):
        errors.append("Partial array files remain")
    started = time.monotonic()
    failed, total_bytes, checked = [], 0, 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(check_one, [(folder, row) for row in successful], buffersize=32):
            checked += 1
            total_bytes += result["bytes"]
            if not result["passed"]:
                failed.append(result)
            if checked % 1000 == 0:
                print(f"Verified {checked}/{len(successful)} arrays; invalid={len(failed)}", flush=True)
    split_counts = Counter(row["set_split"] for row in successful)
    warnings = [dict(track_id=int(row["track_id"]), duration_seconds=float(row["duration_seconds"]), detail=row["detail"])
                for row in successful if row["quality_status"] != "passed"]
    failures = [dict(track_id=int(row["track_id"]), status=row["status"], detail=row["detail"])
                for row in rows if row["status"] != "ok"]
    expected_previews = {f"{tid:06d}.png" for tid in (2,10,136,139,190) if f"{tid:06d}.npy" in expected_files}
    if not expected_previews.issubset({p.name for p in (folder / "previews").glob("*.png")}):
        errors.append("Expected preview files are missing")
    report = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(), requested=len(ids),
                  unique_requested_ids=len(set(ids)), converted=len(successful), failed_conversions=len(failures),
                  checked_arrays=checked, invalid_arrays=len(failed), total_array_bytes=total_bytes,
                  array_gib=total_bytes/2**30, split_counts=dict(split_counts),
                  label_group_split_alignment_passed=not any("mismatch" in e for e in errors),
                  cohort_coverage_passed=set(ids)==cleaned and len(ids)==len(set(ids)),
                  duration_review_count=len(warnings), duration_reviews=warnings, failures=failures,
                  structural_errors=errors, array_errors=failed, all_checks_passed=not errors and not failed,
                  elapsed_seconds=time.monotonic()-started,
                  limitation="Decode failures and unusual-duration tracks are explicit exceptions; cohort freezing and CNN training are separate tasks.")
    (folder / "verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("VERIFICATION_COMPLETE="+json.dumps(report), flush=True)
    if not report["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
