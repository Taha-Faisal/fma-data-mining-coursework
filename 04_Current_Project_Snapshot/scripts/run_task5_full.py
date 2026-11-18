"""Resumable full-cohort FMA conversion with bounded parallel workers.

AI-generated code (OpenAI Codex). Full initiating user prompt:
"Do the full spectrogram conversion."
The user previously confirmed permission for research, ML and AI use.
Student independent evaluation and applicable course acknowledgement remain required.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from zipfile import ZipFile

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
COLLECTION = PROJECT.parent
FIELDS = ["track_id", "group_id", "set_split", "hit", "eligibility", "audio_path", "available",
          "status", "detail", "quality_status", "duration_seconds", "adjustment_samples",
          "array_path", "dtype", "n_mels", "n_frames", "source_sha1", "source_sha256",
          "source_checksum_verified", "array_sha256", "array_min_db", "array_max_db", "resumed"]
_archive = None
_converter = None
_folder = None


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def init_worker(audio, folder, converter_path):
    global _archive, _converter, _folder
    _archive = ZipFile(audio)
    _folder = Path(folder)
    spec = importlib.util.spec_from_file_location("task5_conversion", converter_path)
    _converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_converter)


def validate_array(array):
    if array.shape != (128, 1292) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Unexpected shape, dtype or nonfinite values")
    if float(array.min()) < -80.001 or float(array.max()) > 0.001:
        raise ValueError("Values outside the specified relative-dB range")


def process_track(task):
    record, expected_sha1, prior = task
    tid = int(record["track_id"])
    name = f"{tid:06d}"
    member = f"fma_medium/{name[:3]}/{name}.mp3"
    row = {key: "" for key in FIELDS}
    row.update({key: record[key] for key in ("track_id", "group_id", "set_split", "hit", "eligibility")})
    row.update(audio_path=member, available=member in _archive.NameToInfo,
               status="conversion_error", quality_status="review_required", resumed=False)
    if not row["available"]:
        row.update(status="missing_file", detail="Requested track is absent from archive")
        return row
    try:
        raw = _archive.read(member)  # ZipFile also validates the member CRC.
        source_sha1 = hashlib.sha1(raw).hexdigest()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        row.update(source_sha1=source_sha1, source_sha256=source_sha256,
                   source_checksum_verified=bool(expected_sha1 and source_sha1 == expected_sha1))
        if not row["source_checksum_verified"]:
            row.update(status="source_checksum_mismatch", detail="Missing or mismatched bundled audio SHA-1")
            return row
        target = _folder / "arrays" / (name + ".npy")
        reuse = (prior.get("status") == "ok" and prior.get("source_sha256") == source_sha256
                 and target.is_file() and digest(target) == prior.get("array_sha256"))
        if reuse:
            array = np.load(target, allow_pickle=False)
            validate_array(array)
            duration = float(prior["duration_seconds"])
            adjustment = int(prior["adjustment_samples"])
            row["resumed"] = True
        else:
            try:
                array, duration, adjustment = _converter.convert(raw)
            except Exception as exc:
                row.update(status="decode_or_conversion_error", detail=f"{type(exc).__name__}: {exc}")
                return row
            validate_array(array)
            temp = target.with_suffix(".npy.partial")
            with temp.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            reloaded = np.load(temp, allow_pickle=False)
            validate_array(reloaded)
            if not np.array_equal(array, reloaded):
                raise ValueError("Saved array differs after reload")
            temp.replace(target)
        warning = abs(duration - 30.0) > 1.0
        row.update(status="ok", quality_status="review_required" if warning else "passed",
                   detail="Review unusual decoded duration; array was end-padded/cropped" if warning else "",
                   duration_seconds=duration, adjustment_samples=adjustment,
                   array_path=f"arrays/{target.name}", dtype=str(array.dtype), n_mels=128, n_frames=1292,
                   array_sha256=digest(target), array_min_db=float(array.min()), array_max_db=float(array.max()))
    except Exception as exc:
        row.update(status="conversion_error", detail=f"{type(exc).__name__}: {exc}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Debugging subset only; zero means full cohort")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/spectrograms")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit nonnegative")
    scratch = PROJECT / ".work_tmp/full_spectrograms"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(scratch)
    os.environ["NUMBA_CACHE_DIR"] = str(scratch)
    import librosa
    import soundfile
    import scipy
    import soxr

    audio = COLLECTION / "01_Original_Files/FMA_Audio/fma_medium_audio.zip"
    registered = json.loads((audio.parent / "AUDIO_VERIFICATION.json").read_text(encoding="utf-8"))
    if audio.stat().st_size != registered["bytes"] or not registered["official_archive_sha1_matches"]:
        raise ValueError("Audio does not match registered source information")
    bundle_dir = PROJECT / "data/processed/task2/candidate_e630aee72cc4"
    manifest = bundle_dir / "manifest_candidate.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        selected = list(csv.DictReader(stream))
    if args.limit:
        selected = selected[:args.limit]
    ids = [int(row["track_id"]) for row in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected unique nonempty track IDs")
    with (PROJECT / "data/raw/fma_medium_clean2.csv").open(encoding="utf-8-sig", newline="") as stream:
        cleaned = {int(row["track_id"]) for row in csv.DictReader(stream)}
    if (not args.limit and set(ids) != cleaned) or not set(ids).issubset(cleaned):
        raise ValueError("Conversion list does not match final cleaned cohort")
    for row in selected:
        row["eligibility"] = "eligible"
        row["permission_basis"] = "User confirmed research/ML/AI permission and requested full conversion"
    converter_path = PROJECT / "scripts/run_task5.py"
    config = dict(schema_version=1, converter_sha256=digest(converter_path), runner_sha256=digest(Path(__file__)),
                  shared_manifest_sha256=digest(manifest), candidate_id=json.loads((bundle_dir / "bundle.json").read_text())["candidate_id"],
                  archive_sha256=registered["sha256"], archive_sha1=registered["sha1"],
                  source_checksum_policy="Verify each MP3 against bundled SHA-1, ZIP CRC and record SHA-256",
                  cohort_count=len(ids), selected_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                  limit=args.limit, labels_and_splits="Preserve shared candidate manifest; do not use pilot source splits",
                  permission_basis="User confirmed research/ML/AI permission and requested full conversion",
                  representation=dict(sr=22050, samples=661500, mono=True, n_fft=2048, win_length=2048,
                                      hop_length=512, window="hann", center=True, pad_mode="constant", power=2.0,
                                      n_mels=128, fmin=0, fmax=11025, htk=False, norm="slaney",
                                      top_db=80, db_reference="per-clip maximum", resampler="soxr_hq",
                                      dtype="float32", shape=[128,1292], duration_warning_seconds=1.0),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, librosa=librosa.__version__,
                                soundfile=soundfile.__version__, libsndfile=soundfile.__libsndfile_version__,
                                scipy=scipy.__version__, soxr=soxr.__version__),
                  threads_per_worker=1)
    version = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder = args.output.resolve() / ("full_" + version if not args.limit else "debug_" + version)
    folder.mkdir(parents=True, exist_ok=True)
    for child in ("arrays", "previews"):
        (folder / child).mkdir(exist_ok=True)
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing conversion configuration mismatch")
    write_json(config_path, config)
    input_path = folder / "input_tracks.csv"
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    expected_bytes = len(ids) * (128 * 1292 * 4 + 128)
    if shutil.disk_usage(folder).free < expected_bytes + 2 * 2**30:
        raise OSError("Insufficient free space for output plus 2 GiB headroom")
    prior = {}
    for name in ("manifest.csv", "manifest.inprogress.csv"):
        path = folder / name
        if path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("track_id") and row.get("status"):
                        prior[int(row["track_id"])] = row
    with ZipFile(audio) as archive:
        checksums = {name.strip().lstrip("*"): value for value, name in
                     (line.split(maxsplit=1) for line in archive.read("fma_medium/checksums").decode().splitlines())}
        tasks = [(row, checksums.get(f"{int(row['track_id']):06d}"[:3] + f"/{int(row['track_id']):06d}.mp3"), prior.get(int(row["track_id"]), {})) for row in selected]
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    counts, quality_counts = Counter(), Counter()
    completed = []
    print(f"RUN_FOLDER={folder}", flush=True)
    print(f"Starting {len(tasks)} tracks with {args.workers} workers; using shared candidate splits", flush=True)
    progress_path = folder / "progress.json"
    def progress():
        elapsed = time.monotonic() - started
        done = len(completed)
        state = dict(status="running", started_utc=started_utc, requested=len(tasks), processed=done,
                     outcomes=dict(counts), quality=dict(quality_counts), elapsed_seconds=elapsed,
                     tracks_per_second=done/elapsed if elapsed else 0,
                     estimated_remaining_seconds=(len(tasks)-done)*elapsed/done if done else None,
                     workers=args.workers, manifest="manifest.inprogress.csv")
        write_json(progress_path, state)
        print(json.dumps(state), flush=True)
    progress()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(audio), str(folder), str(converter_path))) as pool:
        # At most two tasks per worker queued in addition to map's current chunk.
        with (folder / "manifest.inprogress.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row in pool.map(process_track, tasks, chunksize=1, buffersize=args.workers * 2):
                writer.writerow(row)
                completed.append(row)
                counts[row["status"]] += 1
                quality_counts[row["quality_status"]] += 1
                if len(completed) % 100 == 0:
                    stream.flush()
                    progress()
    (folder / "manifest.inprogress.csv").replace(folder / "manifest.csv")
    warnings = [row for row in completed if row["quality_status"] != "passed"]
    with (folder / "exceptions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(warnings)
    # Generate only five previews; full-array storage remains bounded per worker.
    spec = importlib.util.spec_from_file_location("task5_previews", converter_path)
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    by_id = {int(row["track_id"]): row for row in completed}
    for tid in (2, 10, 136, 139, 190):
        row = by_id.get(tid)
        if row and row["status"] == "ok":
            plotter.preview(np.load(folder / row["array_path"], allow_pickle=False), tid, folder / "previews" / f"{tid:06d}.png")
    summary = dict(status="conversion_complete", started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
                   requested=len(tasks), processed=len(completed), converted=counts.get("ok",0),
                   failed=len(completed)-counts.get("ok",0), available_audio=sum(bool(row["available"]) for row in completed),
                   outcomes=dict(counts), quality=dict(quality_counts), exceptions=len(warnings),
                   elapsed_seconds=time.monotonic()-started, workers=args.workers, candidate_id=config["candidate_id"],
                   input_sha256=digest(input_path), manifest_sha256=digest(folder/"manifest.csv"),
                   source_archive=str(audio), archive_sha256=registered["sha256"],
                   config_sha256=digest(config_path), folder=str(folder),
                   notes="Per-clip conversion completed; final cohort freeze/CNN training not performed.")
    write_json(folder / "summary.json", summary)
    write_json(progress_path, summary)
    if not args.limit:
        write_json(args.output.resolve() / "LATEST_FULL_CONVERSION.json", dict(folder=folder.name, summary="summary.json", manifest="manifest.csv"))
    print("CONVERSION_COMPLETE=" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
