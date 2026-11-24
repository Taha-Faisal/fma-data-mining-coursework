"""Build the task 5 FMA mel spectrogram collection. Run from the project root."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


SETTINGS = dict(sr=22050, n_fft=2048, win_length=2048, hop_length=512,
                window="hann", center=True, pad_mode="constant", power=2.0,
                n_mels=128, fmin=0, fmax=11025, htk=False, norm="slaney")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert(raw):
    wave, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Empty or nonfinite audio")
    mono = librosa.resample(wave.mean(axis=1), orig_sr=rate, target_sr=22050, res_type="soxr_hq")
    fixed = librosa.util.fix_length(mono, size=661500)
    if np.max(np.abs(fixed)) <= 1e-8:
        raise ValueError("Silent excerpt")
    mel = librosa.feature.melspectrogram(y=fixed, **SETTINGS)
    array = librosa.power_to_db(mel, ref=np.max, top_db=80).astype("float32")
    if array.shape != (128, 1292) or not np.isfinite(array).all():
        raise ValueError("Unexpected output shape or nonfinite energy")
    return array, len(wave) / rate, 661500 - len(mono)


def preview(array, tid, path):
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    centers = librosa.mel_frequencies(n_mels=130, fmin=0, fmax=11025)[1:-1]
    color = ax.pcolormesh(np.arange(1292)*512/22050, centers, array, shading="auto",
                          vmin=-80, vmax=0, rasterized=True)
    ax.set(xlabel="Time (seconds)", ylabel="Frequency (Hz)", title=f"Track {tid:06} mel spectrogram")
    fig.colorbar(color, ax=ax, label="dB relative to clip maximum")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(audio, tracks, output):
    with tracks.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "track_id" not in (reader.fieldnames or []):
            raise ValueError("Supply a CSV of cleared track IDs with a track_id column")
        requested = list(reader)
    ids = [int(r["track_id"]) for r in requested]
    if not ids or min(ids) < 1 or max(ids) > 999999 or len(ids) != len(set(ids)):
        raise ValueError("Expected unique positive FMA IDs")
    config = dict(mel=SETTINGS, samples=661500, top_db=80, db_reference="per-clip maximum",
                  dtype="float32", resampler="soxr_hq", librosa=librosa.__version__,
                  soundfile=sf.__version__, numpy=np.__version__, script_sha256=digest(Path(__file__)),
                  input_sha256=digest(tracks))
    version = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder = output / version
    for child in ("arrays", "previews"):
        (folder / child).mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    prior = {}
    manifest = folder / "manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            prior = {int(row["track_id"]): row for row in csv.DictReader(f)}
    archive = ZipFile(audio) if audio.is_file() and audio.suffix.lower()==".zip" else None
    fields = ["track_id","eligibility","set_split","hit","audio_path","available","status","detail",
              "duration_seconds","adjustment_samples","array_path","source_sha256","array_sha256"]
    counts = Counter()
    plotted = 0
    available = 0
    try:
        with manifest.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            for tid, record in zip(ids, requested):
                name = f"{tid:06}"
                rel = f"{name[:3]}/{name}.mp3"
                member = "fma_medium/" + rel
                exists = member in archive.NameToInfo if archive else (audio / rel).is_file()
                available += int(exists)
                row = dict(track_id=tid, audio_path=rel, available=exists, eligibility=record.get("eligibility","eligible"),
                           set_split=record.get("set_split",""), hit=record.get("hit",""))
                if record.get("exclusion_reason") or record.get("audio_status") == "excluded":
                    row.update(status="excluded", detail=record.get("exclusion_reason") or record.get("reason", ""))
                elif row["eligibility"] != "eligible":
                    row.update(status="eligibility_" + row["eligibility"])
                elif not exists:
                    row.update(status="missing_file")
                else:
                    try:
                        raw = archive.read(member) if archive else (audio / rel).read_bytes()
                        source_hash = hashlib.sha256(raw).hexdigest()
                        target = folder / "arrays" / (name + ".npy")
                        old = prior.get(tid, {})
                        if (old.get("status") == "ok" and old.get("source_sha256")==source_hash and
                                target.is_file() and digest(target)==old.get("array_sha256")):
                            array = np.load(target, allow_pickle=False)
                            duration, adjustment = float(old["duration_seconds"]), int(old["adjustment_samples"])
                        else:
                            array, duration, adjustment = convert(raw)
                            with target.with_suffix(".tmp").open("wb") as dest:
                                np.save(dest, array)
                            target.with_suffix(".tmp").replace(target)
                            np.testing.assert_array_equal(np.load(target, allow_pickle=False), array)
                        if plotted < 5:
                            preview(array, tid, folder / "previews" / (name+".png"))
                            plotted += 1
                        row.update(status="ok", duration_seconds=duration, adjustment_samples=adjustment,
                                   array_path="arrays/"+target.name, source_sha256=source_hash, array_sha256=digest(target),
                                   detail="Review unusual duration" if abs(duration-30)>1 else "")
                    except (ValueError, RuntimeError, OSError) as exc:
                        row.update(status="review_required", detail=str(exc))
                counts[row["status"]] += 1
                writer.writerow(row)
        manifest.with_suffix(".tmp").replace(manifest)
    finally:
        if archive:
            archive.close()
    summary = dict(requested=len(ids), available_audio=available, saved=counts.get("ok",0), outcomes=dict(counts),
                   audio_source=str(audio), track_list=str(tracks), manifest=str(manifest))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("data/raw/fma_medium"), help="Extracted FMA root or official medium ZIP")
    parser.add_argument("--tracks", type=Path, help="Task 2 decisions, or a CSV list of team-cleared pilot IDs")
    parser.add_argument("--output", type=Path, default=Path("data/processed/spectrograms"))
    args = parser.parse_args()
    if args.tracks is None:
        root = Path("data/processed/task2")
        pointer = root / "LATEST_FROZEN.json"
        if not pointer.exists():
            pointer = root / "LATEST_CANDIDATE.json"
        args.tracks = root / json.loads(pointer.read_text())["bundle"] / "cohort_decisions.csv"
    run(args.audio, args.tracks, args.output)
