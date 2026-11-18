"""Freeze the reviewed audio cohort while preserving candidate group/split assignments.

AI-generated code (OpenAI Codex). Full initiating user prompt:
"Continue with the second phase. And create a dashboard for me to show what tasks are done and what are said to be done."
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
import sys
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from fma_project.audit import normalized_names
from fma_project.manifest import validate_review


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    original = PROJECT / "data/processed/task2/candidate_e630aee72cc4"
    previous = json.loads((original / "config.json").read_text())
    ledger = json.loads((original / "bundle.json").read_text())
    for name, checksum in ledger["files"].items():
        assert sha(original / name) == checksum, f"Changed original bundle: {name}"
    conversion = PROJECT / "data/processed/spectrograms/full_c2fcad16360ed43b"
    verification = json.loads((conversion / "verification.json").read_text())
    assert verification["all_checks_passed"] and verification["checked_arrays"] == 24873
    candidate = pd.read_csv(original / "manifest_candidate.csv").set_index("track_id", verify_integrity=True)
    audio = pd.read_csv(conversion / "manifest.csv").set_index("track_id", verify_integrity=True)
    cleaned = pd.read_csv(PROJECT / "data/raw/fma_medium_clean2.csv").set_index("track_id", verify_integrity=True)
    assert set(candidate.index) == set(audio.index) == set(cleaned.index)
    keep = audio.status.eq("ok") & audio.quality_status.eq("passed")
    selected = candidate.loc[audio.index[keep]].copy().sort_index()
    rejected = candidate.loc[audio.index[~keep]].copy().sort_index()
    assert len(selected) == 24872 and len(rejected) == 24
    old_labels = selected.hit.copy()
    cutoff = float(selected.loc[selected.set_split.eq("training"), "track_listens"].quantile(.75, interpolation="linear"))
    selected["hit"] = selected.track_listens.ge(cutoff).astype(int)
    selected["manifest_status"] = "frozen"
    selected["eligibility"] = "eligible"
    selected["audio_status"] = "ok"
    selected["array_path"] = selected.index.map(lambda tid: f"data/processed/spectrograms/{conversion.name}/arrays/{tid:06d}.npy")
    selected["array_sha256"] = audio.loc[selected.index, "array_sha256"]
    selected["duration_seconds"] = audio.loc[selected.index, "duration_seconds"]
    assert (selected.set_split == candidate.loc[selected.index, "set_split"]).all()
    assert (selected.group_id == candidate.loc[selected.index, "group_id"]).all()
    assert selected.index.is_unique and not selected.isna().any().any()
    checks = {}
    for key in ("group_id", "artist_id", "album_id"):
        checks[key] = int(selected.groupby(key).set_split.nunique().gt(1).sum())
    names = normalized_names(cleaned.loc[selected.index, "artist_name"])
    checks["normalized_artist_name"] = int(pd.DataFrame({"key": names, "split": selected.set_split}).groupby("key")["split"].nunique().gt(1).sum())
    for key in ("source_sha256", "array_sha256"):
        frame = pd.DataFrame({"key": audio.loc[selected.index,key], "split": selected.set_split})
        checks[key] = int(frame.groupby("key")["split"].nunique().gt(1).sum())
    links = pd.read_csv(original / "grouping_links.csv")
    retained_links = links[links.track_id_a.isin(selected.index) & links.track_id_b.isin(selected.index)]
    checks["inherited_grouping_links"] = sum(selected.loc[int(row.track_id_a),"set_split"] != selected.loc[int(row.track_id_b),"set_split"] for row in retained_links.itertuples())
    assert not any(checks.values()), checks
    assert all(selected.loc[selected.set_split.eq(split),"hit"].nunique()==2 for split in ("training","validation","test"))
    array_dupes = audio.loc[selected.index].groupby("array_sha256").filter(lambda x: len(x)>1)
    identity = dict(schema_version=2, parent_candidate_id=previous["candidate_id"],
                    runner_sha256=sha(Path(__file__)), parent_manifest_sha256=sha(original / "manifest_candidate.csv"),
                    conversion_manifest_sha256=sha(conversion / "manifest.csv"),
                    conversion_verification_sha256=sha(conversion / "verification.json"),
                    exclusion_rule="Exclude conversion failures and predeclared >1-second duration warnings",
                    split_policy="Preserve parent candidate group IDs and split membership; no seed/search/reassignment",
                    target_policy="Refit training-only linear 75th percentile on reviewed core cohort",
                    listens_cutoff=cutoff, permission_basis="User confirmed research/ML/AI permission")
    cohort_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    output = PROJECT / "data/processed/task2" / ("frozen_audio_" + cohort_id[:12])
    if output.exists():
        raise ValueError("Frozen output already exists; use its verified bundle")
    review_dir = PROJECT / "data/review/audio_phase2"
    review_dir.mkdir(parents=True,exist_ok=True)
    (review_dir / "PERMISSION_RECORD.md").write_text("# Reported team permission\n\nThe user stated: 'Kaggle has given us permission to use the dataset under as full research as well as for machine learning and AI.' The user subsequently confirmed: 'I have got permission and everything.' The user supplied the full audio and explicitly requested full spectrogram conversion and continuation into phase two. This is a team-provided inclusion basis, not an independent legal determination. Source licence metadata remains preserved.\n",encoding="utf-8")
    write_json(review_dir / "RELATED_RECORD_CHECKS.json",dict(cross_split_counts=checks,
               inherited_links_checked=len(retained_links), identical_saved_array_tracks=len(array_dupes),
               identical_saved_array_groups=int(array_dupes.array_sha256.nunique()),
               all_exact_array_matches_within_existing_splits=True,
               scope="Inherited artist/album/name/descriptor links, exact source MP3 hashes and saved array hashes",
               limitation="No exhaustive acoustic near-duplicate detection; different encodings/crops may remain related."))
    def evidence(path):
        return dict(path=os.path.relpath(path, review_dir).replace(os.sep,"/"), sha256=sha(path))
    review = dict(candidate_id=cohort_id,reviews={
        "cleaning":dict(status="complete",reviewed_by="Automated phase-two review of prior cleaning evidence",
                        notes="Final 24,896 cleaned selection and 104 duplicate removals retained as provenance; 24 audio-quality exclusions recorded separately.",
                        evidence=[evidence(PROJECT/"data/review/cleaning/REVIEW.md"),evidence(PROJECT/"data/raw/fma_medium_clean2.csv")]),
        "eligibility":dict(status="complete",reviewed_by="Project user confirmation, recorded by Codex",
                           notes="User confirmed research/ML/AI permission; retained tracks are eligible under that reported team decision.",
                           evidence=[evidence(review_dir/"PERMISSION_RECORD.md")]),
        "audio":dict(status="complete",reviewed_by="Automated full-cohort conversion and independent saved-array verification",
                     notes="23 failures and one unusually short clip excluded without reference to model performance; retained array checks pass.",
                     evidence=[evidence(conversion/"manifest.csv"),evidence(conversion/"verification.json")]),
        "related_recordings":dict(status="complete",reviewed_by="Automated scoped dependency review",
                                  notes="Preserved parent grouping and verified no cross-split artist/album/name/inherited link or exact MP3/array hash dependencies. Acoustic near-duplicates beyond these checks remain a documented limitation.",
                                  evidence=[evidence(review_dir/"RELATED_RECORD_CHECKS.json"),evidence(original/"grouping_links.csv")])})
    write_json(review_dir/"REVIEW.json",review)
    validate_review(review_dir/"REVIEW.json",cohort_id)
    output.mkdir()
    selected.reset_index().to_csv(output/"manifest.csv",index=False)
    decisions = pd.DataFrame(index=candidate.index)
    decisions["eligibility"]="eligible"
    decisions["audio_status"]="ok"
    decisions["reason"]="User-confirmed permission; full conversion passed source/array and duration checks"
    for tid in rejected.index:
        decisions.loc[tid,"audio_status"]="excluded"
        decisions.loc[tid,"reason"] = str(audio.loc[tid,"detail"])
    decisions.reset_index().to_csv(output/"cohort_decisions.csv",index=False)
    exclusions = rejected.join(audio[["status","quality_status","detail","duration_seconds"]])
    exclusions["exclusion_reason"] = exclusions["detail"]
    exclusions.reset_index().to_csv(output/"exclusions.csv",index=False)
    changed = selected.loc[selected.hit.ne(old_labels),["set_split","track_listens","hit"]].copy()
    changed["previous_hit"] = old_labels.loc[changed.index]
    changed.reset_index().to_csv(output/"label_changes.csv",index=False)
    stats=[]
    for split, part in selected.groupby("set_split",sort=False):
        stats.append(dict(split=split,tracks=len(part),groups=int(part.group_id.nunique()),hits=int(part.hit.sum()),hit_pct=float(100*part.hit.mean())))
    pd.DataFrame(stats).to_csv(output/"split_summary.csv",index=False)
    retained_links.to_csv(output/"grouping_links.csv",index=False)
    config=dict(candidate_id=cohort_id,status="frozen",created_at_utc=datetime.now(timezone.utc).isoformat(),
                method="audio_review_with_preserved_candidate_splits",identity=identity,inputs=previous["inputs"],
                cohort=dict(input_tracks=24896,included_tracks=len(selected),excluded_tracks=len(rejected),
                            groups=int(selected.group_id.nunique()),changed_split_tracks=0),
                target=dict(listens_cutoff=cutoff,minimum_integer_listens=math.ceil(cutoff),
                            rule="track_listens >= reviewed_training_quantile_0.75",fitted_on="training",
                            previous_cutoff=2178,label_changes=len(changed)),
                overlap_counts_after=checks,split_summary=stats,pending_eligibility=0,pending_audio=0,
                review=review,review_file=dict(path=str(review_dir/"REVIEW.json"),sha256=sha(review_dir/"REVIEW.json")),
                open_items=[],limitations=["Acoustic near-duplicates outside exact hashes/inherited grouping were not exhaustively searched.",
                                          "Prior candidate model metrics are historical; refit before comparing on this cohort.",
                                          "Student independent code evaluation and report/presentation completion are not established by these checks."])
    write_json(output/"config.json",config)
    report=f"""# Frozen common audio cohort

Retained **24,872** of 24,896 cleaned tracks. Excluded the 23 conversion failures and track 108924, which decoded to 27.338 seconds and exceeded the predeclared one-second duration tolerance. Selection uses audio-quality outcomes only, without model scores or test outcomes.

All parent group IDs and split assignments are preserved. Training: 20,300; validation: 2,546; reserved test: 2,026. The newly fitted training-only 75th percentile is {cutoff:g} listens, so integer counts are hits at **{math.ceil(cutoff):,} listens**. Eight labels changed from the provisional candidate; `label_changes.csv` records them. Prior model metrics require refitting on this version before a direct final comparison.

The manifest includes portable array paths from the working project root and hashes from the fully verified conversion. No spectrogram duplication is required. Source files and the earlier candidate remain unchanged. Review evidence is at `data/review/audio_phase2/REVIEW.json`; exact evidence hashes were validated before freezing.

Artist/album/name/inherited descriptor links, exact MP3 hashes and exact saved-array matches do not cross splits. Forty-four tracks share an exact saved-array hash within their existing splits; these remain grouped conservatively by the inherited grouping. No exhaustive acoustic near-duplicate search was performed. This is a scoped computational review, not proof that no other related recordings exist.

CNN training, revised classical/MLP fits and final test evaluation remain pending. No model test predictions have been generated by this phase.
"""
    (output/"TASK2_REPORT.md").write_text(report,encoding="utf-8")
    write_json(output/"bundle.json",dict(candidate_id=cohort_id,status="frozen",files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    write_json(PROJECT/"data/processed/task2/LATEST_FROZEN.json",dict(candidate_id=cohort_id,status="frozen",bundle=output.name,
               manifest=f"{output.name}/manifest.csv",report=f"{output.name}/TASK2_REPORT.md"))
    print(json.dumps(dict(folder=str(output),included=len(selected),excluded=len(rejected),cutoff=cutoff,
                         minimum_integer_listens=math.ceil(cutoff),label_changes=len(changed),checks=checks,split_summary=stats),indent=2))


if __name__=="__main__":
    main()
