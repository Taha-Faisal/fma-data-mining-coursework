"""Behavioural checks for task 1, using synthetic data rather than FMA results."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fma_project.audit import (
    AuditError, feature_audit, group_overlaps, load_cleaned, main,
    normalized_names, provisional_labels, run_audit, source_audit,
)


class AuditTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".work_tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cleaned = self.folder / "cleaned.csv"
        self.rows = pd.DataFrame({
            "track_id": range(1, 9),
            "track_listens": [100, 200, 300, 400, 325, 1, 326, 2],
            "set_split": ["training"] * 4 + ["validation"] * 2 + ["test"] * 2,
            "set_subset": ["small", "medium"] * 4,
            "track_duration": [120] * 8,
            "artist_name": ["BAND", "NA", "c", "d", "  band  ", "f", "g", "h"],
            "track_tags": [None] * 8,
        })
        self.rows.to_csv(self.cleaned, index=False)

    def frame(self):
        return load_cleaned(self.cleaned)

    def write_source(self):
        path = self.folder / "tracks.csv"
        columns = pd.MultiIndex.from_tuples([
            ("artist", "id"), ("album", "id"), ("set", "subset"),
            ("set", "split"), ("track", "listens"), ("track", "duration"),
        ])
        records = []
        for row in self.rows.to_dict("records"):
            tid = row["track_id"]
            records.append([1 if tid in (1, 5) else tid, tid, row["set_subset"],
                            row["set_split"], row["track_listens"], row["track_duration"]])
        records += [[9, 9, "small", "training", 5, 10], [10, 10, "large", "test", 5, 10]]
        data = pd.DataFrame(records, columns=columns, index=pd.Index(range(1, 11), name="track_id"))
        data.to_csv(path)
        return path

    def write_features(self):
        path = self.folder / "features.csv"
        columns = pd.MultiIndex.from_tuples([
            ("mfcc", "mean", "01"), ("mfcc", "mean", "02"), ("rms", "mean", "01")],
            names=["feature", "statistics", "number"])
        data = pd.DataFrame([
            [np.nan, 2, 3], [np.nan, 3, 4], [np.nan, 4, 5], [np.nan, 5, 6],
            [1, 2, 3], [np.inf, np.nan, np.nan], [4, 5, 6]], columns=columns,
            index=pd.Index([1, 2, 3, 4, 5, 6, 8], name="track_id"))
        data.to_csv(path)
        return path

    def test_cutoff_uses_training_only_and_includes_ties(self):
        data = self.frame()
        cutoff, labels = provisional_labels(data)
        self.assertEqual(cutoff, 325)
        self.assertEqual(labels.loc[5], 1)
        self.assertEqual(labels.loc[6], 0)
        data.loc[data["set_split"].ne("training"), "track_listens"] = "99999999"
        self.assertEqual(provisional_labels(data)[0], cutoff)

    def test_invalid_listens_do_not_silently_become_nonhits(self):
        for value in ("", "wrong", "inf", "-1", "2.5"):
            with self.subTest(value=value):
                data = self.frame()
                data.loc[5, "track_listens"] = value
                with self.assertRaises(AuditError):
                    provisional_labels(data)

    def test_missing_or_unknown_splits_refuse_target(self):
        for value in ("dev", pd.NA):
            with self.subTest(value=value):
                data = self.frame()
                data.loc[1, "set_split"] = value
                with self.assertRaises(AuditError):
                    provisional_labels(data)
        with self.assertRaises(AuditError):
            provisional_labels(self.frame().iloc[:4])

    def test_duplicate_and_invalid_ids_fail_before_joins(self):
        for value in ("1", "0", "-2", "1.5", "bad", ""):
            with self.subTest(value=value):
                rows = self.rows.astype({"track_id": "string"})
                rows.loc[1, "track_id"] = value
                rows.to_csv(self.cleaned, index=False)
                with self.assertRaises(AuditError):
                    load_cleaned(self.cleaned)

    def test_duplicate_headers_are_rejected(self):
        self.cleaned.write_text("track_id,track_id,track_listens,set_split,set_subset\n1,1,2,training,small\n")
        with self.assertRaises(AuditError):
            load_cleaned(self.cleaned)

    def test_blank_cells_and_literal_na_artist_are_distinguished(self):
        rows = self.rows.copy()
        rows.loc[0, "track_tags"] = "   "
        rows.to_csv(self.cleaned, index=False)
        data = self.frame()
        self.assertTrue(data["track_tags"].isna().all())
        self.assertEqual(data.loc[2, "artist_name"], "NA")

    def test_unicode_name_overlaps_preserve_evidence(self):
        data = self.frame()
        data.loc[5, "artist_name"] = "Ｂａｎｄ"
        result = group_overlaps(normalized_names(data["artist_name"]), data["set_split"], "artist_name")
        self.assertEqual(set(result["track_id"]), {1, 5})
        self.assertEqual(set(result["group_value"]), {"band"})

    def test_source_reconciliation_keeps_both_nested_subsets(self):
        ids, removed, differences, info = source_audit(self.frame(), self.write_source())
        self.assertEqual(info["original_medium_rows"], 9)
        self.assertEqual(removed["track_id"].tolist(), [9])
        self.assertEqual(removed.iloc[0]["removal_reason"], "unknown_without_cleaning_history")
        self.assertTrue(differences.empty)
        self.assertEqual(ids.loc[5, "artist_id"], 1)

    def test_source_changes_and_unmatched_ids_are_reported(self):
        path = self.write_source()
        data = self.frame()
        data.loc[2, "track_listens"] = "999"
        data = data.rename(index={8: 99})
        ids, _, differences, info = source_audit(data, path)
        self.assertEqual(info["cleaned_ids_missing_from_original"], [99])
        self.assertTrue(pd.isna(ids.loc[99, "artist_id"]))
        self.assertEqual(differences[["track_id", "field"]].values.tolist(), [[2, "track_listens"]])

    def test_missing_feature_rows_differ_from_empty_feature_rows(self):
        availability, quality, info = feature_audit(self.frame(), self.write_features(), chunksize=2)
        self.assertEqual(info["missing_feature_track_ids"], [7])
        self.assertEqual(info["present_but_empty_track_ids"], [6])
        self.assertEqual(info["nonfinite_values"], 1)
        self.assertEqual(info["all_training_missing_columns"], ["audio__mfcc__mean__01"])
        self.assertEqual(availability.loc[1, "finite_feature_count"], 2)
        self.assertEqual(len(quality), 3)

    def test_duplicate_feature_ids_across_chunks_are_rejected(self):
        path = self.write_features()
        data = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        pd.concat([data, data.iloc[[0]]]).to_csv(path)
        with self.assertRaises(AuditError):
            feature_audit(self.frame(), path, chunksize=2)

    def test_complete_run_preserves_input_and_replaces_stale_optional_results(self):
        original_bytes = self.cleaned.read_bytes()
        output = self.folder / "results"
        summary = run_audit(self.cleaned, output, self.write_source(), self.write_features())
        self.assertEqual(summary["overlap_group_counts"]["artist_id"], 1)
        self.assertFalse(summary["ready_for_frozen_manifest"])
        self.assertEqual(self.cleaned.read_bytes(), original_bytes)
        self.assertEqual(len(pd.read_csv(output / "track_audit_provisional.csv")), 8)
        self.assertTrue((output / "figures/training_listens.png").is_file())
        json.loads((output / "summary.json").read_text())
        rerun = run_audit(self.cleaned, output)
        self.assertEqual(rerun["audio_features"]["status"], "not_available")
        self.assertTrue(pd.read_csv(output / "feature_quality.csv").empty)
        self.assertTrue(pd.read_csv(output / "removed_tracks.csv").empty)

    def test_invalid_target_still_produces_diagnostic_audit(self):
        rows = self.rows.astype({"track_listens": "string"})
        rows.loc[5, "track_listens"] = "wrong"
        rows.to_csv(self.cleaned, index=False)
        output = self.folder / "results"
        summary = run_audit(self.cleaned, output)
        self.assertEqual(summary["target_proposal"]["status"], "unavailable")
        self.assertTrue(pd.read_csv(output / "track_audit_provisional.csv")["provisional_hit"].isna().all())

    def test_cli_missing_input_returns_actionable_failure(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error), contextlib.redirect_stdout(io.StringIO()):
            status = main(["--cleaned", str(self.folder / "missing.csv")])
        self.assertEqual(status, 2)
        self.assertIn("Audit failed", error.getvalue())

    def test_output_cannot_contain_inputs(self):
        with self.assertRaises(AuditError):
            run_audit(self.cleaned, self.folder)


if __name__ == "__main__":
    unittest.main()
