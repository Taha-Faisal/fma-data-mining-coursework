"""Replay Qasim's cleaning rules on the official FMA medium metadata selection.

AI-generated code (OpenAI Codex). User confirmation: "yea this is final".
"""
from io import StringIO
from pathlib import Path

import pandas as pd


def read_export(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, index_col="track_id")
    if not frame.index.is_unique:
        raise ValueError("Repeated track IDs in export")
    return frame.sort_index()


def main():
    review = Path("data/review/cleaning")
    clean = read_export("data/raw/fma_medium_clean2.csv")
    removed = read_export(review / "fma_medium_removed_duplicates.csv")
    raw = pd.read_csv("data/raw/fma_metadata/tracks.csv", header=[0, 1], index_col=0, low_memory=False)
    source = raw.loc[raw[("set", "subset")].isin(["small", "medium"])].sort_index()
    source.columns = ["_".join(column) for column in source.columns]
    frame = source.loc[:, clean.columns].copy()
    for column in ("track_title", "album_title"):
        frame[column] = frame[column].str.strip()
    frame["album_date_released"] = pd.to_datetime(frame["album_date_released"], errors="coerce")
    frame = frame.fillna({"album_title": "Unknown album", "track_title": "Untitled",
                          "track_language_code": "unknown", "track_license": "Unknown"})
    frame = frame.loc[frame.artist_name.notna() & frame.track_duration.gt(0)]
    key = ["track_title", "artist_name", "track_duration"]
    duplicates = frame.duplicated(key)
    frame["track_tags"] = frame.track_tags.replace("[]", "")
    for rows, delivered in ((frame.loc[~duplicates], clean), (frame.loc[duplicates], removed)):
        expected = read_export(StringIO(rows.rename_axis("track_id").to_csv()))
        pd.testing.assert_frame_equal(delivered, expected)
    matches = frame.loc[duplicates, key].reset_index(names="track_id").merge(
        frame.loc[~duplicates, key].reset_index(names="retained_track_id"), on=key, validate="many_to_one")
    matches["removal_reason"] = "duplicate_title_artist_duration"
    matches.to_csv(review / "final_duplicate_matches.csv", index=False)
    print(f"Matched all fields for {len(clean):,} kept and {len(removed):,} duplicate tracks "
          f"from {len(source):,} original tracks.")


if __name__ == "__main__":
    main()
