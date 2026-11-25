"""Replay the recorded cleaning rules on the official FMA medium metadata selection."""
import argparse
import hashlib
from io import StringIO
from pathlib import Path

import pandas as pd


def read_export(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, index_col="track_id")
    if not frame.index.is_unique:
        raise ValueError("Repeated track IDs in export")
    return frame.sort_index()


EXPORT_COLUMNS = (
    "track_title artist_name album_title track_genre_top track_genres track_duration track_bit_rate "
    "track_listens track_favorites track_interest track_date_created album_date_released track_language_code "
    "track_license track_tags set_split artist_id album_id"
).split()


def write_verified(path, frame, expected_digest):
    data = frame.rename_axis("track_id").to_csv(lineterminator="\r\n").encode("utf-8")
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError(f"Reconstructed CSV differs from the recorded export: {path}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def main(rebuild=False):
    review = Path("data/review/cleaning")
    raw = pd.read_csv("data/raw/fma_metadata/tracks.csv", header=[0, 1], index_col=0, low_memory=False)
    source = raw.loc[raw[("set", "subset")].isin(["small", "medium"])].sort_index()
    source.columns = ["_".join(column) for column in source.columns]
    frame = source.loc[:, EXPORT_COLUMNS].copy()
    for column in ("track_title", "album_title"):
        frame[column] = frame[column].str.strip()
    frame["album_date_released"] = pd.to_datetime(frame["album_date_released"], errors="coerce")
    frame = frame.fillna({"album_title": "Unknown album", "track_title": "Untitled",
                          "track_language_code": "unknown", "track_license": "Unknown"})
    frame = frame.loc[frame.artist_name.notna() & frame.track_duration.gt(0)]
    key = ["track_title", "artist_name", "track_duration"]
    duplicates = frame.duplicated(key)
    frame["track_tags"] = frame.track_tags.replace("[]", "")
    clean_path = Path("data/raw/fma_medium_clean2.csv")
    removed_path = review / "fma_medium_removed_duplicates.csv"
    if rebuild:
        write_verified(clean_path, frame.loc[~duplicates], "f094a8df9b4eea58b993e99440763da33d4a47fd9c5c9e5f71537208dae6a178")
        write_verified(removed_path, frame.loc[duplicates], "d741c0227b47b8fb694084f116e3af8130f4e308beab33d4529dd58594f47e7b")
    clean, removed = read_export(clean_path), read_export(removed_path)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Reconstruct both hash-checked team CSVs from official tracks.csv")
    main(parser.parse_args().rebuild)
