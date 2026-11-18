"""Download and verify the official FMA metadata, without downloading audio.

AI-generated code (OpenAI Codex). Full user prompt:
"I want you to implement the first task of the data mining project."
The coursework requires the student to evaluate generated code independently.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request
from zipfile import ZipFile


URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
# Published by https://github.com/mdeff/fma#usage; this is an integrity check.
EXPECTED_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
MEMBERS = ("tracks.csv", "features.csv", "genres.csv", "README.txt")


def sha1(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha1").hexdigest()


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "fma_metadata.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if sha1(partial) != EXPECTED_SHA1:
            raise ValueError(f"Checksum mismatch; untrusted download retained at {partial}")
        partial.replace(archive)
    if sha1(archive) != EXPECTED_SHA1:
        raise ValueError(f"Checksum mismatch for {archive}; extraction refused")
    with ZipFile(archive) as zipped:
        available = set(zipped.namelist())
        for name in MEMBERS:
            member = f"fma_metadata/{name}"
            if member not in available:
                if name == "README.txt":
                    continue
                raise ValueError(f"Archive missing required member: {member}")
            target = destination / name
            partial = target.with_suffix(target.suffix + ".part")
            # Only explicitly named files are extracted, never arbitrary ZIP paths.
            with zipped.open(member) as source, partial.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)
            if target.exists():
                if sha1(target) != sha1(partial):
                    partial.unlink()
                    raise ValueError(f"Refusing to replace different existing input: {target}")
                partial.unlink()
            else:
                partial.replace(target)
            print(f"Verified {target}", flush=True)
    print(f"Archive SHA-1 verified: {EXPECTED_SHA1}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/fma_metadata"))
    args = parser.parse_args()
    fetch(args.output)
