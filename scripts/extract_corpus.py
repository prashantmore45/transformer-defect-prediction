"""Full single streaming pass over Project_CodeNet.tar.gz: extract every file
listed in data/interim/wanted_paths.txt to data/processed/sources/.

Run as a script, not from the notebook. The archive is not seekable, so this
must be Ctrl+C-able without losing kernel state if something goes wrong.

Usage:
    python scripts/extract_corpus.py
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

ARCHIVE = Path(r"D:\Datasets\CodeNet\Project_CodeNet.tar.gz")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WANTED_PATHS_FILE = PROJECT_ROOT / "data" / "interim" / "wanted_paths.txt"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "sources"

# The prefix every wanted path carries, e.g.
#   "Project_CodeNet/data/p00000/C++/s582427538.cpp"
# We strip this so the output tree is just:
#   data/processed/sources/p00000/C++/s582427538.cpp
ARCHIVE_PREFIX = "Project_CodeNet/data/"


def main() -> None:
    if not ARCHIVE.exists():
        raise SystemExit(f"archive not found: {ARCHIVE}")
    if not WANTED_PATHS_FILE.exists():
        raise SystemExit(f"wanted paths not found: {WANTED_PATHS_FILE}")

    wanted = {w for w in WANTED_PATHS_FILE.read_text(encoding="utf-8").splitlines() if w}
    total_wanted = len(wanted)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Archive:      {ARCHIVE}  ({ARCHIVE.stat().st_size / 1e9:.2f} GB)")
    print(f"Wanted files: {total_wanted:,}")
    print(f"Output:       {OUTPUT_ROOT}")
    print("Starting single streaming pass...\n")

    extracted = 0
    members_scanned = 0
    bytes_written = 0
    start = time.monotonic()
    last_report = start

    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            members_scanned += 1

            if member.name in wanted:
                if not member.isfile():
                    print(f"  WARNING: {member.name} matched but is not a regular file")
                    continue

                rel = (
                    member.name[len(ARCHIVE_PREFIX) :]
                    if member.name.startswith(ARCHIVE_PREFIX)
                    else member.name
                )
                dest = OUTPUT_ROOT / rel
                dest.parent.mkdir(parents=True, exist_ok=True)

                src = tar.extractfile(member)
                if src is None:
                    print(f"  WARNING: could not open {member.name}")
                    continue

                data = src.read()
                dest.write_bytes(data)
                bytes_written += len(data)
                extracted += 1

            now = time.monotonic()
            if now - last_report >= 10.0:
                elapsed = now - start
                print(
                    f"  [{elapsed:6.1f}s] scanned {members_scanned:,} members, "
                    f"extracted {extracted:,}/{total_wanted:,}, "
                    f"{bytes_written/1e6:.1f} MB written"
                )
                last_report = now

            if extracted >= total_wanted:
                break

    elapsed = time.monotonic() - start
    print(f"\nExtraction complete in {elapsed:.1f}s ({elapsed/60:.1f} min).")
    print(f"  Members scanned: {members_scanned:,}")
    print(f"  Files extracted: {extracted:,} / {total_wanted:,}")
    print(f"  Bytes written:   {bytes_written/1e6:.1f} MB")

    if extracted < total_wanted:
        print(
            f"\n  SHORTFALL: {total_wanted - extracted:,} wanted files were not found "
            f"in the archive. Investigate before proceeding — this likely means a "
            f"path-construction mismatch, not a sampling problem."
        )
    else:
        print("\n  All wanted files extracted.")


if __name__ == "__main__":
    main()
