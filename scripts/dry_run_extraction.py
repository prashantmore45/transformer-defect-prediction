"""One-off dry run: time how long it takes to find the first 1% of wanted
files in a single streaming pass over Project_CodeNet.tar.gz.

This is a go/no-go check before Task 7 commits to a full pass. Run it as a
script, not from the notebook — the archive is not seekable, so a hang here
should be Ctrl+C-able without losing kernel state.

Usage:
    python scripts/dry_run_extraction.py
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

ARCHIVE = Path(r"D:\Datasets\CodeNet\Project_CodeNet.tar.gz")
WANTED_PATHS_FILE = Path(__file__).resolve().parent.parent / "data" / "interim" / "wanted_paths.txt"

# tar member names have no leading slash and no "Project_CodeNet/data/" prefix
# ambiguity to worry about -- they match wanted_paths.txt exactly, since that
# file was built with ARCHIVE_ROOT = "Project_CodeNet/data" as the prefix.
DRY_RUN_FRACTION = 0.01


def main() -> None:
    if not ARCHIVE.exists():
        raise SystemExit(f"archive not found: {ARCHIVE}")
    if not WANTED_PATHS_FILE.exists():
        raise SystemExit(f"wanted paths not found: {WANTED_PATHS_FILE}")

    wanted = set(WANTED_PATHS_FILE.read_text(encoding="utf-8").splitlines())
    wanted = {w for w in wanted if w}
    total_wanted = len(wanted)
    target = max(1, int(total_wanted * DRY_RUN_FRACTION))

    print(f"Archive:        {ARCHIVE}  ({ARCHIVE.stat().st_size / 1e9:.2f} GB)")
    print(f"Wanted files:   {total_wanted:,}")
    print(f"Dry-run target: {target:,} files ({DRY_RUN_FRACTION*100:.0f}%)")
    print("Starting single streaming pass. This may take a few minutes with no output...\n")

    found = 0
    members_scanned = 0
    bytes_seen = 0
    start = time.monotonic()
    last_report = start

    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            members_scanned += 1
            bytes_seen += member.size

            if member.name in wanted:
                found += 1

            now = time.monotonic()
            if now - last_report >= 5.0:
                elapsed = now - start
                rate = bytes_seen / elapsed / 1e6 if elapsed > 0 else 0
                print(
                    f"  [{elapsed:6.1f}s] scanned {members_scanned:,} members, "
                    f"found {found:,}/{target:,} wanted, "
                    f"{bytes_seen/1e9:.2f} GB decompressed, "
                    f"{rate:.1f} MB/s"
                )
                last_report = now

            if found >= target:
                break

    elapsed = time.monotonic() - start
    if found < target:
        print(
            f"\nReached end of archive after {elapsed:.1f}s "
            f"having found only {found:,}/{target:,} wanted files."
        )
        print(
            "This means fewer wanted files exist in the archive than expected — "
            "investigate before proceeding to Task 7."
        )
        return

    rate_mb_s = bytes_seen / elapsed / 1e6
    projected_full_scan_s = ARCHIVE.stat().st_size / 1e9 / (bytes_seen / 1e9) * elapsed

    print(f"\nDry run complete.")
    print(f"  Time to find {target:,} files ({DRY_RUN_FRACTION*100:.0f}%): {elapsed:.1f}s")
    print(f"  Members scanned:        {members_scanned:,}")
    print(
        f"  Bytes decompressed:     {bytes_seen/1e9:.2f} GB "
        f"of {ARCHIVE.stat().st_size/1e9:.2f} GB archive"
    )
    print(f"  Decompression rate:     {rate_mb_s:.1f} MB/s")
    print(
        f"\n  Linear projection for full pass: {elapsed / DRY_RUN_FRACTION:.0f}s "
        f"(~{elapsed / DRY_RUN_FRACTION / 60:.1f} min)"
    )
    print(
        f"  Rate-based projection (decompress whole archive): "
        f"{projected_full_scan_s:.0f}s (~{projected_full_scan_s/60:.1f} min)"
    )


if __name__ == "__main__":
    main()
