"""Exact-hash deduplication and the cross-split leakage audit.

Scope, per the M2 configuration (see project decisions)
---------------------------------------------------------
Exact SHA-256 only. MinHash-LSH was considered and dropped: problem-level
splitting already eliminates the dominant leakage vector (resubmission chains)
by construction, since every submission to one problem lands in the same
split. What remains is (a) rare byte-identical duplicates possibly appearing
under different problems, and (b) the audit that proves (a) never crosses a
split boundary. Both are answered by exact hashing; near-duplicate detection
would add a threshold to defend at viva for a vector that mostly doesn't exist
once problems don't straddle splits.

Normalization before hashing
-----------------------------
Bytes are lightly normalized before hashing: CRLF -> LF, BOM stripped, trailing
whitespace stripped per line. Nothing else. This treats two files that differ
only in line-ending convention as identical, without touching anything that
could be construed as *preprocessing* the code itself. Comment stripping and
whitespace collapsing are explicitly out of scope here — those are training-time
decisions (M3), applied to train only, after the split is frozen. Doing them
now would blur the split/preprocessing boundary this project's methodology
depends on.

The cross-split audit
----------------------
`cross_split_duplicate_hashes` must return an empty result for the frozen
problem-level split. This is not a redundant check: it is the verification that
the split guarantee (`assert_problem_disjoint` in splitting.py) actually
prevents *content* leakage, not just *problem_id* leakage. The two are related
but distinct claims, and this module verifies the second.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from sdp.data.splitting import SPLIT_ORDER


def normalize_bytes(raw: bytes) -> bytes:
    """CRLF -> LF, strip BOM, strip trailing whitespace per line. Nothing else.

    Deliberately minimal: this is not a preprocessing step. It exists only so
    that two files differing purely in line-ending convention are not counted
    as distinct duplicates.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lines = raw.split(b"\n")
    lines = [line.rstrip() for line in lines]
    return b"\n".join(lines)


def hash_file(path: Path) -> str:
    """SHA-256 of a source file's normalized content, as a hex string."""
    return hashlib.sha256(normalize_bytes(path.read_bytes())).hexdigest()


def hash_corpus(paths: Iterable[Path], *, root: Path | None = None) -> pd.DataFrame:
    """Hash every file in `paths`.

    Args:
        paths: files to hash. Missing files are reported, not raised on — a
            missing file at this stage is a data problem to report, not a
            reason to abort a pass over tens of thousands of files.
        root: if given, paths in the result are stored relative to this root
            rather than absolute, for a more readable report.

    Returns:
        DataFrame with columns: path, sha256, bytes, exists. Always has these
        four columns, even for empty input — callers must not need to special
        case an empty corpus.
    """
    rows = []
    for p in paths:
        exists = p.exists()
        rel = str(p.relative_to(root)) if root is not None else str(p)
        if exists:
            rows.append(
                {
                    "path": rel,
                    "sha256": hash_file(p),
                    "bytes": p.stat().st_size,
                    "exists": True,
                }
            )
        else:
            rows.append({"path": rel, "sha256": None, "bytes": None, "exists": False})

    columns = {"path": "object", "sha256": "object", "bytes": "float64", "exists": "bool"}
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in columns.items()})
    return pd.DataFrame(rows)


def duplicate_groups(hashes: pd.DataFrame) -> pd.DataFrame:
    """Group files sharing a hash. Only groups of size > 1 are returned.

    Returns:
        DataFrame with columns: sha256, count, paths (list of paths in the
        group), sorted by count descending.
    """
    present = hashes[hashes["exists"]]
    grouped = present.groupby("sha256")["path"].agg(["count", list])
    grouped = grouped.rename(columns={"list": "paths"})
    dupes = grouped[grouped["count"] > 1].sort_values("count", ascending=False)
    return dupes.reset_index()


def dedup_report(hashes: pd.DataFrame) -> dict[str, object]:
    """Summary statistics for docs/LABELING.md."""
    present = hashes[hashes["exists"]]
    n_total = len(present)
    n_unique = present["sha256"].nunique()
    n_missing = (~hashes["exists"]).sum()
    groups = duplicate_groups(hashes)

    return {
        "files_hashed": n_total,
        "files_missing": int(n_missing),
        "unique_hashes": n_unique,
        "duplicate_groups": len(groups),
        "files_in_duplicate_groups": int(groups["count"].sum()) if len(groups) else 0,
        "redundant_files": (int((groups["count"] - 1).sum()) if len(groups) else 0),
        "duplicate_rate_pct": (round((1 - n_unique / n_total) * 100, 4) if n_total else 0.0),
    }


# --------------------------------------------------------------------------- #
# Cross-split audit
# --------------------------------------------------------------------------- #


def cross_split_duplicate_hashes(
    hashes_with_split: pd.DataFrame, *, split_col: str = "split"
) -> dict[str, int]:
    """Count hashes that appear in more than one split.

    `hashes_with_split` must have `sha256` and `split_col` columns, one row per
    sampled file (the output of `hash_corpus` joined back to the manifest).

    Returns:
        {"train&val": n, "train&test": n, "val&test": n} — always all three
        pairs, using the canonical SPLIT_ORDER, regardless of which splits are
        actually present in the input. A split with no rows contributes an
        empty set, not a missing key. Every value must be 0 for a
        leakage-safe corpus. This checks *content*, not `problem_id`; it is
        the audit that verifies content-level, not just problem-level,
        disjointness.
    """
    df = hashes_with_split[hashes_with_split["sha256"].notna()]
    by_split = {
        s.value: set(df.loc[df[split_col].astype(str) == s.value, "sha256"]) for s in SPLIT_ORDER
    }
    result: dict[str, int] = {}
    for i, a in enumerate(SPLIT_ORDER):
        for b in SPLIT_ORDER[i + 1 :]:
            result[f"{a.value}&{b.value}"] = len(by_split[a.value] & by_split[b.value])
    return result


def assert_no_cross_split_duplicates(
    hashes_with_split: pd.DataFrame, *, split_col: str = "split"
) -> None:
    """Raise unless no content hash appears in two splits.

    This is the executable form of the audit claim in docs/LABELING.md: content
    disjointness, verified rather than assumed from problem-id disjointness
    alone.
    """
    overlap = cross_split_duplicate_hashes(hashes_with_split, split_col=split_col)
    offending = {k: v for k, v in overlap.items() if v}
    if offending:
        raise AssertionError(f"content hash leaks across splits: {offending}")
