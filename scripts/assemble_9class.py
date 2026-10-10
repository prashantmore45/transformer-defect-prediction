"""M6.3b -- assemble the unified 9-class manifest from the frozen M2/M4/M5 artifacts.

Dry run by default: prints every check and report, writes nothing.
Pass --write to save outputs (only if all checks passed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.labeling.assemble import assemble_manifest
from sdp.data.taxonomy import LEAF_ORDER, TAXONOMY_VERSION

SPLITS = ("train", "val", "test")
REPORTS = PROJECT_ROOT / "reports"
MANIFEST_IN = PROJECT_ROOT / "data" / "processed" / "splits" / "sample_manifest_hashed.parquet"
MANIFEST_OUT = PROJECT_ROOT / "data" / "processed" / "splits" / "sample_manifest_9class.parquet"
M4_IN = REPORTS / "m4_full_reserve_results_final.csv"
M5_IN = REPORTS / "m5_full_reserve_classified.csv"
CONFIG = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"

# Labelled counts per leaf as reported at the M2/M4/M5 closeouts.
CLOSEOUT = {
    "ERROR_FREE": 9_999,
    "LOGICAL": 9_994,
    "SYNTAX": 6_698,
    "SEMANTIC": 15_319,
    "LINKER": 372,
    "SIGSEGV": 2_132,
    "SIGABRT": 1_194,
    "SIGFPE": 579,
    "NZEC": 444,
}
EXPECTED_PRE_CONFLICT = sum(CLOSEOUT.values())  # 46,731


def _sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="save outputs (default: dry run)")
    args = parser.parse_args()

    manifest = pd.read_parquet(MANIFEST_IN)
    tier2 = pd.read_csv(
        M4_IN, usecols=["submission_id", "classification"], dtype=str, keep_default_na=False
    )
    tier3 = pd.read_csv(
        M5_IN,
        usecols=["submission_id", "outcome", "input_source"],
        dtype=str,
        keep_default_na=False,
    )

    res = assemble_manifest(manifest, tier2, tier3)
    kept, dropped = res.retained, res.excluded

    # ---- checks ---------------------------------------------------------- #
    counts = kept["leaf_label"].value_counts()
    deltas = {k: v - int(counts.get(k, 0)) for k, v in CLOSEOUT.items()}
    leaks = int((kept.groupby("sha256")["split"].nunique() > 1).sum())
    checks = {
        f"pre-conflict labelled == {EXPECTED_PRE_CONFLICT:,}": res.pre_conflict_labelled
        == EXPECTED_PRE_CONFLICT,
        "per-class deltas >= 0 and sum == conflict_removed": all(d >= 0 for d in deltas.values())
        and sum(deltas.values()) == res.conflict_removed,
        "cross-split sha256 collisions == 0": leaks == 0,
    }

    print(f"manifest rows: {len(manifest):,}")
    print(f"labelled before conflict rule: {res.pre_conflict_labelled:,}")
    print(f"removed by CONFLICTING_DUPLICATE: {res.conflict_removed:,}")
    print(f"retained: {len(kept):,} | excluded: {len(dropped):,}")
    print("\nper-class delta vs closeout (rows removed by conflict rule):")
    print({k: d for k, d in deltas.items() if d})

    order = [c.value for c in LEAF_ORDER]
    table = pd.crosstab(kept["leaf_label"], kept["split"]).reindex(
        index=order, columns=list(SPLITS), fill_value=0
    )
    table["total"] = table.sum(axis=1)
    table.loc["TOTAL"] = table.sum()
    print("\nclass x split (retained):")
    print(table.to_string())

    excl = pd.crosstab(dropped["exclusion_reason"], dropped["coarse_label"])
    print("\nexclusions by reason x coarse class:")
    print(excl.to_string())

    same_label_groups = int((kept.groupby("sha256").size() > 1).sum())
    print(f"\nsame-label duplicate groups still retained: {same_label_groups:,}")

    excess = int((kept.groupby("sha256").size() - 1).sum())
    print(f"retained rows that are extra copies inside those groups: {excess:,}")

    leaf_names = {c.value for c in LEAF_ORDER}
    seen = pd.concat(
        [
            kept[["sha256", "leaf_label"]].set_axis(["sha256", "outcome"], axis=1),
            dropped[["sha256", "resolved_outcome"]].set_axis(["sha256", "outcome"], axis=1),
        ]
    )
    by_sha = seen.groupby("sha256")["outcome"].agg(set)
    removed = dropped[dropped["exclusion_reason"] == "CONFLICTING_DUPLICATE"]
    kinds = {"2+ different leaf labels": 0, "1 leaf label vs excluded twin": 0}
    for sha, n in removed.groupby("sha256").size().items():
        many = len(by_sha[sha] & leaf_names) >= 2
        kinds["2+ different leaf labels" if many else "1 leaf label vs excluded twin"] += int(n)
    print("\nconflict removals by kind (rows):", kinds)

    print("retained runtime rows by input_source:")
    print(
        kept.loc[kept["coarse_label"] == "RUNTIME_ERROR", "input_source"].value_counts().to_string()
    )

    print("\nchecks:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("checks failed -- nothing written")

    if not args.write:
        print("\ndry run complete (use --write to save).")
        return

    # ---- write ----------------------------------------------------------- #
    kept.to_parquet(MANIFEST_OUT, index=False)
    digest = _sha256_of_file(MANIFEST_OUT)
    table.to_csv(REPORTS / "m6_class_split_counts.csv")
    excl.to_csv(REPORTS / "m6_exclusion_counts.csv")
    dropped.to_csv(REPORTS / "m6_excluded_rows.csv", index=False)
    summary = [
        "# M6 assembly summary",
        "",
        f"- taxonomy version: {TAXONOMY_VERSION}",
        f"- manifest: `{MANIFEST_OUT.name}` ({len(kept):,} rows)",
        f"- manifest sha256: `{digest}`",
        f"- labelled before conflict rule: {res.pre_conflict_labelled:,}",
        f"- removed as CONFLICTING_DUPLICATE: {res.conflict_removed:,}",
        f"- excluded in total: {len(dropped):,}",
        f"- cross-split sha256 collisions: {leaks}",
        f"- same-label duplicate groups retained: {same_label_groups:,}",
        f"- retained rows that are extra copies in same-label groups: {excess:,}",
        f"- conflict removals by kind (rows): {kinds}",
    ]
    (REPORTS / "m6_assembly_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg["artifact_hashes"][MANIFEST_OUT.name] = digest
    cfg["m6_assembly"] = {
        "milestone": "M6",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": TAXONOMY_VERSION,
        "manifest": MANIFEST_OUT.name,
        "rows_retained": len(kept),
        "labelled_before_conflict_rule": res.pre_conflict_labelled,
        "conflicting_duplicate_removed": res.conflict_removed,
        "cross_split_sha256_collisions": leaks,
    }
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    print(f"\nwritten. manifest sha256 = {digest}")


if __name__ == "__main__":
    main()
