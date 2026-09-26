"""Measure real CodeBERT tokenization stats against the frozen M2 corpus.

Run locally (not in CI — needs the real 74,850-file corpus on disk and network
access to the Hugging Face Hub on first run):

    python scripts/measure_tokenization.py

Mirrors the style of `scripts/linker_pilot.py`: a plain script, not a notebook,
because this measurement is a one-shot pipeline input rather than exploratory
analysis, and it needs to be re-runnable exactly if the corpus is ever
regenerated.

Reads:
    data/processed/splits/sample_manifest_hashed.parquet  (74,850 rows)
    data/processed/sources/<rel_path>                       (extracted .cpp files)

Writes:
    reports/tokenization_report.csv   — per-split summary stats
    reports/tokenization_full.csv     — one row per file (submission_id, byte_length,
                                          token_length) — feeds docs/TOKENIZATION.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sdp.data.tokenization import load_tokenizer, measure_corpus, summarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "processed" / "splits" / "sample_manifest_hashed.parquet"
SOURCES = PROJECT_ROOT / "data" / "processed" / "sources"
REPORTS = PROJECT_ROOT / "reports"


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"manifest not found: {MANIFEST}")
    if not SOURCES.exists():
        raise SystemExit(f"extracted sources not found: {SOURCES}")

    manifest = pd.read_parquet(MANIFEST)
    print(f"Manifest rows: {len(manifest):,}")

    files = [(row.submission_id, SOURCES / row.rel_path) for row in manifest.itertuples()]
    missing = [str(p) for _, p in files if not p.exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} files listed in the manifest are missing on disk "
            f"(first few: {missing[:3]}) — re-run the M2 extraction check first."
        )

    print("Loading microsoft/codebert-base tokenizer (first run downloads it)...")
    tokenizer = load_tokenizer()

    print(f"Tokenizing {len(files):,} files — this will take a while...")
    results = measure_corpus(tokenizer, files)
    results = results.merge(
        manifest[["submission_id", "split", "coarse_label"]], on="submission_id", how="left"
    )
    results["tokens_per_byte"] = results["token_length"] / results["byte_length"].replace(0, pd.NA)

    REPORTS.mkdir(parents=True, exist_ok=True)
    full_out = REPORTS / "tokenization_full.csv"
    results.to_csv(full_out, index=False)
    print(f"\nWrote {full_out} ({len(results):,} rows)")

    # Overall summary
    overall = summarize(results["token_length"])
    overall["mean_tokens_per_byte"] = float(results["tokens_per_byte"].mean())
    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k:22} {v:,.4f}" if isinstance(v, float) else f"  {k:22} {v}")

    # Per-split summary (this is what actually matters for the DataLoader design)
    rows = []
    for split, group in results.groupby("split", observed=True):
        stats = summarize(group["token_length"])
        stats["split"] = split
        stats["mean_tokens_per_byte"] = float(group["tokens_per_byte"].mean())
        rows.append(stats)
    per_split = pd.DataFrame(rows).set_index("split")

    summary_out = REPORTS / "tokenization_report.csv"
    per_split.to_csv(summary_out)
    print(f"\n=== PER SPLIT === (written to {summary_out})")
    print(per_split.to_string())

    print(
        "\nNext: paste this output back so docs/TOKENIZATION.md and the "
        "Dataset's max_length/truncation config can be finalized against real numbers."
    )


if __name__ == "__main__":
    main()
