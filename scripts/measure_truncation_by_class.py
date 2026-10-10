"""M6.5 -- how does head-only truncation fall on each of the 9 classes?

Measurement only: the truncation strategy (head-only, 512 tokens) is NOT
changed here.

Part (a), all 9 classes: share of files longer than 512 tokens (M3's own
definition, token_length > 512), plus median and p90 full token length.
Before measuring, it asserts that every file's cached length equals
min(token_length, 512), which proves the 9-class cache and the M3 measurement
agree.

Part (b), SYNTAX and SEMANTIC only: is the compiler's first error line inside
the 510 content tokens the model sees? Each file with a usable error line is
classed fully / partly / not visible (see `sdp.data.visibility`), using exact
character offsets from the fast tokenizer. Left out of those percentages and
counted separately:
  * short_file    -- at most 2 lines: the whole program is "line 1", so a
                     line-level answer says nothing;
  * no_error_line -- the first error is not on a line of the submission's own
                     file (e.g. it is in a system header), or the line is
                     beyond the end of the file.
SYNTAX is reported three ways: all files, only the missing-include branch
("No such file or directory", labelled SYNTAX by tier2.classify), and
without it, because a missing include is almost always near line 1 and
would otherwise flatter the SYNTAX figure.

Run locally (needs the real corpus, the M4 results and the HF tokenizer cache):

    python scripts/measure_truncation_by_class.py

Reads:
    data/processed/splits/sample_manifest_9class.parquet
    data/processed/tokenized_9class/{train,val,test}.parquet
    data/processed/sources/<rel_path>
    reports/tokenization_full.csv
    reports/m4_full_reserve_results_final.csv

Writes:
    reports/m6_truncation_by_class.csv
    reports/m6_error_visibility.csv
"""

from __future__ import annotations

import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import MANIFEST_9CLASS_PATH, SOURCES_DIR, TOKENIZED_9CLASS_DIR

# Private helper reused deliberately: it is the exact function tier2.classify
# uses to pick "the first error line", so the measurement looks at the same line.
from sdp.data.labeling.tier2 import _first_error_line, first_error_line_number
from sdp.data.taxonomy import LEAF_ORDER
from sdp.data.tokenization import DEFAULT_MAX_LENGTH, _decode, load_tokenizer
from sdp.data.visibility import (
    Visibility,
    classify_visibility,
    count_lines,
    echo_status,
    line_span,
)

REPORTS = PROJECT_ROOT / "reports"
TOKENS_CSV = REPORTS / "tokenization_full.csv"
M4_CSV = REPORTS / "m4_full_reserve_results_final.csv"
SHORT_FILE_MAX_LINES = 2


def _cache_lengths() -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            TOKENIZED_9CLASS_DIR / f"{split}.parquet", columns=["submission_id", "input_ids"]
        )
        for split in ("train", "val", "test")
    ]
    df = pd.concat(frames, ignore_index=True)
    df["cache_len"] = df["input_ids"].map(len)
    return df[["submission_id", "cache_len"]]


def _summ(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "n": len(frame),
        "truncated_pct": frame["truncated"].mean() * 100,
        "median_tokens": frame["token_length"].median(),
        "p90_tokens": frame["token_length"].quantile(0.9),
    }


def _truncation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {c.value: _summ(df[df["leaf_label"] == c.value]) for c in LEAF_ORDER}
    rows["ALL"] = _summ(df)
    table = pd.DataFrame(rows).T
    table["n"] = table["n"].astype(int)
    table = table.round(1)
    table.index.name = "leaf_label"
    return table


def _measure(tokenizer, row) -> dict[str, object]:
    """Classify one SYNTAX/SEMANTIC file; see the module docstring for the outcomes."""
    text = _decode((SOURCES_DIR / row.rel_path).read_bytes())
    low = row.stderr.lower()
    record: dict[str, object] = {
        "leaf": row.leaf_label,
        "missing_include": "fatal error" in low and "no such file or directory" in low,
        "outcome": None,
        "echo": None,
    }
    if count_lines(text) <= SHORT_FILE_MAX_LINES:
        record["outcome"] = "short_file"
        return record

    line_no = first_error_line_number(row.submission_id, row.stderr)
    span = line_span(text, line_no) if line_no is not None else None
    if span is None:
        record["outcome"] = "no_error_line"
        return record

    start, end = span
    record["echo"] = echo_status(
        row.stderr, _first_error_line(row.stderr), line_no, text[start:end]
    )
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    record["outcome"] = classify_visibility(
        [token_end for _, token_end in offsets], start, end
    ).value
    return record


def _summarize(name: str, frame: pd.DataFrame) -> dict[str, object]:
    counts = frame["outcome"].value_counts()
    measured = int(sum(counts.get(v.value, 0) for v in Visibility))
    summary: dict[str, object] = {
        "stratum": name,
        "n": len(frame),
        "short_file": int(counts.get("short_file", 0)),
        "no_error_line": int(counts.get("no_error_line", 0)),
        "measured": measured,
    }
    for v in Visibility:
        count = int(counts.get(v.value, 0))
        summary[v.value] = count
        summary[f"{v.value}_pct"] = round(100 * count / measured, 1) if measured else float("nan")
    echo = frame["echo"].value_counts()
    summary["echo_no_snippet"] = int(echo.get("no_echo", 0))
    summary["echo_mismatch"] = int(echo.get("mismatch", 0))
    return summary


def main() -> None:
    manifest = pd.read_parquet(
        MANIFEST_9CLASS_PATH, columns=["submission_id", "leaf_label", "rel_path"]
    )
    full = pd.read_csv(TOKENS_CSV, usecols=["submission_id", "token_length"])
    df = manifest.merge(_cache_lengths(), on="submission_id", validate="1:1")
    df = df.merge(full, on="submission_id", how="left", validate="1:1")
    if len(df) != len(manifest) or df["token_length"].isna().any():
        raise SystemExit("join incomplete: some manifest rows lack a cache or token-length entry")

    expected = df["token_length"].clip(upper=DEFAULT_MAX_LENGTH)
    bad = int((df["cache_len"] != expected).sum())
    if bad:
        raise SystemExit(f"{bad} files: cached length != min(token_length, {DEFAULT_MAX_LENGTH})")
    print(
        f"[PASS] cached length == min(token_length, {DEFAULT_MAX_LENGTH}) for all {len(df):,} files"
    )

    # ---- (a) truncation by class ------------------------------------------- #
    df["truncated"] = df["token_length"] > DEFAULT_MAX_LENGTH
    table = _truncation_table(df)
    print("\n(a) truncation by class (truncated = token_length > 512)")
    print(table.to_string())
    table.to_csv(REPORTS / "m6_truncation_by_class.csv")

    # ---- (b) is the first compile error inside the model's window? --------- #
    m4 = pd.read_csv(M4_CSV, usecols=["submission_id", "stderr"], dtype=str, keep_default_na=False)
    wanted = manifest[manifest["leaf_label"].isin(["SYNTAX", "SEMANTIC"])]
    sel = wanted.merge(m4, on="submission_id", validate="1:1")
    if len(sel) != len(wanted):
        raise SystemExit("some SYNTAX/SEMANTIC files have no M4 stderr row")

    tokenizer = load_tokenizer()
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("a fast tokenizer is required for exact character offsets")

    print(f"\nMeasuring {len(sel):,} SYNTAX/SEMANTIC files...")
    records = []
    for i, row in enumerate(sel.itertuples(), start=1):
        records.append(_measure(tokenizer, row))
        if i % 5000 == 0:
            print(f"  {i:,}/{len(sel):,}")
    res = pd.DataFrame(records)

    syntax = res[res["leaf"] == "SYNTAX"]
    strata = [
        ("SYNTAX (all)", syntax),
        ("SYNTAX (missing-include only)", syntax[syntax["missing_include"]]),
        ("SYNTAX (excl. missing-include)", syntax[~syntax["missing_include"]]),
        ("SEMANTIC", res[res["leaf"] == "SEMANTIC"]),
    ]
    summary = pd.DataFrame([_summarize(name, frame) for name, frame in strata])
    print("\n(b) is the first compiler error inside the 510 content tokens the model sees?")
    print(summary.set_index("stratum").T.to_string())
    summary.to_csv(REPORTS / "m6_error_visibility.csv", index=False)


if __name__ == "__main__":
    main()
