"""Export qualitative examples from the TEST split for manual reading (Milestone 8.4).

    python scripts/export_examples.py [--per-category 5]

Writes reports/m8_examples/examples.md (keep it out of git until the CodeNet license for
redistributing snippets is confirmed). Examples are drawn at random (seed 42) inside each
category, not hand-picked. Each example shows the true and predicted class, the model's top
two probabilities, the token count, the compiler's first error text for compile classes, and
the numbered source with a marker where the model's 512-token window ends.

Line numbers here use Python's splitlines, which can differ slightly from g++'s on unusual
line endings; they are for reading, not for measuring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import MANIFEST_9CLASS_PATH, SOURCES_DIR
from sdp.data.taxonomy import LEAF_TO_ID, DefectClass
from sdp.data.tokenization import _decode, load_tokenizer
from sdp.data.visibility import CONTENT_WINDOW
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.evaluation.predictions import LOGIT_COLUMNS, read_dump, require_hash

REPORTS = PROJECT_ROOT / "reports"
M4_CSV = REPORTS / "m4_full_reserve_results_final.csv"
SPLIT_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"
MANIFEST_KEY = "sample_manifest_9class.parquet"
DUMP_NAME = "test_predictions.parquet"
SEED = 42
MAX_LINES = 120

NZEC = LEAF_TO_ID[DefectClass.NZEC]
LINKER = LEAF_TO_ID[DefectClass.LINKER]
SYNTAX = LEAF_TO_ID[DefectClass.SYNTAX]
LOGICAL = LEAF_TO_ID[DefectClass.LOGICAL]
CATEGORIES = (
    ("NZEC files the model got right", lambda t, p: (t == NZEC) & (p == NZEC)),
    ("LINKER files the model missed", lambda t, p: (t == LINKER) & (p != LINKER)),
    ("SYNTAX files predicted as LOGICAL", lambda t, p: (t == SYNTAX) & (p == LOGICAL)),
)


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def window_end_offset(tokenizer, text: str) -> int | None:
    """Character offset where the model's content window ends, or None if the file fits."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    if len(offsets) <= CONTENT_WINDOW:
        return None
    return int(offsets[CONTENT_WINDOW - 1][1])


def render_source(text: str, cut_line: int | None, max_lines: int = MAX_LINES) -> str:
    """Numbered source lines; a marker follows line ``cut_line`` (where the window ends)."""
    lines = text.splitlines()
    shown: list[str] = []
    for number, line in enumerate(lines[:max_lines], start=1):
        shown.append(f"{number:4d} | {line}")
        if cut_line is not None and number == cut_line:
            shown.append("     | ===== the model's 512-token window ends on the line above =====")
    if len(lines) > max_lines:
        shown.append(f"     | ... {len(lines) - max_lines} more lines not shown")
    return "\n".join(shown)


def first_error_text(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if "error" in line.lower()]
    return "\n".join((lines or stderr.splitlines())[:2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--out", type=Path, default=REPORTS / "m8_examples" / "examples.md")
    args = parser.parse_args(argv)

    reports_dir = REPORTS / "m8"
    expected_dump = json.loads((reports_dir / "m8_1_provenance.json").read_text("utf-8"))[
        "dump_sha256"
    ][DUMP_NAME]
    expected_manifest = json.loads(SPLIT_CONFIG_PATH.read_text("utf-8"))["artifact_hashes"][
        MANIFEST_KEY
    ]
    require_hash(reports_dir / DUMP_NAME, expected_dump, "test dump")
    require_hash(MANIFEST_9CLASS_PATH, expected_manifest, "9-class manifest")
    dump = read_dump(reports_dir / DUMP_NAME)
    manifest = pd.read_parquet(
        MANIFEST_9CLASS_PATH, columns=["submission_id", "problem_id", "rel_path"]
    ).set_index("submission_id")
    meta = manifest.loc[dump["submission_id"]]
    y_true = dump["y_true"].to_numpy()
    y_pred = dump["y_pred"].to_numpy()
    probs = probabilities(dump[list(LOGIT_COLUMNS)].to_numpy())

    rng = np.random.default_rng(SEED)
    chosen: list[tuple[str, int]] = []
    for title, rule in CATEGORIES:
        candidates = np.flatnonzero(rule(y_true, y_pred))
        size = min(args.per_category, candidates.size)
        picks = rng.choice(candidates, size=size, replace=False)
        chosen.extend((title, int(index)) for index in sorted(picks))

    wanted = {str(dump["submission_id"].iloc[index]) for _, index in chosen}
    m4 = pd.read_csv(M4_CSV, usecols=["submission_id", "stderr"], dtype=str, keep_default_na=False)
    stderr_by_id = m4[m4["submission_id"].isin(wanted)].set_index("submission_id")["stderr"]
    tokenizer = load_tokenizer()

    out = [
        "# M8 qualitative examples (test split)",
        "",
        f"Drawn at random (seed {SEED}) within each category; not hand-picked.",
        "",
    ]
    current, count = None, 0
    for title, index in chosen:
        if title != current:
            out += [f"## {title}", ""]
            current, count = title, 0
        count += 1
        submission_id = str(dump["submission_id"].iloc[index])
        text = _decode((SOURCES_DIR / meta.iloc[index]["rel_path"]).read_bytes())
        end = window_end_offset(tokenizer, text)
        cut_line = None if end is None else text.count("\n", 0, end) + 1
        order = np.argsort(probs[index])[::-1]
        top, second = int(order[0]), int(order[1])
        out.append(f"### {count}. {submission_id} (problem {meta.iloc[index]['problem_id']})")
        true_name = LEAF_NAMES[int(y_true[index])]
        pred_name = LEAF_NAMES[int(y_pred[index])]
        out.append(
            f"- true **{true_name}**, predicted **{pred_name}** "
            f"(top {LEAF_NAMES[top]} {probs[index, top]:.2f}; "
            f"next {LEAF_NAMES[second]} {probs[index, second]:.2f})"
        )
        if cut_line is None:
            out.append("- the whole file fits in the model's window")
        else:
            out.append(f"- the model's window ends on line {cut_line}; later code was not seen")
        if submission_id in stderr_by_id.index:
            error = first_error_text(stderr_by_id[submission_id])
            if error:
                out += ["- compiler output:", "", "```", error, "```"]
        out += ["", "```cpp", render_source(text, cut_line), "```", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(args.out)
    print("keep reports/m8_examples/ out of git until the CodeNet license is confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
