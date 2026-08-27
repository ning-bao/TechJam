#!/usr/bin/env python3
"""Flatten Bao's per-condition metrics JSON files into Zhang's long CSV."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys


EVAL_15 = [
    "clean",
    "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30",
    "blur_05", "blur_10", "blur_20",
    "resize_050", "resize_025",
    "noise_002", "noise_005", "noise_010",
    "jitter_pm20",
    "crop_80",
]

# The producer's --condition accepts either spelling (REF_eval_atoms.canonical_atom),
# so its output files may be named either way. Superset of REF_eval_atoms.TC2_ALIASES:
# that table does not cover `resize_0.50` or `jitter_20`, both of which appear in the
# consumer's own condition vector, so resolving only via TC2_ALIASES would drop them.
ALIASES = {
    "jpeg_q90": "jpeg_90", "jpeg_q70": "jpeg_70",
    "jpeg_q50": "jpeg_50", "jpeg_q30": "jpeg_30",
    "blur_0.5": "blur_05", "blur_1.0": "blur_10", "blur_2.0": "blur_20",
    "resize_0.5": "resize_050", "resize_0.50": "resize_050",
    "resize_0.25": "resize_025",
    "noise_0.02": "noise_002", "noise_0.05": "noise_005",
    "noise_0.10": "noise_010",
    "color_jitter_0.20": "jitter_pm20", "jitter_20": "jitter_pm20",
    "center_crop_0.80": "crop_80",
}

FIELDNAMES = ["image_path", "label", "pred", "condition", "generator"]
REQUIRED_PRED_KEYS = ["path", "label", "pred", "generator_family"]


def canonical(name):
    """Canonical condition name from either spelling; unknown names pass through
    so the caller can decide to exclude them (or admit them via --atoms)."""
    if name in EVAL_15:
        return name
    return ALIASES.get(name, name)


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold-out")
    ap.add_argument("--atoms", action="append", default=[],
                    help="extra non-EVAL_15 condition to include; repeatable")
    return ap.parse_args()


def json_paths(metrics_dir):
    return sorted(
        p for p in metrics_dir.glob("*.json")
        if not p.name.endswith(".done.json")
    )


def read_threshold(obj, cond):
    calib = obj.get("calibration")
    if isinstance(calib, dict) and "threshold" in calib:
        value = calib["threshold"]
    elif "threshold" in obj:
        value = obj["threshold"]
    else:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{cond}: threshold is not a float")


def validate_label(value, cond, index):
    if isinstance(value, bool):
        raise ValueError(f"{cond}: prediction {index} has label not in {{0,1}}")
    try:
        label = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{cond}: prediction {index} has label not in {{0,1}}")
    if label not in (0, 1) or str(value) not in ("0", "1"):
        raise ValueError(f"{cond}: prediction {index} has label not in {{0,1}}")
    return label


def validate_pred(value, cond, index):
    try:
        pred = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{cond}: prediction {index} has pred outside [0,1]")
    if not 0.0 <= pred <= 1.0:
        raise ValueError(f"{cond}: prediction {index} has pred outside [0,1]")
    return pred


def load_condition_file(path):
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)

    # Compare after canonicalizing both sides: a file named jpeg_q30.json holding
    # body condition "jpeg_30" is the same condition spelled two legal ways, not a
    # mismatch. A genuine disagreement (different conditions) still hard-errors.
    file_cond = canonical(path.stem)
    raw_body = obj.get("condition", file_cond)
    if not isinstance(raw_body, str):
        raise ValueError(f"{path.name}: body condition is not a string "
                         f"({raw_body!r})")
    body_cond = canonical(raw_body)
    if body_cond != file_cond:
        raise ValueError(
            f"{path.name}: body/filename condition mismatch "
            f"({body_cond!r} != {file_cond!r})"
        )
    cond = body_cond

    if "predictions" not in obj:
        raise ValueError(f"{cond}: missing predictions")
    predictions = obj["predictions"]
    if not isinstance(predictions, list):
        raise ValueError(f"{cond}: predictions is not a list")

    rows = []
    n_real = 0
    n_fake = 0
    for i, pred_row in enumerate(predictions, start=1):
        if not isinstance(pred_row, dict):
            raise ValueError(f"{cond}: prediction {i} is not an object")
        missing = [k for k in REQUIRED_PRED_KEYS if k not in pred_row]
        if missing:
            raise ValueError(
                f"{cond}: prediction {i} missing required key(s): "
                f"{', '.join(missing)}"
            )

        image_path = pred_row["path"]
        if image_path is None:
            raise ValueError(f"{cond}: prediction {i} missing image path")
        image_path = str(image_path)

        label = validate_label(pred_row["label"], cond, i)
        score = validate_pred(pred_row["pred"], cond, i)
        generator = str(pred_row["generator_family"])
        if generator == "":
            generator = "real"

        if label == 0:
            n_real += 1
        else:
            n_fake += 1

        rows.append({
            "image_path": image_path,
            "label": label,
            "pred": f"{score:.6f}",
            "condition": cond,
            "generator": generator,
        })

    return cond, rows, n_real, n_fake, read_threshold(obj, cond)


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_path = Path(args.out)

    def bail(message):
        """Fail AND remove any output from an earlier successful run. The consumer
        cannot distinguish a stale CSV from a current one, so a failed re-run must
        not leave one behind: absent is recoverable, silently-outdated is not."""
        if out_path.exists():
            out_path.unlink()
            print(f"[warn] removed stale output {out_path.name} from an earlier "
                  f"run; it no longer reflects the metrics dir", file=sys.stderr)
        return fail(message)

    paths = json_paths(metrics_dir)
    if not paths:
        return bail(f"no JSON files found in {metrics_dir}")

    include_order = list(EVAL_15)
    for atom in args.atoms:
        atom = canonical(atom)
        if atom not in include_order:
            include_order.append(atom)
    include_set = set(include_order)

    by_condition = {}
    counts = {}
    thresholds = {}
    seen_pairs = set()

    try:
        for path in paths:
            cond, rows, n_real, n_fake, threshold = load_condition_file(path)
            if cond not in include_set:
                print(f"[warn] excluding condition not requested: {cond}",
                      file=sys.stderr)
                continue
            if cond in by_condition:
                raise ValueError(f"{cond}: duplicate result file")
            # A file with zero predictions is NOT a written condition: emitting it
            # would put a header-only CSV downstream, where the consumer's own
            # thin-data guard (n > 0 & n < 100) does not fire and the figures come
            # out empty without a single warning. Treat it as missing.
            if not rows:
                print(f"[warn] condition {cond} has zero predictions; "
                      f"treating as MISSING, not written", file=sys.stderr)
                continue
            for row in rows:
                pair = (row["image_path"], row["condition"])
                if pair in seen_pairs:
                    raise ValueError(
                        f"{cond}: duplicate (image_path, condition) pair "
                        f"{pair!r}"
                    )
                seen_pairs.add(pair)
            by_condition[cond] = rows
            counts[cond] = (len(rows), n_real, n_fake)
            if threshold is not None:
                thresholds[cond] = threshold
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return bail(str(exc))

    missing = [c for c in EVAL_15 if c not in by_condition]
    for cond in missing:
        print(f"[warn] missing condition: {cond}", file=sys.stderr)

    ordered_conditions = [c for c in include_order if c in by_condition]
    if not ordered_conditions:
        return bail("no conditions written")

    if thresholds:
        first_cond = next(iter(thresholds))
        first_value = thresholds[first_cond]
        for cond, value in thresholds.items():
            if value != first_value:
                return bail(
                    "conditions disagree on threshold: "
                    f"{first_cond}={first_value:g}, {cond}={value:g}"
                )
    threshold_value = None
    if args.threshold_out:
        missing_threshold = [c for c in ordered_conditions if c not in thresholds]
        if missing_threshold:
            return bail(
                "missing threshold for condition(s): "
                f"{', '.join(missing_threshold)}"
            )
        threshold_value = thresholds[ordered_conditions[0]]

    # Atomic write: the consumer cannot tell a stale successful run from a fresh
    # one by looking at the file, so a failed re-run must never leave the previous
    # CSV in place looking current. Same discipline as the producer's
    # atomic_write_json. Every validation above has already run at this point.
    out_path = Path(args.out)
    tmp_path = out_path.with_name(out_path.name + ".partial")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES,
                                    lineterminator="\n")
            writer.writeheader()
            for cond in ordered_conditions:
                writer.writerows(by_condition[cond])
        os.replace(tmp_path, out_path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        return bail(str(exc))

    # After the CSV, never before: threshold.txt is the operating point FOR that
    # CSV, and a threshold file left over from a run whose table never landed is
    # exactly the kind of mismatch the consumer would apply without noticing.
    if args.threshold_out:
        try:
            Path(args.threshold_out).write_text(
                f"{threshold_value:g}\n", encoding="utf-8", newline="\n")
        except OSError as exc:
            return fail(f"CSV written to {out_path} but threshold file failed: "
                        f"{exc}")

    total = 0
    total_real = 0
    total_fake = 0
    for cond in ordered_conditions:
        n_rows, n_real, n_fake = counts[cond]
        total += n_rows
        total_real += n_real
        total_fake += n_fake
        print(f"{cond}\t{n_rows}\t{n_real}\t{n_fake}")
    print(f"TOTAL\t{total}\t{total_real}\t{total_fake}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
