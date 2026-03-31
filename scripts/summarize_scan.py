#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect_json_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def flatten_metadata(meta: dict) -> dict:
    """
    Keep this simple: assume the metadata is already flat.
    Convert all values to plain CSV-friendly representations.
    """
    row = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
        else:
            row[key] = json.dumps(value, sort_keys=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing run metadata JSON files")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    json_files = collect_json_files(input_dir)
    if not json_files:
        raise FileNotFoundError(f"No JSON metadata files found under {input_dir}")

    rows = []
    all_keys = set()

    for path in json_files:
        meta = json.loads(path.read_text())
        row = flatten_metadata(meta)
        row["metadata_file"] = str(path)
        rows.append(row)
        all_keys.update(row.keys())

    # Stable column order: common fields first, everything else alphabetically after
    preferred = [
        "model",
        "noise_model",
        "observable",
        "distance",
        "samples",
        "max_steps",
        "reset_time",
        "p_data",
        "p_meas",
        "p_signal",
        "U",
        "fN",
        "fC",
        "seed",
        "mean_lifetime",
        "logical_error_rate",
        "data_file",
        "metadata_file",
    ]

    ordered_keys = [k for k in preferred if k in all_keys]
    ordered_keys += sorted(k for k in all_keys if k not in ordered_keys)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
