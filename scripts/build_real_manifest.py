"""Build a small manifest from a directory of official-format MRIxFields NIfTI files.

Meant for ad-hoc dry runs (e.g. a Drive-mounted Colab runtime) against a handful of real
volumes. Never commit the output manifest to the repo -- it contains real file paths,
which AGENTS.md forbids. Write it to a scratch/tmp location instead.

Usage:
    python scripts/build_real_manifest.py --data-root /content/drive/MyDrive/.../Data \
        --out /content/manifest.json --max-records 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fieldbridge.data.manifests import Manifest
from fieldbridge.data.sources import records_from_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=8)
    args = parser.parse_args()

    records = records_from_directory(args.data_root, max_records=args.max_records)
    if not records:
        raise SystemExit(f"No NIfTI files found under {args.data_root}.")

    manifest = Manifest.from_records(records, name="real-dry-run")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} records to {args.out}")
    for record in records:
        print(f"  {record.case_id}: {record.domain.label} -> {record.image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
