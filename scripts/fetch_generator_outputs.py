#!/usr/bin/env python
"""Fetch the thesis datasets and the CTGAN / TVAE baseline outputs from the public
Hugging Face dataset repo into ``data/generators/<dataset>/`` (git-ignored).

    python scripts/fetch_generator_outputs.py [--datasets adult cardio credit] [--force]

Writes, per dataset: ``train.csv``, ``test.csv``, ``ctgan.csv``, ``tvae.csv``. The
generator outputs are stored upstream as HDF5 (``/synthetic_data``); this converts them
to CSV so the comparison needs no HDF5 reader at run time. Requires ``tables`` (PyTables)
for the conversion step only.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "generators"
BASE = "https://huggingface.co/datasets/dayav/pipeline_tabular_data_results/resolve/main"

DATASETS: dict[str, dict[str, str]] = {
    "adult": {"train": "data/adult_train.csv", "test": "data/adult_test.csv"},
    "cardio": {"train": "data/cardio_train.csv", "test": "data/cardio_test.csv"},
    "credit": {"train": "data/credit_card_Train.csv", "test": "data/credit_card_Test.csv"},
}
GENERATORS = ("ctgan", "tvae")


def download(rel: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{rel}"
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as fh:  # noqa: S310 - fixed public host
        while chunk := r.read(1 << 20):
            fh.write(chunk)


def h5_to_csv(h5: Path, csv: Path) -> None:
    frame = pd.read_hdf(h5, key="synthetic_data")
    frame.to_csv(csv, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"unknown dataset {ds}", file=sys.stderr)
            return 2
        folder = args.out / ds
        print(f"{ds}:")
        for split, rel in DATASETS[ds].items():
            download(rel, folder / f"{split}.csv", args.force)
        for gen in GENERATORS:
            h5 = folder / f"{gen}_generation.h5"
            csv = folder / f"{gen}.csv"
            download(f"baseline/{ds}_{gen}_baseline/{ds}_{gen}_baseline_generation.h5", h5, args.force)
            if not csv.exists() or args.force:
                h5_to_csv(h5, csv)
                print(f"  converted {h5.name} -> {csv.name} ({sum(1 for _ in csv.open()) - 1} rows)")
    print(f"done; files under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
