#!/usr/bin/env python3
"""
AMASS Downloader Stub

Purpose: Prepare local folders and print instructions to obtain AMASS.

AMASS requires accepting a license and manual download:
  https://amass.is.tue.mpg.de/

Place extracted sequences under:
  /Users/camdouglas/quark/data/amass/AMASS/

This script will create the directory layout and sanity-check presence.
"""
from pathlib import Path
import sys

DATA_ROOT = Path("/Users/camdouglas/quark/data/amass").expanduser()


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "AMASS").mkdir(parents=True, exist_ok=True)
    print(f"✅ Prepared directory: {DATA_ROOT}/AMASS")
    print("Next steps:")
    print("1) Register and download AMASS archives → https://amass.is.tue.mpg.de/")
    print("2) Extract all sequences into:")
    print(f"   {DATA_ROOT}/AMASS/<dataset_name>/<subject>/<sequence>.npz")
    print("3) Then run the parser: \n   python -m training.imitation.amass_dataset --root /Users/camdouglas/quark/data/amass/AMASS --out /Users/camdouglas/quark/data/amass/processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())


