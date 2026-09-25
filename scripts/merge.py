"""Usage: python scripts/merge.py --config configs/pi05.json (after installation)."""

import sys

from tcr_merging.cli import main

if __name__ == "__main__":
    main(["merge", *sys.argv[1:]])
