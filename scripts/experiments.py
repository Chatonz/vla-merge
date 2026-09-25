"""Generate main-ablation or parameter-study configurations without launching jobs."""

import sys

from tcr_merging.cli import main

if __name__ == "__main__":
    main(["sweep", *sys.argv[1:]])
