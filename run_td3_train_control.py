"""PyCharm entry: FC quadratic variation and SOC deviation rewards only."""
import sys
from experiments.sb3_manual.td3_aggregate.simple_training import main

if __name__ == "__main__":
    raise SystemExit(main(["--control-only", *sys.argv[1:]]))
