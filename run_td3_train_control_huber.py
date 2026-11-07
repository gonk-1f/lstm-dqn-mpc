"""PyCharm entry for the SOC12 control experiment with Huber critic loss."""
import sys

from experiments.sb3_manual.td3_aggregate.simple_training import main


if __name__ == "__main__":
    raise SystemExit(main(["--control-huber", *sys.argv[1:]]))
