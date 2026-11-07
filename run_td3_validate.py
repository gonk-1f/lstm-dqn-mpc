"""PyCharm green-button entry for deterministic TD3 Validation8."""
import sys

from experiments.sb3_manual.td3_aggregate.evaluation import entrypoint_arguments

from experiments.sb3_manual.td3_aggregate.validate import main as validate


def main() -> None:
    validate(entrypoint_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
