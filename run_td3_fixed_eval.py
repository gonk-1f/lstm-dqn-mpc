"""PyCharm green-button entry for deterministic evaluation of the fixed Train voyage."""
import sys

from experiments.sb3_manual.td3_aggregate.evaluation import entrypoint_arguments

from experiments.sb3_manual.td3_aggregate.fixed_train_eval import main as evaluate


def main() -> None:
    evaluate(entrypoint_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
