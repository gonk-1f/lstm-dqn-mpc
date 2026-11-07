"""PyCharm green-button entry for final deterministic TD3 Test4 only."""
import sys

from experiments.sb3_manual.td3_aggregate.evaluation import entrypoint_arguments

from experiments.sb3_manual.td3_aggregate.test4 import main as test4


def main() -> None:
    test4(entrypoint_arguments(sys.argv[1:], final_test=True))


if __name__ == "__main__":
    main()
