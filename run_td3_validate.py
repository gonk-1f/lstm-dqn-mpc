"""PyCharm green-button entry for deterministic TD3 Validation8."""
from pathlib import Path
import sys

from experiments.sb3_manual.td3_aggregate.validate import main as validate


def main() -> None:
    arguments = sys.argv[1:]
    if not any(a in ("--model-path", "--help", "-h") or a.startswith("--model-path=")
               for a in arguments):
        root = Path(__file__).resolve().parent
        models = list((root / "outputs/td3_train_500ep/td3").glob(
            "AggregateShipTD3-v0_*/AggregateShipTD3-v0.zip"
        ))
        if not models:
            raise FileNotFoundError("No final TD3 model found; specify --model-path explicitly")
        arguments = ["--model-path", str(max(models, key=lambda p: p.stat().st_mtime)), *arguments]
    validate(arguments)


if __name__ == "__main__":
    main()
