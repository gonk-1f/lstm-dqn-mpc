"""Print the current formal-training gate without reading a data payload."""

from __future__ import annotations

from collections.abc import Sequence

from ..preflight import assess_formal_training_preflight


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    report = assess_formal_training_preflight()
    for check in report.checks:
        print(f"[{check.status.value}] {check.key}: {check.evidence}")
    print(f"FORMAL_TRAINING={report.formal_training}")
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
