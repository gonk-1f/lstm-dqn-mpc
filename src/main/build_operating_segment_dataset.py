"""Compatibility entrypoint for the formal rebuilt operating-segment builder."""

from __future__ import annotations

import json

from build_rebuilt_operating_segment_dataset import build_dataset


def main() -> None:
    print(json.dumps(build_dataset(), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
