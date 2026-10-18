"""Read recorded JSON or gzip trajectories one sample at a time, without Torch."""
from __future__ import annotations

import gzip
import json
from pathlib import Path


def resolve_trajectory_path(path: str | Path) -> Path:
    """Prefer an existing JSON; otherwise use its lossless .json.gz archive."""
    path = Path(path)
    if path.is_file():
        return path
    compressed = Path(str(path) + '.gz')
    if path.suffix != '.gz' and compressed.is_file():
        return compressed
    raise FileNotFoundError(path)


def open_trajectory(path: str | Path):
    """Return a text context manager, transparently resolving archived JSON."""
    path = resolve_trajectory_path(path)
    return (gzip.open(path, 'rt', encoding='utf-8-sig') if path.suffix == '.gz'
            else path.open(encoding='utf-8-sig'))


class _JSONStream:
    def __init__(self, handle):
        self.handle, self.buffer, self.position = handle, '', 0
        self.decoder = json.JSONDecoder()

    def read_more(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        chunk = self.handle.read(1024 * 1024)
        self.buffer += chunk
        return bool(chunk)

    def peek(self):
        while True:
            while self.position < len(self.buffer):
                value = self.buffer[self.position]
                if not value.isspace():
                    return value
                self.position += 1
            if not self.read_more():
                return ''

    def expect(self, token):
        if self.peek() != token:
            raise ValueError(f'invalid trajectory JSON: expected {token!r}')
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, self.position = self.decoder.raw_decode(self.buffer, self.position)
                return value
            except json.JSONDecodeError:
                if not self.read_more():
                    raise ValueError('invalid or incomplete trajectory JSON') from None


def iter_trajectory_records(path: str | Path, *, sample_id: str | None = None):
    """Yield (completed/failed, profile), retaining at most one decoded profile.

    Accepts both the original indented format and compact JSON. This reads
    already recorded evidence only; it never loads a dataset or runs a policy.
    """
    with open_trajectory(path) as handle:
        stream = _JSONStream(handle)
        stream.expect('{')
        seen = set()
        while stream.peek() != '}':
            section = stream.value()
            if not isinstance(section, str) or section in seen:
                raise ValueError('invalid or duplicate trajectory section')
            seen.add(section)
            stream.expect(':')
            if section in ('completed', 'failed'):
                stream.expect('[')
                while stream.peek() != ']':
                    record = stream.value()
                    if not isinstance(record, dict):
                        raise ValueError('trajectory profile must be a JSON object')
                    if sample_id is None or record.get('sample_id') == sample_id:
                        yield section, record
                    if stream.peek() == ']':
                        break
                    stream.expect(',')
                    if stream.peek() == ']':
                        raise ValueError('invalid trailing comma in trajectory array')
                stream.expect(']')
            else:
                raise ValueError(f'unrecognized trajectory section: {section!r}')
            if stream.peek() == '}':
                break
            stream.expect(',')
            if stream.peek() == '}':
                raise ValueError('invalid trailing comma in trajectory object')
        stream.expect('}')
        if stream.peek():
            raise ValueError('unexpected data after trajectory JSON')
