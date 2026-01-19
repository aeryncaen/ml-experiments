from dataclasses import dataclass
from typing import Iterator


@dataclass
class Chunk:
    data: bytes
    start: int
    end: int
    is_first: bool
    is_last: bool


class Chunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap

    def chunk(self, data: bytes) -> Iterator[Chunk]:
        if len(data) == 0:
            return

        if len(data) <= self.chunk_size:
            yield Chunk(
                data=data,
                start=0,
                end=len(data),
                is_first=True,
                is_last=True,
            )
            return

        pos = 0
        is_first = True

        while pos < len(data):
            end = min(pos + self.chunk_size, len(data))
            is_last = end >= len(data)

            yield Chunk(
                data=data[pos:end],
                start=pos,
                end=end,
                is_first=is_first,
                is_last=is_last,
            )

            if is_last:
                break

            pos += self.stride
            is_first = False
