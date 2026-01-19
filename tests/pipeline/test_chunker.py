import pytest
from heuristic_secrets.pipeline.chunker import Chunker, Chunk


class TestChunker:
    def test_small_input_single_chunk(self):
        chunker = Chunker(chunk_size=512, overlap=64)
        data = b"hello world"

        chunks = list(chunker.chunk(data))

        assert len(chunks) == 1
        assert chunks[0].data == data
        assert chunks[0].start == 0
        assert chunks[0].end == len(data)

    def test_exact_chunk_size(self):
        chunker = Chunker(chunk_size=10, overlap=2)
        data = b"0123456789"

        chunks = list(chunker.chunk(data))

        assert len(chunks) == 1
        assert chunks[0].data == data

    def test_two_chunks_with_overlap(self):
        chunker = Chunker(chunk_size=10, overlap=2)
        data = b"0123456789ABCDEF"

        chunks = list(chunker.chunk(data))

        assert len(chunks) == 2
        assert chunks[0].data == b"0123456789"
        assert chunks[0].start == 0
        assert chunks[1].start == 8
        assert chunks[1].data == data[8:]

    def test_many_chunks(self):
        chunker = Chunker(chunk_size=100, overlap=10)
        data = b"x" * 500

        chunks = list(chunker.chunk(data))

        assert len(chunks) >= 5

        covered = set()
        for chunk in chunks:
            for i in range(chunk.start, chunk.end):
                covered.add(i)
        assert covered == set(range(500))

    def test_empty_input(self):
        chunker = Chunker(chunk_size=512, overlap=64)

        chunks = list(chunker.chunk(b""))

        assert len(chunks) == 0

    def test_chunk_has_is_first_is_last(self):
        chunker = Chunker(chunk_size=10, overlap=2)
        data = b"0123456789ABCDEFGHIJ"

        chunks = list(chunker.chunk(data))

        assert chunks[0].is_first
        assert not chunks[0].is_last
        assert not chunks[-1].is_first
        assert chunks[-1].is_last
