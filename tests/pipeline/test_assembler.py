import pytest
import torch
from heuristic_secrets.pipeline.assembler import SpanAssembler, Span


class TestSpanAssembler:
    def test_simple_span_detection(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)

        predictions = torch.zeros(10, 2)
        predictions[2, 0] = 0.9
        predictions[5, 1] = 0.9

        spans = assembler.extract_spans(predictions, chunk_start=0)

        assert len(spans) == 1
        assert spans[0].start == 2
        assert spans[0].end == 6

    def test_multiple_spans(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)

        predictions = torch.zeros(20, 2)
        predictions[2, 0] = 0.9
        predictions[4, 1] = 0.9
        predictions[10, 0] = 0.9
        predictions[15, 1] = 0.9

        spans = assembler.extract_spans(predictions, chunk_start=0)

        assert len(spans) == 2
        assert spans[0].start == 2
        assert spans[0].end == 5
        assert spans[1].start == 10
        assert spans[1].end == 16

    def test_threshold_filtering(self):
        assembler = SpanAssembler(threshold=0.7, min_length=1, max_length=100)

        predictions = torch.zeros(10, 2)
        predictions[2, 0] = 0.8
        predictions[5, 1] = 0.6
        predictions[7, 1] = 0.9

        spans = assembler.extract_spans(predictions, chunk_start=0)

        assert len(spans) == 1
        assert spans[0].end == 8

    def test_min_length_filtering(self):
        assembler = SpanAssembler(threshold=0.5, min_length=5, max_length=100)

        predictions = torch.zeros(10, 2)
        predictions[2, 0] = 0.9
        predictions[3, 1] = 0.9
        predictions[5, 1] = 0.9
        predictions[8, 1] = 0.9

        spans = assembler.extract_spans(predictions, chunk_start=0)

        assert len(spans) == 1
        assert spans[0].end - spans[0].start >= 5

    def test_max_length_filtering(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=5)

        predictions = torch.zeros(20, 2)
        predictions[2, 0] = 0.9
        predictions[15, 1] = 0.9
        predictions[5, 1] = 0.9

        spans = assembler.extract_spans(predictions, chunk_start=0)

        assert len(spans) == 1
        assert spans[0].end - spans[0].start <= 5

    def test_chunk_offset_applied(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)

        predictions = torch.zeros(10, 2)
        predictions[2, 0] = 0.9
        predictions[5, 1] = 0.9

        spans = assembler.extract_spans(predictions, chunk_start=100)

        assert spans[0].start == 102
        assert spans[0].end == 106

    def test_partial_span_start_only(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)

        predictions = torch.zeros(10, 2)
        predictions[7, 0] = 0.9

        spans = assembler.extract_spans(predictions, chunk_start=0, is_last_chunk=False)

        assert len(spans) == 1
        assert spans[0].end is None

    def test_partial_span_end_only(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)

        predictions = torch.zeros(10, 2)
        predictions[2, 1] = 0.9

        spans = assembler.extract_spans(predictions, chunk_start=0, is_first_chunk=False)

        assert len(spans) == 1
        assert spans[0].start is None

    def test_no_spans(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)

        predictions = torch.zeros(10, 2)

        spans = assembler.extract_spans(predictions, chunk_start=0)

        assert len(spans) == 0
