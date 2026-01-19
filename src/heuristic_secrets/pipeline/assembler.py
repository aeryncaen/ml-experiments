from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Span:
    start: Optional[int]
    end: Optional[int]

    @property
    def is_partial(self) -> bool:
        return self.start is None or self.end is None

    @property
    def length(self) -> Optional[int]:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start


class SpanAssembler:
    def __init__(
        self,
        threshold: float = 0.5,
        min_length: int = 8,
        max_length: int = 256,
    ):
        self.threshold = threshold
        self.min_length = min_length
        self.max_length = max_length

    def extract_spans(
        self,
        predictions: torch.Tensor,
        chunk_start: int = 0,
        is_first_chunk: bool = True,
        is_last_chunk: bool = True,
    ) -> list[Span]:
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        start_positions = (predictions[:, 0] >= self.threshold).nonzero(as_tuple=True)[0].tolist()
        end_positions = (predictions[:, 1] >= self.threshold).nonzero(as_tuple=True)[0].tolist()

        spans = []

        if not is_first_chunk and end_positions:
            first_end = end_positions[0]
            if not start_positions or start_positions[0] > first_end:
                spans.append(Span(start=None, end=chunk_start + first_end + 1))

        for start_pos in start_positions:
            best_end = None
            for end_pos in end_positions:
                if end_pos < start_pos:
                    continue
                length = end_pos - start_pos + 1
                if length < self.min_length:
                    continue
                if length > self.max_length:
                    continue
                best_end = end_pos
                break

            if best_end is not None:
                spans.append(Span(
                    start=chunk_start + start_pos,
                    end=chunk_start + best_end + 1,
                ))
            elif not is_last_chunk:
                spans.append(Span(start=chunk_start + start_pos, end=None))

        return spans
