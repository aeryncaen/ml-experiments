from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch

from heuristic_secrets.spanfinder.model import SpanFinderModel
from heuristic_secrets.validator.model import ValidatorModel
from heuristic_secrets.validator.features import (
    FrequencyTable,
    extract_features,
)
from heuristic_secrets.pipeline.chunker import Chunker
from heuristic_secrets.pipeline.assembler import SpanAssembler


@dataclass
class Detection:
    start: int
    end: int
    text: str
    probability: float


@dataclass
class DetectorConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    spanfinder_threshold: float = 0.5
    validator_threshold: float = 0.5
    min_span_length: int = 8
    max_span_length: int = 256


class Detector:
    def __init__(
        self,
        spanfinder: SpanFinderModel,
        validator: ValidatorModel,
        config: DetectorConfig,
        text_freq: FrequencyTable,
        innocuous_freq: FrequencyTable,
        secret_freq: FrequencyTable,
    ):
        self.spanfinder = spanfinder
        self.validator = validator
        self.config = config
        self.text_freq = text_freq
        self.innocuous_freq = innocuous_freq
        self.secret_freq = secret_freq

        self.chunker = Chunker(config.chunk_size, config.chunk_overlap)
        self.assembler = SpanAssembler(
            threshold=config.spanfinder_threshold,
            min_length=config.min_span_length,
            max_length=config.max_span_length,
        )

        self.spanfinder.eval()
        self.validator.eval()

    def scan(self, text: str) -> list[Detection]:
        if not text:
            return []

        data = text.encode('utf-8')

        candidate_spans = self._find_spans(data)

        detections = []
        for span in candidate_spans:
            if span.is_partial:
                continue

            span_data = data[span.start:span.end]
            probability = self._validate_span(span_data)

            if probability >= self.config.validator_threshold:
                detections.append(Detection(
                    start=span.start,
                    end=span.end,
                    text=span_data.decode('utf-8', errors='replace'),
                    probability=probability,
                ))

        return detections

    def scan_batch(self, texts: list[str]) -> list[list[Detection]]:
        return [self.scan(text) for text in texts]

    def _find_spans(self, data: bytes) -> list:
        all_spans = []

        for chunk in self.chunker.chunk(data):
            byte_values = list(chunk.data)
            x = torch.tensor([byte_values], dtype=torch.long)

            with torch.no_grad():
                predictions = self.spanfinder(x)
                predictions = torch.sigmoid(predictions[0])

            spans = self.assembler.extract_spans(
                predictions,
                chunk_start=chunk.start,
                is_first_chunk=chunk.is_first,
                is_last_chunk=chunk.is_last,
            )
            all_spans.extend(spans)

        return all_spans

    def _validate_span(self, data: bytes) -> float:
        features = extract_features(
            data,
            self.text_freq,
            self.innocuous_freq,
            self.secret_freq,
        )

        x = torch.tensor([features.to_list()], dtype=torch.float32)

        with torch.no_grad():
            probability = self.validator(x)

        return probability.item()

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: Optional[DetectorConfig] = None,
    ) -> "Detector":
        from heuristic_secrets.io import load_model_bundle

        bundle = load_model_bundle(Path(path))

        return cls(
            spanfinder=bundle.spanfinder,
            validator=bundle.validator,
            config=config or bundle.config,
            text_freq=bundle.frequencies["text"],
            innocuous_freq=bundle.frequencies["innocuous"],
            secret_freq=bundle.frequencies["secret"],
        )
