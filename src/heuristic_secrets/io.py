import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig
from heuristic_secrets.validator.features import FrequencyTable
from heuristic_secrets.pipeline.detector import DetectorConfig


@dataclass
class ModelBundle:
    spanfinder: SpanFinderModel
    validator: ValidatorModel
    config: DetectorConfig
    frequencies: dict[str, FrequencyTable]


def save_model_bundle(
    path: Path,
    spanfinder: SpanFinderModel,
    validator: ValidatorModel,
    detector_config: DetectorConfig,
    frequencies: dict[str, FrequencyTable],
    version: str = "1.0.0",
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    config_data = {
        "version": version,
        "chunk_size": detector_config.chunk_size,
        "chunk_overlap": detector_config.chunk_overlap,
        "spanfinder_threshold": detector_config.spanfinder_threshold,
        "validator_threshold": detector_config.validator_threshold,
        "min_span_length": detector_config.min_span_length,
        "max_span_length": detector_config.max_span_length,
    }
    with open(path / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    sf_state = {k: v for k, v in spanfinder.state_dict().items()}
    save_file(sf_state, path / "spanfinder.safetensors")

    sf_meta = spanfinder.config.to_dict()
    with open(path / "spanfinder.meta.json", "w") as f:
        json.dump(sf_meta, f, indent=2)

    val_state = {
        k: v for k, v in validator.state_dict().items()
        if not k.startswith("attention.")  # Skip alias (attention = head)
    }
    save_file(val_state, path / "validator.safetensors")

    val_meta = {
        **validator.config.to_dict(),
        "frequencies": {
            name: {str(k): v for k, v in table.frequencies.items()}
            for name, table in frequencies.items()
        },
    }
    with open(path / "validator.meta.json", "w") as f:
        json.dump(val_meta, f, indent=2)


def load_model_bundle(path: Path) -> ModelBundle:
    path = Path(path)

    with open(path / "config.json") as f:
        config_data = json.load(f)

    detector_config = DetectorConfig(
        chunk_size=config_data["chunk_size"],
        chunk_overlap=config_data["chunk_overlap"],
        spanfinder_threshold=config_data["spanfinder_threshold"],
        validator_threshold=config_data["validator_threshold"],
        min_span_length=config_data["min_span_length"],
        max_span_length=config_data["max_span_length"],
    )

    with open(path / "spanfinder.meta.json") as f:
        sf_meta = json.load(f)
    sf_config = SpanFinderConfig.from_dict(sf_meta)
    spanfinder = SpanFinderModel(sf_config)
    sf_state = load_file(path / "spanfinder.safetensors")
    spanfinder.load_state_dict(sf_state)

    with open(path / "validator.meta.json") as f:
        val_meta = json.load(f)

    frequencies = {
        name: FrequencyTable({int(k): v for k, v in freqs.items()})
        for name, freqs in val_meta.pop("frequencies").items()
    }

    val_config = ValidatorConfig.from_dict(val_meta)
    validator = ValidatorModel(val_config)
    val_state = load_file(path / "validator.safetensors")
    validator.load_state_dict(val_state, strict=False)

    return ModelBundle(
        spanfinder=spanfinder,
        validator=validator,
        config=detector_config,
        frequencies=frequencies,
    )
