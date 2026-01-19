"""Build script for training data."""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.registry import get_default_registry, CollectorRegistry
from heuristic_secrets.data.pipeline import (
    deduplicate_samples,
    split_samples,
    write_jsonl,
)
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample


@dataclass
class Manifest:
    """Build manifest tracking sources and outputs."""

    created_at: str = ""
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    splits: dict[str, dict[str, int]] = field(default_factory=dict)
    deduplication: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "sources": self.sources,
            "splits": self.splits,
            "deduplication": self.deduplication,
        }


class DataBuilder:
    """Build training data from multiple sources."""

    def __init__(
        self,
        output_dir: Path,
        sources: list[str] | None = None,
        no_fetch: bool = False,
        force_download: bool = False,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.no_fetch = no_fetch
        self.force_download = force_download
        self.seed = seed
        self.registry = get_default_registry()

        # Filter to requested sources
        if sources:
            self._filter_sources(sources)

    def _filter_sources(self, sources: list[str]) -> None:
        """Keep only the specified sources."""
        filtered = CollectorRegistry()
        for name in sources:
            collector = self.registry.get(name)
            if collector:
                filtered._collectors[name] = collector
            else:
                print(f"[warn] Unknown source: {name}", file=sys.stderr)
        self.registry = filtered

    def build(self) -> Manifest:
        """Run the full build pipeline."""
        manifest = Manifest(created_at=datetime.now(timezone.utc).isoformat())

        # Setup phase
        if not self.no_fetch:
            self._setup_collectors()

        # Collect phase
        validator_samples: list[ValidatorSample] = []
        spanfinder_samples: list[SpanFinderSample] = []

        collectors = self.registry.all()
        for collector in tqdm(collectors, desc="Collecting data", unit="collector"):
            tqdm.write(f"[collect] {collector.name}...")

            samples = list(collector.collect())
            count = len(samples)

            if collector.data_type == "validator":
                for s in samples:
                    if isinstance(s, ValidatorSample):
                        validator_samples.append(s)
            else:
                for s in samples:
                    if isinstance(s, SpanFinderSample):
                        spanfinder_samples.append(s)

            tqdm.write(f"[collect] {collector.name}: collecting spans...")
            spans = list(collector.collect_spans())
            span_count = len(spans)
            spanfinder_samples.extend(spans)

            manifest.sources[collector.name] = {
                "samples": count,
                "spans": span_count,
            }
            tqdm.write(f"[collect] {collector.name}: {count} samples, {span_count} spans")

        # Deduplicate phase
        print("[dedupe] validator...")
        before = len(validator_samples)
        validator_samples = deduplicate_samples(validator_samples)
        after = len(validator_samples)
        manifest.deduplication["validator_before"] = before
        manifest.deduplication["validator_after"] = after
        print(f"[dedupe] validator: {before} -> {after} (removed {before - after})")

        # Split phase
        if validator_samples:
            print("[split] validator...")
            train, val, test = split_samples(validator_samples, seed=self.seed)
            manifest.splits["validator"] = {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            }
            print(f"[split] validator: train={len(train)} val={len(val)} test={len(test)}")

            # Write output files
            splits_dir = self.output_dir / "splits" / "validator"
            write_jsonl(train, splits_dir / "train.jsonl")
            write_jsonl(val, splits_dir / "val.jsonl")
            write_jsonl(test, splits_dir / "test.jsonl")

        if spanfinder_samples:
            print("[split] documents...")
            train, val, test = split_samples(spanfinder_samples, seed=self.seed)
            manifest.splits["documents"] = {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            }
            print(f"[split] documents: train={len(train)} val={len(val)} test={len(test)}")

            splits_dir = self.output_dir / "splits" / "documents"
            write_jsonl(train, splits_dir / "train.jsonl")
            write_jsonl(val, splits_dir / "val.jsonl")
            write_jsonl(test, splits_dir / "test.jsonl")

        # Write manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        print(f"[done] Wrote manifest to {manifest_path}")
        return manifest

    def _setup_collectors(self) -> None:
        """Run setup on all collectors."""
        collectors = self.registry.all()
        for collector in tqdm(collectors, desc="Setting up collectors", unit="collector"):
            if hasattr(collector, "setup"):
                tqdm.write(f"[setup] {collector.name}...")
                try:
                    collector.setup(force=self.force_download)
                    tqdm.write(f"[setup] {collector.name} done")
                except Exception as e:
                    tqdm.write(f"[setup] {collector.name} failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build training data")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data"),
        help="Output directory",
    )
    parser.add_argument(
        "--sources",
        type=str,
        help="Comma-separated list of sources to use",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Don't fetch/clone source repositories",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of data even if already present",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else None

    builder = DataBuilder(
        output_dir=args.output,
        sources=sources,
        no_fetch=args.no_fetch,
        force_download=args.force_download,
        seed=args.seed,
    )
    builder.build()


if __name__ == "__main__":
    main()
