# Training Data Pipeline Design

**Design Document**  
**Date:** 2026-01-17  
**Status:** Approved

---

## Overview & Goals

A pluggable data collection system that gathers training data from multiple sources, normalizes it, and produces train/val/test splits for both models.

### Goals

- Collect real secrets from existing tool test suites and public leaks
- Collect false positives (high-entropy non-secrets) for Validator training
- Collect contextual data with span positions for SpanFinder training
- Easy to add new sources over time
- Reproducible builds with manifest tracking

### Two Distinct Data Needs

| Model | Data Type | Format |
|-------|-----------|--------|
| **Validator** | Isolated strings | `{"text": "...", "label": 0\|1, "source": "..."}` |
| **SpanFinder** | Text chunks with span positions | `{"text": "...", "starts": [...], "ends": [...], "source": "..."}` |

### Priority

Validator first (simpler data needs), SpanFinder when SecretBench access is available.

---

## Architecture

### Plugin-based Collector System

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Build Script: python -m heuristic_secrets.data.build                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │  Gitleaks   │ │ TruffleHog  │ │   UUIDs     │
            │  Collector  │ │  Collector  │ │  Generator  │
            └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                   │               │               │
                   └───────────────┼───────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  Deduplication (exact match) │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  Split Generator (80/10/10) │
                    └──────────────┬──────────────┘
                                   ▼
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
    data/splits/train.jsonl  data/splits/val.jsonl  data/splits/test.jsonl
```

### Base Collector Class

```python
class Collector(ABC):
    name: str                           # "gitleaks", "trufflehog", etc.
    data_type: str                      # "validator" | "spanfinder"
    
    @abstractmethod
    def collect(self) -> Iterator[Sample]:
        """Yield samples from this source."""
    
    def setup(self) -> None:
        """Auto-fetch: clone repos, download files, etc."""
    
    def cache_path(self) -> Path:
        """Return path in cache dir for this collector's data."""
```

### Sample Types

```python
@dataclass
class ValidatorSample:
    text: str
    label: int      # 1 = secret, 0 = not secret
    source: str

@dataclass  
class SpanFinderSample:
    text: str
    starts: list[int]
    ends: list[int]
    source: str
```

---

## Data Sources

### Validator Sources (Phase 1 Priority)

| Source | Type | Data | Auto-fetch |
|--------|------|------|------------|
| **Gitleaks** | Secrets | Test fixtures with real secret patterns | Clone repo |
| **TruffleHog** | Secrets | Detector test cases | Clone repo |
| **detect-secrets** | Secrets | Plugin test fixtures | Clone repo |
| **Public Leaks** | Secrets | Curated breach disclosures | Download |
| **UUIDs** | False positives | Programmatically generated | Generate |
| **Hashes** | False positives | MD5, SHA1, SHA256 examples | Generate |
| **Base64** | False positives | Random base64 strings | Generate |
| **FPSecretBench** | False positives | 1.5M labeled false positives | Manual (requires DPA) |

### SpanFinder Sources (Phase 2)

| Source | Type | Data | Auto-fetch |
|--------|------|------|------------|
| **SecretBench** | Contextual | 15K secrets with repo/file/position metadata | Manual (requires DPA) |

### SecretBench Processing

1. Parse metadata (repo URL, file path, secret position)
2. Clone repo or fetch file via raw GitHub URL
3. Read entire file, chunk with 512/64 overlap
4. Label each chunk: full spans, partial spans, or negative (no secret)
5. Output SpanFinderSample with sparse positions

---

## File Structure & Caching

### Cache Directory (Auto-fetched Sources)

Default: `~/.cache/heuristic-secrets/`  
Override: `HEURISTIC_SECRETS_CACHE` environment variable

```
~/.cache/heuristic-secrets/
├── sources/
│   ├── gitleaks/                    # Cloned repo
│   ├── trufflehog/                  # Cloned repo
│   ├── detect-secrets/              # Cloned repo
│   └── public-leaks/                # Downloaded files
└── .cache_manifest.json             # Tracks versions, timestamps
```

### Project Data Directory (Outputs)

```
data/
├── sources/                         # Intermediate collected data
│   ├── gitleaks.jsonl
│   ├── trufflehog.jsonl
│   ├── detect_secrets.jsonl
│   ├── public_leaks.jsonl
│   ├── generated_uuids.jsonl
│   ├── generated_hashes.jsonl
│   ├── generated_base64.jsonl
│   └── fpbench.jsonl                # When available
├── splits/
│   ├── validator/
│   │   ├── train.jsonl              # 80%
│   │   ├── val.jsonl                # 10%
│   │   └── test.jsonl               # 10%
│   └── spanfinder/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
└── manifest.json                    # Build metadata
```

### manifest.json

```json
{
  "created_at": "2026-01-17T12:00:00Z",
  "sources": {
    "gitleaks": {"version": "v8.18.0", "samples": 1234},
    "trufflehog": {"version": "v3.63.0", "samples": 892},
    "generated_uuids": {"samples": 5000}
  },
  "splits": {
    "validator": {"train": 8000, "val": 1000, "test": 1000},
    "spanfinder": {"train": 0, "val": 0, "test": 0}
  },
  "deduplication": {"before": 12500, "after": 10000}
}
```

---

## Build Pipeline

### CLI Usage

```bash
# Build all available data
python -m heuristic_secrets.data.build

# Build specific sources only
python -m heuristic_secrets.data.build --sources gitleaks,trufflehog

# Custom output directory
python -m heuristic_secrets.data.build --output ./my-data

# Custom cache location
HEURISTIC_SECRETS_CACHE=/tmp/cache python -m heuristic_secrets.data.build

# Skip auto-fetch (use existing cache only)
python -m heuristic_secrets.data.build --no-fetch
```

### Build Steps

```
1. DISCOVER
   └── Find all registered Collector classes

2. SETUP (auto-fetch)
   ├── gitleaks: git clone https://github.com/gitleaks/gitleaks → cache
   ├── trufflehog: git clone https://github.com/trufflesecurity/trufflehog → cache
   ├── detect-secrets: git clone https://github.com/Yelp/detect-secrets → cache
   └── (skip unavailable sources with warning)

3. COLLECT
   ├── Run each collector's collect() method
   ├── Write intermediate files to data/sources/*.jsonl
   └── Track counts per source

4. DEDUPLICATE
   ├── Group by data_type (validator vs spanfinder)
   ├── Remove exact duplicate texts
   └── Log dedup stats

5. SPLIT
   ├── Shuffle deterministically (seeded random)
   ├── Split 80/10/10 → train/val/test
   └── Write to data/splits/{validator,spanfinder}/*.jsonl

6. MANIFEST
   └── Write data/manifest.json with full build metadata
```

### Example Output

```
$ python -m heuristic_secrets.data.build

[setup] Cloning gitleaks... done (cached)
[setup] Cloning trufflehog... done (cached)
[setup] Cloning detect-secrets... done (cached)
[setup] fpbench not available, skipping

[collect] gitleaks: 1,847 secrets
[collect] trufflehog: 1,203 secrets
[collect] detect-secrets: 412 secrets
[collect] public_leaks: 892 secrets
[collect] generated_uuids: 5,000 false positives
[collect] generated_hashes: 3,000 false positives
[collect] generated_base64: 2,000 false positives

[dedupe] validator: 14,354 → 11,892 (removed 2,462 duplicates)

[split] validator: train=9,513 val=1,189 test=1,190

[done] Wrote manifest to data/manifest.json
```

---

## Collector Implementations

### Gitleaks Collector

```python
class GitleaksCollector(Collector):
    name = "gitleaks"
    data_type = "validator"
    repo_url = "https://github.com/gitleaks/gitleaks"
    
    def setup(self):
        clone_or_pull(self.repo_url, self.cache_path())
    
    def collect(self) -> Iterator[ValidatorSample]:
        # Parse test fixtures in cmd/generate/config/rules/
        # Each rule has test cases with known secrets
        for rule_file in (self.cache_path() / "cmd/generate/config").rglob("*.go"):
            for secret in parse_gitleaks_test_secrets(rule_file):
                yield ValidatorSample(text=secret, label=1, source="gitleaks")
```

### TruffleHog Collector

```python
class TrufflehogCollector(Collector):
    name = "trufflehog"
    data_type = "validator"
    repo_url = "https://github.com/trufflesecurity/trufflehog"
    
    def collect(self) -> Iterator[ValidatorSample]:
        # Parse detector test files in pkg/detectors/*/
        for test_file in (self.cache_path() / "pkg/detectors").rglob("*_test.go"):
            for secret in parse_trufflehog_test_secrets(test_file):
                yield ValidatorSample(text=secret, label=1, source="trufflehog")
```

### UUID Generator

```python
class UUIDGenerator(Collector):
    name = "generated_uuids"
    data_type = "validator"
    
    def setup(self):
        pass  # Nothing to fetch
    
    def collect(self) -> Iterator[ValidatorSample]:
        import uuid
        for _ in range(5000):
            yield ValidatorSample(
                text=str(uuid.uuid4()),
                label=0,
                source="generated_uuids"
            )
```

### SecretBench Collector (SpanFinder)

```python
class SecretBenchCollector(Collector):
    name = "secretbench"
    data_type = "spanfinder"
    
    def collect(self) -> Iterator[SpanFinderSample]:
        for record in load_secretbench_metadata():
            # Fetch file content from GitHub
            content = fetch_github_file(record.repo, record.commit, record.filepath)
            
            # Chunk the file
            for chunk in Chunker(512, 64).chunk(content.encode('utf-8')):
                starts, ends = find_spans_in_chunk(
                    chunk, record.secret_start, record.secret_end
                )
                yield SpanFinderSample(
                    text=chunk.data.decode('utf-8', errors='replace'),
                    starts=starts,
                    ends=ends,
                    source="secretbench"
                )
```

---

## Implementation Order

1. Base `Collector` class and build script infrastructure
2. Generated false positives (UUIDs, hashes, base64) — instant data
3. Gitleaks collector — largest source of real secrets
4. TruffleHog collector
5. detect-secrets collector
6. Public leaks collector
7. FPSecretBench collector (when DPA obtained)
8. SecretBench collector for SpanFinder (when DPA obtained)

---

## Summary

| Component | Decision |
|-----------|----------|
| **Architecture** | Plugin-based Collector system |
| **Validator data** | Tool fixtures + public leaks + generated false positives |
| **SpanFinder data** | SecretBench with full file chunking (when available) |
| **Format** | JSON Lines, minimal metadata (text, label, source) |
| **Deduplication** | Exact match |
| **Splits** | 80/10/10 train/val/test |
| **Execution** | Single build script |
| **Caching** | `~/.cache/heuristic-secrets/`, configurable via env var |
| **Auto-fetch** | Clone repos, download files on first run |
