# heuristic-secrets

Heuristic-based secret detection using ML.

## Overview

A two-stage ML pipeline for secret detection:
1. **SpanFinder**: Byte-level model that finds candidate spans in text
2. **Heuristic Validator**: Classifies candidates using 6 precomputed features

Based on the paper "Beyond RegEx - Heuristic-based Secret Detection" (Burdick-Pless, 2025).

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
