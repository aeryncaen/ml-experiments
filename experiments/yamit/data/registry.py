"""
Dataset registry for YAMIT training pipeline.

All sources from the SmolLM3 3-stage pretraining recipe, with HuggingFace paths
and per-stage sampling weights extracted from the nanotron YAML configs.

Usage:
    from data.registry import DATASETS, get_stage_mix

    stage1 = get_stage_mix(1)
    for entry in stage1:
        print(entry.name, entry.hf_path, entry.weight)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Category(Enum):
    WEB = "web"
    WEB_MULTILINGUAL = "web_multilingual"
    CODE = "code"
    MATH = "math"
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"


@dataclass(frozen=True)
class Dataset:
    """A single data source in the training mix."""

    name: str
    hf_path: str
    hf_subset: Optional[str]  # None means default config
    text_column: str
    category: Category
    # Sampling weight per stage. Missing key = not used in that stage.
    stage_weights: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StageMixEntry:
    """A dataset with its resolved weight for a specific stage."""

    name: str
    hf_path: str
    hf_subset: Optional[str]
    text_column: str
    category: Category
    weight: float


def get_stage_mix(stage: int) -> list[StageMixEntry]:
    """Return the dataset mix for a given stage (1, 2, or 3).

    Weights are returned as-is from the configs (they sum to ~1.0 per stage).
    """
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2, or 3, got {stage}")
    entries = []
    for ds in DATASETS:
        if stage in ds.stage_weights:
            entries.append(
                StageMixEntry(
                    name=ds.name,
                    hf_path=ds.hf_path,
                    hf_subset=ds.hf_subset,
                    text_column=ds.text_column,
                    category=ds.category,
                    weight=ds.stage_weights[stage],
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Full dataset registry.
#
# Sources and weights are from the SmolLM3 nanotron configs:
#   stage1_8T.yaml, stage2_8T_9T.yaml, stage3_9T_11T.yaml
#
# Notes on HF paths / subsets:
#   - Some datasets have multiple subsets (e.g., dolmino-mix has pes2o, wiki,
#     stackexchange as separate viewers). We list each as a separate entry.
#   - FineWeb2-HQ is used for most multilingual sources; FineWeb-2 (not HQ)
#     is used for Hindi, Thai, Korean.
#   - Stack v2 languages are subsets of bigcode/the-stack-v2, filtered by
#     programming language. The HF dataset supports language filtering.
#   - Stage 2 introduces Stack-Edu (HuggingFaceTB/stack-edu) for some
#     languages, replacing Stack v2. Stage 3 uses Stack-Edu for all languages.
#   - text_column values are best-effort from HF dataset cards. The download
#     script should verify at runtime.
# ---------------------------------------------------------------------------

DATASETS: list[Dataset] = [
    # =========================================================================
    # WEB (English)
    # =========================================================================
    Dataset(
        name="fineweb-edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_subset=None,
        text_column="text",
        category=Category.WEB,
        stage_weights={1: 0.333, 2: 0.333, 3: 0.2},
    ),
    Dataset(
        name="dclm",
        hf_path="mlfoundations/dclm-baseline-1.0",
        hf_subset=None,
        text_column="text",
        category=Category.WEB,
        stage_weights={1: 0.37, 2: 0.37, 3: 0.3},
    ),
    Dataset(
        name="pes2o",
        hf_path="allenai/dolmino-mix-1124",
        hf_subset="pes2o",
        text_column="text",
        category=Category.KNOWLEDGE,
        stage_weights={1: 0.02, 2: 0.02, 3: 0.002},
    ),
    Dataset(
        name="wiki",
        hf_path="allenai/dolmino-mix-1124",
        hf_subset="wiki",
        text_column="text",
        category=Category.KNOWLEDGE,
        stage_weights={1: 0.001, 2: 0.001, 3: 0.0002},
    ),
    Dataset(
        name="stackexchange",
        hf_path="allenai/dolmino-mix-1124",
        hf_subset="stackexchange",
        text_column="text",
        category=Category.KNOWLEDGE,
        stage_weights={1: 0.004, 2: 0.004, 3: 0.001},
    ),
    # =========================================================================
    # WEB (Multilingual) — FineWeb2-HQ except Hindi, Thai, Korean use FineWeb-2
    # =========================================================================
    Dataset(
        name="fw2-fra",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="hun_Latn",  # per SmolLM3 config comment — likely typo, actually fra_Latn
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.016, 2: 0.016, 3: 0.018},
    ),
    Dataset(
        name="fw2-spa",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="spa_Latn",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.02, 2: 0.02, 3: 0.022},
    ),
    Dataset(
        name="fw2-deu",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="deu_Latn",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.022, 2: 0.0232, 3: 0.023},
    ),
    Dataset(
        name="fw2-ita",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="ita_Latn",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.0105, 2: 0.0105, 3: 0.0125},
    ),
    Dataset(
        name="fw2-por",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="por_Latn",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.01, 2: 0.01, 3: 0.0045},
    ),
    Dataset(
        name="fw2-cmn",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="cmn_Hani",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.01, 2: 0.01, 3: 0.01},
    ),
    Dataset(
        name="fw2-rus",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="rus_Cyrl",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.01, 2: 0.01, 3: 0.01},
    ),
    Dataset(
        name="fw2-fas",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="fas_Arab",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.003, 2: 0.002, 3: 0.009},
    ),
    Dataset(
        name="fw2-jpn",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="jpn_Jpan",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.00325, 2: 0.00325, 3: 0.0032},
    ),
    # Hindi, Thai, Korean use FineWeb-2 (not HQ)
    Dataset(
        name="fw2-kor",
        hf_path="HuggingFaceFW/fineweb-2",
        hf_subset="kor_Hang",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.00325, 2: 0.00325, 3: 0.0032},
    ),
    Dataset(
        name="fw2-hin",
        hf_path="HuggingFaceFW/fineweb-2",
        hf_subset="hin_Deva",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.00325, 2: 0.00325, 3: 0.0032},
    ),
    Dataset(
        name="fw2-tha",
        hf_path="HuggingFaceFW/fineweb-2",
        hf_subset="tha_Thai",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.00325, 2: 0.00005, 3: 0.0032},
    ),
    Dataset(
        name="fw2-vie",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="vie_Latn",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.00325, 2: 0.00325, 3: 0.00005},
    ),
    Dataset(
        name="fw2-ell",
        hf_path="epfml/FineWeb2-HQ",
        hf_subset="ell_Grek",
        text_column="text",
        category=Category.WEB_MULTILINGUAL,
        stage_weights={1: 0.00225, 2: 0.00225, 3: 0.0022},
    ),
    # =========================================================================
    # MATH
    # =========================================================================
    Dataset(
        name="infiwebmath",
        hf_path="HuggingFaceTB/finemath",
        hf_subset="infiwebmath-3plus",
        text_column="text",
        category=Category.MATH,
        stage_weights={1: 0.01, 2: 0.01, 3: 0.002},
    ),
    Dataset(
        name="finemath",
        hf_path="HuggingFaceTB/finemath",
        hf_subset="finemath-3plus",
        text_column="text",
        category=Category.MATH,
        stage_weights={1: 0.017, 2: 0.01, 3: 0.002},
    ),
    # Stage 2+ additions
    Dataset(
        name="infiwebmath-4plus",
        hf_path="HuggingFaceTB/finemath",
        hf_subset="infiwebmath-4plus",
        text_column="text",
        category=Category.MATH,
        stage_weights={2: 0.01, 3: 0.02},
    ),
    Dataset(
        name="finemath-4plus",
        hf_path="HuggingFaceTB/finemath",
        hf_subset="finemath-4plus",
        text_column="text",
        category=Category.MATH,
        stage_weights={2: 0.02, 3: 0.025},
    ),
    Dataset(
        name="megamath-web-pro",
        hf_path="LLM360/MegaMath",
        hf_subset="web-pro",
        text_column="text",
        category=Category.MATH,
        stage_weights={2: 0.02, 3: 0.014},
    ),
    Dataset(
        name="megamath-qa-qwen",
        hf_path="LLM360/MegaMath",
        hf_subset="qa-qwen",
        text_column="text",
        category=Category.MATH,
        stage_weights={2: 0.0008, 3: 0.002},
    ),
    Dataset(
        name="megamath-text-code-block",
        hf_path="LLM360/MegaMath",
        hf_subset="text-code-block",
        text_column="text",
        category=Category.MATH,
        stage_weights={2: 0.02, 3: 0.05},
    ),
    # =========================================================================
    # CODE — StarCoder / The Stack
    # =========================================================================
    Dataset(
        name="starcoder",
        hf_path="bigcode/starcoderdata",
        hf_subset=None,
        text_column="content",
        category=Category.CODE,
        stage_weights={1: 0.06, 2: 0.06, 3: 0.05},
    ),
    # =========================================================================
    # CODE — GitHub auxiliary (all stages)
    # =========================================================================
    # NOTE: pull-requests and jupyter-scripts were internal HuggingFace
    # datasets used for SmolLM3 training but not publicly released.
    # Their weights are redistributed to github-issues and kaggle.
    # Original weights: pull-requests {1: 0.006, 2: 0.0114, 3: 0.005},
    #                   jupyter-scripts {1: 0.0055, 2: 0.01, 3: 0.012}
    Dataset(
        name="kaggle",
        hf_path="HuggingFaceTB/issues-kaggle-notebooks",
        hf_subset="kaggle",
        text_column="text",
        category=Category.CODE,
        stage_weights={1: 0.0005, 2: 0.0005, 3: 0.0006},
    ),
    Dataset(
        name="github-issues",
        hf_path="HuggingFaceTB/issues-kaggle-notebooks",
        hf_subset="issues",
        text_column="text",
        category=Category.CODE,
        stage_weights={1: 0.0032, 2: 0.004, 3: 0.004},
    ),
    # =========================================================================
    # KNOWLEDGE — Stage 3 additions
    # =========================================================================
    Dataset(
        name="multilingual-wiki",
        hf_path="allenai/dolmino-mix-1124",
        hf_subset="wiki",  # TBD: may need specific multilingual wiki dataset
        text_column="text",
        category=Category.KNOWLEDGE,
        stage_weights={3: 0.008},
    ),
    Dataset(
        name="cosmopedia2",
        hf_path="HuggingFaceTB/smollm-corpus",
        hf_subset="cosmopedia-v2",
        text_column="text",
        category=Category.KNOWLEDGE,
        stage_weights={3: 0.004},
    ),
    # =========================================================================
    # REASONING — Stage 3 additions
    # =========================================================================
    Dataset(
        name="openmathinstruct-2",
        hf_path="nvidia/OpenMathReasoning",
        hf_subset="openmathinstruct-2",
        text_column="text",
        category=Category.REASONING,
        stage_weights={3: 0.005},
    ),
    Dataset(
        name="openmathreasoning-4k",
        hf_path="nvidia/OpenMathReasoning",
        hf_subset=None,
        text_column="text",
        category=Category.REASONING,
        stage_weights={3: 0.005},
    ),
    Dataset(
        name="opencodereasoning-4k",
        hf_path="nvidia/OpenCodeReasoning",
        hf_subset=None,
        text_column="text",
        category=Category.REASONING,
        stage_weights={3: 0.0005},
    ),
    Dataset(
        name="natural-reasoning",
        hf_path="facebook/natural_reasoning",
        hf_subset=None,
        text_column="text",
        category=Category.REASONING,
        stage_weights={3: 0.001},
    ),
    Dataset(
        name="tinygsm-problem-solving",
        hf_path="allenai/dolmino-mix-1124",
        hf_subset="tinygsm-problem-solving",
        text_column="text",
        category=Category.REASONING,
        stage_weights={3: 0.003},
    ),
    Dataset(
        name="tinygsm-2students",
        hf_path="allenai/dolmino-mix-1124",
        hf_subset="tinygsm-2students",
        text_column="text",
        category=Category.REASONING,
        stage_weights={3: 0.003},
    ),
    Dataset(
        name="dolmino-math-synth-gsm8k",
        hf_path="allenai/dolmino-mix-1124",
        hf_subset="math-synth-gsm8k",
        text_column="text",
        category=Category.REASONING,
        stage_weights={3: 0.0004},
    ),
    Dataset(
        name="dolmino-math-synth-basic",
        hf_path="allenai/dolmino-mix-1124",
        hf_subset="math-synth-basic",
        text_column="text",
        category=Category.REASONING,
        stage_weights={3: 0.0002},
    ),
]


def print_stage_summary(stage: int) -> None:
    """Print a summary of a stage's dataset mix."""
    mix = get_stage_mix(stage)
    total = sum(e.weight for e in mix)
    print(f"\nStage {stage}: {len(mix)} datasets, total weight = {total:.4f}")
    print(f"{'Name':<30} {'Category':<20} {'Weight':>8} {'Pct':>7}")
    print("-" * 67)
    for e in sorted(mix, key=lambda x: -x.weight):
        pct = 100 * e.weight / total if total > 0 else 0
        print(f"{e.name:<30} {e.category.value:<20} {e.weight:>8.5f} {pct:>6.2f}%")


if __name__ == "__main__":
    for s in (1, 2, 3):
        print_stage_summary(s)
