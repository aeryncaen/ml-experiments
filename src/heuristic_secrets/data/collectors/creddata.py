"""Collector for Samsung CredData dataset.

CredData is a curated dataset of ~67k labeled credential samples from GitHub.
https://github.com/Samsung/CredData
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.git_utils import clone_or_pull
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample, SecretCategory


CREDDATA_CATEGORY_MAP = {
    "password": SecretCategory.PASSWORD,
    "auth": SecretCategory.AUTH_TOKEN,
    "token": SecretCategory.AUTH_TOKEN,
    "secret": SecretCategory.GENERIC_SECRET,
    "key": SecretCategory.API_KEY,
    "api": SecretCategory.API_KEY,
    "private": SecretCategory.PRIVATE_KEY,
    "certificate": SecretCategory.PRIVATE_KEY,
    "credentials": SecretCategory.PASSWORD,
    "generic": SecretCategory.GENERIC_SECRET,
    "seed": SecretCategory.GENERIC_SECRET,
    "salt": SecretCategory.GENERIC_SECRET,
}


def map_creddata_category(category_str: str) -> SecretCategory:
    """Map CredData category string to SecretCategory.
    
    CredData categories can be colon-separated (e.g., "Auth:Token").
    """
    if not category_str:
        return SecretCategory.GENERIC_SECRET
    
    category_lower = category_str.lower()
    
    for key, secret_cat in CREDDATA_CATEGORY_MAP.items():
        if key in category_lower:
            return secret_cat
    
    return SecretCategory.GENERIC_SECRET


class CredDataCollector(Collector):
    """Collect labeled credentials from Samsung CredData dataset."""

    name = "creddata"
    data_type = "validator"
    repo_url = "https://github.com/Samsung/CredData.git"

    LABEL_TRUE = "T"
    LABEL_FALSE = "F"

    def setup(self, force: bool = False) -> None:
        """Clone CredData repo and run download_data.py to generate data files."""
        clone_or_pull(self.repo_url, self.cache_path())
        
        data_dir = self.cache_path() / "data"
        if not force and data_dir.exists() and any(data_dir.iterdir()):
            return
        
        if sys.platform == "win32":
            print("Warning: CredData download_data.py fails on Windows due to "
                  "filenames with special characters (e.g., '>'). Skipping.")
            return
        
        from heuristic_secrets.data.collectors.creddata_download import run_download
        
        tmp_dir = self.cache_path() / "tmp"
        skip_download = tmp_dir.exists() and any(tmp_dir.iterdir())
        
        if skip_download:
            print("Repos already downloaded, processing only...")
        else:
            print("Downloading and processing repos...")
        
        try:
            stats = run_download(
                creddata_dir=self.cache_path(),
                skip_download=skip_download,
            )
            print(f"CredData: {stats['process_success']} repos processed, {stats['process_failed']} failed")
        except Exception as e:
            print(f"Warning: CredData processing failed: {e}")

    def is_available(self) -> bool:
        """Check if CredData has been cloned and data generated."""
        meta_dir = self.cache_path() / "meta"
        data_dir = self.cache_path() / "data"
        return meta_dir.exists() and data_dir.exists()

    def _iter_meta_rows(self, desc: str = "Reading meta CSVs") -> Iterator[dict]:
        """Iterate over all rows from meta/*.csv files."""
        meta_dir = self.cache_path() / "meta"
        if not meta_dir.exists():
            return
        
        csv_files = sorted(meta_dir.glob("*.csv"))
        for csv_file in tqdm(csv_files, desc=desc, unit="file"):
            try:
                with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        yield row
            except Exception:
                continue

    def _extract_secret_text(self, row: dict) -> str | None:
        """Extract the actual secret text from the data file.
        
        Returns None if file cannot be read or value position is invalid.
        """
        data_dir = self.cache_path() / "data"
        file_path = data_dir / row["FilePath"].lstrip("data/")
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return None
        
        line_start = int(row["LineStart"])
        line_end = int(row["LineEnd"])
        
        if line_start < 1 or line_start > len(lines):
            return None
        
        value_start_str = row.get("ValueStart", "")
        value_end_str = row.get("ValueEnd", "")
        
        if line_start == line_end and value_start_str and value_end_str:
            try:
                value_start = int(value_start_str)
                value_end = int(value_end_str)
                if value_start >= 0 and value_end > value_start:
                    line = lines[line_start - 1]
                    return line[value_start:value_end].strip()
            except (ValueError, IndexError):
                pass
        
        relevant_lines = lines[line_start - 1:line_end]
        return "".join(relevant_lines).strip()

    def collect(self) -> Iterator[ValidatorSample]:
        """Yield labeled credential samples from CredData."""
        if not self.is_available():
            return

        seen = set()

        for row in self._iter_meta_rows(desc="Collecting samples"):
            ground_truth = row.get("GroundTruth", "")
            
            if ground_truth == self.LABEL_TRUE:
                label = 1
            elif ground_truth == self.LABEL_FALSE:
                label = 0
            else:
                continue
            
            text = self._extract_secret_text(row)
            if not text or len(text) < 8:
                continue
            
            if text in seen:
                continue
            seen.add(text)
            
            creddata_category = row.get("Category", "")
            if label == 1:
                category = map_creddata_category(creddata_category)
            else:
                category = SecretCategory.FP_OTHER
            
            yield ValidatorSample(
                text=text,
                label=label,
                source=self.name,
                category=category.value,
            )

    def collect_spans(self) -> Iterator[SpanFinderSample]:
        """Yield SpanFinder samples with full file context."""
        if not self.is_available():
            return

        data_dir = self.cache_path() / "data"
        
        file_secrets: dict[str, list[tuple[int, int, int, str]]] = {}
        
        for row in self._iter_meta_rows(desc="Indexing spans"):
            if row.get("GroundTruth") != self.LABEL_TRUE:
                continue
            
            file_path = row["FilePath"].lstrip("data/")
            
            value_start_str = row.get("ValueStart", "")
            value_end_str = row.get("ValueEnd", "")
            
            if not value_start_str or not value_end_str:
                continue
            
            try:
                value_start = int(value_start_str)
                value_end = int(value_end_str)
                if value_start < 0 or value_end <= value_start:
                    continue
            except ValueError:
                continue
            
            line_num = int(row["LineStart"])
            category = row.get("Category", "")
            
            if file_path not in file_secrets:
                file_secrets[file_path] = []
            file_secrets[file_path].append((line_num, value_start, value_end, category))
        
        for file_path, secrets in tqdm(file_secrets.items(), desc="Extracting spans", unit="file"):
            full_path = data_dir / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    lines = content.splitlines(keepends=True)
            except Exception:
                continue
            
            starts = []
            ends = []
            categories = []
            
            for line_num, col_start, col_end, category in secrets:
                if line_num < 1 or line_num > len(lines):
                    continue
                
                char_offset = sum(len(lines[i]) for i in range(line_num - 1))
                
                abs_start = char_offset + col_start
                abs_end = char_offset + col_end
                
                if abs_end <= len(content):
                    starts.append(abs_start)
                    ends.append(abs_end)
                    categories.append(map_creddata_category(category).value)
            
            if starts:
                yield SpanFinderSample(
                    text=content,
                    starts=starts,
                    ends=ends,
                    source=self.name,
                    categories=categories,
                )
