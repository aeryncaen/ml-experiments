"""Patched CredData download/processing script with better error handling.

Based on Samsung/CredData download_data.py but continues on per-repo failures
instead of dying.
"""

import binascii
import hashlib
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level="INFO")
logger = logging.getLogger(__name__)


def get_words_in_path(creddata_dir: pathlib.Path) -> list[str]:
    with open(creddata_dir / "word_in_path.json") as f:
        return json.load(f)


def get_file_scope(path_without_extension: str, words_in_path: list[str]) -> str:
    result = '/'
    local_file_path_lower = f"./{path_without_extension.lower()}"
    for word in words_in_path:
        if word in local_file_path_lower:
            result += word[1:] if word.startswith('/') else word
            if not result.endswith('/'):
                result += '/'
    if '/' == result:
        result = "/_/"
    return result


def get_new_repo_id(repo_id: str) -> str:
    repo_id_bytes = binascii.unhexlify(repo_id)
    return f"{binascii.crc32(repo_id_bytes):08x}"


def download_repo(repo_id: str, repo_url: str, tmp_dir: pathlib.Path) -> bool:
    """Download one repo. Returns True on success."""
    commit_sha = repo_id[:40]
    repo_dir = tmp_dir / repo_id

    try:
        if repo_dir.exists():
            subprocess.check_call(
                f"git checkout {commit_sha}",
                cwd=repo_dir, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            result = subprocess.run(
                'git status --porcelain',
                cwd=repo_dir, shell=True,
                capture_output=True, text=True
            )
            if not result.stdout.strip():
                return True
    except subprocess.CalledProcessError:
        pass

    try:
        shutil.rmtree(repo_dir, ignore_errors=True)
        repo_dir.mkdir(parents=True, exist_ok=True)
        for cmd in [
            "git init",
            "git config advice.detachedHead false",
            f"git remote add origin {repo_url}",
            f"git fetch --depth 1 origin {commit_sha}",
            f"git checkout {commit_sha}",
        ]:
            subprocess.check_call(
                cmd, cwd=repo_dir, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to download {repo_url}: {e}")
        return False


def read_meta_csv(meta_file: pathlib.Path) -> list[dict]:
    """Read meta CSV file into list of dicts."""
    import csv
    if not meta_file.exists():
        return []
    
    rows = []
    with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def process_repo(
    repo_id: str,
    repo_url: str,
    tmp_dir: pathlib.Path,
    meta_dir: pathlib.Path,
    data_dir: pathlib.Path,
    words_in_path: list[str],
) -> tuple[bool, str]:
    """Process a single repo. Returns (success, message)."""
    new_repo_id = get_new_repo_id(repo_id)
    meta_file = meta_dir / f"{new_repo_id}.csv"
    repo_dir = tmp_dir / repo_id

    if not repo_dir.exists():
        return False, "repo not downloaded"

    meta_rows = read_meta_csv(meta_file)
    
    interesting_files: dict[str, str] = {}
    for row in meta_rows:
        file_id = row.get("FileID", "")
        file_path = row.get("FilePath", "")
        if file_id:
            interesting_files[file_id] = file_path

    all_files = [p for p in repo_dir.rglob("*") if p.is_file() and not p.is_symlink()]
    
    files_to_copy: dict[pathlib.Path, tuple[str, str, str]] = {}
    
    for full_path in all_files:
        short_path = full_path.relative_to(repo_dir).as_posix()
        
        if "/.git/" in f"/{short_path}" or short_path.startswith(".git/"):
            continue
        if short_path.endswith(".xml"):
            continue
            
        file_id = hashlib.sha256(short_path.encode()).hexdigest()[:8]
        file_path_name, file_extension = os.path.splitext(short_path)
        file_extension = file_extension.lower()
        file_scope = get_file_scope(file_path_name, words_in_path)
        
        if file_id in interesting_files or not meta_rows:
            files_to_copy[full_path] = (file_id, file_scope, file_extension)

    if meta_rows:
        found_ids = set(x[0] for x in files_to_copy.values())
        expected_ids = set(interesting_files.keys())
        missing = expected_ids - found_ids
        if missing:
            return False, f"missing {len(missing)} files from meta"

    copied = 0
    skipped = 0
    for full_path, (file_id, file_scope, file_extension) in files_to_copy.items():
        dest_dir = data_dir / new_repo_id / file_scope.strip("/")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{file_id}{file_extension}"
        
        try:
            shutil.copy(full_path, dest_file)
            copied += 1
        except PermissionError:
            skipped += 1
        except OSError as e:
            if e.errno == 1:  # Operation not permitted (antivirus)
                skipped += 1
            else:
                logger.debug(f"Failed to copy {full_path}: {e}")
        except Exception as e:
            logger.debug(f"Failed to copy {full_path}: {e}")

    for license_pattern in ["*LICEN*", "*Licen*", "*licen*", "*COPYING*"]:
        for license_file in repo_dir.glob(license_pattern):
            if license_file.is_file():
                dest = data_dir / new_repo_id / license_file.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy(license_file, dest)
                except Exception:
                    pass

    msg = f"copied {copied} files"
    if skipped:
        msg += f", skipped {skipped} (permission denied)"
    return True, msg


def run_download(
    creddata_dir: pathlib.Path,
    skip_download: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    """Main entry point for downloading and processing CredData.
    
    Returns stats dict with success/failure counts.
    """
    tmp_dir = creddata_dir / "tmp"
    meta_dir = creddata_dir / "meta"
    data_dir = creddata_dir / "data"
    
    with open(creddata_dir / "snapshot.json") as f:
        snapshot = json.load(f)
    
    words_in_path = get_words_in_path(creddata_dir)
    
    stats = {
        "total": len(snapshot),
        "download_success": 0,
        "download_failed": 0,
        "process_success": 0,
        "process_failed": 0,
        "failed_repos": [],
    }
    
    if not skip_download:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {len(snapshot)} repos...")
        
        for repo_id, repo_url in tqdm(snapshot.items(), desc="Downloading", unit="repo"):
            if download_repo(repo_id, repo_url, tmp_dir):
                stats["download_success"] += 1
            else:
                stats["download_failed"] += 1
                stats["failed_repos"].append(("download", repo_id, repo_url))
    
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(snapshot)} repos...")
    
    for repo_id, repo_url in tqdm(snapshot.items(), desc="Processing", unit="repo"):
        try:
            success, msg = process_repo(
                repo_id, repo_url, tmp_dir, meta_dir, data_dir, words_in_path
            )
            if success:
                stats["process_success"] += 1
            else:
                stats["process_failed"] += 1
                stats["failed_repos"].append(("process", repo_id, msg))
        except Exception as e:
            stats["process_failed"] += 1
            stats["failed_repos"].append(("process", repo_id, str(e)))
    
    logger.info(f"Done. Success: {stats['process_success']}, Failed: {stats['process_failed']}")
    
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--creddata-dir", type=pathlib.Path, required=True)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    
    args = parser.parse_args()
    stats = run_download(args.creddata_dir, args.skip_download, args.jobs)
    
    print(f"\nResults:")
    print(f"  Download: {stats['download_success']} ok, {stats['download_failed']} failed")
    print(f"  Process:  {stats['process_success']} ok, {stats['process_failed']} failed")
    
    if stats["failed_repos"]:
        print(f"\nFailed repos ({len(stats['failed_repos'])}):")
        for stage, repo_id, msg in stats["failed_repos"][:10]:
            print(f"  [{stage}] {repo_id[:16]}... : {msg}")
        if len(stats["failed_repos"]) > 10:
            print(f"  ... and {len(stats['failed_repos']) - 10} more")


if __name__ == "__main__":
    main()
