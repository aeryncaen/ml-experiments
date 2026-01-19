import subprocess
from pathlib import Path


def clone_or_pull(repo_url: str, dest: Path) -> None:
    """Clone a git repository, or pull if it already exists.

    Args:
        repo_url: URL of the git repository
        dest: Destination directory
    """
    dest = Path(dest)

    if (dest / ".git").exists():
        # Pull existing repo
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=dest,
            capture_output=True,
            check=True,
        )
    else:
        # Clone new repo
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            capture_output=True,
            check=True,
        )


def get_repo_version(repo_path: Path) -> str:
    """Get the current commit hash of a git repository.

    Args:
        repo_path: Path to the repository

    Returns:
        The current commit hash (40 hex characters)
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
