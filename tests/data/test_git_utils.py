import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from heuristic_secrets.data.git_utils import clone_or_pull, get_repo_version


class TestCloneOrPull:
    def test_clone_new_repo(self, tmp_path):
        # Use a small, fast repo for testing
        repo_url = "https://github.com/octocat/Hello-World.git"
        dest = tmp_path / "hello-world"

        clone_or_pull(repo_url, dest)

        assert dest.exists()
        assert (dest / ".git").exists()

    def test_pull_existing_repo(self, tmp_path):
        repo_url = "https://github.com/octocat/Hello-World.git"
        dest = tmp_path / "hello-world"

        # Clone first
        clone_or_pull(repo_url, dest)
        # Pull again (should not fail)
        clone_or_pull(repo_url, dest)

        assert dest.exists()


class TestGetRepoVersion:
    def test_get_version_from_repo(self, tmp_path):
        repo_url = "https://github.com/octocat/Hello-World.git"
        dest = tmp_path / "hello-world"

        clone_or_pull(repo_url, dest)
        version = get_repo_version(dest)

        # Should return a commit hash (40 hex chars)
        assert len(version) == 40
        assert all(c in "0123456789abcdef" for c in version)
