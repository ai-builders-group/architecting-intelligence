import shutil
from pathlib import Path

import pytest


@pytest.fixture
def toy_repo(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    shutil.copytree(Path(__file__).parent / "fixtures" / "repo", target)
    return target
