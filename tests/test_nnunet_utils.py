"""nnU-Net's input file listing must not start processes.

It used to run in a fork-based Pool, which forked the Celery worker at the start of
every prediction and could hang the task.
"""

import multiprocessing.process
from pathlib import Path
from typing import Any

import pytest

import body_organ_analysis  # noqa: F401  (puts _external on sys.path)

# isort: split
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def test_create_lists_runs_without_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ["s01_0000.nii.gz", "s02_0000.nii.gz", "s02_0001.nii.gz"]:
        (tmp_path / name).touch()

    def no_process(*_: Any, **__: Any) -> None:
        pytest.fail("a process was started")

    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", no_process)

    lists = create_lists_from_splitted_dataset_folder(str(tmp_path), ".nii.gz")

    assert lists == [
        [str(tmp_path / "s01_0000.nii.gz")],
        [str(tmp_path / "s02_0000.nii.gz"), str(tmp_path / "s02_0001.nii.gz")],
    ]
