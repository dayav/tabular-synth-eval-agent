"""Golden case loading and the prompt the agent is given.

A :class:`Case` is a directory under ``evals/cases/`` holding ``real.csv``,
``synth.csv`` and ``expected.json``. Case directories are named after their
corruption and the model reads file paths, so :func:`neutral_copy` copies a case
into a temporary directory with a neutral name before it is prompted.
"""

from __future__ import annotations

import json
import random
import shutil
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASES_DIR = Path(__file__).resolve().parent / "cases"

PROMPT_TEMPLATE = (
    "Evaluate the synthetic dataset at {synth} against the real dataset at {real}. "
    "The target column for utility evaluation is '{target}'."
)
PRIVACY_SUFFIX = " Also assess the privacy of the synthetic data: the disclosure risk of real records."


@dataclass(frozen=True)
class Case:
    name: str
    real_path: Path
    synth_path: Path
    expected: dict[str, Any]

    @property
    def target_column(self) -> str:
        return str(self.expected["target_column"])

    @property
    def expected_tools(self) -> list[str]:
        return list(self.expected["tools"])

    @property
    def ask_privacy(self) -> bool:
        return bool(self.expected.get("ask_privacy", False))

    def prompt(self, real: Path | None = None, synth: Path | None = None) -> str:
        text = PROMPT_TEMPLATE.format(
            real=real or self.real_path,
            synth=synth or self.synth_path,
            target=self.target_column,
        )
        return text + PRIVACY_SUFFIX if self.ask_privacy else text


def load_case(case_dir: Path) -> Case:
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    return Case(
        name=case_dir.name,
        real_path=case_dir / "real.csv",
        synth_path=case_dir / "synth.csv",
        expected=expected,
    )


def load_cases(cases_dir: Path = CASES_DIR, names: list[str] | None = None) -> list[Case]:
    dirs = sorted(d for d in cases_dir.iterdir() if (d / "expected.json").exists())
    cases = [load_case(d) for d in dirs]
    if names:
        wanted = set(names)
        missing = wanted - {c.name for c in cases}
        if missing:
            raise ValueError(f"unknown cases: {sorted(missing)}")
        cases = [c for c in cases if c.name in wanted]
    return cases


@dataclass(frozen=True)
class NeutralCopy:
    """A case copied under neutral names so the prompt carries no answer key."""

    directory: Path
    real_path: Path
    synth_path: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def neutral_copy(case: Case, parent: Path | None = None) -> NeutralCopy:
    """Copy the case files to ``<parent>/eval_<letters>/{real,synth}.csv``.

    The directory name uses letters only, so it also introduces no digit that
    the numeric faithfulness check could mistake for a quoted number.
    """
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    base = Path(parent) if parent else Path(tempfile.gettempdir())
    directory = base / f"eval_{suffix}"
    directory.mkdir(parents=True, exist_ok=False)
    real = directory / "real.csv"
    synth = directory / "synth.csv"
    shutil.copyfile(case.real_path, real)
    shutil.copyfile(case.synth_path, synth)
    return NeutralCopy(directory=directory, real_path=real, synth_path=synth)
