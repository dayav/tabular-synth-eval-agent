"""Write the agent's report to a Markdown file."""

from datetime import datetime
from pathlib import Path
from typing import Protocol


class HasText(Protocol):
    """Anything carrying a report under ``.text`` (an ``AgentRun``, for instance)."""

    text: str


def save_report(report: str | HasText, path: str = "report.md") -> Path:
    """Save the agent's evaluation report as a Markdown file.

    Accepts either the report text or an object with a ``text`` attribute,
    so ``save_report(run_agent(prompt))`` keeps working.
    """
    text = report if isinstance(report, str) else report.text
    output = Path(path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Synthetic Data Evaluation Report\n\n_Generated {timestamp}_\n\n---\n\n"
    output.write_text(header + text, encoding="utf-8")
    return output.resolve()
