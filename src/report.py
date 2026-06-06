"""Write the agent's report to a Markdown file."""

from pathlib import Path
from datetime import datetime


def save_report(text: str, path: str = "report.md") -> Path:
    """Save the agent's evaluation report as a Markdown file."""
    output = Path(path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Synthetic Data Evaluation Report\n\n_Generated {timestamp}_\n\n---\n\n"
    output.write_text(header + text, encoding="utf-8")
    return output.resolve()