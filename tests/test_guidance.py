"""Unit tests for the metric-documentation index."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from guidance import DOCS_DIR, GuidanceIndex, chunk_dicts, lookup, parse_page, thresholds  # noqa: E402


def test_every_page_has_front_matter_and_the_five_sections() -> None:
    pages = [p for p in DOCS_DIR.glob("*.md") if p.name != "README.md"]
    assert len(pages) >= 5
    for page in pages:
        chunks = parse_page(page)
        assert [c.section for c in chunks] == ["Definition", "Threshold", "Interpretation", "Caveats", "Citation"], page
        assert all(c.metric and c.title for c in chunks)
        assert all("2602.06390" in c.text for c in chunks if c.section == "Citation"), page


def test_thresholds_are_numbers_where_expected() -> None:
    t = thresholds()
    assert t["js_distance"]["threshold"] == 0.1
    assert t["cohens_d"]["threshold"] == 0.2
    assert t["utility_gap"]["threshold"] == 0.05
    assert t["identifiability"]["threshold"] == 0.7 and t["identifiability"]["exact_match_threshold"] == 0.05
    assert t["verdict_rules"]["threshold"] is None


@pytest.mark.parametrize(
    "query, metric, section",
    [
        ("JS distance threshold", "js_distance", "Threshold"),
        ("Jensen-Shannon", "js_distance", None),
        ("Cohen's d threshold", "cohens_d", "Threshold"),
        ("effect size small medium large", "cohens_d", None),
        ("utility gap threshold", "utility_gap", "Threshold"),
        ("negative utility gap memorisation", "utility_gap", None),
        ("identifiability threshold", "identifiability", "Threshold"),
        ("how do the axes combine into an overall verdict", "verdict_rules", None),
    ],
)
def test_lookup_top_hit(query: str, metric: str, section: str | None) -> None:
    hit = lookup(query, top_k=1)["results"][0]
    assert hit["metric"] == metric, (query, hit)
    if section:
        assert hit["section"] == section


def test_filters_and_shape() -> None:
    r = lookup("anything", top_k=2, metric="cohens_d", section="Caveats")
    assert len(r["results"]) == 1 and r["results"][0]["section"] == "Caveats" and r["results"][0]["metric"] == "cohens_d"
    assert r["index"].startswith("tf-idf") and r["n_pages"] >= 5 and r["n_sections"] == len(chunk_dicts())
    assert r["results"][0]["source"] == "docs/metrics/cohens_d.md#caveats"
    assert 0.0 <= r["results"][0]["score"] <= 1.0


def test_index_from_another_folder(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("---\nmetric: x\ntitle: X\nthreshold: 3\n---\n\n## Definition\n\nAbout x.\n\n## Threshold\n\nFlag above 3.\n", encoding="utf-8")
    idx = GuidanceIndex(tmp_path)
    hit = idx.lookup("x threshold", top_k=1)["results"][0]
    assert hit["metric"] == "x" and hit["threshold"] == 3.0 and hit["section"] == "Threshold"
    with pytest.raises(ValueError):
        GuidanceIndex(tmp_path / "empty")
