"""Retrieval over the metric documentation in ``docs/metrics/``.

Each page has YAML-style front matter (``metric``, ``title``, ``tool``, ``axis``,
``threshold``, ...) and a fixed set of ``##`` sections. This module:

* parses the pages once and exposes their thresholds (:func:`thresholds`), which the
  agent prompt and the eval harness both read, so there is one source of truth;
* builds a small local index over the sections and answers free-text queries
  (:func:`lookup`). The index is lexical: TF-IDF vectors over word unigrams and
  bigrams plus character n-grams, with cosine similarity. It is deliberately not a
  neural embedding model: the corpus is a handful of pages, the vocabulary is
  technical and exact ("Cohen's d", "epsilon_any"), and a dependency-free index
  that builds in milliseconds and runs offline is the honest choice at this size.

The ``lookup_metric_guidance`` tool in ``agent.py`` is a thin wrapper around
:func:`lookup`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "metrics"
SECTION_ORDER = ("Definition", "Threshold", "Interpretation", "Caveats", "Citation")

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    metric: str
    title: str
    tool: str
    axis: str
    section: str
    text: str
    source: str
    threshold: float | None
    extra: dict[str, Any]

    def search_text(self) -> str:
        """What the index sees: title and metric name prepended so queries like
        "Cohen's d threshold" land on the right page's Threshold section."""
        return f"{self.title}. {self.metric}. {self.tool}. {self.section}. {self.text}"


class GuidanceHit(TypedDict):
    metric: str
    title: str
    tool: str
    axis: str
    section: str
    text: str
    source: str
    threshold: float | None
    score: float


class GuidanceResult(TypedDict):
    query: str
    results: list[GuidanceHit]
    n_pages: int
    n_sections: int
    index: str


def _parse_front(block: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            meta[key.strip()] = float(value) if re.fullmatch(r"-?\d+(\.\d+)?", value) else value
        except ValueError:
            meta[key.strip()] = value
    return meta


def parse_page(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT.match(raw)
    if not m:
        raise ValueError(f"{path} has no front matter")
    meta = _parse_front(m.group(1))
    body = raw[m.end():]
    headings = list(_SECTION.finditer(body))
    chunks: list[Chunk] = []
    for i, h in enumerate(headings):
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = " ".join(body[start:end].split())
        if not text:
            continue
        name = h.group(1).strip()
        chunks.append(
            Chunk(
                metric=str(meta.get("metric", path.stem)),
                title=str(meta.get("title", path.stem)),
                tool=str(meta.get("tool", "")),
                axis=str(meta.get("axis", "")),
                section=name,
                text=text,
                source=f"docs/metrics/{path.name}#{name.lower()}",
                threshold=float(meta["threshold"]) if "threshold" in meta else None,
                extra={k: v for k, v in meta.items() if k not in ("metric", "title", "tool", "axis", "threshold")},
            )
        )
    return chunks


def _pages(docs_dir: Path) -> list[Path]:
    return sorted(p for p in docs_dir.glob("*.md") if p.name != "README.md")


class GuidanceIndex:
    """TF-IDF index over the sections of every page under ``docs_dir``."""

    def __init__(self, docs_dir: Path = DOCS_DIR) -> None:
        self.docs_dir = docs_dir
        self.chunks: list[Chunk] = [c for p in _pages(docs_dir) for c in parse_page(p)]
        if not self.chunks:
            raise ValueError(f"no metric pages under {docs_dir}")
        texts = [c.search_text() for c in self.chunks]
        self._word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w[\w'._-]*\b")
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._matrix = hstack([self._word.fit_transform(texts), self._char.fit_transform(texts)]).tocsr()
        self.n_pages = len(_pages(docs_dir))

    def _embed(self, query: str) -> Any:
        return hstack([self._word.transform([query]), self._char.transform([query])]).tocsr()

    def lookup(self, query: str, top_k: int = 3, metric: str | None = None, section: str | None = None) -> GuidanceResult:
        q = self._embed(query)
        scores = (self._matrix @ q.T).toarray().ravel()
        norms = np.sqrt(np.asarray(self._matrix.multiply(self._matrix).sum(axis=1)).ravel()) * np.sqrt(q.multiply(q).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = np.where(norms > 0, scores / norms, 0.0)
        order = np.argsort(-cosine)
        hits: list[GuidanceHit] = []
        for i in order:
            c = self.chunks[int(i)]
            if metric and c.metric != metric:
                continue
            if section and c.section.lower() != section.lower():
                continue
            hits.append(
                GuidanceHit(
                    metric=c.metric, title=c.title, tool=c.tool, axis=c.axis, section=c.section,
                    text=c.text, source=c.source, threshold=c.threshold, score=round(float(cosine[i]), 4),
                )
            )
            if len(hits) >= max(1, top_k):
                break
        return GuidanceResult(
            query=query, results=hits, n_pages=self.n_pages, n_sections=len(self.chunks),
            index="tf-idf (word 1-2 grams + char 3-5 grams), cosine",
        )

    def thresholds(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c in self.chunks:
            if c.metric not in out:
                out[c.metric] = {"threshold": c.threshold, "tool": c.tool, "axis": c.axis, "title": c.title, **c.extra}
        return out


@lru_cache(maxsize=4)
def get_index(docs_dir: str = str(DOCS_DIR)) -> GuidanceIndex:
    return GuidanceIndex(Path(docs_dir))


def lookup(query: str, top_k: int = 3, metric: str | None = None, section: str | None = None) -> GuidanceResult:
    """Retrieve the most relevant documentation sections for ``query``."""
    return get_index().lookup(query, top_k=top_k, metric=metric, section=section)


def thresholds() -> dict[str, dict[str, Any]]:
    """Machine-readable thresholds from the pages' front matter, keyed by metric."""
    return get_index().thresholds()


def chunk_dicts() -> list[dict[str, Any]]:
    return [asdict(c) for c in get_index().chunks]
