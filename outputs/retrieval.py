"""Intake + retrieval: clause-level chunking with citable refs, TF-IDF search per persona."""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).parent / "data" / "personas"
CLAUSE_LINE = re.compile(r"^(\d+\.\d+)\s+(.*)")


def chunk_document(path: Path) -> list[dict]:
    """One chunk per numbered sub-clause, prefixed with doc title + clause heading for context."""
    title, heading, chunks = "", "", []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("# "):
            title = line[2:]
        elif line.startswith("## "):
            heading = line[3:]
        else:
            m = CLAUSE_LINE.match(line)
            if m:
                chunks.append({"ref": f"{path.name}#{m.group(1)}",
                               "text": f"[{title} | {heading}] {m.group(1)} {m.group(2)}"})
    return chunks


def load_persona(persona_id: str) -> dict:
    folder = DATA / persona_id
    profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    chunks = []
    for doc in sorted(folder.glob("*.md")):
        chunks.extend(chunk_document(doc))
    return {"profile": profile, "chunks": chunks}


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform([c["text"] for c in chunks])

    def search(self, queries: list[str], k: int = 4) -> list[dict]:
        """Union of top-k per query, de-duplicated, best score first."""
        best: dict[str, float] = {}
        for q in queries:
            scores = cosine_similarity(self.vec.transform([q]), self.matrix)[0]
            for i in scores.argsort()[::-1][:k]:
                if scores[i] > 0:
                    ref = self.chunks[i]["ref"]
                    best[ref] = max(best.get(ref, 0.0), float(scores[i]))
        by_ref = {c["ref"]: c for c in self.chunks}
        return [by_ref[r] for r in sorted(best, key=best.get, reverse=True)]
