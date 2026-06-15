"""
STAT 418 · Chapter 6.3 — Text Preprocessing for LLM Pipelines
Script 03: A complete, configurable preprocessing pipeline.

Combines cleaning + chunking into one testable class, then validates it on sample
documents. Unlike classical NLP, we deliberately do NOT lowercase, stem, or remove
stop words — LLMs benefit from natural, well-formed text; we strip only *noise*
(HTML, URLs, control characters, duplicate whitespace). A subclass adds domain
rules for academic abstracts (LaTeX, citations) — Exercise 6.3.4.

Pure Python — no API key required.

Prereqs / run: nothing beyond the standard library.
"""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Standard cleaning for LLM input: strip noise, preserve natural language."""
    text = re.sub(r"<[^>]+>", "", text)                            # HTML tags
    text = re.sub(r"https?://\S+", "[URL]", text)                  # URLs
    text = re.sub(r"\s+", " ", text)                               # collapse whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)   # control chars
    return text.strip()


class TextPreprocessor:
    """Configurable clean -> (size check) -> chunk pipeline for LLM workflows."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50, strategy: str = "semantic"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def clean(self, text: str) -> str:
        return clean_text(text)

    def chunk(self, text: str) -> list:
        if self.strategy == "fixed":
            return self._chunk_fixed(text)
        if self.strategy == "overlap":
            return self._chunk_overlap(text)
        if self.strategy == "semantic":
            return self._chunk_semantic(text)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _chunk_fixed(self, text: str) -> list:
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size])
                for i in range(0, len(words), self.chunk_size)]

    def _chunk_overlap(self, text: str) -> list:
        words = text.split()
        chunks, step = [], self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def _chunk_semantic(self, text: str) -> list:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks, current, size = [], [], 0
        for para in paragraphs:
            para_size = len(para.split())
            if size + para_size > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                current, size = [para], para_size
            else:
                current.append(para)
                size += para_size
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def process(self, text: str) -> list:
        """Full pipeline: clean, then chunk only if the text exceeds chunk_size."""
        cleaned = self.clean(text)
        if int(len(cleaned.split()) * 1.3) <= self.chunk_size:
            return [cleaned]
        return self.chunk(cleaned)

    def validate(self, chunks: list) -> dict:
        return {
            "n_chunks": len(chunks),
            "sizes": [len(c.split()) for c in chunks],
            "all_within_limit": all(len(c.split()) <= self.chunk_size * 1.1 for c in chunks),
        }


class AcademicPreprocessor(TextPreprocessor):
    """Adds domain rules for academic abstracts: LaTeX commands and citations."""

    def clean(self, text: str) -> str:
        text = re.sub(r"\\textbf\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\textit\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\cite\{[^}]+\}", "", text)
        text = re.sub(r"\\ref\{[^}]+\}", "[ref]", text)
        text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)              # [1], [1,2,3]
        text = re.sub(r"\([A-Z][a-z]+ et al\.,? \d{4}\)", "", text)  # (Smith et al., 2023)
        return super().clean(text)


def validate_pipeline(preprocessor: TextPreprocessor, sample_texts: list) -> None:
    """Run the pipeline on samples and report aggregate chunk statistics."""
    reports = [preprocessor.validate(preprocessor.process(t)) for t in sample_texts]
    total = sum(r["n_chunks"] for r in reports)
    sizes = [s for r in reports for s in r["sizes"]]
    print(f"  documents processed: {len(sample_texts)}")
    print(f"  total chunks:        {total}  (avg {total / len(sample_texts):.1f}/doc)")
    print(f"  chunk size range:    [{min(sizes)}, {max(sizes)}] words")
    print(f"  all within limit:    {all(r['all_within_limit'] for r in reports)}")


def main() -> None:
    # --- Cleaning: strip noise, keep signal ---
    raw = ("<p>This is a <b>test</b> of the   cleaning pipeline.</p>\n"
           "Visit https://example.com/very/long/url for more info.\n"
           "Extra    spaces   and\ttabs\there.")
    print("Cleaning (noise removed, case/grammar/punctuation preserved):")
    print(f"  raw   ({len(raw)} chars): {raw[:60]!r}...")
    print(f"  clean ({len(clean_text(raw))} chars): {clean_text(raw)}")

    # --- Full pipeline on a long document ---
    pre = TextPreprocessor(chunk_size=100, overlap=15, strategy="overlap")
    long_doc = " ".join(f"Paragraph {i} covers topic {i % 3}. " * 5 for i in range(20))
    report = pre.validate(pre.process(long_doc))
    print(f"\nPipeline (overlap, 100w): {len(long_doc.split())} words "
          f"-> {report['n_chunks']} chunks, all within limit: {report['all_within_limit']}")

    # --- Validate across documents of varied length ---
    print("\nBatch validation:")
    validate_pipeline(pre, [
        "Short text that fits in one chunk.",
        " ".join(["Medium text."] * 100),
        " ".join(["Long document with many paragraphs."] * 500),
    ])

    # --- Domain-specific cleaning for academic abstracts (Exercise 6.3.4) ---
    academic = AcademicPreprocessor(chunk_size=200, strategy="semantic")
    abstract = (r"We present a \textbf{novel} approach to bootstrap inference "
                r"\cite{efron1979} that improves coverage [1,2]. Our method "
                r"(Smith et al., 2023) achieves 95\% coverage in simulations.")
    print("\nAcademic cleaning (LaTeX + citations stripped):")
    print(f"  before ({len(abstract.split())} words): {abstract[:70]}...")
    print(f"  after  ({len(academic.clean(abstract).split())} words): {academic.clean(abstract)}")


if __name__ == "__main__":
    main()
