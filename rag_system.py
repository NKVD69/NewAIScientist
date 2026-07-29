"""
RAG System for Scientific Literature Analysis
Provides advanced document retrieval and semantic search capabilities
"""

import asyncio
import hashlib
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anti-SSRF allowlist for paper downloads
# ---------------------------------------------------------------------------

# Hosts we are willing to fetch PDFs from. The literature search layer can
# legitimately only return ArXiv URLs through this code path; PubMed goes
# through Entrez, not urlretrieve. Anything else is a red flag (LLM-
# induced URL or upstream metadata poisoning) and must be refused before
# urlretrieve sees it — otherwise an attacker can pivot to
# file:///etc/passwd, http://169.254.169.254/... (cloud metadata), etc.
_PDF_DOWNLOAD_ALLOWED_HOSTS: frozenset[str] = frozenset({
    "arxiv.org",
    "www.arxiv.org",
    "export.arxiv.org",
})
_PDF_DOWNLOAD_ALLOWED_SCHEMES: frozenset[str] = frozenset({"https"})


def _is_pdf_url_safe(url: str) -> bool:
    """Return True iff *url* is fetchable for a PDF.

    Enforces (a) HTTPS only, (b) host in the explicit allowlist, and
    (c) no embedded credentials or path traversal indicators.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:  # noqa: BLE001 — defensive
        return False

    if parsed.scheme.lower() not in _PDF_DOWNLOAD_ALLOWED_SCHEMES:
        return False
    host = (parsed.hostname or "").lower()
    if host not in _PDF_DOWNLOAD_ALLOWED_HOSTS:
        return False
    # No userinfo (https://attacker@arxiv.org/...)
    if parsed.username or parsed.password:
        return False
    return True


# ---------------------------------------------------------------------------
# URL normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_arxiv_url(url: str) -> str:
    """
    Canonicalise any ArXiv URL variant to ``https://arxiv.org/abs/<id>``.

    Handled inputs:
      - https://arxiv.org/abs/2103.12345
      - https://arxiv.org/abs/2103.12345v2
      - https://arxiv.org/pdf/2103.12345.pdf
      - https://arxiv.org/pdf/2103.12345v2.pdf
      - http://arxiv.org/abs/...
      - arxiv:2103.12345
      - plain ArXiv IDs embedded in other URLs

    Non-ArXiv URLs are returned unchanged.
    """
    url = url.strip()

    # arxiv:<id> shorthand
    if url.lower().startswith("arxiv:"):
        arxiv_id = url.split(":", 1)[1].strip()
        return f"https://arxiv.org/abs/{arxiv_id}"

    if "arxiv.org" not in url:
        return url

    # Extract the ArXiv ID from /abs/ or /pdf/ paths
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?$", url, re.IGNORECASE)
    if match:
        arxiv_id = match.group(1)
        return f"https://arxiv.org/abs/{arxiv_id}"

    # Fallback: return as-is
    return url


def _url_to_paper_id(url: str) -> str:
    """Return a short, stable hex ID derived from the *normalised* URL."""
    return hashlib.md5(_normalize_arxiv_url(url).encode()).hexdigest()[:16]

# PDF and text processing
try:
    import pypdf
except ImportError:
    pypdf = None

# Optional: pdfplumber for table extraction. Falls back to pypdf-only when absent.
try:
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover
    pdfplumber = None

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Vector database
try:
    import chromadb
except ImportError:
    chromadb = None

# Text chunking
try:
    import tiktoken
except ImportError:
    tiktoken = None

# PubMed Tools
try:
    from Bio import Entrez
    Entrez.email = config.get_entrez_email()
except ImportError:
    Entrez = None


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    text: str
    paper_id: str
    paper_title: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    chunk_type: str = "text"   # "text" | "table" | "figure_caption"
    metadata: dict = None


# ---------------------------------------------------------------------------
# Table & figure-caption extraction
# ---------------------------------------------------------------------------

# Captures lines starting with "Figure 1", "Fig. 2:", "Table 3 -" etc. and
# the rest of the line. Non-greedy so it stops at the next newline.
_FIGURE_CAPTION_RE = re.compile(
    r"(?m)^\s*(?:Figure|Fig\.?|Table)\s+\d+[\.\:\-—]?\s*(.+?)$",
    re.IGNORECASE,
)


def _serialise_table(rows) -> str:
    """Render a 2-D table (list of rows) as a compact pipe-separated string."""
    out_lines = []
    for row in rows:
        if row is None:
            continue
        cells = [(str(c).strip() if c is not None else "") for c in row]
        if any(cells):
            out_lines.append(" | ".join(cells))
    return "\n".join(out_lines)


def extract_tables_from_pdf(file_path) -> list[dict]:
    """Extract every table from a PDF as ``{page, text}`` records.

    Uses pdfplumber when available; returns ``[]`` otherwise. Empty tables
    are skipped. Failures are logged and swallowed.
    """
    if pdfplumber is None:
        return []
    out: list[dict] = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    for tbl in page.extract_tables() or []:
                        text = _serialise_table(tbl)
                        if text:
                            out.append({"page": page_num, "text": text})
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Table extraction error on page %d: %s", page_num, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber failed on %s: %s", file_path, exc)
    return out


def extract_figure_captions(text: str) -> list[str]:
    """Find figure / table captions in raw extracted PDF text.

    Returns the full caption lines (including the leading "Figure N:" so
    callers know where each came from).
    """
    if not text:
        return []
    return [m.group(0).strip() for m in _FIGURE_CAPTION_RE.finditer(text)]


class PDFDownloader:
    """Downloads PDFs from ArXiv and PubMed"""

    def __init__(self, cache_dir: str = "./papers"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL (keyed on the normalised URL)."""
        url_hash = hashlib.md5(_normalize_arxiv_url(url).encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.pdf"

    async def download_arxiv_pdf(self, paper_url: str) -> Path | None:
        """Download PDF from ArXiv URL with robust conversion."""
        try:
            # Clean URL
            paper_url = paper_url.strip()

            # Convert abstract URL to PDF URL
            # Handles:
            # - http://arxiv.org/abs/2103.12345
            # - https://arxiv.org/pdf/2103.12345.pdf
            # - arxiv:2103.12345

            if "arxiv.org/abs/" in paper_url:
                pdf_url = paper_url.replace("/abs/", "/pdf/")
                if not pdf_url.endswith(".pdf"):
                    pdf_url += ".pdf"
            elif "arxiv.org/pdf/" in paper_url:
                pdf_url = paper_url
                if not pdf_url.endswith(".pdf"):
                    pdf_url += ".pdf"
            elif paper_url.startswith("arxiv:"):
                arxiv_id = paper_url.split(":")[1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            else:
                pdf_url = paper_url
                if "/abs/" in pdf_url:
                    pdf_url = pdf_url.replace("/abs/", "/pdf/")
                if not pdf_url.endswith(".pdf") and "arxiv.org" in pdf_url:
                    pdf_url += ".pdf"

            # Force http→https for arXiv (the literature layer may still
            # feed us legacy http:// URLs; the allowlist below requires
            # https). This is safe because arxiv.org redirects http→https
            # anyway.
            if pdf_url.startswith("http://") and "arxiv.org" in pdf_url:
                pdf_url = "https://" + pdf_url[len("http://"):]

            # Anti-SSRF: refuse anything not on the explicit allowlist
            # *before* it reaches urlretrieve. Without this, a malicious
            # URL (e.g. file:///etc/passwd or
            # http://169.254.169.254/latest/meta-data/) would be fetched.
            if not _is_pdf_url_safe(pdf_url):
                logger.warning(
                    "Refusing to download from unsafe URL: %s "
                    "(scheme/host not in allowlist).",
                    pdf_url,
                )
                return None

            cache_path = self._get_cache_path(pdf_url)

            # Return cached if exists
            if cache_path.exists():
                return cache_path

            # Download
            def download():
                urllib.request.urlretrieve(pdf_url, cache_path)

            await asyncio.to_thread(download)
            return cache_path

        except Exception as e:
            logger.warning("Failed to download ArXiv PDF: %s", e)
            return None

    async def _get_pmcid_from_pmid(self, pmid: str) -> str | None:
        """Convert PMID to PMCID using Entrez API"""
        if not Entrez:
            return None

        try:
            def call_entrez():
                handle = Entrez.elink(dbfrom="pubmed", db="pmc", linkname="pubmed_pmc", id=pmid)
                record = Entrez.read(handle)
                handle.close()
                if record and len(record) > 0 and record[0].get("LinkSetDb") and len(record[0]["LinkSetDb"]) > 0:
                    links = record[0]["LinkSetDb"][0].get("Link", [])
                    if links:
                        return links[0]["Id"]
                return None

            pmcid = await asyncio.to_thread(call_entrez)
            return pmcid
        except Exception as e:
            logger.warning("Failed to convert PMID %s to PMCID: %s", pmid, e)
            return None

    async def download_pubmed_pdf(self, paper_url: str) -> Path | None:
        """Download PDF from PubMed (via PMC when available)"""
        # Extract PMID from URL
        # Format: https://pubmed.ncbi.nlm.nih.gov/38218645/
        pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", paper_url)
        if not pmid_match:
            logger.info("Could not extract PMID from %s", paper_url)
            return None

        pmid = pmid_match.group(1)

        # Get PMCID
        pmcid = await self._get_pmcid_from_pmid(pmid)
        if not pmcid:
            logger.info("No PMCID found for PMID %s (paper may not be Open Access)", pmid)
            return None

        # Construct PMC XML download using Entrez
        clean_pmcid = pmcid.replace("PMC", "")
        cache_path = self.cache_dir / f"pmc_{clean_pmcid}.txt"

        try:
            if cache_path.exists():
                return cache_path

            def download():
                handle = Entrez.efetch(db="pmc", id=clean_pmcid, retmode="xml")
                record = handle.read()
                handle.close()

                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(record)
                    paragraphs = []
                    for p in root.iter('p'):
                        text = "".join(p.itertext()).strip()
                        if text:
                            paragraphs.append(text)

                    text_content = "\n\n".join(paragraphs)

                    # Fallback to abstract
                    if not text_content.strip():
                        for abstract in root.iter('abstract'):
                            text = "".join(abstract.itertext()).strip()
                            if text:
                                paragraphs.append(text)
                        text_content = "\n\n".join(paragraphs)

                    if not text_content.strip():
                        logger.info("No text content found in XML for PMID %s", pmid)
                        return None

                    with open(cache_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(text_content)

                    return cache_path
                except Exception as e:
                    logger.warning("Failed to parse PMC XML for PMID %s: %s", pmid, e)
                    return None

            return await asyncio.to_thread(download)

        except Exception as e:
            logger.warning("Failed to download PMC text for PMID %s: %s", pmid, e)
            return None

    async def download_paper(self, paper: dict) -> Path | None:
        """Download paper PDF based on source"""
        url = paper.get("url", "")
        source = paper.get("source", "")

        if source == "ArXiv":
            return await self.download_arxiv_pdf(url)
        elif source == "PubMed":
            return await self.download_pubmed_pdf(url)
        else:
            return None


class DocumentProcessor:
    """Extracts and processes text from PDFs"""

    def __init__(self):
        if not pypdf:
            logger.warning("pypdf not installed — PDF processing disabled.")

    async def extract_text(self, file_path: Path) -> str | None:
        """Extract all text from PDF or TXT"""
        try:
            if file_path.suffix.lower() == '.txt':
                def extract_txt():
                    with open(file_path, encoding='utf-8') as f:
                        return f.read()
                text = await asyncio.to_thread(extract_txt)
                return self._clean_text(text)

            elif file_path.suffix.lower() == '.pdf':
                if not pypdf:
                    return None

                def extract_pdf():
                    reader = pypdf.PdfReader(str(file_path))
                    text_parts = []
                    for page in reader.pages:
                        text_parts.append(page.extract_text())
                    return "\n\n".join(text_parts)

                text = await asyncio.to_thread(extract_pdf)
                return self._clean_text(text)
            else:
                return None

        except Exception as e:
            logger.warning("Failed to extract text from %s: %s", file_path.name, e)
            return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove page numbers and headers (simple heuristic) - do this BEFORE collapsing whitespace
        text = re.sub(r'\n\d+\n', '\n', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


#: Embedding models in preference order.
#:
#: ``all-MiniLM-L6-v2`` is a 384-d generalist trained on web text. On
#: biomedical prose it conflates entities that are close in surface form and
#: distant in meaning -- gene nomenclature, compound names, pathway
#: terminology -- which is precisely the vocabulary this system retrieves
#: over. Domain models are tried first; MiniLM remains the fallback so the
#: graceful-degradation pattern used throughout the codebase still holds.
EMBEDDING_CANDIDATES: list[tuple[str, str]] = [
    ("FremyCompany/BioLORD-2023", "biomedical concept similarity"),
    ("pritamdeka/S-PubMedBert-MS-MARCO", "biomedical passage retrieval"),
    ("allenai/specter2_base", "scientific paper-level similarity"),
    ("all-MiniLM-L6-v2", "generalist fallback"),
]


def _load_embedding_model():
    """Load the best available embedding model.

    Honours ``NEWAISCI_EMBEDDING_MODEL`` when set, otherwise walks the
    candidate list. Returns ``(model, name)``; ``(None, None)`` if none load.
    """
    import os

    override = os.environ.get("NEWAISCI_EMBEDDING_MODEL", "").strip()
    candidates = (
        [(override, "explicit override")] + EMBEDDING_CANDIDATES
        if override else EMBEDDING_CANDIDATES
    )

    for name, purpose in candidates:
        try:
            logger.info("Loading embedding model '%s' (%s)...", name, purpose)
            model = SentenceTransformer(name)
            logger.info("Embedding model loaded: %s", name)
            if name == "all-MiniLM-L6-v2":
                logger.warning(
                    "Using the generalist fallback embedding model. For "
                    "biomedical corpora a domain model gives substantially "
                    "better retrieval; install one of: %s",
                    ", ".join(n for n, _ in EMBEDDING_CANDIDATES[:3]),
                )
            return model, name
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedding model '%s' unavailable: %s", name, exc)

    logger.warning("No embedding model could be loaded; dense retrieval disabled.")
    return None, None


class SemanticChunker:
    """Intelligent text chunking with semantic awareness"""

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Try to use tiktoken for accurate token counting
        self.encoder = None
        if tiktoken:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 chars
            return len(text) // 4

    def chunk_text(self, text: str, paper_id: str, paper_title: str) -> list[DocumentChunk]:
        """Split text into semantic chunks, tagged with their IMRaD section.

        Section tagging is what stops a sentence from the Discussion -- where
        authors speculate in the subjunctive about what their results might
        imply -- from being indistinguishable downstream from a sentence in
        Results, where they report what they measured. GenerationAgent grounds
        hypotheses in retrieved chunks, so without this, author speculation
        entered the pipeline as established fact and came back out as
        "grounding evidence".

        Chunks from the References section are dropped: they are citation
        strings, useless as retrieval targets and pure index pollution.
        """
        chunks = self._chunk_text_raw(text, paper_id, paper_title)
        return self._annotate_sections(chunks, text)

    @staticmethod
    def _annotate_sections(
        chunks: list[DocumentChunk], full_text: str,
    ) -> list[DocumentChunk]:
        """Attach section labels and evidential weights; drop References."""
        try:
            from utils.imrad import (
                Section,
                annotate_chunk,
                evidential_score,
                section_at,
                segment,
                should_index,
            )
        except ImportError:
            return chunks

        spans = segment(full_text)
        if not spans:
            return chunks

        kept: list[DocumentChunk] = []
        cursor = 0
        for chunk in chunks:
            head = (chunk.text or "")[:120]
            offset = full_text.find(head, cursor) if head else -1
            if offset < 0:
                offset = full_text.find(head) if head else cursor
            if offset < 0:
                offset = cursor
            else:
                cursor = offset

            section = section_at(spans, offset)
            if not should_index(section):
                continue

            chunk.section = section
            meta = dict(chunk.metadata or {})
            meta.update(annotate_chunk(chunk.text, section))
            meta["evidential_score"] = evidential_score(chunk.text, section)
            chunk.metadata = meta
            kept.append(chunk)

        # Never return an empty index because heading detection misfired.
        if not kept and chunks:
            for chunk in chunks:
                chunk.section = Section.UNKNOWN
            return chunks
        return kept

    def _chunk_text_raw(self, text: str, paper_id: str, paper_title: str) -> list[DocumentChunk]:
        """Split text into semantic chunks (section-agnostic)."""

        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._count_tokens(para)

            # If paragraph itself is too large, split it
            if para_tokens > self.chunk_size:
                # Add current chunk if exists
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        paper_id=paper_id,
                        paper_title=paper_title,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = self._count_tokens(sent)
                    if current_tokens + sent_tokens > self.chunk_size:
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            chunks.append(DocumentChunk(
                                text=chunk_text,
                                paper_id=paper_id,
                                paper_title=paper_title,
                                chunk_index=chunk_index
                            ))
                            chunk_index += 1

                        # Start new chunk with overlap
                        if chunks and self.overlap > 0:
                            # Take last sentences for overlap
                            overlap_text = current_chunk[-1] if current_chunk else ""
                            current_chunk = [overlap_text, sent] if overlap_text else [sent]
                            current_tokens = self._count_tokens(" ".join(current_chunk))
                        else:
                            current_chunk = [sent]
                            current_tokens = sent_tokens
                    else:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens
            else:
                # Normal paragraph
                if current_tokens + para_tokens > self.chunk_size:
                    # Save current chunk
                    if current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunks.append(DocumentChunk(
                            text=chunk_text,
                            paper_id=paper_id,
                            paper_title=paper_title,
                            chunk_index=chunk_index
                        ))
                        chunk_index += 1

                    # Start new chunk with overlap
                    if chunks and self.overlap > 0:
                        overlap_text = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_text, para] if overlap_text else [para]
                        current_tokens = self._count_tokens("\n\n".join(current_chunk))
                    else:
                        current_chunk = [para]
                        current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(DocumentChunk(
                text=chunk_text,
                paper_id=paper_id,
                paper_title=paper_title,
                chunk_index=chunk_index
            ))

        return chunks


class RAGEngine:
    """Main RAG engine orchestrating all components"""

    def __init__(
        self,
        collection_name: str = "papers",
        persist_dir: str = "./chroma_db",
        enable_hybrid: bool = True,
        enable_rerank: bool = True,
    ):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.enable_hybrid = enable_hybrid
        self.enable_rerank = enable_rerank

        # Initialize components
        self.downloader = PDFDownloader()
        self.processor = DocumentProcessor()
        self.chunker = SemanticChunker()

        # Hybrid retrieval state — populated lazily by _ensure_bm25_index().
        self._bm25 = None
        self._bm25_dirty = True
        self._reranker = None

        # Initialize embedding model
        self.embedding_model = None
        self.embedding_model_name = None
        if SentenceTransformer:
            self.embedding_model, self.embedding_model_name = _load_embedding_model()

        # Initialize vector store
        self.chroma_client = None
        self.collection = None
        if chromadb:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.persist_dir)
                )
                # Get or create collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Scientific papers for RAG"}
                )
                logger.info("Vector database initialised: %d chunks indexed.", self.collection.count())
            except Exception as e:
                logger.warning("Failed to initialise ChromaDB: %s", e)

    async def process_papers(self, papers: list[dict]) -> int:
        """Download, process, and index papers"""
        if not self.embedding_model or not self.collection:
            logger.warning("RAG system not fully initialised — skipping paper processing.")
            return 0

        total_chunks = 0

        for paper in papers:
            try:
                url = paper.get("url", "")
                if not url:
                    logger.warning("Paper '%s' has no URL — skipping.", paper.get("title", "<unknown>"))
                    continue

                # Generate a stable, normalised paper ID
                paper_id = _url_to_paper_id(url)

                # Skip papers already present in the vector store
                existing = self.collection.get(
                    where={"paper_id": paper_id},
                    limit=1,
                )
                if existing and existing["ids"]:
                    logger.info("Skipping '%s' (already indexed).", paper.get("title", url)[:60])
                    continue

                logger.info("Processing: %s", paper.get("title", url)[:60])

                # Download PDF/TXT
                file_path = await self.downloader.download_paper(paper)
                if not file_path:
                    logger.warning("  Skipping '%s' — no PDF available.", paper.get("title", url)[:60])
                    continue

                # Extract text
                text = await self.processor.extract_text(file_path)
                if not text:
                    logger.warning("  Skipping '%s' — text extraction failed.", paper.get("title", url)[:60])
                    continue

                # Chunk text
                chunks = self.chunker.chunk_text(text, paper_id, paper.get("title", ""))

                # Enrich with extracted tables and figure captions
                title = paper.get("title", "")
                if file_path.suffix.lower() == ".pdf":
                    for _t_idx, tbl in enumerate(extract_tables_from_pdf(file_path)):
                        chunks.append(DocumentChunk(
                            text=tbl["text"],
                            paper_id=paper_id,
                            paper_title=title,
                            chunk_index=len(chunks),
                            page_number=tbl.get("page"),
                            chunk_type="table",
                        ))
                for caption in extract_figure_captions(text):
                    chunks.append(DocumentChunk(
                        text=caption,
                        paper_id=paper_id,
                        paper_title=title,
                        chunk_index=len(chunks),
                        chunk_type="figure_caption",
                    ))

                logger.info("  Created %d chunks (text+tables+captions).", len(chunks))

                # Generate embeddings and add to vector store
                await self._index_chunks(chunks)
                total_chunks += len(chunks)

            except Exception as e:
                logger.warning("Error processing paper '%s': %s", paper.get("title", ""), e)
                continue

        logger.info("Indexed %d total chunks from %d papers.", total_chunks, len(papers))
        return total_chunks

    async def _index_chunks(self, chunks: list[DocumentChunk]):
        """Generate embeddings and add chunks to vector store"""
        if not chunks:
            return

        # Prepare data
        texts = [chunk.text for chunk in chunks]
        ids = [f"{chunk.paper_id}_chunk_{chunk.chunk_index}" for chunk in chunks]
        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "paper_title": chunk.paper_title,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number or -1,
            }
            for chunk in chunks
        ]

        # Generate embeddings
        def generate_embeddings():
            return self.embedding_model.encode(texts, convert_to_tensor=False).tolist()

        embeddings = await asyncio.to_thread(generate_embeddings)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        # New chunks ⇒ BM25 index needs to be rebuilt before next hybrid query.
        self._bm25_dirty = True

    # ------------------------------------------------------------------
    # Hybrid retrieval (BM25 + dense + RRF + optional cross-encoder)
    # ------------------------------------------------------------------

    def _ensure_bm25_index(self) -> None:
        """Build (or rebuild) the BM25 index from the current Chroma corpus."""
        if not self.enable_hybrid or self.collection is None:
            return
        if self._bm25 is not None and not self._bm25_dirty:
            return
        try:
            from utils.hybrid_retrieval import BM25Index
            data = self.collection.get(include=["documents", "metadatas"])
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            if not ids or not docs:
                self._bm25 = BM25Index()
                self._bm25_dirty = False
                return
            idx = BM25Index()
            for cid, text in zip(ids, docs, strict=False):
                idx.add(cid, text or "")
            idx.build()
            self._bm25 = idx
            self._bm25_dirty = False
            logger.info("BM25 index built: %d chunks.", len(idx))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 index build failed: %s — hybrid disabled.", exc)
            self._bm25 = None
            self._bm25_dirty = False

    def _ensure_reranker(self):
        """Lazy-load the cross-encoder reranker on first hybrid query."""
        if not self.enable_rerank:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            from utils.hybrid_retrieval import CrossEncoderReranker
            self._reranker = CrossEncoderReranker()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker init failed: %s", exc)
            self._reranker = None
        return self._reranker

    async def query_hybrid(
        self,
        query_text: str,
        top_k: int = 5,
        fusion_candidates: int = 50,
    ) -> list[dict]:
        """Hybrid retrieval: BM25 + dense + RRF fusion + optional rerank.

        Falls back transparently to pure dense (``query``) when the BM25
        layer is unavailable.
        """
        if not self.embedding_model or not self.collection:
            logger.warning("RAG system not available.")
            return []

        from utils.hybrid_retrieval import hybrid_search

        # Pull a deeper dense slate for fusion.
        try:
            def encode_query():
                return self.embedding_model.encode(
                    [query_text], convert_to_tensor=False,
                ).tolist()

            query_embedding = await asyncio.to_thread(encode_query)
            dense_raw = self.collection.query(
                query_embeddings=query_embedding,
                n_results=fusion_candidates,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hybrid query (dense leg) failed: %s", exc)
            return []

        # Build (doc_id, text) pairs in dense rank order.
        dense_pairs: list[tuple[str, str]] = []
        meta_by_id: dict[str, dict] = {}
        if dense_raw.get("documents"):
            ids = dense_raw.get("ids", [[]])[0]
            docs = dense_raw["documents"][0]
            metas = dense_raw.get("metadatas", [[]])[0]
            for cid, text, meta in zip(ids, docs, metas, strict=False):
                dense_pairs.append((cid, text))
                meta_by_id[cid] = meta or {}

        # Build / refresh the BM25 index, instantiate the reranker.
        self._ensure_bm25_index()
        reranker = self._ensure_reranker()

        fused = hybrid_search(
            query_text,
            dense_results=dense_pairs,
            bm25=self._bm25,
            top_k=top_k,
            fusion_candidates=fusion_candidates,
            reranker=reranker,
        )

        # Format results in the same shape as ``query``.
        out: list[dict] = []
        for cid, score, text in fused:
            meta = meta_by_id.get(cid, {})
            out.append({
                "text": text,
                "paper_title": meta.get("paper_title", ""),
                "paper_id": meta.get("paper_id", ""),
                "score": float(score),
            })
        return out

    async def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Semantic search over indexed papers"""
        if not self.embedding_model or not self.collection:
            logger.warning("RAG system not available.")
            return []

        try:
            # Generate query embedding
            def encode_query():
                return self.embedding_model.encode([query_text], convert_to_tensor=False).tolist()

            query_embedding = await asyncio.to_thread(encode_query)

            # Search
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )

            # Format results
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "text": doc,
                        "paper_title": results['metadatas'][0][i]['paper_title'],
                        "paper_id": results['metadatas'][0][i]['paper_id'],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    })

            return formatted_results

        except Exception as e:
            logger.warning("RAG query failed: %s", e)
            return []

    def is_paper_indexed(self, paper_url: str) -> bool:
        """Return True if the paper identified by *paper_url* is already in the vector store."""
        if not self.collection:
            return False
        paper_id = _url_to_paper_id(paper_url)
        existing = self.collection.get(where={"paper_id": paper_id}, limit=1)
        return bool(existing and existing["ids"])

    def close(self) -> None:
        """Release the ChromaDB PersistentClient and its underlying SQLite database.

        On Windows the SQLite WAL file stays locked while the Chroma
        system is alive, which prevents ``shutil.rmtree`` (or
        ``tempfile.TemporaryDirectory`` cleanup) from deleting the
        persist directory.  Calling this method stops the internal
        system and resets all references so the directory can be
        safely removed.
        """
        if self.chroma_client is not None:
            try:
                # ChromaDB's PersistentClient wraps a `System` object
                # that manages the SQLite connection pool.
                system = getattr(self.chroma_client, "_system", None)
                if system is not None and hasattr(system, "stop"):
                    system.stop()
                    logger.info("ChromaDB system stopped — file locks released.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping ChromaDB system: %s", exc)
            finally:
                self.chroma_client = None
                self.collection = None

    def get_stats(self) -> dict:
        """Get RAG system statistics"""
        if not self.collection:
            return {"status": "unavailable"}

        return {
            "status": "ready",
            "total_chunks": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "vector_db": "ChromaDB" if self.chroma_client else None
        }
