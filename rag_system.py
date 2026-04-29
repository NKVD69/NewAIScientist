"""
RAG System for Scientific Literature Analysis
Provides advanced document retrieval and semantic search capabilities
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import urllib.request
import hashlib
import re

import config

logger = logging.getLogger(__name__)


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

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Vector database
try:
    import chromadb
    from chromadb.config import Settings
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
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict = None


class PDFDownloader:
    """Downloads PDFs from ArXiv and PubMed"""
    
    def __init__(self, cache_dir: str = "./papers"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL (keyed on the normalised URL)."""
        url_hash = hashlib.md5(_normalize_arxiv_url(url).encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.pdf"
    
    async def download_arxiv_pdf(self, paper_url: str) -> Optional[Path]:
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
    
    async def _get_pmcid_from_pmid(self, pmid: str) -> Optional[str]:
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

    async def download_pubmed_pdf(self, paper_url: str) -> Optional[Path]:
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
    
    async def download_paper(self, paper: Dict) -> Optional[Path]:
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
    
    async def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract all text from PDF or TXT"""
        try:
            if file_path.suffix.lower() == '.txt':
                def extract_txt():
                    with open(file_path, 'r', encoding='utf-8') as f:
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
    
    def chunk_text(self, text: str, paper_id: str, paper_title: str) -> List[DocumentChunk]:
        """Split text into semantic chunks"""
        
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
    
    def __init__(self, collection_name: str = "papers", persist_dir: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.downloader = PDFDownloader()
        self.processor = DocumentProcessor()
        self.chunker = SemanticChunker()
        
        # Initialize embedding model
        self.embedding_model = None
        if SentenceTransformer:
            try:
                logger.info("Loading embedding model (first run may take a moment)...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded.")
            except Exception as e:
                logger.warning("Failed to load embedding model: %s", e)
        
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
    
    async def process_papers(self, papers: List[Dict]) -> int:
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
                logger.info("  Created %d chunks.", len(chunks))

                # Generate embeddings and add to vector store
                await self._index_chunks(chunks)
                total_chunks += len(chunks)

            except Exception as e:
                logger.warning("Error processing paper '%s': %s", paper.get("title", ""), e)
                continue

        logger.info("Indexed %d total chunks from %d papers.", total_chunks, len(papers))
        return total_chunks
    
    async def _index_chunks(self, chunks: List[DocumentChunk]):
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
                "chunk_index": chunk.chunk_index
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
    
    async def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
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

    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        if not self.collection:
            return {"status": "unavailable"}
        
        return {
            "status": "ready",
            "total_chunks": self.collection.count(),
            "embedding_model": "all-MiniLM-L6-v2" if self.embedding_model else None,
            "vector_db": "ChromaDB" if self.chroma_client else None
        }
