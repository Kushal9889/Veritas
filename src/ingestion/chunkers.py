"""
Unified Chunking Strategies for the Ingestion Pipeline.
Handles Heading-based, Parent-Child, Semantic, and Hybrid chunking.
"""
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import uuid

# ── 1. Section Classification ──────────────────────────────────────────
SECTION_KEYWORDS = {
    "financial": ["revenue", "income", "profit", "loss", "earnings", "cash flow",
                  "balance sheet", "assets", "liabilities", "margin", "expense",
                  "sales", "debt", "cost", "operating"],
    "risk": ["risk factor", "uncertainty", "litigation", "regulatory", "volatility",
             "competition", "cybersecurity", "compliance", "threat", "danger"],
    "operations": ["manufacturing", "production", "factory", "gigafactory", "vehicle",
                   "energy", "solar", "battery", "supply chain", "segment",
                   "product", "strategy", "business"],
    "legal": ["legal proceeding", "lawsuit", "court", "SEC", "governance",
              "regulation", "filing", "compliance"],
}

def classify_chunk_section(text: str) -> str:
    """Fast keyword classifier — tags a chunk with one of: financial/risk/operations/legal/general."""
    text_lower = text.lower()
    scores = {}
    for section, keywords in SECTION_KEYWORDS.items():
        scores[section] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

# ── 2. Chunking Classes ────────────────────────────────────────────────
class HeadingBasedChunker:
    """Chunk documents based on heading hierarchy (H1, H2, H3)"""
    def __init__(self):
        self.heading_patterns = [
            (r'^PART\s+[IV]+', 1),
            (r'^(?:ITEM|Item)\s+(1[A-C]?|2|3|4|5|6|7[A]?|8|9[A-C]?|10|11|12|13|14|15|16)\.?', 2),
            (r'^[A-Z][a-zA-Z0-9\s]{10,}:?$', 3),
            (r'^\d+\.\s+[A-Z]', 3),
        ]
    
    def detect_heading(self, line: str) -> int:
        line = line.strip()
        if not line:
            return 0
        for pattern, level in self.heading_patterns:
            if re.match(pattern, line):
                return level
        return 0
    
    def chunk_by_headings(self, text: str, metadata: Dict) -> List[Document]:
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_heading = "Introduction"
        current_level = 0
        
        for line in lines:
            heading_level = self.detect_heading(line)
            if heading_level > 0:
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if len(chunk_text) > 100:
                        chunks.append(Document(
                            page_content=chunk_text,
                            metadata={
                                **metadata,
                                "heading": current_heading,
                                "heading_level": current_level,
                                "chunk_type": "heading_based"
                            }
                        ))
                current_heading = line.strip()
                current_level = heading_level
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) > 100:
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **metadata,
                        "heading": current_heading,
                        "heading_level": current_level,
                        "chunk_type": "heading_based"
                    }
                ))
        return chunks

class ParentChildChunker:
    """Create parent-child hierarchy for precise retrieval + full context"""
    def __init__(self, parent_size: int = 2000, child_size: int = 400):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_hierarchy(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, str]]:
        child_docs = []
        parent_map = {}
        for doc in documents:
            parents = self.parent_splitter.split_documents([doc])
            for parent in parents:
                parent_id = str(uuid.uuid4())
                parent_text = parent.page_content
                children = self.child_splitter.split_text(parent_text)
                for child_text in children:
                    child_id = str(uuid.uuid4())
                    child_doc = Document(
                        page_content=child_text,
                        metadata={
                            **parent.metadata,
                            "child_id": child_id,
                            "parent_id": parent_id,
                            "chunk_type": "child"
                        }
                    )
                    child_docs.append(child_doc)
                    parent_map[child_id] = parent_text
        return child_docs, parent_map

class SemanticChunker:
    """Chunk based on semantic boundaries (paragraphs, sentences)"""
    def __init__(self, target_size: int = 800):
        self.target_size = target_size
    
    def chunk_by_semantics(self, text: str, metadata: Dict) -> List[Document]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_size = len(para)
            if para_size > self.target_size * 1.5:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = []
                temp_size = 0
                for sent in sentences:
                    if temp_size + len(sent) > self.target_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = [sent]
                        temp_size = len(sent)
                    else:
                        temp_chunk.append(sent)
                        temp_size += len(sent)
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            elif current_size + para_size > self.target_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        return [
            Document(page_content=chunk, metadata={**metadata,"chunk_type": "semantic","chunk_size": len(chunk)})
            for chunk in chunks if len(chunk) > 100
        ]

class HybridChunker:
    """Combine multiple chunking strategies for optimal retrieval"""
    def __init__(self):
        self.heading_chunker = HeadingBasedChunker()
        self.parent_child_chunker = ParentChildChunker(parent_size=2000, child_size=400)
        self.semantic_chunker = SemanticChunker(target_size=800)
    
    def chunk_document(self, text: str, metadata: Dict, strategy: str = "parent_child") -> Tuple[List[Document], Dict]:
        if strategy == "heading":
            chunks = self.heading_chunker.chunk_by_headings(text, metadata)
            return chunks, {}
        elif strategy == "semantic":
            chunks = self.semantic_chunker.chunk_by_semantics(text, metadata)
            return chunks, {}
        elif strategy == "parent_child":
            base_doc = Document(page_content=text, metadata=metadata)
            chunks, parent_map = self.parent_child_chunker.create_hierarchy([base_doc])
            return chunks, parent_map
        elif strategy == "hybrid":
            heading_chunks = self.heading_chunker.chunk_by_headings(text, metadata)
            all_children = []
            parent_map = {}
            for heading_chunk in heading_chunks:
                children, p_map = self.parent_child_chunker.create_hierarchy([heading_chunk])
                all_children.extend(children)
                parent_map.update(p_map)
            return all_children, parent_map
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

def chunk_multimodal_content(page_contents: List[Dict], strategy: str = "parent_child") -> Tuple[List[Document], Dict[str, str]]:
    chunker = HybridChunker()
    all_chunks = []
    all_parent_maps = {}
    for page_content in page_contents:
        text = page_content["text"]
        metadata = page_content["metadata"]
        chunks, parent_map = chunker.chunk_document(text, metadata, strategy)
        all_chunks.extend(chunks)
        all_parent_maps.update(parent_map)
    return all_chunks, all_parent_maps
