"""
Unified Document Loaders for the Ingestion Pipeline.
Handles Commercial-Grade Multimodal Extraction via Unstructured.io, 
enforcing strict Non-Null Metadata Schemas.
"""
import re
import glob
import uuid
from pathlib import Path
from typing import List, Dict, Iterator, Tuple

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from unstructured.partition.pdf import partition_pdf
except ImportError:
    partition_pdf = None

# ── 1. Commercial Metadata Standard ──────────────────────────────────────
def build_commercial_metadata(
    source: str,
    page_num: int,
    chunk_type: str,
    section_id: str = "[]",
    collection_id: str = "[]",
    ticker: str = "[]",
    year: str = "[]"
) -> Dict:
    """Enforce strict non-null industry standard metadata for absolute traceability."""
    return {
        "doc_id": str(uuid.uuid4()),
        "source": source if source else "[]",
        "company_ticker": ticker if ticker else "[]",
        "filing_type": "10-K", # Defaulting to 10-K for SEC analysis
        "filing_date": year if year else "[]",
        "section_id": section_id if section_id else "[]",
        "chunk_type": chunk_type if chunk_type else "[]",
        "page": int(page_num) if page_num else 1,
        "page_num": int(page_num) if page_num else 1,
        "collection_id": collection_id if collection_id else "[]"
    }

def _extract_metadata_from_filename(file_path: str) -> Tuple[str, str, str, str]:
    path = Path(file_path)
    filename = path.stem
    parts = filename.split('_')
    ticker = parts[0].upper() if len(parts) >= 2 else "[]"
    year = parts[1] if len(parts) >= 2 and re.match(r'^\d{4}$', parts[1]) else "[]"
    return str(path.name), ticker, year, filename

# ── 2. Unstructured Financial Loader (Multimodal) ──────────────────────
class FinancialLoader(BaseLoader):
    """Enterprise 10-K parsing using Unstructured.io for hi-res multimodal extraction."""
    def __init__(self, file_path: str, strategy: str = "fast"):
        self.file_path = file_path
        self.strategy = strategy
        self.output_dir = Path("data/extracted_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
            
    def lazy_load(self) -> Iterator[Document]:
        if partition_pdf is None:
            raise ImportError("Please install unstructured: pip install unstructured[pdf] unstructured-inference pdf2image pdfminer.six")
            
        source_name, ticker, year, collection_id = _extract_metadata_from_filename(self.file_path)
        
        # Partition the PDF using Unstructured
        elements = partition_pdf(
            filename=self.file_path,
            strategy=self.strategy,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=str(self.output_dir),
            extract_image_block_types=["Image", "Table"]
        )
        
        current_section = "[]"
        
        for el in elements:
            el_type = type(el).__name__
            text = str(el).strip()
            if not text:
                continue
                
            # Heuristic to detect SEC sections (Item 1A, etc.) in titles or narrative text
            if "Title" in el_type or "Heading" in el_type:
                match = re.search(r'(?:ITEM|Item)\s+(1[A-C]?|2|3|4|5|6|7[A]?|8|9[A-C]?|10|11|12|13|14|15|16)\.?', text, re.IGNORECASE)
                if match:
                    current_section = f"Item {match.group(1).upper()}"
            
            chunk_type = "Text"
            if "Table" in el_type:
                chunk_type = "Table"
            elif "Image" in el_type or "Figure" in el_type:
                chunk_type = "Image"
                
            page_num = el.metadata.page_number if hasattr(el, 'metadata') and hasattr(el.metadata, 'page_number') else 1
            
            meta = build_commercial_metadata(
                source=source_name,
                page_num=page_num,
                chunk_type=chunk_type,
                section_id=current_section,
                collection_id=collection_id,
                ticker=ticker,
                year=year
            )
            
            # If it's a table, try to get the HTML representation if unstructured extracted it
            if chunk_type == "Table" and hasattr(el.metadata, 'text_as_html') and el.metadata.text_as_html:
                text = el.metadata.text_as_html
                
            # If it's an image, unstructured saves it and puts the path in metadata
            if chunk_type == "Image" and hasattr(el.metadata, 'image_path') and el.metadata.image_path:
                meta["image_path"] = el.metadata.image_path
                
            yield Document(page_content=text, metadata=meta)

    def load(self) -> List[Document]:
        return list(self.lazy_load())


def load_multimodal_pdf(file_path: str, strategy: str = "fast") -> Tuple[List[Dict], List[str], List[Dict]]:
    """Legacy wrapper for multimodal extraction, adapted to Unstructured."""
    loader = FinancialLoader(file_path, strategy=strategy)
    docs = loader.load()
    
    page_contents = []
    image_paths = []
    table_data = []
    
    for doc in docs:
        c_type = doc.metadata.get("chunk_type", "Text")
        if c_type == "Text":
            page_contents.append({"text": doc.page_content, "metadata": doc.metadata})
        elif c_type == "Image":
            img_path = doc.metadata.get("image_path")
            if img_path:
                image_paths.append(img_path)
            page_contents.append({"text": doc.page_content, "metadata": doc.metadata})
        elif c_type == "Table":
            table_data.append({
                "text": doc.page_content, 
                "page": doc.metadata.get("page_num", 1), 
                "row_count": len(doc.page_content.split('\n'))
            })
            page_contents.append({"text": doc.page_content, "metadata": doc.metadata})
            
    return page_contents, image_paths, table_data

# ── 3. Standard Splitter Utilities ──────────────────────────────────────────
def load_and_split_pdf(file_path: str):
    """Ingests a PDF using the commercial Unstructured loader and splits it."""
    loader = FinancialLoader(file_path, strategy="fast")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(raw_docs)

def load_complete_pdf(file_path: str):
    """Enhanced PDF loader that ensures complete document processing with larger chunks."""
    loader = FinancialLoader(file_path, strategy="fast")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(raw_docs)

def load_all_complete_documents():
    """Load all PDFs from data/raw/ with complete processing."""
    pdf_files = glob.glob("data/raw/*.pdf")
    if not pdf_files:
        return []
    all_documents = []
    for pdf_file in pdf_files:
        try:
            docs = load_complete_pdf(pdf_file)
            all_documents.extend(docs)
        except Exception as e:
            print(f"⚠️ Error loading {pdf_file}: {e}")
    return all_documents
