import fitz
import pdfplumber
import pytesseract
from PIL import Image
import re
import os
import json
import hashlib
from typing import List, Dict, Tuple
from collections import Counter
from deep_translator import GoogleTranslator
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")
DOCS_FOLDER = "/mount/src/spo/documents"
TESSERACT_PATH = os.getenv("TESSDATA_PATH")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Initialize translator
translator = GoogleTranslator(source='auto', target='en')

# ============================================
# ADVANCED CHUNKING CONFIGURATION
# ============================================

# Max tokens for embedding model (intfloat/multilingual-e5-large supports 512 tokens)
MAX_EMBEDDING_TOKENS = 450  # Keep buffer for safety
MIN_CHUNK_TOKENS = 50  # Minimum meaningful chunk size
OVERLAP_TOKENS = 50  # Overlap between chunks for context

# Initialize semantic splitter
try:
    splitter = TextSplitter.from_huggingface_tokenizer(
        Tokenizer.from_pretrained("intfloat/multilingual-e5-large"),
        capacity=MAX_EMBEDDING_TOKENS,
        overlap=OVERLAP_TOKENS
    )
    print("✅ Semantic text splitter initialized successfully")
except Exception as e:
    print(f"⚠️ Could not initialize semantic splitter: {e}")
    splitter = None


# ============================================
# FILE UTILITIES
# ============================================

def get_files_from_folder():
    """Get all PDF files from documents folder"""
    files = []
    for f in os.listdir(DOCS_FOLDER):
        if f.lower().endswith(".pdf"):
            files.append(os.path.join(DOCS_FOLDER, f))
    return files


def get_file_hash(filepath: str) -> str:
    """Generate hash for file caching"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_cache(key: str):
    """Load cached document processing results"""
    cache_path = os.path.join(CACHE_FOLDER, f"{key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(key: str, data):
    """Save document processing results to cache"""
    cache_path = os.path.join(CACHE_FOLDER, f"{key}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================
# TEXT UTILITIES
# ============================================

def normalize(s: str) -> str:
    """Normalize text for comparison"""
    return re.sub(r"\s+", " ", s.lower()).strip()


def normalize_multilang(line: str) -> str:
    """Normalize text across languages using translation"""
    line = line.strip()
    if not line:
        return ""
    
    try:
        translated = translator.translate(line)
        return normalize(translated)
    except Exception as e:
        return normalize(line)


def remove_dynamic_noise(text: str, doc_name: str) -> str:
    """Remove repeated filename and page numbers"""
    out = []
    for line in text.splitlines():
        ln = normalize(line)
        
        if normalize(doc_name) in ln:
            continue
        
        if re.search(r"(seite|page|pagina|página|صفحة)\s+\d+(\s+(von|of|de)\s+\d+)?", ln):
            continue
        
        out.append(line)
    
    return "\n".join(out)


def remove_empty_lines(text: str) -> str:
    """Remove empty lines from text"""
    return "\n".join(l for l in text.splitlines() if l.strip())


def is_header(line: str) -> bool:
    """Detect if line is a section header"""
    line = line.strip()
    
    if len(line) > 120:
        return False
    
    if line.isupper():
        return True
    
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
        return True
    
    return False


# ============================================
# SMART CHUNKING UTILITIES
# ============================================

def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)"""
    # Average: 1 token ≈ 4 characters for multilingual text
    return len(text) // 4


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into meaningful paragraphs"""
    # Split by double newlines or major breaks
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def smart_chunk_text(text: str, metadata: dict, max_tokens: int = MAX_EMBEDDING_TOKENS) -> List[Dict]:
    """
    Intelligently split text into chunks based on semantic meaning
    """
    chunks = []
    
    # Check if text is within limits
    text_tokens = estimate_tokens(text)
    
    print(f"    📏 Text length: {len(text)} chars, ~{text_tokens} tokens")
    
    # If text is small enough, keep as is
    if text_tokens <= max_tokens:
        print(f"    ✅ Chunk fits within limit, keeping as single chunk")
        return [{
            "content": text,
            "metadata": metadata
        }]
    
    print(f"    ⚠️ Chunk exceeds limit! Splitting intelligently...")
    
    # Try semantic splitter first
    if splitter:
        try:
            semantic_chunks = splitter.chunks(text)
            print(f"    🧠 Semantic splitter created {len(semantic_chunks)} chunks")
            
            for idx, chunk_text in enumerate(semantic_chunks, 1):
                chunk_tokens = estimate_tokens(chunk_text)
                print(f"      • Chunk {idx}: {len(chunk_text)} chars, ~{chunk_tokens} tokens")
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": idx,
                        "total_chunks": len(semantic_chunks),
                        "is_split": True
                    }
                })
            return chunks
        except Exception as e:
            print(f"    ⚠️ Semantic splitting failed: {e}, using paragraph-based fallback")
    
    # Fallback: Split by paragraphs
    paragraphs = split_into_paragraphs(text)
    print(f"    📝 Found {len(paragraphs)} paragraphs")
    
    current_chunk = []
    current_tokens = 0
    chunk_idx = 1
    
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        
        # If single paragraph exceeds limit, split it further
        if para_tokens > max_tokens:
            print(f"    ⚠️ Large paragraph detected ({para_tokens} tokens), splitting by sentences...")
            
            # Save current chunk if exists
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_idx,
                        "is_split": True
                    }
                })
                chunk_idx += 1
                current_chunk = []
                current_tokens = 0
            
            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = []
            sentence_tokens = 0
            
            for sent in sentences:
                sent_tokens = estimate_tokens(sent)
                
                if sentence_tokens + sent_tokens > max_tokens:
                    if sentence_chunk:
                        chunk_text = " ".join(sentence_chunk)
                        chunks.append({
                            "content": chunk_text,
                            "metadata": {
                                **metadata,
                                "chunk_index": chunk_idx,
                                "is_split": True
                            }
                        })
                        chunk_idx += 1
                    sentence_chunk = [sent]
                    sentence_tokens = sent_tokens
                else:
                    sentence_chunk.append(sent)
                    sentence_tokens += sent_tokens
            
            if sentence_chunk:
                chunk_text = " ".join(sentence_chunk)
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_idx,
                        "is_split": True
                    }
                })
                chunk_idx += 1
            
            continue
        
        # Normal paragraph processing
        if current_tokens + para_tokens > max_tokens:
            # Save current chunk
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_idx,
                    "is_split": True
                }
            })
            chunk_idx += 1
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append({
            "content": chunk_text,
            "metadata": {
                **metadata,
                "chunk_index": chunk_idx,
                "is_split": True
            }
        })
    
    # Update total_chunks in metadata
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    print(f"    ✅ Created {len(chunks)} semantic chunks")
    return chunks


def merge_small_chunks(chunks: List[Dict], min_tokens: int = MIN_CHUNK_TOKENS) -> List[Dict]:
    """
    Merge consecutive small chunks to improve efficiency
    """
    if not chunks:
        return chunks
    
    print(f"\n🔗 Merging small chunks (minimum {min_tokens} tokens)...")
    
    merged = []
    current = None
    
    for chunk in chunks:
        # Never merge tables - keep them intact
        if chunk["metadata"].get("type") == "table_with_context":
            if current:
                merged.append(current)
                current = None
            merged.append(chunk)
            continue
        
        chunk_tokens = estimate_tokens(chunk["content"])
        
        if current is None:
            current = chunk
            continue
        
        current_tokens = estimate_tokens(current["content"])
        
        # Merge if both are small and same page
        if (chunk_tokens < min_tokens and current_tokens < min_tokens and
            chunk["metadata"]["page"] == current["metadata"]["page"]):
            
            print(f"  🔗 Merging chunks from page {chunk['metadata']['page']}")
            current["content"] = current["content"] + "\n\n" + chunk["content"]
            current["metadata"]["is_merged"] = True
        else:
            merged.append(current)
            current = chunk
    
    if current:
        merged.append(current)
    
    print(f"✅ Merged into {len(merged)} chunks (from {len(chunks)})")
    return merged


# ============================================
# HEADER/FOOTER DETECTION
# ============================================

def detect_repeated_headers_footers(pages_text: List[str], top_k: int = 2, 
                                     bottom_k: int = 2, threshold: float = 0.8):
    """Statistically detect repeated headers and footers"""
    total_pages = len(pages_text)
    header_counter = Counter()
    footer_counter = Counter()
    
    for text in pages_text:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < top_k + bottom_k:
            continue
        
        for l in lines[:top_k]:
            header_counter[normalize_multilang(l)] += 1
        
        for l in lines[-bottom_k:]:
            footer_counter[normalize_multilang(l)] += 1
    
    headers = {h for h, c in header_counter.items() if c / total_pages >= threshold}
    footers = {f for f, c in footer_counter.items() if c / total_pages >= threshold}
    
    return headers, footers


# ============================================
# TABLE UTILITIES
# ============================================

def clean_cell(cell):
    """Clean table cell content"""
    return str(cell).strip() if cell is not None else ""


def normalize_table(table: List[List[str]]) -> List[List[str]]:
    """Normalize table structure"""
    if not table:
        return []
    
    cleaned = [[clean_cell(c) for c in row] for row in table]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    
    if len(cleaned) <= 1:
        return []
    
    max_cols = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (max_cols - len(row)) for row in cleaned]
    
    cols_to_keep = [i for i in range(max_cols) if any(row[i] for row in cleaned)]
    
    return [[row[i] for i in cols_to_keep] for row in cleaned]


def headers_similar(h1, h2) -> bool:
    """Check if two table headers are similar"""
    h1 = [c.lower() for c in h1]
    h2 = [c.lower() for c in h2]
    same = sum(1 for a, b in zip(h1, h2) if a == b)
    return same >= max(1, len(h1) // 2)


def table_bbox(table_obj):
    """Get table bounding box"""
    return table_obj.bbox[1], table_obj.bbox[3]


def has_text_between(page, y1, y2) -> bool:
    """Check if there's text between two y coordinates"""
    for w in page.extract_words():
        if y1 < w["top"] < y2:
            return True
    return False


def table_to_markdown(table: List[List[str]]) -> str:
    """Convert table to markdown format"""
    if not table or len(table) < 2:
        return ""
    
    lines = []
    
    # Header
    lines.append("| " + " | ".join(table[0]) + " |")
    lines.append("|" + "|".join(["---" for _ in table[0]]) + "|")
    
    # Rows
    for row in table[1:]:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


# ============================================
# OCR DETECTION
# ============================================

def render_page_for_ocr(fitz_doc, page_index: int) -> Image.Image:
    """Render PDF page as image for OCR"""
    page = fitz_doc.load_page(page_index)
    pix = page.get_pixmap(dpi=300, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def ocr_says_table(fitz_doc, page_index: int) -> bool:
    """Use OCR to detect if page contains table"""
    img = render_page_for_ocr(fitz_doc, page_index)
    text = pytesseract.image_to_string(img, lang="deu+eng", config="--psm 6")
    
    score = 0
    for l in text.splitlines():
        digits = sum(c.isdigit() for c in l)
        letters = sum(c.isalpha() for c in l)
        if digits >= 2:
            score += 1
        if digits > 0 and letters > 0:
            score += 1
    
    return score >= 6


# ============================================
# MAIN EXTRACTION FUNCTION WITH SMART CHUNKING
# ============================================

def extract_pdf_detailed(pdf_path: str):
    """
    Extract text and tables from PDF with SMART CHUNKING
    Returns: (document_info, error_message)
    """
    try:
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{'='*60}")
        print(f"📄 Processing: {doc_name}")
        print(f"{'='*60}")
        
        raw_chunks = []
        
        # Open with both libraries
        fitz_doc = fitz.open(pdf_path)
        
        # PASS 1: Clean pages for statistics
        print("\n🔍 PASS 1: Cleaning pages and detecting patterns...")
        cleaned_pages = []
        for page_num, page in enumerate(fitz_doc, 1):
            print(f"  📄 Page {page_num}/{len(fitz_doc)}: Extracting text...")
            raw_text = page.get_text()
            cleaned = remove_empty_lines(remove_dynamic_noise(raw_text, doc_name))
            cleaned_pages.append(cleaned)
            print(f"    ✅ Extracted {len(cleaned)} characters")
        
        # Detect repeated headers/footers
        print("\n🔍 Detecting repeated headers/footers...")
        repeated_headers, repeated_footers = detect_repeated_headers_footers(cleaned_pages)
        print(f"  ✅ Found {len(repeated_headers)} repeated headers")
        print(f"  ✅ Found {len(repeated_footers)} repeated footers")
        
        # PASS 2: Extract content with smart chunking
        print(f"\n🔍 PASS 2: Extracting content with smart chunking...")
        with pdfplumber.open(pdf_path) as pdf:
            current_section = ""
            last_context_text = ""
            
            for page_idx, (clean_text, plumber_page) in enumerate(zip(cleaned_pages, pdf.pages), start=1):
                print(f"\n📄 Processing Page {page_idx}/{len(pdf.pages)}")
                
                # Extract tables from this page
                tables = plumber_page.find_tables()
                
                # Use OCR if no tables found but OCR detects them
                if not tables and ocr_says_table(fitz_doc, page_idx - 1):
                    print(f"  🔍 OCR detected potential tables, re-scanning...")
                    tables = plumber_page.find_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 3,
                            "join_tolerance": 3
                        }
                    )
                
                # Process tables
                table_blocks = []
                if tables:
                    print(f"  📊 Found {len(tables)} table(s)")
                    merged_tables = []
                    current_table = None
                    current_bottom = None
                    
                    for t_idx, t in enumerate(tables, 1):
                        normalized = normalize_table(t.extract())
                        if not normalized:
                            continue
                        
                        top, bottom = table_bbox(t)
                        
                        if current_table is None:
                            current_table = normalized
                            current_bottom = bottom
                            continue
                        
                        if not has_text_between(plumber_page, current_bottom, top):
                            if headers_similar(current_table[0], normalized[0]):
                                current_table.extend(normalized[1:])
                            else:
                                current_table.extend(normalized)
                            current_bottom = bottom
                        else:
                            merged_tables.append(current_table)
                            current_table = normalized
                            current_bottom = bottom
                    
                    if current_table:
                        merged_tables.append(current_table)
                    
                    for idx, table in enumerate(merged_tables, 1):
                        if len(table) > 1 and len(table[0]) > 1:
                            table_md = table_to_markdown(table)
                            table_blocks.append(table_md)
                            print(f"    ✅ Table {idx}: {len(table)} rows × {len(table[0])} cols")
                
                # Process text lines
                page_lines = []
                for ln in clean_text.splitlines():
                    ln_norm = normalize_multilang(ln)
                    
                    if ln_norm in repeated_headers or ln_norm in repeated_footers:
                        continue
                    
                    if is_header(ln):
                        current_section = ln.strip()
                    else:
                        if len(ln.strip().split()) >= 5:
                            last_context_text = ln.strip()
                    
                    page_lines.append(ln)
                
                page_text = "\n".join(page_lines)
                
                # Create chunks with SMART SPLITTING
                if table_blocks:
                    # TABLES: Never split, keep complete
                    combined_content = f"{page_text}\n\n### TABLES:\n\n" + "\n\n".join(table_blocks)
                    table_tokens = estimate_tokens(combined_content)
                    print(f"  📊 Table chunk: ~{table_tokens} tokens (WILL NOT BE SPLIT)")
                    
                    raw_chunks.append({
                        "content": combined_content,
                        "metadata": {
                            "source": doc_name,
                            "page": page_idx,
                            "type": "table_with_context",
                            "section": current_section,
                            "table_count": len(table_blocks)
                        }
                    })
                else:
                    # TEXT: Apply smart chunking
                    if page_text.strip():
                        print(f"  📝 Text content found")
                        text_chunks = smart_chunk_text(
                            page_text,
                            metadata={
                                "source": doc_name,
                                "page": page_idx,
                                "type": "semantic_text",
                                "section": current_section,
                                "context": last_context_text
                            }
                        )
                        raw_chunks.extend(text_chunks)
        
        fitz_doc.close()
        
        # PASS 3: Merge small chunks
        print(f"\n🔗 PASS 3: Optimizing chunk distribution...")
        final_chunks = merge_small_chunks(raw_chunks)
        
        print(f"\n{'='*60}")
        print(f"✅ Processing Complete!")
        print(f"{'='*60}")
        print(f"📊 Total pages: {len(cleaned_pages)}")
        print(f"📦 Raw chunks: {len(raw_chunks)}")
        print(f"✨ Optimized chunks: {len(final_chunks)}")
        print(f"{'='*60}\n")
        
        return {"chunks": final_chunks}, None
    
    except Exception as e:
        print(f"\n❌ Error processing {pdf_path}: {str(e)}")

        return None, str(e)

