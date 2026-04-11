import os
import logging
import warnings

os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Suppress verbose logging from OCR and Docling
logging.getLogger("RapidOCR").setLevel(logging.ERROR)
logging.getLogger("rapidocr").setLevel(logging.ERROR)
logging.getLogger("rapidocr_onnxruntime").setLevel(logging.ERROR)
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling.pipeline").setLevel(logging.WARNING)

# Suppress "RapidOCR returned empty result!" warnings
warnings.filterwarnings("ignore", message=".*RapidOCR.*")

from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from langchain_core.documents import Document
from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
import base64
import io

KEEP_CLASSES= {"chart", "flow_diagram", "diagram", "table", "plot", "graph", "map"}
SKIP_CLASSES= {"logo", "signature", "icon", "decorative", "photograph","other","engineering_drawing"}

# Noise patterns to filter out garbage text
NOISE_WORDS = {"photograph", "logo", "icon", "image", "glyph"}
MIN_MEANINGFUL_LENGTH = 50  # Minimum characters of real content


def is_garbage_chunk(text: str) -> bool:
    """
    Detect garbage/noise chunks that should be filtered out.
    Returns True if the chunk is garbage.
    """
    # Check for GLYPH font encoding garbage
    if "GLYPH<" in text:
        return True
    
    # Check for font encoding patterns
    if "font=/" in text.lower():
        return True
    
    # Remove known noise words and check what's left
    clean = text.lower()
    for word in NOISE_WORDS:
        clean = clean.replace(word, "")
    
    # Remove common separators and whitespace
    clean = clean.replace("\n", " ").replace("  ", " ").strip()
    
    # If almost nothing meaningful remains, it's garbage
    if len(clean) < MIN_MEANINGFUL_LENGTH:
        return True
    
    return False


def load_from_pdf_docling(file_path: str) -> Tuple[List[Document], List[Document]]:
    """
    Parse a PDF using Docling's native API with picture detection + classification.
    Handles text chunking (via Docling's HierarchicalChunker) and image extraction.
    
    Returns:
        all_text_docs: Chunked text + table documents (ready for embed_text)
        image_docs: Chart/diagram images with base64 in metadata (ready for VLM + embed_text)
    """
    pdf_path = str(file_path)
    file_name = Path(pdf_path).name

    # Configure Docling pipeline
    pipeline_options = PdfPipelineOptions(allow_external_plugins=True)
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_classification = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    print(f"📄 [Docling] Parsing PDF: {file_name}")
    conv_res = doc_converter.convert(pdf_path)

    table_docs = []
    image_docs = []

    for element, _level in conv_res.document.iterate_items():
        # Get page number from provenance
        page_no = element.prov[0].page_no if hasattr(element, 'prov') and element.prov else 0

        if isinstance(element, PictureItem):
            # Get classification
            class_name = "unknown"
            if hasattr(element, 'annotations') and element.annotations:
                for ann in element.annotations:
                    if hasattr(ann, 'predicted_classes') and ann.predicted_classes:
                        class_name = ann.predicted_classes[0].class_name
                        break
            
            # Skip logos, icons, decorative elements
            if class_name.lower() in SKIP_CLASSES:
                print(f"   ⏭️ Skipping {class_name} on page {page_no}")
                continue

            # Keep charts, diagrams, tables — convert to base64
            pil_img = element.get_image(conv_res.document)
            if pil_img is None:
                continue

            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            image_doc = Document(
                page_content=f"[image: {class_name} on page {page_no}]",
                metadata={
                    "page": page_no,
                    "type": "image",
                    "image_class": class_name,
                    "source": pdf_path,
                    "file_name": file_name,
                    "image_base64": img_base64,
                    "is_full_page": False,

                }
            )
            image_docs.append(image_doc)
            print(f"   🖼️ Kept {class_name} on page {page_no}")

        elif isinstance(element, TableItem):
            # Tables: extract as text (Docling parses table structure)
            table_text = element.export_to_markdown()
            if table_text.strip():
                table_docs.append(Document(
                    page_content=table_text,
                    metadata={
                        "page": page_no,
                        "type": "text",
                        "source": pdf_path,
                        "file_name": file_name,
                    }
                ))
    # Chunk all text using Docling's chunker
    text_docs = chunk_text_with_docling(conv_res, file_path)

    # Combine text chunks + table docs
    all_text_docs = text_docs + table_docs
        

    print(f"   ✅ [Docling] Done: {len(all_text_docs)} text docs, {len(image_docs)} image docs")
    return all_text_docs, image_docs


PAGE_MERGE_CHAR_LIMIT = 3000


def chunk_text_with_docling(conv_res, file_path: str) -> List[Document]:
    """
    Use Docling's HybridChunker to chunk the text from a parsed document.
    
    NO REDUNDANCY STRATEGY:
        - Pages with >4 chunks: save as ONE page-level document only
        - Pages with <=4 chunks: save individual chunks only
    
    Each page's content appears exactly ONCE in the vector store.
    """
    file_name = Path(file_path).name
    chunker = HybridChunker()
    chunks = list(chunker.chunk(conv_res.document))

    # First pass: collect all chunks by page with their metadata
    page_buckets: dict[int, List[tuple]] = defaultdict(list)  # page -> [(content, headings), ...]

    skipped_garbage = 0
    
    for chunk in chunks:
        body_text = chunk.text.strip()
        if not body_text:
            continue

        # Skip garbage chunks (GLYPH encoding, cover pages, etc.)
        if is_garbage_chunk(body_text):
            skipped_garbage += 1
            continue

        page_no = 0
        if chunk.meta and chunk.meta.doc_items:
            for item in chunk.meta.doc_items:
                if item.prov:
                    page_no = item.prov[0].page_no
                    break

        headings = []
        if chunk.meta and chunk.meta.headings:
            headings = list(chunk.meta.headings)

        if headings:
            heading_prefix = "\n".join(headings)
            page_content = f"{heading_prefix}\n{body_text}"
        else:
            page_content = body_text

        page_buckets[page_no].append((page_content, headings))
    
    if skipped_garbage > 0:
        print(f"   🧹 [Docling] Filtered out {skipped_garbage} garbage chunks (GLYPH/cover pages)")

    # Second pass: for each page, decide chunk-level OR page-level (not both)
    fine_docs = []
    page_docs = []
    
    for page_no in sorted(page_buckets):
        chunks_on_page = page_buckets[page_no]
        
        if len(chunks_on_page) > 4:
            # Many chunks → merge into ONE page-level document (no individual chunks)
            merged_text = "\n\n".join([content for content, _ in chunks_on_page])
            if len(merged_text) > PAGE_MERGE_CHAR_LIMIT:
                merged_text = merged_text[:PAGE_MERGE_CHAR_LIMIT]
            
            page_docs.append(Document(
                page_content=f"[Full page {page_no} content]\n{merged_text}",
                metadata={
                    "page": page_no,
                    "type": "text",
                    "source": str(file_path),
                    "file_name": file_name,
                    "headings": [],
                    "granularity": "page",
                }
            ))
        else:
            # Few chunks → save each chunk individually (no page-level doc)
            for content, headings in chunks_on_page:
                fine_docs.append(Document(
                    page_content=content,
                    metadata={
                        "page": page_no,
                        "type": "text",
                        "source": str(file_path),
                        "file_name": file_name,
                        "headings": headings,
                        "granularity": "chunk",
                    }
                ))

    all_docs = fine_docs + page_docs
    print(
        f"   📝 [Docling] {len(fine_docs)} chunk-level + {len(page_docs)} page-level = "
        f"{len(all_docs)} total text docs (no redundancy)"
    )
    return all_docs



