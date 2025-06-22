import re, json, logging
import pandas as pd
from typing import List
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from config import CSV_OUTPUT

logger = logging.getLogger(__name__)

def setup_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.do_cell_matching = True
    # opts.ocr_options = EasyOcrOptions()
    opts.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    # opts.ocr_options.lang = ["en"]
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts, backend=PyPdfiumDocumentBackend)})

def chunk_markdown(md: str, max_chars: int = 5000) -> List[str]:
    sections = re.split(r"\n(?=##\s)", md)
    chunks, buf = [], ""
    for sec in sections:
        if len(buf) + len(sec) < max_chars:
            buf += sec.strip() + "\n"
        else:
            chunks.append(buf.strip())
            buf = sec.strip() + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

def generate_prompt(chunk: str, prev_items: List[dict], component: List[dict]) -> str:
    prev = json.dumps(prev_items, indent=2) if prev_items else "[]"
    comp = json.dumps(component, indent=2) if component else "[]"
    return f"""
    You are given:
    1. A list of previously extracted items from earlier sections of a technical document.
    2. A Markdown-formatted chunk of the document.

    Your task is to return a **single, updated list of extracted items** that:
    - **Keeps all previously extracted items** intact.
    - **Adds any new items** found in the current document chunk.
    - **Do not hallucinate** or infer values — only include items clearly present in the document.
    - **Does not remove or omit** previously found items, even if the current chunk contains no new data.
    - **Returning the exact previous list** unchanged if there are **no new valid items** in this chunk.
    - **Avoids duplicates**. Keep the more complete version if duplicates exist.
    - **Small variations in `mpn` or `top_marking`** (e.g., suffixes, added characters, etc.) **must be treated as unique items**.
    - For each item, include an optional `confidence` field with one of: `"high"`, `"medium"`, or `"low"`.
    - Use:
        - `"high"` when all fields are clearly present and unambiguous.
        - `"medium"` when some fields are inferred from context but not explicitly labeled.
        - `"low"` when any part of the item may be uncertain or unclear.

    For each item, return:

    - mpn: Manufacturer Part Number (Manufacturer Part Number, Type Number, or similar terms). This is like the **full name of a specific variant of a component**, often derived from a known base component {comp}.
    - top_marking: Short alphanumeric code on the component (Top Marking Code, Marking Code, or similar identifiers)
    - package_case: Standardized mechanical format (e.g., DO-214AB, SOD-123)
    - description: Functional description (e.g., "Transient Voltage Suppression Diode")

    Respond strictly in a valid JSON format, with no explanation or extra text:

    [
    {{
        "mpn": "...",
        "top_marking": "...",
        "package_case": "...",
        "description": "...",
        "confidence": "..."
    }},
    ...
    ]
    
    Previously Extracted Items:
    {prev}

    Document Chunk:
    {chunk}
    """

def save_items(items: List[dict]):
    df = pd.DataFrame(items)
    mode = "a" if CSV_OUTPUT.exists() else "w"
    header = not CSV_OUTPUT.exists()
    df.to_csv(CSV_OUTPUT, mode=mode, header=header, index=False)
    logger.info(f"Saved {len(items)} items to CSV")
