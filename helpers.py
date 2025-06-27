import re, json, logging
import pandas as pd
from typing import List
from typing import Dict, List, Any, Optional
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from config import CSV_OUTPUT, CSV_VALIDATED_OUTPUT

logger = logging.getLogger(__name__)

def setup_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.do_cell_matching = True
    opts.images_scale = 2.0
    opts.generate_page_images = True
    opts.generate_picture_images = True
    opts.ocr_options = EasyOcrOptions()
    # opts.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    opts.ocr_options.lang = ["en"]
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

def score_chunk(chunk: str, components: list[str]) -> int:
    """
    Scores a chunk based on a combination of component name matches and
    the presence of important keywords found via regex.
    """
    score = 0
    chunk_lower = chunk.lower()

    KEYWORD_PATTERN = re.compile(
    r"\b("
    r"part( numbers?)?|"
    r"type numbers?|"
    r"ordering|"
    r"markings?|"
    r"package options?"
    r")\b",
    re.IGNORECASE)

    if components:
        score += sum(1 for m in components if m.lower() in chunk_lower)

    keyword_matches = KEYWORD_PATTERN.findall(chunk)
    score += len(keyword_matches)
    
    return score

def clean_markdown_text(document_text: str) -> str:
    """
    Safely removes common non-content artifacts from the markdown text.
    This version is less aggressive to avoid damaging valid table structures.
    """
    # This pattern is safe. It only removes lines with 5 or more dots,
    # which is unique to dot-leader style Tables of Contents.
    toc_pattern = r'^.*\.{5,}.*\n?'
    cleaned_text = re.sub(toc_pattern, '', document_text, flags=re.MULTILINE)

    # Clean up any resulting excess blank lines
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    return cleaned_text.strip()

def extract_all_tables_with_optional_header(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Extracts all valid Markdown tables (header + separator + at least one data row).
    For each table, it includes the preceding '##' header and filters out ToC-like tables.
    """
    # Stricter regex: requires at least one data row by using '+'
    table_pattern_with_capture = re.compile(
        r'('
        r'^\s*\|.*\|\s*\n'      # Header row
        r'\s*\|[-|: ]+\|.*\n'   # Separator row
        r'(?:^\s*\|.*\|\s*\n?)+' # Body rows (one or more required)
        r')',
        re.MULTILINE
    )
    parts = table_pattern_with_capture.split(markdown_content)
    tables = parts[1::2]
    preceding_texts = parts[0::2]
    extracted_data: List[Dict[str, Any]] = []

    for i, table_str in enumerate(tables):
        if '.....' in table_str:
            continue

        preceding_text_block = preceding_texts[i].strip()
        header: Optional[str] = None
        if preceding_text_block:
            possible_headers = [
                line.strip() for line in preceding_text_block.splitlines()
                if line.strip().startswith('##')
            ]
            if possible_headers:
                header = possible_headers[-1]
        
        extracted_data.append({
            'header': header,
            'table': table_str.strip()
        })
    return extracted_data

def chunk_markdown(md: str, max_words: int = 1000) -> List[str]:
    print(f"Total words: {len(md.split())}")
    sections = re.split(r"\n(?=##\s)", md)
    chunks, buf = [], ""
    
    for sec in sections:
        sec = sec.strip()
        buf_words = len(buf.split())
        sec_words = len(sec.split())
        
        if buf_words + sec_words < max_words:
            buf += sec + "\n"
        else:
            chunks.append(buf.strip())
            buf = sec + "\n"
    
    if buf.strip():
        chunks.append(buf.strip())
    print(f"Number of words in chunks: {[len(chunk.split()) for chunk in chunks]}")
    return chunks

def generate_anchor_prompt(excerpt: str) -> str:
    return f"""
    You are given the beginning of a Markdown-formatted technical document for an electronic component.

    From the given text, perform two tasks:

    1.  **Extraction**: Extract the following information:
        - The **main component name(s)** (e.g., MMBT3906). If a range is shown (e.g., `BZX84C2V4W - BZX84C39W`), extract the **start and end MPNs**.
        - A **short technical description** of the component (e.g., "40 V, 200 mA PNP switching transistor").

    2.  **Classification**: Set the following boolean flags based on the component type.
        - `is_chip_component`: Set to **true** ONLY if the component is explicitly described as a **resistor, capacitor (MLCC), inductor, or ferrite bead**. If the type is anything else or is not clearly mentioned, you MUST set this to **false**.
        - `is_through_hole`: Set to **true** ONLY if the excerpt explicitly mentions that the component is THT (Through Hole Technology). If the text describes the component as SMD/SMT (Surface Mount Technology / Device)), or does not specify the type, set this to **false**.

    Respond **strictly** in the correct JSON format, including the boolean classification:

    [
      {{
        "component": ["StartMPN", "EndMPN"],
        "description": "Short description of the component",
        "is_chip_component": boolean,
        "is_through_hole": boolean
      }}
    ]

    Markdown excerpt:
    {excerpt}
    """

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
    - A single component's information may be split across multiple tables (chunks). For example, the `mpn` might be in one table, while its `top_marking` is in another. If you find new information for an `mpn` that already exists in the "Previously Extracted Items" list, **you must update the existing item** with the new information instead of creating a new, separate entry.
    - **Small variations in `mpn` or `top_marking`** (e.g., suffixes, added characters, etc., SN74LVC1G17DBVR is different from SN74LVC1G17DBVR.Z) **must be treated as unique items**.
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

    Respond **only** with a JSON array of items. Do **not** include any explanation, thought process, or markdown formatting.

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

    STRICT INSTRUCTION: Return **only** a valid JSON list (i.e., starting with `[` and ending with `]`) and **nothing else**.
    """

def generate_repair_prompt(raw: str) -> str:
    return f"""
    The following JSON array is invalid, incomplete, or malformed.

    Your task is to:
    - Fix any syntax issues (e.g., unclosed braces, trailing commas, incorrect quotes).
    - Ensure it follows **exactly** this format:

    [
    {{
        "mpn": "...",
        "top_marking": "...",
        "package_case": "...",
        "description": "...",
        "confidence": "...", 
        "validation_comment": "..."
    }},
    ...
    ]

    Only return the repaired JSON array. Do not include any other text or explanation.

    Fix this JSON:

    {raw}
    """

def generate_validation_prompt(items: list) -> str:
    return f"""
You are given a list of extracted items from a technical document.

Your task is to:
1. Validate each item as-is.
2. Update the `confidence` field (`high`, `medium`, or `low`) depending on the consistency and plausibility of the fields.
3. Add a `validation_comment`:
   - Briefly explain why confidence is not high (e.g., missing `top_marking`, inconsistent `package_case`, suspicious values).
   - Leave it blank if the data is strong.

Respond strictly in the following JSON format — no explanations, markdown code blocks, or extra text. The response **must** start with `[` and end with `]`.

[
  {{
    "mpn": "...",
    "top_marking": "...",  // or null
    "package_case": "...", // or null
    "description": "...",
    "confidence": "...",   // high, medium, or low
    "validation_comment": "Explain confidence or highlight issues in 1 sentence if there are, leave blank if confident."
  }},
  ...
]

Extracted Items:
{json.dumps(items, indent=2)}
"""

def save_items(items: List[dict]):
    df = pd.DataFrame(items)
    mode = "a" if CSV_OUTPUT.exists() else "w"
    header = not CSV_OUTPUT.exists()
    df.to_csv(CSV_OUTPUT, mode=mode, header=header, index=False)
    logger.info(f"Saved {len(items)} items to CSV")

def save_validated_items(items: List[dict]):
    df = pd.DataFrame(items)
    mode = "a" if CSV_VALIDATED_OUTPUT.exists() else "w"
    header = not CSV_VALIDATED_OUTPUT.exists()
    df.to_csv(CSV_VALIDATED_OUTPUT, mode=mode, header=header, index=False)
    logger.info(f"Saved {len(items)} validated items to CSV")
