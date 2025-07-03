import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, EasyOcrOptions, PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from config import CSV_OUTPUT, CSV_VALIDATED_OUTPUT, FAILURE_LOG_PATH

logger = logging.getLogger(__name__)

def setup_converter() -> DocumentConverter:
    """
    Initializes and configures the DocumentConverter for processing PDFs.

    This function sets up the pipeline with specific options, including enabling
    Optical Character Recognition (OCR) and table structure analysis. It is
    configured to use EasyOCR by default.

    Returns:
        DocumentConverter: A configured instance of the DocumentConverter.
    """
    opts = PdfPipelineOptions()
    opts.do_ocr = True  # Enable OCR to extract text from images.
    opts.do_table_structure = True  # Enable table structure analysis.
    opts.table_structure_options.do_cell_matching = True    # Enable cell matching for better table structure recognition.
    opts.images_scale = 2.0  # Scale images for better OCR accuracy.
    opts.generate_page_images = True    # Generate images for each page in the PDF.
    opts.generate_picture_images = True   # Generate images for pictures in the PDF.
    opts.ocr_options = EasyOcrOptions()   # Use EasyOCR for text extraction from images.
    # opts.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS, num_threads=4)
    opts.ocr_options.lang = ["en"]  # Set the OCR language to English.

    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

def score_chunk(chunk: str, components: list[str], title: str) -> int:
    """
    Scores a text chunk based on the presence of relevant keywords and component names.

    This function is used to identify the most relevant parts of a document for extraction.
    It assigns a score by counting occurrences of predefined keywords (e.g., "part number",
    "package") and the component names identified in the anchor extraction step.

    Args:
        chunk (str): The text chunk to score.
        components (list[str]): A list of component names to search for.

    Returns:
        int: The calculated score for the chunk.
    """
    score = 0
    chunk_lower = chunk.lower()

    # A regex pattern to find keywords related to component specifications.
    KEYWORD_PATTERN = re.compile(
    r"\b("
    r"part( numbers?)?|"
    r"type numbers?|"
    r"ordering|"
    r"markings?|"
    r"package options?|"
    r"product series|"
    r"packages?"
    r")\b",
    re.IGNORECASE
    )
    if len(title) >= 5:
        title_prefix = title[:5].lower()
        score += chunk_lower.count(title_prefix)*0.5
    if components:
        score += sum(1 for m in components if m.lower() in chunk_lower)
    keyword_matches = KEYWORD_PATTERN.findall(chunk)
    score += len(keyword_matches)
    
    return score

def clean_markdown_text(document_text: str) -> str:
    """
    Cleans raw markdown text by removing unwanted artifacts.

    Specifically, this function removes lines that look like a table of contents
    (e.g., "Section 1 .......... 5") and normalizes multiple newlines into a
    standard double newline.

    Args:
        document_text (str): The raw markdown content.

    Returns:
        str: The cleaned markdown content.
    """
    toc_pattern = r'^.*\.{5,}.*\n?'
    cleaned_text = re.sub(toc_pattern, '', document_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)

    return cleaned_text.strip()

def extract_all_tables_with_optional_header(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Extracts all markdown tables and their immediate preceding '##' header.

    This function uses a regular expression to find all markdown-formatted tables.
    It also captures the text immediately before each table to find a potential
    header, which is assumed to be the last line starting with '##'.

    Args:
        markdown_content (str): The markdown text to search through.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               contains a 'table' and an optional 'header'.
    """
    table_pattern_with_capture = re.compile(
        r'('
        r'^\s*\|.*\|\s*\n'      
        r'\s*\|[-|: ]+\|.*\n'   
        r'(?:^\s*\|.*\|\s*\n?)+' 
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
    """
    Splits a large markdown document into smaller chunks based on a word limit.

    This function attempts to keep markdown sections (starting with '##')
    together. It iterates through sections, adding them to a buffer until the
    word count exceeds the maximum, at which point it creates a new chunk.

    Args:
        md (str): The markdown content to be chunked.
        max_words (int): The maximum number of words allowed per chunk.

    Returns:
        List[str]: A list of markdown text chunks.
    """
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
        - A **short technical description** of the component (e.g., "40 V, 200 mA PNP switching transistor"). If not found, leave this blank.
        - The **package case** or type, if available (e.g., SOT-23, DO-214AB, QFN). If not found, leave this blank.

    2.  **Classification**: Set the following boolean flag based on the component type.
        - `is_chip_component`: Set to **true** ONLY if the component is explicitly described as a **resistor, capacitor (MLCC), inductor, or ferrite bead**. If the type is anything else or is not clearly mentioned, you MUST set this to **false**.

    3.  **Justification**: If you set `is_chip_component` to `true`, you MUST add an `explanation` field briefly stating the reason (e.g., "Component is described as a chip resistor").

    Respond **strictly** in the correct JSON format, including the boolean classification:

    [
      {{
        "component": ["StartMPN", "EndMPN"],
        "description": "Short description of the component",
        "package_case": "Package type if available(e.g., SOT-23, DO-214AB)",
        "is_chip_component": boolean,
        "explanation": "Brief reason if is_chip_component or is_through_hole is true."
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
    - **CRUCIAL RULE: You MUST treat small variations in a part number (`mpn`) or `top_marking` as completely separate and unique items.** For example, if you find "TPS6285010MQDRLRQ1" and "TPS6285010MQDRLRQ1.A", they are two different items and you must include both. Extract every distinct part number you can find may it be active or obsolete.
    - **Capture the FULL Part Number:** The `mpn` value **MUST** be the complete, orderable part number exactly as it appears in the text. This includes all suffixes, prefixes, spaces, and special characters (`+`, `-`, `/`, etc.).
    - Do not hallucinate or infer values — only include items clearly present in the document.
    - **Avoids duplicates**. Keep the more complete version if duplicates exist.
    - A single component's information may be split across multiple tables (chunks). For example, the `mpn` might be in one table, while its `top_marking` is in another. 
    - **Small variations in `mpn` or `top_marking`** (e.g., suffixes, added characters, etc., SN74LVC1G17DBVR is different from SN74LVC1G17DBVR.Z) **must be treated as unique items**.
    - For each item, include an optional `confidence` field with one of: `"high"`, `"medium"`, or `"low"`.
    - Use:
        - `"high"` when all fields are clearly present and unambiguous.
        - `"medium"` when some fields are inferred from context but not explicitly labeled.
        - `"low"` when any part of the item may be uncertain or unclear.

    For each item, return:

    - mpn: Manufacturer Part Number (Manufacturer Part Number, Type Number, or similar terms). This field is mandatory and must not be null.
    - top_marking: Short alphanumeric code on the component (Device Marking, Top Marking Code, Marking Code, or similar identifiers). If not found, leave this blank.
    - package_case: Standardized mechanical format (e.g., DO-214AB, SOD-123). If not found, leave this blank.
    - description: Functional description (e.g., "Transient Voltage Suppression Diode"). If not found, leave this blank.

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
    }},
    ...
    ]

    Only return the repaired JSON array. Do not include any other text or explanation.

    Fix this JSON:

    {raw}
    """

# def generate_validation_prompt(items: list) -> str:
#     return f"""
#     You are given a list of extracted items from a technical document.

#     Your task is to:
#     1. Validate each item as-is.
#     2. Update the `confidence` field (`high`, `medium`, or `low`) depending on the consistency and plausibility of the fields.
#     3. Add a `validation_comment`:
#     - Briefly explain why confidence is not high (e.g., missing `top_marking`, inconsistent `package_case`, suspicious values).
#     - Leave it blank if the data is strong.

#     Respond strictly in the following JSON format — no explanations, markdown code blocks, or extra text. The response **must** start with `[` and end with `]`.

#     [
#     {{
#         "mpn": "...",
#         "top_marking": "...",  // or null
#         "package_case": "...", // or null
#         "description": "...",
#         "confidence": "...",   // high, medium, or low
#         "validation_comment": "Explain confidence or highlight issues in 1 sentence if there are, leave blank if confident."
#     }},
#     ...
#     ]

#     Extracted Items:
#     {json.dumps(items, indent=2)}
#     """

def log_failure(pdf_path: Path, error: Exception): # <-- Add this entire function
    """Logs a PDF processing failure to a CSV file."""
    FAILURE_LOG_PATH.parent.mkdir(exist_ok=True)
    with open(FAILURE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), pdf_path.name, str(error)])

def save_items(items: List[dict]):
    """
    Saves a list of extracted items to a CSV file.

    Appends to the file if it exists, otherwise creates a new file with a header.

    Args:
        items (List[dict]): The list of dictionaries to save.
    """
    df = pd.DataFrame(items)
    mode = "a" if CSV_OUTPUT.exists() else "w"
    header = not CSV_OUTPUT.exists()
    df.to_csv(CSV_OUTPUT, mode=mode, header=header, index=False)
    logger.info(f"Saved {len(items)} items to CSV")

def save_validated_items(items: List[dict]):
    """
    Saves a list of validated items to a separate CSV file.

    Appends to the file if it exists, otherwise creates a new file with a header.

    Args:
        items (List[dict]): The list of validated dictionaries to save.
    """
    df = pd.DataFrame(items)
    mode = "a" if CSV_VALIDATED_OUTPUT.exists() else "w"
    header = not CSV_VALIDATED_OUTPUT.exists()
    df.to_csv(CSV_VALIDATED_OUTPUT, mode=mode, header=header, index=False)
    logger.info(f"Saved {len(items)} validated items to CSV")
