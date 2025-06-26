import re, json, logging
import pandas as pd
from typing import List
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
