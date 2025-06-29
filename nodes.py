import json, logging
import re
import ast
from pathlib import Path
from helpers import chunk_markdown, generate_prompt, generate_anchor_prompt, generate_repair_prompt, generate_validation_prompt, save_items, save_validated_items, clean_markdown_text, extract_all_tables_with_optional_header, score_chunk
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import pandas as pd
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
from config import MARKDOWN_DIR, PROCESSED_DIR, SKIPPED_DIR, METADATA_DIR, FAILED_DIR, CSV_SKIPPED_OUTPUT, CSV_FAILED_OUTPUT

logger = logging.getLogger(__name__)

def load_and_split(state: Dict) -> Dict:
    """
    Node: Loads a PDF, converts it to markdown, and splits it into chunks.

    This is the entry point of the processing graph. It performs the following steps:
    1. Checks if the PDF has already been processed and skips if so.
    2. If no markdown file exists, it converts the PDF to markdown, saving referenced
       images and tables.
    3. It cleans the markdown and extracts only the tables and their headers.
    4. The table-focused markdown is then split into smaller, manageable chunks.
    5. Updates the state with the markdown content and the list of chunks.

    Args:
        state (Dict): The current state, must contain 'pdf_path' and 'converter'.

    Returns:
        Dict: The updated state with 'markdown' and 'chunks' added.
    """
    pdf = Path(state["pdf_path"])
    converter = state["converter"]

    MARKDOWN_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    processed_path = PROCESSED_DIR / pdf.name
    if processed_path.exists():
        logger.info(f"Skipping already processed: {pdf.name}")
        return state

    output_dir = MARKDOWN_DIR / pdf.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{pdf.stem}-with-image-refs.md"

    logger.info(f"Processing: {pdf.name}")

    try:
        if not md_path.exists():
            logger.info("Markdown not found. Converting PDF...")
            start_time = time.time()
            conv_res = converter.convert(pdf)

            doc_filename = pdf.stem

            table_counter = 0
            picture_counter = 0

            for element, _level in conv_res.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = output_dir / f"{doc_filename}-table-{table_counter}.png"
                    with element_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")

                if isinstance(element, PictureItem):
                    picture_counter += 1
                    element_image_filename = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                    with element_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")

            conv_res.document.save_as_markdown(md_path,image_mode=ImageRefMode.REFERENCED)
            logger.info(f"Markdown with image refs saved to: {md_path}")

            logger.info(f"Document converted and figures exported in {time.time() - start_time:.2f} seconds.")
        else:
            logger.info("Markdown already exists. Skipping conversion.")

        # Save picture text into a separate JSON file
        # logger.info("Extracting picture text...")
        # conv_doc = converter.convert(pdf).document
        # picture_texts = {}
        # for i, picture in enumerate(conv_doc.pictures, start=1):
        #     image_filename = f"{pdf.stem}-picture-{i}.png"
        #     lines = []
        #     for item, _ in conv_doc.iterate_items(root=picture, traverse_pictures=True):
        #         if isinstance(item, TextItem):
        #             txt = item.text.strip()
        #             if txt:
        #                 lines.append(txt)
        #     picture_texts[image_filename] = lines

        # ocr_text_path = output_dir / f"{pdf.stem}-picture-text.json"
        # ocr_text_path.write_text(json.dumps(picture_texts, indent=2), encoding="utf-8")
        # logger.info(f"Picture text saved to: {ocr_text_path}") 

        logger.info(f"Attempting to extract tables from {md_path.name}...")
        full_md_content = md_path.read_text(encoding="utf-8").strip()
        print(f"Full Markdown content length: {len(full_md_content)} characters")
        cleaned_md = clean_markdown_text(full_md_content)

        state["full_markdown_content"] = cleaned_md
        state["attempt_number"] = 1
        content_for_chunking = ""
        extracted_data = extract_all_tables_with_optional_header(cleaned_md)
        
        if extracted_data:
            logger.info(f"Attempt 1: Tables found. Using table-only content for first pass.")
            output_blocks = []
            for item in extracted_data:
                block_parts = []
                if item.get('header'):
                    block_parts.append(item['header'])
                block_parts.append(item['table'])
                output_blocks.append("\n\n".join(block_parts))
            content_for_chunking = "\n\n---\n\n".join(output_blocks)
            tables_only_md_path = output_dir / f"{pdf.stem}_tables.md"
            tables_only_md_path.write_text(content_for_chunking, encoding="utf-8")
            logger.info(f"Successfully saved extracted tables to: {tables_only_md_path.name}")
        else:
            logger.info("Attempt 1: No tables found. Using full document content for first pass.")
            content_for_chunking = cleaned_md

        print(f"Extracted {len(extracted_data)} tables from the cleaned Markdown content.")

        chunks = chunk_markdown(content_for_chunking)
        logger.info(f"Markdown split into {len(chunks)} chunk(s)")
        return {**state, "markdown": content_for_chunking, "chunks": chunks}

    except Exception as e:
        logger.error(f"Failed to process {pdf.name}: {e}")
        with open("failed_files.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now()}] {pdf.name} | Error: {e}\n")

        return {**state, "chunks": []}
    
def extract_anchor(state: Dict) -> Dict:
    """
    Node: Extracts the "anchor" component information from the start of the document.

    This node calls the LLM with a specialized prompt to get the main component name,
    description, and classification. It uses this information to decide whether to
    skip processing the document entirely (e.g., if it's a chip component or THT).

    Args:
        state (Dict): The current state, must contain 'markdown' and 'client_anchor'.

    Returns:
        Dict: The updated state with 'component', 'description', and potentially a 'skip_reason'.
    """
    client = state["client_anchor"]
    excerpt = "\n".join(state["markdown"].splitlines()[:100]).strip()

    prompt = generate_anchor_prompt(excerpt)

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an information extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = resp.choices[0].message.content.strip()

    match = re.search(r"```json\s*(.*?)```", raw, re.DOTALL)
    if match:
        json_to_parse = match.group(1).strip()
    else:
        json_to_parse = raw

    try:
        data = json.loads(json_to_parse)
        if isinstance(data, list) and len(data) > 0:
            item_data = data[0]
            component = item_data.get("component", [])
            description = item_data.get("description", "")
            is_chip = item_data.get("is_chip_component", False)
            is_tht = item_data.get("is_through_hole", False) 
            explanation = item_data.get("explanation")
            print(explanation)
        else:
            component, description, is_chip, is_tht = [], "", False, False
    except Exception as e:
        logger.warning(f"Anchor parse failed: {e}")
        component, description, is_chip, is_tht = [], "", False, False
    
    skip_reason = None
    if is_chip:
        skip_reason = "LLM classified it as a chip component."
    elif is_tht:
        skip_reason = "LLM classified it as a Through-Hole (THT) component."

    if skip_reason:
        logger.warning(f"Excluding document '{Path(state['pdf_path']).name}': {skip_reason}")
        return {**state, "component": component, "description": description, "skip_reason": skip_reason, "current_idx": 0, "chunks": []}
    
    return {**state, "component": component, "description": description, "explanation": explanation, "current_idx": 0, "items": []}

def filter_chunks(state: Dict) -> Dict:
    """
    Node: Filters and combines chunks to create a single, high-relevance input for the LLM.

    This optimization step scores all chunks, selects the top 4 highest-scoring chunks,
    and then combines them into a single, dense text block. This reduces the number
    of LLM calls and focuses the model on the most important parts of the document.

    Args:
        state (Dict): The current state, must contain 'chunks' and 'component'.

    Returns:
        Dict: The updated state with 'final_chunks' containing the combined top chunks.
    """
    chunks = state.get("chunks", [])
    raw_components = state.get("component", "[]")
    
    if isinstance(raw_components, str):
        raw_components = raw_components.strip()
        try:
            parsed = ast.literal_eval(raw_components)
            components = parsed if isinstance(parsed, list) else [str(parsed)]
        except (ValueError, SyntaxError):
            components = [raw_components.strip('"')]
    elif isinstance(raw_components, list):
        components = raw_components
    else:
        components = []

    # 1. Score each chunk and store it with its original index.
    indexed_scored_chunks = [
        (i, chunk, score_chunk(chunk, components))
        for i, chunk in enumerate(chunks)
    ]

    print("\n--- Chunk Scoring Details ---")
    if not indexed_scored_chunks:
        print("  No chunks to score.")
    else:
        for i, chunk, score in indexed_scored_chunks:
            print(f"  - Chunk {i+1:02d}/{len(chunks):02d} | Score: {score}")
    print("---------------------------\n")

    high_scoring_chunks = [
        (i, chunk, score) 
        for i, chunk, score in indexed_scored_chunks if score > 1
    ]
    
    # 2. Temporarily sort the high-scoring chunks by score to find the top 3.
    sorted_by_score = sorted(high_scoring_chunks, key=lambda x: x[2], reverse=True)

    # 3. Slice to get only the top 4 entries.
    top_entries = sorted_by_score[:4]

    # 4. Sort the top entries back by their original index.
    top_entries_in_original_order = sorted(top_entries, key=lambda x: x[0])

    # 5. Extract the text of the top chunks.
    final_top_chunks = [chunk for i, chunk, score in top_entries_in_original_order]

    # 6. Combine the top chunks into a single final_chunk ---
    if final_top_chunks:
        final_chunk = "\n\n---\n\n".join(final_top_chunks)
        logger.info(f"Combined the top {len(final_top_chunks)} chunks into a single chunk for processing.")
        chunks_for_llm = [final_chunk]
    else:
        logger.warning("No relevant chunks found after filtering. Nothing to process.")
        chunks_for_llm = []

    chunk_scores_data = [
        {"chunk_number": i + 1, "score": score, "chunk": chunk} 
        for i, chunk, score in indexed_scored_chunks
    ]

    return {**state, "final_chunks": chunks_for_llm, "chunk_scores": chunk_scores_data, "current_idx": 0, "items": []}

def call_llm(state: Dict) -> Dict:
    """
    Node: Calls the main language model for information extraction.

    This node takes the (filtered and combined) chunk, constructs the main extraction
    prompt, sends it to the LLM, and stores the raw string response in the state.

    Args:
        state (Dict): The current state, containing 'final_chunks', 'items', 'component', etc.

    Returns:
        Dict: The updated state with the 'raw_response' from the LLM.
    """
    chunk = state["final_chunks"][state["current_idx"]]
    prompt = generate_prompt(chunk, state["items"], state["component"])
    
    resp = state["client"].chat.completions.create(
        model=state["model_name"],
        messages=[
            {"role": "system", "content": "You are an information extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    print(resp.choices[0].message.content.strip())

    return {**state, "raw_response": resp.choices[0].message.content.strip()}

def parse_and_repair(state: Dict) -> Dict:
    """
    Node: Parses the LLM's response and attempts to repair it if it's invalid JSON.

    It first tries to parse the 'raw_response'. If parsing fails, it calls the LLM
    again with a 'repair' prompt. If the repaired response is valid JSON, it updates
    the state. Otherwise, it logs a warning.

    Args:
        state (Dict): The current state, must contain 'raw_response' and 'client'.

    Returns:
        Dict: The updated state with the parsed 'items'.
    """
    client = state["client"]
    raw = state["raw_response"]

    try:
        data = json.loads(raw)
        print(f"Parsed {len(data)} items from chunk {state['current_idx'] + 1}")
    except Exception:
        print("Initial JSON parsing failed. Attempting repair...")

        repair_prompt = generate_repair_prompt(raw)

        resp = client.chat.completions.create(
            model=state["model_name"],
            messages=[
                {"role": "system", "content": "You are a JSON repair assistant."},
                {"role": "user","content": repair_prompt}
            ],
            temperature=0
        )

        fixed_raw = resp.choices[0].message.content.strip()

        match = re.search(r"```json\s*(.*?)```", fixed_raw, re.DOTALL)
        if match:
            raw_json = match.group(1).strip()
        else:
            raw_json = fixed_raw

        try:
            data = json.loads(raw_json)
        except Exception as e:
            logger.warning(f"Final JSON parse after repair failed: {e}")
            data = []

    if isinstance(data, list) and data:
        state["items"] = data
    else:
        logger.info("No valid items recovered after repair.")

    return {**state, "current_idx": state["current_idx"] + 1}

def decide_what_to_do_next(state: Dict) -> str:
    """
    Node: A central router that decides the next step after an extraction attempt.
    - If items were extracted, proceed to validation.
    - If no items were found on the 1st attempt, trigger a retry using the full markdown.
    - If no items were found on the 2nd attempt, log a failure.
    """
    logger.info("--- Deciding next step ---")
    
    if state.get("items"):
        logger.info(f"SUCCESS: Found {len(state['items'])} items. Proceeding to validation.")
        state['next_action'] = 'validate'
        
        return state

    if state.get("attempt_number", 1) == 1:
        logger.warning("Attempt 1 yielded no items. Triggering a retry with full document markdown.")
        
        state["attempt_number"] = 2
        
        full_md = state.get("full_markdown_content", "")
        state["chunks"] = chunk_markdown(full_md)
        state["items"] = []
        
        if state["chunks"]:
            state['next_action'] = 'retry'   
        else:
            state['next_action'] = 'log_failure'
            
        return state 
    
    else:
        logger.error("FAILURE: Attempt 2 also yielded no items. Logging extraction failure.")
        state['next_action'] = 'log_failure'

        return state

def log_extraction_failure(state: Dict) -> Dict:
    """
    Node: Logs documents from which no items could be extracted after all attempts.
    Moves the source PDF to the FAILED_EXTRACTION_DIR.
    """
    source_path = Path(state["pdf_path"])
    logger.info(f"Logging complete extraction failure for: {source_path.name}")

    failure_info = {
        "source": source_path.name,
        "timestamp": datetime.now().isoformat(),
        "reason": "No structured items could be extracted after two attempts (table-only and full-text)."
    }
    df = pd.DataFrame([failure_info])
    
    mode = "a" if CSV_FAILED_OUTPUT.exists() else "w"
    header = not CSV_FAILED_OUTPUT.exists()
    df.to_csv(CSV_FAILED_OUTPUT, mode=mode, header=header, index=False)
    FAILED_DIR.mkdir(exist_ok=True)
    source_path.rename(FAILED_DIR / source_path.name)
    
    return state   

def validate_items(state: Dict) -> Dict:
    """
    Node: Validates, deduplicates, and enriches the final list of extracted items.

    This node performs a rule-based deduplication on the extracted items, grouping
    them by 'mpn' (Manufacturer Part Number) and merging their information.
    The commented-out code shows how an LLM could be used for a more advanced
    validation step to add confidence scores and comments.

    Args:
        state (Dict): The current state, must contain 'items'.

    Returns:
        Dict: The updated state with the 'validated_items' list.
    """
    # client = state["client"]
    items = state.get("items", [])

    grouped = {}
    for item in items:
        mpn = item["mpn"]
        grouped.setdefault(mpn, []).append(item)

    deduped = []
    for mpn, group in grouped.items():
        top_markings = set()
        package_cases = set()
        description = ""
        confidence = ""
        for entry in group:
            if entry.get("top_marking"):
                if isinstance(entry["top_marking"], list):
                    top_markings.update(entry["top_marking"])
                else:
                    top_markings.add(entry["top_marking"])
            if entry.get("package_case"):
                if isinstance(entry["package_case"], list):
                    package_cases.update(entry["package_case"])
                else:
                    package_cases.add(entry["package_case"])
            if not description and entry.get("description"):
                description = entry["description"]
            if not confidence and entry.get("confidence"):
                confidence = entry["confidence"]

        deduped.append({
            "mpn": mpn,
            "top_marking": ", ".join(sorted(set(top_markings))) if top_markings else None,
            "package_case": ", ".join(sorted(set(package_cases))) if package_cases else None,
            "description": description,
            "confidence": confidence,
            "validation_comment": ""
        })
    validated_items = deduped 
    # prompt = generate_validation_prompt(deduped)
    # resp = client.chat.completions.create(
    #     model=state["model_name"],
    #     messages=[
    #         {"role": "system", "content": "You are a critical reviewer of component extraction data."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0
    # )

    # raw = resp.choices[0].message.content.strip()
    # match = re.search(r"```json\s*(.*?)```", raw, re.DOTALL)
    # if match:
    #     raw_json = match.group(1).strip()
    # else:
    #     raw_json = raw
    # print("Validation Response:\n", raw)

    # try:
    #     validated_items = json.loads(raw_json)
    # except Exception as e:
    #     logger.warning(f"Validation parse failed: {e}")
    #     validated_items = deduped 
    return {**state, "validated_items": validated_items}

def finalize(state: Dict) -> Dict:
    """
    Node: Saves the final results and moves the processed file.

    This is a terminal node for a successful run. It adds the source filename
    to each extracted item, saves both the raw and validated items to CSV files,
    and moves the original PDF to the 'processed' directory to prevent reprocessing.

    Args:
        state (Dict): The final state containing 'items' and 'validated_items'.

    Returns:
        Dict: The final state.
    """
    for item in state["items"]:
        item["source"] = Path(state["pdf_path"]).name
        item["description"] = item.get("description") or state.get("description", "")
    
    for item in state["validated_items"]:
        item["source"] = Path(state["pdf_path"]).name
        item["description"] = item.get("description") or state.get("description", "")
    
    save_items(state["items"])
    save_validated_items(state["validated_items"])

    Path(state["pdf_path"]).rename(PROCESSED_DIR / Path(state["pdf_path"]).name)
    return state

def save_full_state(state: Dict) -> Dict:
    print("Save Full State STATE KEYS:", list(state.keys()))
    METADATA_DIR.mkdir(exist_ok=True)

    keys_to_save = ['title', 'model_name', 'component', 'description', 'chunks', 'final_chunks', 'current_idx']

    metadata = {
        k: json.dumps(state[k], ensure_ascii=False)
        if isinstance(state[k], (list, dict))
        else state[k]
        for k in keys_to_save if k in state
    }

    title = state.get("title").strip()
    json_path = METADATA_DIR / f"{title}_metadata.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata to {json_path}")
    
    return state

def save_skipped_component(state: Dict) -> Dict:
    """
    Node: Logs information about a skipped document and moves the file.

    This is a terminal node for a skipped run. It records the filename and the
    reason for skipping to a dedicated CSV log file. It then moves the PDF to
    the 'skipped' directory.

    Args:
        state (Dict): The state, must contain 'pdf_path' and 'skip_reason'.

    Returns:
        Dict: The final state for the skipped item.
    """
    skipped_info = {
        "source": Path(state["pdf_path"]).name,
        "component": ", ".join(state.get("component", [])),
        "description": state.get("description", ""),
        "explanation": state.get("explanation", ""),
        "reason": state.get("skip_reason", "")
    }
    df = pd.DataFrame([skipped_info])
    
    mode = "a" if CSV_SKIPPED_OUTPUT.exists() else "w"
    header = not CSV_SKIPPED_OUTPUT.exists()
    df.to_csv(CSV_SKIPPED_OUTPUT, mode=mode, header=header, index=False)
    
    logger.info(f"Logged skipped component from '{skipped_info['source']}' to {CSV_SKIPPED_OUTPUT}")
    SKIPPED_DIR.mkdir(exist_ok=True)
    Path(state["pdf_path"]).rename(SKIPPED_DIR / Path(state["pdf_path"]).name)

    return state