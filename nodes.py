import json, logging
import re
import ast
from pathlib import Path
from helpers import chunk_markdown, generate_prompt, generate_anchor_prompt, generate_repair_prompt, generate_validation_prompt, save_items, save_validated_items
from typing import Dict
from datetime import datetime
import time
import pandas as pd
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
from config import MARKDOWN_DIR, PROCESSED_DIR, SKIPPED_DIR, METADATA_DIR, CSV_SKIPPED_OUTPUT

logger = logging.getLogger(__name__)

def load_and_split(state: Dict) -> Dict:
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

        md = md_path.read_text(encoding="utf-8").strip()
        chunks = chunk_markdown(md)
        logger.info(f"Markdown split into {len(chunks)} chunk(s)")
        return {**state, "markdown": md, "chunks": chunks}

    except Exception as e:
        logger.error(f"Failed to process {pdf.name}: {e}")
        with open("failed_files.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now()}] {pdf.name} | Error: {e}\n")
        return state

def extract_anchor(state: Dict) -> Dict:
    print("Extract Anchor STATE KEYS:", list(state.keys()))
    client = state["client_anchor"]
    excerpt = "\n".join(state["markdown"].splitlines()[:100]).strip()

    prompt = generate_anchor_prompt(excerpt)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
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
    
    return {**state, "component": component, "description": description, "current_idx": 0, "items": []}

def call_llm(state: Dict) -> Dict:
    print("Call LLM STATE KEYS:", list(state.keys()))
    chunk = state["chunks"][state["current_idx"]]
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

        print("Repaired JSON:\n", raw_json)

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

def validate_items(state: Dict) -> Dict:
    print("Validate Items STATE KEYS:", list(state.keys()))
    client = state["client"]
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
    print("Finalize STATE KEYS:", list(state.keys()))
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

    keys_to_save = ['title', 'model_name', 'component', 'description', 'chunks', 'current_idx']

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
    print("Save Skipped Component STATE KEYS:", list(state.keys()))

    skipped_info = {
        "source": Path(state["pdf_path"]).name,
        "component": ", ".join(state.get("component", [])),
        "description": state.get("description", ""),
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

# def filter_chunks(state: Dict) -> Dict:
#     print("Filter Chunks STATE KEYS:", list(state.keys()))
#     raw_components = state.get("component", "[]")
#     if isinstance(raw_components, str):
#         raw_components = raw_components.strip()
#         try:
#             parsed = ast.literal_eval(raw_components)
#             components = parsed if isinstance(parsed, list) else [str(parsed)]
#         except Exception:
#             components = [raw_components.strip('"')]
#     elif isinstance(raw_components, list):
#         components = raw_components
#     else:
#         components = []
#     scored = [(chunk, score_chunk(chunk, components)) for chunk in state["chunks"]]
#     top_chunks = [chunk for chunk, score in sorted(scored, key=lambda x: -x[1]) if score > 1][:5]
#     title = state.get("title", Path(state["pdf_path"]).stem)
#     print(title.lower())
#     pattern = re.compile(
#     r"\b("
#     r"part( numbers?)?|"
#     r"type numbers?|"
#     r"ordering|"
#     r"markings?|"
#     r"package options?"
#     r")\b",
#     re.IGNORECASE
# )
#     cands = [c for c in state["chunks"]
#              if any(m.lower() in c.lower() for m in components)
#              or title.lower() in c.lower()
#              or pattern.search(c)]
#     logger.info(f"{len(top_chunks)} candidate chunks")
#     return {**state, "candidate_chunks": top_chunks, "current_idx": 0, "items": []}

# def save_candidates(state: Dict) -> Dict:
#     print("Save Candidates STATE KEYS:", list(state.keys()))
#     out_dir = Path("candidates")
#     out_dir.mkdir(exist_ok=True)
#     out_path = out_dir / f"{Path(state['pdf_path']).stem}_candidates.json"
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(state["chunks"], f, indent=2)
#     return state

# def score_chunk(chunk: str, components: list[str]) -> int:
#     score = 0
#     chunk_lower = chunk.lower()
#     # Score for MPN matches
#     score += sum(1 for m in components if m.lower() in chunk_lower)
#     # Score for important keywords
#     keywords = ["part number", "ordering", "marking", "package option", "overview", "description"]
#     score += sum(1 for kw in keywords if kw in chunk_lower)
#     return score