import json, logging
from pathlib import Path
from config import PROCESSED_DIR
from helpers import chunk_markdown, generate_prompt, save_items
from typing import Dict
from datetime import datetime
import time
import pandas as pd
from config import MARKDOWN_DIR, PROCESSED_DIR, METADATA_DIR

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

    md_path = MARKDOWN_DIR / pdf.with_suffix(".md").name
    logger.info(f"Processing: {pdf.name}")

    try:
        if not md_path.exists():
            logger.info("Markdown not found. Converting PDF...")
            start = time.time()
            conv_result = converter.convert(pdf)
            logger.info(f"Converted in {time.time() - start:.2f} seconds")
            md_content = conv_result.document.export_to_markdown()
            md_path.write_text(md_content, encoding="utf-8")
            logger.info(f"Markdown saved to: {md_path}")
        else:
            logger.info("Markdown already exists. Skipping conversion.")

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
    client = state["client"]
    excerpt = "\n".join(state["markdown"].splitlines()[:100]).strip()

    prompt = f"""
You are given the beginning of a Markdown-formatted technical document for an electronic component.

From the given text, extract:

- The **main component name(s)** (e.g., MMBT3906). If a **range** of part numbers is shown (e.g., `BZX84C2V4W - BZX84C39W`), extract the **start and end MPNs** as a list under the `"component"` field.
- A **short technical description** of the component (e.g., "40 V, 200 mA PNP switching transistor").

Ignore:
- Tables of part number variants or packages.
- Ordering info, dimensions, and electrical specs unless directly part of the short description.

Respond **strictly** in the correct JSON format:

[
  {{
    "component": ["StartMPN", "EndMPN"],  // or a single string if not a range
    "description": "Short description of the component"
  }}
]

Markdown excerpt:
{excerpt}
"""

    resp = client.chat.completions.create(
        model=state["model_name"],
        messages=[
            {"role": "system", "content": "You are an information extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = resp.choices[0].message.content.strip()
    print(raw)

    try:
        data = json.loads(raw)
        if isinstance(data, list) and len(data) > 0:
            component = data[0].get("component", [])
            description = data[0].get("description", "")
        else:
            component, description = [], ""
    except Exception as e:
        logger.warning(f"Anchor parse failed: {e}")
        component, description = [], ""

    return {**state, "component": component, "description": description, "current_idx": 0, "items": []}

# def filter_chunks(state: Dict) -> Dict:
#     print("Filter Chunks STATE KEYS:", list(state.keys()))
#     components = state.get("component", [])
#     scored = [(chunk, score_chunk(chunk, components)) for chunk in state["chunks"]]
#     top_chunks = [chunk for chunk, score in sorted(scored, key=lambda x: -x[1]) if score > 1][:5]
#     title = state.get("title", Path(state["pdf_path"]).stem)
#     print(title.lower())
#     # cands = [c for c in state["chunks"]
#     #          if any(m.lower() in c.lower() for m in components)
#     #          or title.lower() in c.lower()
#     #          or any(k in c.lower() for k in ("part number", "ordering", "marking", "package option"))]
#     logger.info(f"{len(top_chunks)} candidate chunks")
#     return {**state, "candidate_chunks": top_chunks, "current_idx": 0, "items": []}

def save_candidates(state: Dict) -> Dict:
    print("Save Candidates STATE KEYS:", list(state.keys()))
    out_dir = Path("candidates")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(state['pdf_path']).stem}_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state["chunks"], f, indent=2)
    return state

def call_llm(state: Dict) -> Dict:
    print("Call LLM STATE KEYS:", list(state.keys()))
    # chunk = state["candidate_chunks"][state["current_idx"]]
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
    except:
        print(raw)
        fix = client.chat.completions.create(
            model=state["model_name"],
            messages=[
                {"role": "system", "content": "You are a JSON repair assistant."},
                {"role": "user", "content": f"Fix this JSON array:\n{raw}"}
            ],
            temperature=0
        ).choices[0].message.content.strip()
        print(fix)
        data = json.loads(fix)
    if isinstance(data, list) and data:
        state["items"] = data
    else:
        logger.info("No new items; keeping previous")
    return {**state, "current_idx": state["current_idx"] + 1}

# def score_chunk(chunk: str, components: list[str]) -> int:
#     score = 0
#     chunk_lower = chunk.lower()
#     # Score for MPN matches
#     score += sum(1 for m in components if m.lower() in chunk_lower)
#     # Score for important keywords
#     keywords = ["part number", "ordering", "marking", "package option", "overview", "description"]
#     score += sum(1 for kw in keywords if kw in chunk_lower)
#     return score

def finalize(state: Dict) -> Dict:
    print("Finalize STATE KEYS:", list(state.keys()))
    for item in state["items"]:
        item["source"] = Path(state["pdf_path"]).name
        item["description"] = item.get("description") or state.get("description", "")
    save_items(state["items"])
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