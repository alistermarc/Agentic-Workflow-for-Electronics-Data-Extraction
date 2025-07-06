import csv
import logging
import shutil 
import ssl
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from groq import Groq
from openai import OpenAI
from PyPDF2 import PdfReader

from config import DOCUMENTS_DIR, PROCESSED_DIR, MODEL_NAME, ANCHOR_MODEL_NAME, SKIPPED_LARGE_FILES_DIR, FAILURE_LOG_PATH
from graph_builder import build_graph
from helpers import setup_converter, log_failure

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

def main():
    """Main function to process PDF documents through the graph."""
    load_dotenv(find_dotenv())
    client = Groq()
    client_anchor = OpenAI()
    converter = setup_converter()
    PROCESSED_DIR.mkdir(exist_ok=True)
    app = build_graph()

    file_limit = 700
    processed_count = 0

    for pdf in Path(DOCUMENTS_DIR).glob("*.pdf"):
        logging.info(f"Processing file: {pdf.name}")
        if processed_count >= file_limit:
            logging.info(f"Reached the limit of {file_limit} files to process. Stopping.")
            break
        if (PROCESSED_DIR / pdf.name).exists():
            continue
        try: 
            reader = PdfReader(pdf)
            page_count = len(reader.pages)
            if page_count > 300:
                reason = f"File has {page_count} pages, exceeding the limit of 300."
                logging.warning(f"SKIPPING '{pdf.name}': {reason}")
                SKIPPED_LARGE_FILES_DIR.mkdir(exist_ok=True)
                shutil.move(pdf, SKIPPED_LARGE_FILES_DIR / pdf.name)
                logging.info(f"Moved large file to {SKIPPED_LARGE_FILES_DIR}")
                continue
        except Exception as e:
            logging.error(f"Failed to read {pdf.name}: {e}")
            log_failure(pdf, e)
        state = {
            "pdf_path": str(pdf),
            "title": pdf.stem,
            "client": client,
            "client_anchor": client_anchor,
            "converter": converter,
            "model_name": MODEL_NAME,
            "anchor_model_name": ANCHOR_MODEL_NAME
        }
        try:
            app.invoke(state, {"recursion_limit": 100})
        except Exception as e:
            logging.error(f"Failed processing {pdf.name}: {e}")
            log_failure(pdf, e)
        processed_count += 1

if __name__ == "__main__":
    if not FAILURE_LOG_PATH.exists():
        with open(FAILURE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "pdf_name", "error"])
    main()
