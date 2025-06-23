import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import logging
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from config import DOCUMENTS_DIR, PROCESSED_DIR, MODEL_NAME
from helpers import setup_converter
from graph_builder import build_graph
from datetime import datetime
import csv
from groq import Groq

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

FAILURE_LOG_PATH = Path("failed_pdfs.csv")

def log_failure(pdf_path: Path, error: Exception):
    FAILURE_LOG_PATH.parent.mkdir(exist_ok=True)
    with open(FAILURE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), pdf_path.name, str(error)])

def main():
    load_dotenv(find_dotenv())
    client = Groq()
    converter = setup_converter()
    PROCESSED_DIR.mkdir(exist_ok=True)
    app = build_graph()

    for pdf in Path(DOCUMENTS_DIR).glob("*.pdf"):
        if (PROCESSED_DIR / pdf.name).exists():
            continue
        state = {
            "pdf_path": str(pdf),
            "title": pdf.stem,
            "client": client,
            "converter": converter,
            "model_name": MODEL_NAME
        }
        try:
            app.invoke(state, {"recursion_limit": 100})
        except Exception as e:
            logging.error(f"Failed processing {pdf.name}: {e}")
            log_failure(pdf, e)

if __name__ == "__main__":
    # Initialize failure log with headers if not exists
    if not FAILURE_LOG_PATH.exists():
        with open(FAILURE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "pdf_name", "error"])

    main()
