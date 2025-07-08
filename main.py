import concurrent.futures
import csv
import logging
import os
import shutil
import ssl
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from groq import Groq
from openai import OpenAI
from PyPDF2 import PdfReader

from config import (ANCHOR_MODEL_NAME, DOCUMENTS_DIR, FAILURE_LOG_PATH,
                    MODEL_NAME, PROCESSED_DIR, SKIPPED_LARGE_FILES_DIR)
from graph_builder import build_graph
from helpers import log_failure, setup_converter

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

MAX_WORKERS = os.cpu_count() or 4

def process_single_pdf(pdf_path: Path):
    """
    This function encapsulates the entire workflow for processing a single PDF file.
    It will be executed by each worker process in the pool.
    """
    if (PROCESSED_DIR / pdf_path.name).exists():
        logging.info(f"Skipping already processed file: {pdf_path.name}")
        return

    logging.info(f"Starting processing for: {pdf_path.name}")

    try:
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
        if page_count > 500:
            reason = f"File has {page_count} pages, exceeding the limit of 500."
            logging.warning(f"SKIPPING '{pdf_path.name}': {reason}")
            SKIPPED_LARGE_FILES_DIR.mkdir(exist_ok=True)
            shutil.move(pdf_path, SKIPPED_LARGE_FILES_DIR / pdf_path.name)
            logging.info(f"Moved large file to {SKIPPED_LARGE_FILES_DIR}")
            return
    except Exception as e:
        logging.error(f"Failed to read {pdf_path.name}: {e}")
        log_failure(pdf_path, e)

    try:
        client = Groq()
        client_anchor = OpenAI()
        converter = setup_converter()
        app = build_graph()

        state = {
            "pdf_path": str(pdf_path),
            "title": pdf_path.stem,
            "client": client,
            "client_anchor": client_anchor,
            "converter": converter,
            "model_name": MODEL_NAME,
            "anchor_model_name": ANCHOR_MODEL_NAME
        }

        app.invoke(state, {"recursion_limit": 100})
        logging.info(f"Successfully finished processing: {pdf_path.name}")

    except Exception as e:
        logging.error(f"A critical error occurred while processing {pdf_path.name}: {e}")
        log_failure(pdf_path, e)


def main():
    """
    Main function to manage the batch processing of PDF documents.
    It finds all PDFs and distributes them to a pool of worker processes.
    """
    load_dotenv(find_dotenv())

    all_pdfs = list(Path(DOCUMENTS_DIR).glob("*.pdf"))
    logging.info(f"Found {len(all_pdfs)} total PDFs in the documents directory.")

    if not all_pdfs:
        logging.info("No PDFs to process. Exiting.")
        return

    file_limit = 700
    files_to_process = all_pdfs[:file_limit]
    logging.info(f"Processing up to {len(files_to_process)} files due to limit of {file_limit}.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        logging.info(f"Starting PDF processing with {MAX_WORKERS} parallel workers...")

        future_to_pdf = {executor.submit(process_single_pdf, pdf): pdf for pdf in files_to_process}

        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f'{pdf.name} generated an exception in the worker: {exc}')

    logging.info("All PDF processing tasks have been completed.")


if __name__ == "__main__":
    if not FAILURE_LOG_PATH.exists():
        with open(FAILURE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "pdf_name", "error"])
    main()
