from pathlib import Path

# --- Directories ---
DOCUMENTS_DIR = Path("documents")
MARKDOWN_DIR = Path("markdown")
PROCESSED_DIR = Path("processed")
SKIPPED_DIR = Path("skipped")
FAILED_DIR = Path("failed")
SKIPPED_LARGE_FILES_DIR = Path("skipped_large_files")
METADATA_DIR = Path("metadata")

# --- CSV Outputs ---
CSV_OUTPUT = Path("extracted_items.csv")
CSV_VALIDATED_OUTPUT = Path("extracted_validated_items.csv")
CSV_SKIPPED_OUTPUT = Path("skipped_components.csv")
CSV_FAILED_OUTPUT = Path("failed_extractions.csv")
FAILURE_LOG_PATH = Path("failed_pdfs.csv")

# --- Model Configuration ---ç
MODEL_NAME = "llama-3.3-70b-versatile"
ANCHOR_MODEL_NAME = "gpt-4o"
# MODEL_NAME = "deepseek-r1-distill-llama-70b"
# MODEL_NAME = "qwen-qwq-32b"
# MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
# MODEL_NAME = "gemma2-9b-it"
# MODEL_NAME = "llama-3.1-8b-instant"
