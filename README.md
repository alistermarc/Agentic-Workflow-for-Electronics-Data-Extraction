# 🧠 Automated Workflow for Electronics Data Extraction

This project implements a robust agentic workflow for extracting structured information (e.g., MPNs, descriptions, packaging, markings) from unstructured electronics datasheets (PDFs). It uses a combination of OCR, markdown conversion, chunking, and LLM-driven parsing in a graph-based pipeline.

---

## 📁 Project Structure

```
.
👉 documents/                  # Input PDF datasheets
👉 failed/                     # PDFs where extraction failed after retries
👉 markdown/                   # Extracted markdown and images from each PDF
👉 metadata/                   # Per-document metadata (JSON) for debugging
👉 processed/                  # Successfully processed PDFs
👉 skipped/                    # PDFs skipped by the initial classification (e.g., chip components)
👉 skipped_large_files/        # PDFs skipped for exceeding the page count limit
👉 config.py                   # Path and model configuration
👉 graph_builder.py            # Builds the conditional LangGraph pipeline
👉 helpers.py                  # Utilities: OCR setup, chunking, prompt generation, saving
👉 main.py                     # Main entry point for processing PDFs
👉 nodes.py                    # Modular graph nodes (load, extract, validate, save, etc.)
📄 README.md                   # This documentation file
👉 requirements.txt            # Required Python dependencies
📄 extracted_items.csv         # Raw, unprocessed LLM extractions
📄 extracted_validated_items.csv # ✅ Final, deduplicated, and validated structured data
📄 failed_extractions.csv      # ❌ Log of PDFs where extraction failed (e.g no structured items could be extracted from the document)
📄 failed_pdfs.csv             # ❌ Log of PDFs that failed during initial reading or conversion
📄 skipped_components.csv      # 🟡 Log of PDFs that were intentionally skipped due to large size

```

---

---
## ⚙️ How It Works

1.  **Input**: Place any PDF datasheet in the `documents/` folder.
2.  **Conversion & Pre-filtering**: Each PDF is converted into markdown via OCR. Very large files are moved to `skipped_large_files/`.
3.  **Anchor Extraction & Classification**: The start of the document is analyzed to extract the main component name and to classify it. Non-target components (e.g., chip components) are moved to the `skipped/` directory.
4.  **Table-First Extraction**: The agent first attempts to extract data **only from tables** found in the document, as this is the most reliable source.
5.  **Intelligent Filtering**: Text chunks are scored based on relevance. Only the highest-scoring chunks are sent to the LLM to optimize for quality and efficiency.
6.  **Full-Text Retry**: If the table-first pass yields no results, the agent automatically performs a **second attempt**, using the full text of the document.
7.  **Deduplication & Validation**: Extracted items are deduplicated by MPN and merged to create a clean, final list.
8.  **Finalization**: Validated items are stored in `extracted_validated_items.csv`, and the original PDF is moved to `processed/`.

---

## 🧱 Components

### `main.py`
The entry point that loads the environment, initializes the graph and helper functions, iterates over unprocessed PDFs, and invokes the LangGraph agent on each document.

### `graph_builder.py`
Constructs a **conditional LangGraph** that directs the flow of data based on the outcome of each step. The graph is not a simple linear chain but a state machine with branches for:
* **Skipping** a document based on its classification.
* **Retrying** extraction with different content if the first pass fails.
* **Logging failures** if all attempts are exhausted.
* Proceeding to **validation** and finalization upon successful extraction.

### `nodes.py`
Implements the logic for each graph node:

| Function | Description |
| :--- | :--- |
| `load_and_split()` | Converts PDF to markdown, extracts tables, and creates initial chunks. |
| `extract_anchor()` | Gets the main component name and classifies it to decide if the doc should be skipped. |
| `filter_chunks()` | Scores text chunks and combines the most relevant ones for the LLM. |
| `decide_what_to_do_next()` | Routes the workflow to retry, validate, or log a failure based on extraction results. |
| `call_llm()` | Extracts MPN data using an LLM and a custom prompt. |
| `parse_and_repair()` | Parses or repairs broken JSON received from the LLM. |
| `validate_items()` | Deduplicates and merges extracted items based on the MPN. |
| `finalize()` | Adds metadata and writes validated items to the final CSV. |
| `save_skipped_component()` | Logs and moves files that were intentionally skipped. |
| `log_extraction_failure()` | Logs and moves files where all extraction attempts failed. |
| `save_full_state()` | Dumps intermediate state to `metadata/` for inspection. |

---
## 🔧 Setup & Usage

1.  **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Add your `.env`**
    ```
    GROQ_API_KEY=your_groq_key_here
    OPENAI_API_KEY=your_openai_key_here
    ```
3.  **Add PDFs**
    Place your datasheets in the `documents/` folder.
4.  **Run the pipeline**
    ```bash
    python main.py
    ```
5.  **View Results**
    * ✅ **Final results** in `extracted_validated_items.csv`.
    * 🟡 **Skipped components** logged in `skipped_components.csv`.
    * ❌ **Failed jobs** logged in `failed_pdfs.csv` and `failed_extractions.csv`.
    * 📄 Markdown + image references in `markdown/`.
    * 🧠 Intermediate metadata for each document in `metadata/`.

---
## 📦 Final Output Explained

The final, validated data is stored in `extracted_validated_items.csv`. The `finalize` node in the graph enriches the data with additional context before saving.

### Fields in `extracted_validated_items.csv`

| Field | Origin | Description |
| :--- | :--- | :--- |
| **`mpn`** | LLM | The core **Manufacturer Part Number** extracted from the document text. |
| **`top_marking`** | LLM | The **marking code** found on the component, extracted by the LLM. |
| **`package_case`** | LLM (Anchor / Body) | The component's package type (e.g., `SOT-23`). It's pulled from the initial **anchor extraction** or from the document body. |
| **`description`** | LLM (Anchor) | The short technical description (e.g., "40V PNP switching transistor") extracted from the **document's title or header**. |
| **`confidence`** | LLM | The confidence score (`high`, `medium`, or `low`) assigned by the extraction model. |
| **`source`** | `finalize` node | The original **PDF filename** (e.g., `MMBT3906.pdf`), added to trace the item back to its source document. |
| **`manufacturer`**| `finalize` node | The manufacturer's name, which is parsed **from the PDF filename** (e.g., `MANUFACTURER__PARTNUM.pdf`). |
