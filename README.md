# 🧠 Agentic Workflow for Electronics Data Extraction

This project implements a robust agentic workflow for extracting structured information (e.g., MPNs, descriptions, packaging, markings) from unstructured electronics datasheets (PDFs). It uses a combination of OCR, markdown conversion, chunking, and LLM-driven parsing in a graph-based pipeline.

---

## 📁 Project Structure

```
.
👉 documents/               # Input PDF datasheets
👉 markdown/                # Extracted markdown and images from each PDF
👉 metadata/                # Per-document metadata (JSON)
👉 processed/               # Marks processed PDFs to avoid duplication
👉 extracted_items.csv      # Final structured extractions (MPNs, package, etc.)
👉 failed_pdfs.csv          # Log of PDFs that failed during processing
👉 config.py                # Path and model config
👉 graph_builder.py         # Graph-based pipeline builder
👉 helpers.py               # Utilities: OCR, chunking, prompt generation, saving
👉 main.py                  # Main entry point for processing PDFs
👉 nodes.py                 # Modular graph nodes (load, extract, validate, save)
👉 requirements.txt         # Required Python dependencies
```

---

## ⚙️ How It Works

1. **Input**: Place any PDF datasheet in `documents/`.
2. **Markdown Conversion**: Each PDF is converted into markdown via OCR (with images extracted).
3. **Chunking**: Markdown is split into manageable parts for LLM processing.
4. **Anchor Extraction**: Extract the core component name and description.
5. **Chunk Iteration**: Each chunk is analyzed for item extraction (MPN, marking, etc.).
6. **Validation**: Extracted items are validated with a second LLM call for confidence assessment.
7. **Finalization**: Items are stored in CSV and the PDF is marked as processed.

---

## 🧱 Components

### `main.py`

* Entry point that:

  * Loads the environment
  * Initializes the graph and converter
  * Iterates over unprocessed PDFs
  * Invokes the LangGraph agent on each document

### `graph_builder.py`

* Constructs a **LangGraph** with the following nodes:

  * `load` → `anchor` → `llm` → `parse` → `validate` → `final` → `save`

### `nodes.py`

Implements the logic for each graph node:

| Function             | Description                                             |
| -------------------- | ------------------------------------------------------- |
| `load_and_split()`   | Converts PDF to markdown, extracts tables and images    |
| `extract_anchor()`   | Gets the main component name and short description      |
| `call_llm()`         | Extracts MPN data using an LLM and a custom prompt      |
| `parse_and_repair()` | Parses or repairs broken JSON from the LLM              |
| `validate_items()`   | Double-checks extracted items for quality/confidence    |
| `finalize()`         | Adds metadata and writes items to `extracted_items.csv` |
| `save_full_state()`  | Dumps intermediate state to `metadata/` for inspection  |

---

## 🔧 Setup & Usage

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your `.env`**

   ```
   GROQ_API_KEY=your_groq_key_here
   ```

3. **Add PDFs**
   Place your datasheets in the `documents/` folder.

4. **Run the pipeline**

   ```bash
   python main.py
   ```

5. **View Results**

   * ✅ Structured results in `extracted_items.csv`
   * ❌ Failures logged in `failed_pdfs.csv`
   * 📄 Markdown + image references in `markdown/`
   * 🧠 Metadata for each document in `metadata/`

---

## 📦 Sample Output Format (CSV)

| mpn      | top\_marking | package\_case | description                  | confidence | source       |
| -------- | ------------ | ------------- | ---------------------------- | ---------- | ------------ |
| MMBT3906 | 3906         | SOT-23        | 40V PNP switching transistor | high       | MMBT3906.pdf |

---