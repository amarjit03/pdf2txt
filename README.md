```
PDF Input
    ↓
Is file valid? → NO → Report error & exit
    ↓ YES
Analyze PDF type
    ↓
┌─────────────────────────────────────┐
│ TEXT-BASED  │  SCANNED  │   MIXED   │
│     ↓       │     ↓     │     ↓     │
│ Fast Text   │   OCR     │  Hybrid   │
│ Extraction  │ Processing│ Approach  │
└─────────────────────────────────────┘
    ↓
Estimate memory needs
    ↓
File size > Memory limit? → YES → Use chunking
    ↓ NO                           ↓
Process entire file            Process in chunks
    ↓                               ↓
    └─────────── Extract text ──────┘
                     ↓
            Clean & format text
                     ↓
        ┌─────────────────────────┐
        │ Save as TXT │ Save as JSON │
        └─────────────────────────┘
                     ↓
              Report results

```
pdf-text-agent/
├── main.py                    # Entry point - run this to start
├── config.py                  # All configuration settings
├── requirements.txt           # Dependencies list
│
├── core/
│   ├── __init__.py
│   ├── agent.py              # Main orchestrator
│   ├── memory_manager.py     # Memory monitoring & cleanup
│   └── file_manager.py       # File operations & path handling
│
├── analyzers/
│   ├── __init__.py
│   ├── pdf_detector.py       # Determines PDF type (text/scanned/mixed)
│   └── memory_estimator.py   # Estimates memory needs
│
├── extractors/
│   ├── __init__.py
│   ├── text_extractor.py     # Direct text extraction (PyMuPDF)
│   ├── ocr_extractor.py      # OCR processing (Tesseract)
│   └── hybrid_extractor.py   # Smart combination of both
│
├── processors/
│   ├── __init__.py
│   ├── chunk_manager.py      # Handles chunking for large files
│   ├── page_processor.py     # Page-by-page processing
│   └── text_cleaner.py       # Cleans and formats extracted text
│
├── savers/
│   ├── __init__.py
│   ├── text_saver.py         # Saves as .txt files
│   ├── json_saver.py         # Saves as .json files
│   └── batch_saver.py        # Handles multiple output formats
│
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Logging setup
│   ├── helpers.py            # Common utility functions
│   └── validators.py         # Input validation
│
├── output/                   # Generated output files
│   ├── text_files/          # Plain text outputs
│   ├── json_files/          # JSON formatted outputs
│   ├── logs/                # Processing logs
│   └── failed/              # Failed extraction attempts
│
└── sample_pdfs/             # Test PDFs for development