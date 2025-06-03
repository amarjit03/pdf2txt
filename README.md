# PDF-to-Text Agent

The PDF-to-Text Agent is a Python application designed to extract textual content from various types of PDF documents. It intelligently analyzes PDFs to determine if they are text-based, scanned (image-based), or mixed, and then applies the optimal extraction strategy. The agent incorporates memory management techniques to handle large files efficiently and provides options for different processing modes and output formats.

## Table of Contents

*   [Features](#features)
*   [Quick Start](#quick-start)
*   [Directory Structure](#directory-structure)
*   [Setup and Installation](#setup-and-installation)
*   [Usage](#usage)
*   [Configuration](#configuration)
*   [Contributing](#contributing)
*   [License](#license)

## Features

*   **Versatile PDF Handling**: Supports text-based, scanned (OCR), and mixed PDF documents.
*   **Intelligent Analysis**: Automatically analyzes PDF content to choose the most effective extraction method (direct text, OCR, or hybrid).
*   **Configurable Processing Modes**: Offers different modes (`Fast`, `Balanced`, `Quality`) to balance speed and extraction accuracy.
*   **Memory Efficient**: Implements chunking for large file processing and includes an active memory manager to prevent crashes and optimize resource usage.
*   **Multiple Output Formats**: Extracted content can be saved as plain text (`.txt`) or structured JSON (`.json`) files.
*   **Batch Processing**: Capable of processing multiple PDF files from a directory in a single run.
*   **Customizable Configuration**: Core settings (memory limits, OCR language, output paths, etc.) are configurable via `pdf2text/config.py` and command-line arguments.
*   **Stress Testing Tool**: Includes `tools/stress_test.py` for performance testing and aiding in the detection of potential memory leaks.
*   **Detailed Logging**: Provides comprehensive logs for monitoring and troubleshooting.

## Quick Start

This guide will get you up and running with the PDF-to-Text Agent quickly.

1.  **Prerequisites**:
    *   Python 3.7+ installed.
    *   pip (Python package installer) installed.
    *   Tesseract OCR installed (see [Setup and Installation](#setup-and-installation) for details).

2.  **Clone & Setup**:
    ```bash
    # Clone the repository (if you haven't already)
    # git clone <repository_url>
    # cd <repository_directory>

    # Create and activate a virtual environment
    python -m venv .venv
    # On Windows:
    # .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate

    # Install the package and dependencies
    pip install .
    ```

3.  **Run on a Sample PDF**:
    The project includes a sample PDF. Try converting it:
    ```bash
    pdf2text-agent pdf2text/sample_pdf/sample.pdf
    ```
    This will process `sample.pdf` using default settings (balanced mode, outputting both text and JSON files) and save the results in the `pdf2text/output/` directory.

4.  **Check the Output**:
    Look for `sample_extracted.txt` and `sample_extracted.json` in `pdf2text/output/text_files/` and `pdf2text/output/json_files/` respectively.

For more detailed instructions, options, and batch processing, please refer to the [Usage](#usage) section.

## Directory Structure

```
.
├── pdf2text/
│   ├── main.py                 # Main command-line interface entry point
│   ├── config.py               # Application configuration settings
│   │
│   ├── core/                   # Core application logic
│   │   ├── agent.py            # Orchestrates the PDF processing pipeline
│   │   ├── file_manager.py     # Handles file I/O operations
│   │   └── memory_manager.py   # Manages memory usage and cleanup
│   │
│   ├── analyzers/              # Modules for PDF analysis
│   │   ├── pdf_analyzer.py     # Analyzes PDF type and characteristics
│   │   └── memory_estimator.py # Estimates memory requirements for processing
│   │
│   ├── extractors/             # Modules for text extraction
│   │   ├── text_extractors.py  # Extracts text from text-based PDFs (uses PyMuPDF)
│   │   ├── ocr_extractors.py   # Extracts text from scanned PDFs using OCR (Tesseract)
│   │   └── hybrid_extractors.py # Combines text and OCR extraction strategies
│   │
│   ├── utils/                  # Utility modules
│   │   └── logger.py           # Logging configuration
│   │
│   ├── sample_pdf/             # Contains sample PDF files for testing
│   │   └── sample.pdf
│   └── output/                 # Default directory for generated output files
│       ├── text_files/         # Plain text (.txt) outputs
│       ├── json_files/         # JSON (.json) outputs
│       └── ...                 # Other potential output subdirectories (logs, temp)
│
├── tools/
│   └── stress_test.py          # Script for stress testing and memory leak detection
│
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
├── .gitignore                  # Specifies intentionally untracked files
├── pyproject.toml              # Project metadata (often for build systems like Poetry/Flit)
└── uv.lock                     # Lock file for uv pip installer (if used)
```

## Setup and Installation

Follow these steps to set up and run the PDF-to-Text Agent:

1.  **Get the Code**: If you haven't done so already (e.g., via the Quick Start), clone the repository or download the source code.
    ```bash
    # Example for cloning:
    # git clone <repository_url> # Replace with actual URL if hosted
    # cd <repository_directory>
    ```
    If you have the files locally, navigate to the project's root directory.

2.  **Create a Python Virtual Environment** (Recommended):
    ```bash
    python -m venv .venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install the Package**:
            Ensure your virtual environment is activated. There are a couple of ways to install the `pdf2text-agent`:

            *   **Editable install from local source (for development)**:
                This is useful if you are making changes to the code. From the project's root directory:
                ```bash
                pip install -e .
                ```
            *   **Standard install from local source**:
                From the project's root directory:
                ```bash
                pip install .
                ```
            *   **(Future) Install from PyPI**:
                Once the package is published to the Python Package Index (PyPI), you'll be able to install it like any other package:
                ```bash
                # pip install pdf2text-agent # (This is a placeholder until published)
                ```
            Installing the package will automatically handle the dependencies listed in `pyproject.toml`. The `requirements.txt` file can still be useful for specific development environments or for reference.

4.  **Install Tesseract OCR**:
    This application uses Tesseract OCR for processing scanned documents. You need to install Tesseract on your system.
    *   **Linux (Debian/Ubuntu)**:
        ```bash
        sudo apt-get update
        sudo apt-get install tesseract-ocr
        sudo apt-get install libtesseract-dev # For development headers if needed
        # Install language packs as needed, e.g., for English:
        sudo apt-get install tesseract-ocr-eng
        ```
    *   **macOS (using Homebrew)**:
        ```bash
        brew install tesseract
        # Install language packs, e.g., for English:
        brew install tesseract-lang
        ```
    *   **Windows**:
        Download the Tesseract installer from the [official Tesseract at UB Mannheim page](https://github.com/UB-Mannheim/tesseract/wiki). During installation, make sure to:
            *   Add Tesseract to your system PATH.
            *   Select and install the language data packs you need (e.g., English).
    *   **Verify Installation**:
        After installation, open a new terminal/command prompt and type `tesseract --version`. You should see the Tesseract version information.
    *   For more details and other operating systems, refer to the [official Tesseract installation documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).

5.  **Configuration (Optional)**:
    *   Default settings are provided in `pdf2text/config.py`. You can modify this file to change default behaviors such as output directories, memory limits, or OCR languages.

## Usage

Ensure your virtual environment is activated and Tesseract OCR is installed before running the agent.

### Command-line Interface

The PDF-to-Text Agent is run from the command line using the `pdf2text-agent` command (once the package is installed).
The basic command structure is:

```bash
pdf2text-agent <input_path_or_dir> [options]
```

*   `<input_path_or_dir>`: This is a positional argument. It should be the path to a single PDF file, or if the `--batch` option is used, it should be the path to a directory containing PDF files.

**Common Options**:

The following options are available to customize the processing:

*   `--mode [fast|balanced|quality]`: Sets the processing mode.
    *   `fast`: Prioritizes speed over extraction accuracy. Useful for quick previews or when OCR quality is less critical.
    *   `balanced`: Offers a balance between speed and accuracy. This is the default mode.
    *   `quality`: Prioritizes the highest possible extraction accuracy, which may be slower. Recommended for documents requiring careful OCR.
*   `--output [text|json|both]`: Specifies the output format.
    *   `text`: Saves a plain `.txt` file containing the extracted text.
    *   `json`: Saves a detailed `.json` file with extracted content, metadata, and processing information.
    *   `both`: Saves both `.txt` and `.json` files. This is the default.
*   `--language <language_code>`: Specifies the language(s) for OCR (Optical Character Recognition). For example, use `eng` for English, `deu` for German, or `eng+fra` for English and French. The default is `eng`.
*   `--memory-limit <MB>`: Overrides the maximum memory (in Megabytes) the application should attempt to use. This can be useful for processing very large files or in memory-constrained environments.
*   `--verbose` or `-v`: Enables verbose logging, providing more detailed output about the processing steps, including configuration details and memory usage.
*   `--config`: Displays the current configuration settings (derived from `pdf2text/config.py` and any command-line overrides) and then exits without processing any files.

Output files are saved in the `pdf2text/output/` directory by default. This can be configured in `pdf2text/config.py`.

**Examples**:

*   Process a single PDF with default settings (balanced mode, output both text and JSON):
    ```bash
    pdf2text-agent documents/report.pdf
    ```
*   Process a scanned PDF in quality mode and save only as a text file:
    ```bash
    pdf2text-agent scans/image_based.pdf --mode quality --output text
    ```
*   Process a German PDF using OCR (default output is both text and JSON):
    ```bash
    pdf2text-agent invoices/invoice_de.pdf --language deu
    ```
*   Display the current configuration:
    ```bash
    pdf2text-agent --config
    ```
*   Process a PDF with verbose output:
    ```bash
    pdf2text-agent important_document.pdf --verbose
    ```

### Batch Processing Multiple PDFs

To process all PDF files within a specific directory, use the `--batch` flag along with the directory path.

```bash
pdf2text-agent <path_to_your_pdf_directory> --batch [options]
```

You can combine batch processing with other options like `--mode` or `--output`:

*   Process all PDFs in the `project_files/reports/` directory using fast mode and saving only JSON output:
    ```bash
    pdf2text-agent project_files/reports/ --batch --mode fast --output json
    ```

### Running the Stress Test Tool

The application includes a stress testing tool to help identify potential memory leaks by repeatedly processing PDFs.

```bash
python tools/stress_test.py --pdf_dir <path_to_directory_with_test_pdfs> [--iterations <number_of_runs>] [--output_dir <path_for_stress_test_outputs>]
```
*   `--pdf_dir`: (Required) Path to a directory containing a diverse set of PDF files for testing.
*   `--iterations`: (Optional) Number of times the batch processing will be repeated. Defaults to 5.
*   `--output_dir`: (Optional) If specified, output files from the stress test runs will be saved here in iteration-specific subdirectories. Otherwise, temporary directories are used and cleaned up.

Example:
```bash
python tools/stress_test.py --pdf_dir pdf2text/sample_pdf/ --iterations 10
```
Monitor the console output for memory usage statistics after each iteration. A consistent increase in memory usage might indicate a leak.

## Configuration

Beyond the command-line arguments, more granular control over the application's behavior can be achieved by modifying the settings in `pdf2text/config.py`.

This file includes options such as:

*   Default OCR language.
*   Memory management thresholds and limits.
*   Default output directory paths.
*   Logging levels and formatting.
*   Parameters for PDF analysis and memory estimation.

Please refer to the comments within `pdf2text/config.py` for detailed explanations of each configuration option.

## Contributing

Contributions are welcome! If you'd like to improve the PDF-to-Text Agent or add new features:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Ensure your code adheres to good practices and includes relevant comments or documentation.
5.  If adding new features or fixing bugs, consider adding or updating tests.
6.  Submit a pull request with a clear description of your changes.

We appreciate your help in making this tool better!

## License

This project is licensed under the MIT License.