import time
import fitz  
import pdfplumber
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import sys

# Ensure the root directory is on the import path
# sys.path.append(str(Path(__file__).resolve().parent.parent)) # Removed for packaging

from pdf2text.config import get_config

class PDFType(Enum):
    """PDF content types"""
    TEXT_BASED = "text_based"      # Digital text, directly extractable
    SCANNED = "scanned"            # Images of text, needs OCR
    MIXED = "mixed"                # Some pages text, some scanned
    ENCRYPTED = "encrypted"        # Password protected
    CORRUPTED = "corrupted"        # Damaged or invalid
    EMPTY = "empty"                # No content


class ContentComplexity(Enum):
    """Content structure complexity"""
    SIMPLE = "simple"              # Plain text, basic formatting
    MODERATE = "moderate"          # Tables, lists, multiple columns
    COMPLEX = "complex"            # Forms, complex layouts, mixed content


@dataclass
class PDFAnalysisResult:
    """Complete analysis results for a PDF"""
    # Basic properties
    pdf_type: PDFType
    complexity: ContentComplexity
    page_count: int
    file_size_mb: float
    
    # Content analysis
    text_pages: int                # Pages with extractable text
    scanned_pages: int            # Pages requiring OCR
    empty_pages: int              # Pages with no content
    
    # Structure analysis
    has_tables: bool
    has_images: bool
    has_forms: bool
    has_annotations: bool
    
    # Quality metrics
    text_density_avg: float       # Average text density per page
    image_quality_score: float    # For scanned pages (0-1)
    extraction_confidence: float   # Predicted success rate (0-1)
    
    # Processing estimates
    estimated_processing_time: float    # Seconds
    estimated_memory_mb: float         # Peak memory usage
    recommended_chunk_size: int        # Pages per chunk
    
    # Metadata
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    creation_date: Optional[str] = None
    
    # Issues found
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []



class PDFAnalyzer:
    """Inteligent PDF analysis """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.sample_pages_count = 5  # Pages to sample for analysis
        self.min_text_length = 50    # Minimum text length to consider "text-based"
        self.text_density_threshold = 0.1  # Text area / page area ratio
    
    def analyze_pdf(self, file_path: Path) -> PDFAnalysisResult:
        """Complete PDF analysis - main entry point

        Args:
            file_path: Path to PDF file

        Returns:
            PDFAnalysisResult with complete analysis"""
        start_time = time.time()
        try:
            if not self._validate_file(file_path):
                return self._create_error_result(file_path, "File validation failed")

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            result = PDFAnalysisResult(
                pdf_type=PDFType.CORRUPTED,  # Placeholder
                complexity=ContentComplexity.SIMPLE,
                page_count=0,
                file_size_mb=file_size_mb,
                text_pages=0,
                scanned_pages=0,
                empty_pages=0,
                has_tables=False,
                has_images=False,
                has_forms=False,
                has_annotations=False,
                text_density_avg=0.0,
                image_quality_score=0.0,
                extraction_confidence=0.0,
                estimated_processing_time=0.0,
                estimated_memory_mb=0.0,
                recommended_chunk_size=1
)

            # Try to open and analyze PDF
            doc = None
            try:
                try:
                    doc = fitz.open(str(file_path))
                except Exception as e:
                    return self._create_error_result(file_path, f"PyMuPDF failed to open file: {str(e)}")

                # Check for encryption
                if doc.needs_pass:
                    result.pdf_type = PDFType.ENCRYPTED
                    result.errors.append("PDF is password protected")
                    return result

                # Basic document info
                try:
                    result.page_count = doc.page_count
                except Exception as e:
                    return self._create_error_result(file_path, f"Could not retrieve page count: {str(e)}")

                if result.page_count == 0:
                    result.pdf_type = PDFType.EMPTY
                    result.warnings.append("PDF contains no pages or is invalid")
                    return result

                # Extract metadata
                self._extract_metadata(doc, result)

                # Analyze content type and structure
                self._analyze_content_type(doc, result)
                self._analyze_structure(doc, result)
                self._calculate_complexity(result)

                # Performance estimates
                self._estimate_processing_requirements(result)

                # Confidence
                result.extraction_confidence = self._calculate_confidence(result)

            except Exception as e:
                result.pdf_type = PDFType.CORRUPTED
                result.errors.append(f"PDF analysis failed: {str(e)}")
                return result

            finally:
                if doc:
                    doc.close()

            # Additional structure analysis with pdfplumber
            try:
                self._analyze_with_pdfplumber(file_path, result)
            except Exception as e:
                result.warnings.append(f"Advanced analysis failed: {str(e)}")

            self.logger.info(f"PDF analysis completed in {time.time() - start_time:.2f}s")
            return result

        except Exception as e:
            return self._create_error_result(file_path, f"Top-level analysis failed: {str(e)}")

        
    def _validate_file(self, file_path: Path) -> bool:
        """Basic file validation with debug prints"""
        if not file_path.exists():
            print("❌ Validation failed: File does not exist.")
            return False
        if not file_path.is_file():
            print("❌ Validation failed: Not a file.")
            return False
        if file_path.suffix.lower() != '.pdf':
            print(f"❌ Validation failed: Invalid extension {file_path.suffix}")
            return False
        if file_path.stat().st_size == 0:
            print("❌ Validation failed: File is empty.")
            return False
        return True
    
    def _extract_metadata(self, doc: fitz.Document, result: PDFAnalysisResult):
        """Extract PDF metadata"""
        try:
            metadata = doc.metadata
            result.title = metadata.get('title', '').strip() or None
            result.author = metadata.get('author', '').strip() or None
            result.subject = metadata.get('subject', '').strip() or None
            result.creator = metadata.get('creator', '').strip() or None
            result.creation_date = metadata.get('creationDate', '').strip() or None
        except Exception as e:
            result.warnings.append(f"Metadata extraction failed: {str(e)}")

    def _analyze_content_type(self, doc: fitz.Document, result: PDFAnalysisResult):
        """Determine if PDF is text-based, scanned, or mixed"""
        sample_size = min(self.sample_pages_count, result.page_count)
        text_pages = 0
        scanned_pages = 0
        empty_pages = 0
        total_text_density = 0.0
        
        for page_num in range(sample_size):
            page = doc[page_num]
            
            # Get text content
            text = page.get_text().strip()
            text_length = len(text)
            
            # Calculate text density (text area vs page area)
            page_area = page.rect.width * page.rect.height
            text_blocks = page.get_text("dict")["blocks"]
            text_area = sum([
                (block.get("bbox", [0, 0, 0, 0])[2] - block.get("bbox", [0, 0, 0, 0])[0]) * (block.get("bbox", [0, 0, 0, 0])[3] - block.get("bbox", [0, 0, 0, 0])[1])
                for block in text_blocks 
                if block.get("type") == 0  # Text blocks
            ])
            
            text_density = text_area / page_area if page_area > 0 else 0
            total_text_density += text_density
            
            # Check for images
            image_list = page.get_images()
            has_images = len(image_list) > 0
            
            # Classify page type
            if text_length >= self.min_text_length and text_density >= self.text_density_threshold:
                text_pages += 1
            elif has_images and text_length < self.min_text_length:
                scanned_pages += 1
            else:
                empty_pages += 1
        
        # Scale results to full document
        scale_factor = result.page_count / sample_size
        result.text_pages = int(text_pages * scale_factor)
        result.scanned_pages = int(scanned_pages * scale_factor)
        result.empty_pages = int(empty_pages * scale_factor)
        result.text_density_avg = total_text_density / sample_size
        
        # Determine overall PDF type
        text_ratio = result.text_pages / result.page_count
        scanned_ratio = result.scanned_pages / result.page_count
        
        if text_ratio >= 0.8:
            result.pdf_type = PDFType.TEXT_BASED
        elif scanned_ratio >= 0.8:
            result.pdf_type = PDFType.SCANNED
        elif text_ratio + scanned_ratio >= 0.5:
            result.pdf_type = PDFType.MIXED
        else:
            result.pdf_type = PDFType.EMPTY

    
    
    def _analyze_structure(self, doc: fitz.Document, result: PDFAnalysisResult):
        """Analyze document structure elements"""
        result.has_images = False
        result.has_forms = False
        result.has_annotations = False
        
        # Sample a few pages for structure analysis
        sample_size = min(3, result.page_count)
        
        for page_num in range(sample_size):
            page = doc[page_num]
            
            # Check for images
            if page.get_images():
                result.has_images = True
            
            # Check for form fields
            if page.widgets():
                result.has_forms = True
            
            # Check for annotations
            if page.annots():
                result.has_annotations = True
    
    def _analyze_with_pdfplumber(self, file_path: Path, result: PDFAnalysisResult):
        """Use pdfplumber for advanced structure analysis (especially tables)"""
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                # Sample first few pages for table detection
                sample_size = min(3, len(pdf.pages))
                tables_found = 0
                
                for page_num in range(sample_size):
                    page = pdf.pages[page_num]
                    tables = page.find_tables()
                    if tables:
                        tables_found += len(tables)
                
                result.has_tables = tables_found > 0
                
        except Exception as e:
            # pdfplumber failed, but we can continue without table detection
            result.warnings.append(f"Table detection failed: {str(e)}")
    
    def _calculate_complexity(self, result: PDFAnalysisResult):
        """Determine content complexity level"""
        complexity_score = 0
        
        # Base complexity from PDF type
        if result.pdf_type == PDFType.SCANNED:
            complexity_score += 3
        elif result.pdf_type == PDFType.MIXED:
            complexity_score += 2
        
        # Additional complexity factors
        if result.has_tables:
            complexity_score += 2
        if result.has_forms:
            complexity_score += 2
        if result.has_images:
            complexity_score += 1
        if result.has_annotations:
            complexity_score += 1
        
        # Low text density = more complex layout
        if result.text_density_avg < 0.1:
            complexity_score += 1
        
        # Classify complexity
        if complexity_score <= 2:
            result.complexity = ContentComplexity.SIMPLE
        elif complexity_score <= 5:
            result.complexity = ContentComplexity.MODERATE
        else:
            result.complexity = ContentComplexity.COMPLEX
    
    def _estimate_processing_requirements(self, result: PDFAnalysisResult):
        """Estimate processing time and memory requirements"""
        config = self.config
        
        # Base estimates per page
        if result.pdf_type == PDFType.TEXT_BASED:
            time_per_page = 0.1  # seconds
            memory_per_page = 0.5  # MB
        elif result.pdf_type == PDFType.SCANNED:
            time_per_page = 2.0   # OCR is slow
            memory_per_page = 5.0  # Images use more memory
        else:  # MIXED
            time_per_page = 1.0
            memory_per_page = 2.5
        
        # Adjust for complexity
        if result.complexity == ContentComplexity.COMPLEX:
            time_per_page *= 1.5
            memory_per_page *= 1.3
        elif result.complexity == ContentComplexity.MODERATE:
            time_per_page *= 1.2
            memory_per_page *= 1.1
        
        # Calculate totals
        result.estimated_processing_time = result.page_count * time_per_page
        result.estimated_memory_mb = min(
            result.page_count * memory_per_page,
            config.memory.max_memory_mb * 0.8  # Leave 20% buffer
        )
        
        # Recommend chunk size based on memory constraints
        pages_per_chunk = max(1, int(config.memory.max_memory_mb * 0.6 / memory_per_page))
        result.recommended_chunk_size = min(pages_per_chunk, config.get_chunk_size(result.file_size_mb))
    
    def _calculate_confidence(self, result: PDFAnalysisResult) -> float:
        """Calculate extraction confidence score"""
        if result.pdf_type == PDFType.TEXT_BASED:
            base_confidence = 0.95
        elif result.pdf_type == PDFType.SCANNED:
            base_confidence = 0.75  # OCR reliability
        elif result.pdf_type == PDFType.MIXED:
            # Weighted average based on page types
            total_pages = result.text_pages + result.scanned_pages
            if total_pages > 0:
                text_weight = result.text_pages / total_pages
                scanned_weight = result.scanned_pages / total_pages
                base_confidence = (text_weight * 0.95) + (scanned_weight * 0.75)
            else:
                base_confidence = 0.5
        else:
            base_confidence = 0.1
        
        # Adjust for complexity
        if result.complexity == ContentComplexity.COMPLEX:
            base_confidence *= 0.85
        elif result.complexity == ContentComplexity.MODERATE:
            base_confidence *= 0.92
        
        # Adjust for text density
        if result.text_density_avg < 0.05:
            base_confidence *= 0.9
        
        return max(0.0, min(1.0, base_confidence))
    
    def _create_error_result(self, file_path: Path, error_message: str) -> PDFAnalysisResult:
        """Create result object for failed analysis"""
        file_size_mb = 0
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
        except:
            pass
        
        return PDFAnalysisResult(
            pdf_type=PDFType.CORRUPTED,
            complexity=ContentComplexity.SIMPLE,
            page_count=0,
            file_size_mb=file_size_mb,
            text_pages=0,
            scanned_pages=0,
            empty_pages=0,
            has_tables=False,
            has_images=False,
            has_forms=False,
            has_annotations=False,
            text_density_avg=0.0,
            image_quality_score=0.0,
            extraction_confidence=0.0,
            estimated_processing_time=0.0,
            estimated_memory_mb=0.0,
            recommended_chunk_size=1,
            errors=[error_message]
        )


def analyze_pdf_quick(file_path: Path) -> Dict:
    """Quick analysis for basic PDF info (used by main.py for progress display)"""
    analyzer = PDFAnalyzer()
    try:
        doc = fitz.open(str(file_path))
        page_count = doc.page_count
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Quick type detection (first page only)
        if page_count > 0:
            page = doc[0]
            text = page.get_text().strip()
            has_images = len(page.get_images()) > 0
            
            if len(text) > 50:
                pdf_type = "text-based"
            elif has_images:
                pdf_type = "scanned"
            else:
                pdf_type = "unknown"
        else:
            pdf_type = "empty"
        
        doc.close()
        
        return {
            'page_count': page_count,
            'file_size_mb': file_size_mb,
            'pdf_type': pdf_type,
            'success': True
        }
        
    except Exception as e:
        return {
            'page_count': 0,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
            'pdf_type': 'corrupted',
            'error': str(e),
            'success': False
        }


if __name__ == "__main__":
    # Test the analyzer
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pdf_analyzer.py <pdf_file>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    analyzer = PDFAnalyzer()
    result = analyzer.analyze_pdf(file_path)
    
    print(f"\n=== PDF Analysis Results ===")
    print(f"File: {file_path.name}")
    print(f"Type: {result.pdf_type.value}")
    print(f"Complexity: {result.complexity.value}")
    print(f"Pages: {result.page_count}")
    print(f"Size: {result.file_size_mb:.1f} MB")
    print(f"Text pages: {result.text_pages}")
    print(f"Scanned pages: {result.scanned_pages}")
    print(f"Has tables: {result.has_tables}")
    print(f"Has images: {result.has_images}")
    print(f"Confidence: {result.extraction_confidence:.2f}")
    print(f"Est. processing time: {result.estimated_processing_time:.1f}s")
    print(f"Est. memory usage: {result.estimated_memory_mb:.1f} MB")
    print(f"Recommended chunk size: {result.recommended_chunk_size} pages")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

