# """
# Text Extractor - Direct PDF Text Extraction using PyMuPDF
# High-speed extraction for text-based PDFs with memory management
# """

# import fitz  # PyMuPDF
# import time
# from pathlib import Path
# from typing import Dict, List, Optional, Iterator, Tuple
# from dataclasses import dataclass
# from enum import Enum
# import logging
# import re

# import sys

# # Ensure the root directory is on the import path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

# from config import get_config
# from core.memory_manager import get_memory_manager, ManagedResource, CleanupPriority, MemoryContext
# from analyzers.pdf_analyzer import PDFAnalysisResult, PDFType, ContentComplexity


# class TextExtractionMode(Enum):
#     """Text extraction modes"""
#     FAST = "fast"           # Basic text extraction
#     LAYOUT = "layout"       # Preserve formatting and layout
#     STRUCTURED = "structured"  # Extract with structure (tables, lists)


# @dataclass
# class TextExtractionResult:
#     """Results from text extraction"""
#     success: bool
#     extracted_text: str
#     page_count: int
#     total_characters: int
#     processing_time: float
    
#     # Quality metrics
#     extraction_confidence: float    # 0.0 - 1.0
#     pages_with_text: int
#     pages_empty: int
    
#     # Structure information
#     tables_detected: int
#     images_detected: int
#     links_extracted: List[str]
    
#     # Processing details
#     method_used: str
#     memory_peak_mb: float
#     chunks_processed: int
    
#     # Issues encountered
#     warnings: List[str]
#     errors: List[str]
    
#     # Page-by-page breakdown
#     page_details: List[Dict]


# class TextExtractor:
#     """High-performance direct text extraction from PDFs"""
    
#     def __init__(self):
#         self.config = get_config()
#         self.memory_manager = get_memory_manager()
#         self.logger = logging.getLogger(__name__)
        
#         # Extraction parameters
#         self.extraction_flags = {
#             TextExtractionMode.FAST: fitz.TEXT_PRESERVE_WHITESPACE,
#             TextExtractionMode.LAYOUT: fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES,
#             TextExtractionMode.STRUCTURED: fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_SPANS
#         }
        
#         # Text cleaning patterns
#         self.cleanup_patterns = [
#             (r'\x00', ''),           # Remove null characters
#             (r'\r\n', '\n'),         # Normalize line endings
#             (r'\r', '\n'),           # Normalize line endings
#             (r'\n{3,}', '\n\n'),     # Reduce excessive line breaks
#             (r'[ \t]{2,}', ' '),     # Reduce excessive spaces
#             (r'^\s+$', '', re.MULTILINE),  # Remove empty lines with whitespace
#         ]
        
#         self.logger.info("Text Extractor initialized")
    
#     def extract_text(self, file_path: Path, 
#                     analysis: PDFAnalysisResult,
#                     mode: TextExtractionMode = TextExtractionMode.FAST,
#                     chunk_size: Optional[int] = None) -> TextExtractionResult:
#         """
#         Main text extraction method
        
#         Args:
#             file_path: Path to PDF file
#             analysis: PDF analysis results from Step 2
#             mode: Extraction mode (fast/layout/structured)
#             chunk_size: Pages per chunk (None = use analysis recommendation)
            
#         Returns:
#             TextExtractionResult with extracted text and metadata
#         """
#         start_time = time.time()
        
#         # Validate inputs
#         if analysis.pdf_type not in [PDFType.TEXT_BASED, PDFType.MIXED]:
#             return self._create_error_result(
#                 f"Text extractor cannot handle {analysis.pdf_type.value} PDFs"
#             )
        
#         # Determine chunk size
#         if chunk_size is None:
#             chunk_size = analysis.recommended_chunk_size
        
#         # Initialize result
#         result = TextExtractionResult(
#             success=False,
#             extracted_text="",
#             page_count=analysis.page_count,
#             total_characters=0,
#             processing_time=0.0,
#             extraction_confidence=0.0,
#             pages_with_text=0,
#             pages_empty=0,
#             tables_detected=0,
#             images_detected=0,
#             links_extracted=[],
#             method_used=f"pymupdf_{mode.value}",
#             memory_peak_mb=0.0,
#             chunks_processed=0,
#             warnings=[],
#             errors=[],
#             page_details=[]
#         )
        
#         try:
#             # Memory-managed extraction
#             with MemoryContext(self.memory_manager, "text_extraction", 
#                              analysis.estimated_memory_mb) as ctx:
                
#                 if chunk_size >= analysis.page_count:
#                     # Process entire document at once
#                     result = self._extract_full_document(file_path, analysis, mode, result)
#                 else:
#                     # Process in chunks
#                     result = self._extract_chunked(file_path, analysis, mode, chunk_size, result)
                
#                 # Final processing
#                 result.processing_time = time.time() - start_time
#                 result.extraction_confidence = self._calculate_confidence(result)
#                 result.memory_peak_mb = ctx.memory_manager.get_memory_snapshot().process_mb
                
#                 self.logger.info(
#                     f"Text extraction completed: {result.total_characters} chars "
#                     f"from {result.page_count} pages in {result.processing_time:.2f}s"
#                 )
                
#                 return result
                
#         except Exception as e:
#             self.logger.error(f"Text extraction failed: {str(e)}")
#             result.success = False
#             result.errors.append(str(e))
#             result.processing_time = time.time() - start_time
#             return result
    
#     def _extract_full_document(self, file_path: Path, analysis: PDFAnalysisResult,
#                               mode: TextExtractionMode, result: TextExtractionResult) -> TextExtractionResult:
#         """Extract text from entire document at once (for small files)"""
#         try:
#             doc = fitz.open(str(file_path))
            
#             # Register document as managed resource
#             doc_resource = ManagedResource(
#                 resource_id=f"pdf_doc_{file_path.stem}",
#                 resource_type="pdf_document",
#                 size_mb=analysis.file_size_mb,
#                 created_at=time.time(),
#                 last_accessed=time.time(),
#                 cleanup_callback=lambda: doc.close(),
#                 priority=CleanupPriority.HIGH
#             )
#             self.memory_manager.register_resource(doc_resource)
            
#             text_parts = []
#             links_found = []
            
#             for page_num in range(doc.page_count):
#                 page = doc[page_num]
                
#                 # Extract text based on mode
#                 page_text = self._extract_page_text(page, mode)
                
#                 # Track page details
#                 page_info = {
#                     'page_number': page_num + 1,
#                     'character_count': len(page_text),
#                     'has_text': len(page_text.strip()) > 0,
#                     'has_images': len(page.get_images()) > 0,
#                     'has_links': len(page.get_links()) > 0
#                 }
                
#                 result.page_details.append(page_info)
                
#                 # Update counters
#                 if page_info['has_text']:
#                     result.pages_with_text += 1
#                     text_parts.append(page_text)
#                 else:
#                     result.pages_empty += 1
                
#                 if page_info['has_images']:
#                     result.images_detected += len(page.get_images())
                
#                 if page_info['has_links']:
#                     page_links = [link.get('uri', '') for link in page.get_links() 
#                                 if link.get('uri')]
#                     links_found.extend(page_links)
                
#                 # Memory check during processing
#                 self.memory_manager.update_resource_access(doc_resource.resource_id)
            
#             # Combine all text
#             result.extracted_text = self._combine_page_texts(text_parts, mode)
#             result.total_characters = len(result.extracted_text)
#             result.links_extracted = list(set(links_found))  # Remove duplicates
#             result.chunks_processed = 1
#             result.success = True
            
#             # Cleanup
#             self.memory_manager.unregister_resource(doc_resource.resource_id)
#             doc.close()
            
#             return result
            
#         except Exception as e:
#             result.errors.append(f"Full document extraction failed: {str(e)}")
#             return result
    
#     def _extract_chunked(self, file_path: Path, analysis: PDFAnalysisResult,
#                         mode: TextExtractionMode, chunk_size: int, 
#                         result: TextExtractionResult) -> TextExtractionResult:
#         """Extract text in chunks for memory efficiency"""
#         text_parts = []
#         links_found = []
#         chunk_count = 0
        
#         for chunk_start in range(0, analysis.page_count, chunk_size):
#             chunk_end = min(chunk_start + chunk_size, analysis.page_count)
            
#             try:
#                 # Process chunk with memory management
#                 chunk_text, chunk_info = self._extract_chunk(
#                     file_path, chunk_start, chunk_end, mode
#                 )
                
#                 if chunk_text:
#                     text_parts.append(chunk_text)
                
#                 # Merge chunk information
#                 result.page_details.extend(chunk_info['page_details'])
#                 result.pages_with_text += chunk_info['pages_with_text']
#                 result.pages_empty += chunk_info['pages_empty']
#                 result.images_detected += chunk_info['images_detected']
#                 links_found.extend(chunk_info['links_found'])
                
#                 chunk_count += 1
                
#                 # Memory cleanup between chunks
#                 self.memory_manager.force_garbage_collection()
                
#                 self.logger.debug(f"Processed chunk {chunk_count}: pages {chunk_start+1}-{chunk_end}")
                
#             except Exception as e:
#                 result.warnings.append(f"Chunk {chunk_start+1}-{chunk_end} failed: {str(e)}")
#                 continue
        
#         # Combine all chunks
#         result.extracted_text = self._combine_page_texts(text_parts, mode)
#         result.total_characters = len(result.extracted_text)
#         result.links_extracted = list(set(links_found))
#         result.chunks_processed = chunk_count
#         result.success = chunk_count > 0
        
#         return result
    
#     def _extract_chunk(self, file_path: Path, start_page: int, end_page: int,
#                       mode: TextExtractionMode) -> Tuple[str, Dict]:
#         """Extract text from a specific page range"""
#         doc = None
#         try:
#             doc = fitz.open(str(file_path))
            
#             text_parts = []
#             chunk_info = {
#                 'page_details': [],
#                 'pages_with_text': 0,
#                 'pages_empty': 0,
#                 'images_detected': 0,
#                 'links_found': []
#             }
            
#             for page_num in range(start_page, end_page):
#                 if page_num >= doc.page_count:
#                     break
                
#                 page = doc[page_num]
#                 page_text = self._extract_page_text(page, mode)
                
#                 # Track page details
#                 page_info = {
#                     'page_number': page_num + 1,
#                     'character_count': len(page_text),
#                     'has_text': len(page_text.strip()) > 0,
#                     'has_images': len(page.get_images()) > 0,
#                     'has_links': len(page.get_links()) > 0
#                 }
                
#                 chunk_info['page_details'].append(page_info)
                
#                 if page_info['has_text']:
#                     chunk_info['pages_with_text'] += 1
#                     text_parts.append(page_text)
#                 else:
#                     chunk_info['pages_empty'] += 1
                
#                 if page_info['has_images']:
#                     chunk_info['images_detected'] += len(page.get_images())
                
#                 if page_info['has_links']:
#                     page_links = [link.get('uri', '') for link in page.get_links() 
#                                 if link.get('uri')]
#                     chunk_info['links_found'].extend(page_links)
            
#             chunk_text = self._combine_page_texts(text_parts, mode)
#             return chunk_text, chunk_info
            
#         finally:
#             if doc:
#                 doc.close()
    
#     def _extract_page_text(self, page: fitz.Page, mode: TextExtractionMode) -> str:
#         """Extract text from a single page based on mode"""
#         try:
#             flags = self.extraction_flags.get(mode, fitz.TEXT_PRESERVE_WHITESPACE)
            
#             if mode == TextExtractionMode.STRUCTURED:
#                 # Enhanced extraction with structure preservation
#                 text = self._extract_structured_text(page)
#             else:
#                 # Standard extraction
#                 text = page.get_text(flags=flags)
            
#             # Clean up text
#             return self._clean_text(text)
            
#         except Exception as e:
#             self.logger.warning(f"Page text extraction failed: {str(e)}")
#             return ""
    
#     def _extract_structured_text(self, page: fitz.Page) -> str:
#         """Extract text with enhanced structure preservation"""
#         try:
#             # Get text with detailed structure
#             text_dict = page.get_text("dict")
#             structured_text = []
            
#             for block in text_dict.get("blocks", []):
#                 if block.get("type") == 0:  # Text block
#                     block_text = self._process_text_block(block)
#                     if block_text:
#                         structured_text.append(block_text)
#                 elif block.get("type") == 1:  # Image block
#                     structured_text.append("[IMAGE]")
            
#             return "\n\n".join(structured_text)
            
#         except Exception as e:
#             self.logger.warning(f"Structured text extraction failed: {str(e)}")
#             # Fallback to basic extraction
#             return page.get_text()
    
#     def _process_text_block(self, block: Dict) -> str:
#         """Process a text block with formatting preservation"""
#         lines = []
        
#         for line in block.get("lines", []):
#             line_text_parts = []
            
#             for span in line.get("spans", []):
#                 text = span.get("text", "").strip()
#                 if text:
#                     line_text_parts.append(text)
            
#             if line_text_parts:
#                 lines.append(" ".join(line_text_parts))
        
#         return "\n".join(lines)
    
#     def _combine_page_texts(self, text_parts: List[str], mode: TextExtractionMode) -> str:
#         """Combine text from multiple pages"""
#         if not text_parts:
#             return ""
        
#         if mode == TextExtractionMode.LAYOUT:
#             # Preserve page breaks
#             return "\n\n[PAGE BREAK]\n\n".join(text_parts)
#         else:
#             # Simple concatenation
#             return "\n\n".join(text_parts)
    
#     def _clean_text(self, text: str) -> str:
#         """Clean and normalize extracted text"""
#         if not text:
#             return ""
        
#         cleaned = text
        
#         # Apply cleanup patterns
#         for pattern, replacement, *flags in self.cleanup_patterns:
#             flag = flags[0] if flags else 0
#             cleaned = re.sub(pattern, replacement, cleaned, flags=flag)
        
#         # Remove excessive whitespace
#         cleaned = cleaned.strip()
        
#         return cleaned
    
#     def _calculate_confidence(self, result: TextExtractionResult) -> float:
#         """Calculate extraction confidence score"""
#         if result.page_count == 0:
#             return 0.0
        
#         # Base confidence from successful page extraction
#         page_success_rate = result.pages_with_text / result.page_count
#         base_confidence = page_success_rate * 0.8  # Max 0.8 from page success
        
#         # Bonus for text density
#         if result.total_characters > 0:
#             avg_chars_per_page = result.total_characters / result.page_count
#             if avg_chars_per_page > 100:  # Reasonable amount of text
#                 base_confidence += 0.1
#             if avg_chars_per_page > 500:  # Good amount of text
#                 base_confidence += 0.1
        
#         # Penalty for warnings/errors
#         if result.warnings:
#             base_confidence -= len(result.warnings) * 0.05
#         if result.errors:
#             base_confidence -= len(result.errors) * 0.1
        
#         return max(0.0, min(1.0, base_confidence))
    
#     def _create_error_result(self, error_message: str) -> TextExtractionResult:
#         """Create error result object"""
#         return TextExtractionResult(
#             success=False,
#             extracted_text="",
#             page_count=0,
#             total_characters=0,
#             processing_time=0.0,
#             extraction_confidence=0.0,
#             pages_with_text=0,
#             pages_empty=0,
#             tables_detected=0,
#             images_detected=0,
#             links_extracted=[],
#             method_used="text_extractor",
#             memory_peak_mb=0.0,
#             chunks_processed=0,
#             warnings=[],
#             errors=[error_message],
#             page_details=[]
#         )
    
#     def extract_text_streaming(self, file_path: Path, analysis: PDFAnalysisResult,
#                               chunk_size: int = 5) -> Iterator[str]:
#         """Stream text extraction for very large files"""
#         try:
#             doc = fitz.open(str(file_path))
            
#             for chunk_start in range(0, analysis.page_count, chunk_size):
#                 chunk_end = min(chunk_start + chunk_size, analysis.page_count)
                
#                 chunk_text_parts = []
#                 for page_num in range(chunk_start, chunk_end):
#                     if page_num >= doc.page_count:
#                         break
                    
#                     page = doc[page_num]
#                     page_text = page.get_text()
#                     page_text = self._clean_text(page_text)
                    
#                     if page_text.strip():
#                         chunk_text_parts.append(page_text)
                
#                 if chunk_text_parts:
#                     yield "\n\n".join(chunk_text_parts)
                
#                 # Memory cleanup between chunks
#                 self.memory_manager.force_garbage_collection()
            
#             doc.close()
            
#         except Exception as e:
#             self.logger.error(f"Streaming extraction failed: {str(e)}")
#             yield f"[ERROR: {str(e)}]"


# # Convenience functions for direct usage
# def extract_text_simple(file_path: Path, analysis: PDFAnalysisResult) -> str:
#     """Simple text extraction - returns just the text"""
#     extractor = TextExtractor()
#     result = extractor.extract_text(file_path, analysis, TextExtractionMode.FAST)
#     return result.extracted_text if result.success else ""


# def extract_text_with_layout(file_path: Path, analysis: PDFAnalysisResult) -> str:
#     """Text extraction with layout preservation"""
#     extractor = TextExtractor()
#     result = extractor.extract_text(file_path, analysis, TextExtractionMode.LAYOUT)
#     return result.extracted_text if result.success else ""


# if __name__ == "__main__":
#     # Test text extractor
#     import sys
#     from analyzers.pdf_analyzer import PDFAnalyzer
    
#     if len(sys.argv) != 2:
#         print("Usage: python text_extractor.py <pdf_file>")
#         sys.exit(1)
    
#     file_path = sys.argv[1]
    
#     print("=== Text Extractor Test ===")
#     print(f"File: {file_path}")
    
#     # First analyze the PDF
#     analyzer = PDFAnalyzer()
#     analysis = analyzer.analyze_pdf(file_path)
    
#     print(f"PDF Type: {analysis.pdf_type.value}")
#     print(f"Pages: {analysis.page_count}")
#     print(f"Estimated memory: {analysis.estimated_memory_mb:.1f}MB")
    
#     if analysis.pdf_type in [PDFType.TEXT_BASED, PDFType.MIXED]:
#         # Extract text
#         extractor = TextExtractor()
#         result = extractor.extract_text(file_path, analysis, TextExtractionMode.FAST)
        
#         print(f"\n=== Extraction Results ===")
#         print(f"Success: {result.success}")
#         print(f"Characters: {result.total_characters}")
#         print(f"Pages with text: {result.pages_with_text}")
#         print(f"Processing time: {result.processing_time:.2f}s")
#         print(f"Confidence: {result.extraction_confidence:.2f}")
        
#         if result.extracted_text:
#             preview = result.extracted_text[:500] + "..." if len(result.extracted_text) > 500 else result.extracted_text
#             print(f"\n=== Text Preview ===")
#             print(preview)
        
#         if result.warnings:
#             print(f"\n=== Warnings ===")
#             for warning in result.warnings:
#                 print(f"  - {warning}")
        
#         if result.errors:
#             print(f"\n=== Errors ===")
#             for error in result.errors:
#                 print(f"  - {error}")
"""
Text Extractor - Direct PDF Text Extraction using PyMuPDF
High-speed extraction for text-based PDFs with improved memory management
"""

import fitz  # PyMuPDF
import time
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re

import sys

# Ensure the root directory is on the import path
# sys.path.append(str(Path(__file__).resolve().parent.parent)) # Removed for packaging

from pdf2text.config import get_config
from pdf2text.core.memory_manager import get_memory_manager, ManagedResource, CleanupPriority, MemoryContext
from pdf2text.analyzers.pdf_analyzer import PDFAnalysisResult, PDFType, ContentComplexity


class TextExtractionMode(Enum):
    """Text extraction modes"""
    FAST = "fast"           # Basic text extraction
    LAYOUT = "layout"       # Preserve formatting and layout
    STRUCTURED = "structured"  # Extract with structure (tables, lists)


@dataclass
class TextExtractionResult:
    """Results from text extraction"""
    success: bool
    extracted_text: str
    page_count: int
    total_characters: int
    processing_time: float
    
    # Quality metrics
    extraction_confidence: float    # 0.0 - 1.0
    pages_with_text: int
    pages_empty: int
    
    # Structure information
    tables_detected: int
    images_detected: int
    links_extracted: List[str]
    
    # Processing details
    method_used: str
    memory_peak_mb: float
    chunks_processed: int
    
    # Issues encountered
    warnings: List[str]
    errors: List[str]
    
    # Page-by-page breakdown
    page_details: List[Dict]


class TextExtractor:
    """High-performance direct text extraction from PDFs with improved memory management"""
    
    def __init__(self):
        self.config = get_config()
        self.memory_manager = get_memory_manager()
        self.logger = logging.getLogger(__name__)
        
        # Extraction parameters
        self.extraction_flags = {
            TextExtractionMode.FAST: fitz.TEXT_PRESERVE_WHITESPACE,
            TextExtractionMode.LAYOUT: fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES,
            TextExtractionMode.STRUCTURED: fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_SPANS
        }
        
        # Text cleaning patterns
        self.cleanup_patterns = [
            (r'\x00', ''),           # Remove null characters
            (r'\r\n', '\n'),         # Normalize line endings
            (r'\r', '\n'),           # Normalize line endings
            (r'\n{3,}', '\n\n'),     # Reduce excessive line breaks
            (r'[ \t]{2,}', ' '),     # Reduce excessive spaces
            (r'^\s+$', '', re.MULTILINE),  # Remove empty lines with whitespace
        ]
        
        self.logger.info("Text Extractor initialized")
    
    def extract_text(self, file_path: Path, 
                    analysis: PDFAnalysisResult,
                    mode: TextExtractionMode = TextExtractionMode.FAST,
                    chunk_size: Optional[int] = None) -> TextExtractionResult:
        """
        Main text extraction method with improved memory handling and error recovery
        
        Args:
            file_path: Path to PDF file
            analysis: PDF analysis results from Step 2
            mode: Extraction mode (fast/layout/structured)
            chunk_size: Pages per chunk (None = use analysis recommendation)
            
        Returns:
            TextExtractionResult with extracted text and metadata
        """
        start_time = time.time()
        
        # Validate inputs
        if analysis.pdf_type not in [PDFType.TEXT_BASED, PDFType.MIXED]:
            return self._create_error_result(
                f"Text extractor cannot handle {analysis.pdf_type.value} PDFs"
            )
        
        # Initialize result
        result = TextExtractionResult(
            success=False,
            extracted_text="",
            page_count=analysis.page_count,
            total_characters=0,
            processing_time=0.0,
            extraction_confidence=0.0,
            pages_with_text=0,
            pages_empty=0,
            tables_detected=0,
            images_detected=0,
            links_extracted=[],
            method_used=f"pymupdf_{mode.value}",
            memory_peak_mb=0.0,
            chunks_processed=0,
            warnings=[],
            errors=[],
            page_details=[]
        )
        
        # Smart chunk size determination with memory awareness
        chunk_size = self._determine_optimal_chunk_size(analysis, chunk_size)
        
        try:
            # Check memory availability before starting
            memory_needed = self._calculate_conservative_memory_need(analysis, chunk_size)
            
            if not self._check_memory_availability(memory_needed, result):
                return result
            
            # Try processing with gradually smaller chunks if needed
            success = False
            attempts = 0
            max_attempts = 3
            current_chunk_size = chunk_size
            
            while not success and attempts < max_attempts:
                attempts += 1
                
                try:
                    self.logger.info(f"Attempt {attempts}: Processing with chunk size {current_chunk_size}")
                    
                    # Memory-managed extraction with current chunk size
                    with MemoryContext(self.memory_manager, "text_extraction", 
                                     memory_needed) as ctx:
                        
                        if current_chunk_size >= analysis.page_count:
                            # Process entire document at once
                            result = self._extract_full_document(file_path, analysis, mode, result)
                        else:
                            # Process in chunks
                            result = self._extract_chunked(file_path, analysis, mode, current_chunk_size, result)
                        
                        success = result.success
                        
                except MemoryError as e:
                    # Memory error - try smaller chunks
                    result.warnings.append(f"Attempt {attempts} failed due to memory: {str(e)}")
                    current_chunk_size = max(1, current_chunk_size // 2)
                    memory_needed = memory_needed // 2
                    
                    if attempts < max_attempts:
                        self.logger.warning(f"Memory error, reducing chunk size to {current_chunk_size}")
                        # Clear any partial results
                        result.extracted_text = ""
                        result.total_characters = 0
                        result.page_details = []
                        continue
                    else:
                        result.errors.append("Failed after multiple attempts due to memory constraints")
                        break
                        
                except Exception as e:
                    # Other errors - try once more with minimal settings
                    if attempts == 1:
                        result.warnings.append(f"First attempt failed: {str(e)}, trying conservative approach")
                        current_chunk_size = 1  # Most conservative
                        mode = TextExtractionMode.FAST  # Fastest mode
                        continue
                    else:
                        result.errors.append(f"Processing failed: {str(e)}")
                        break
            
            # Final processing if successful
            if success:
                result.processing_time = time.time() - start_time
                result.extraction_confidence = self._calculate_confidence(result)
                
                # Get memory peak safely
                try:
                    snapshot = self.memory_manager.get_memory_snapshot()
                    result.memory_peak_mb = snapshot.process_mb
                except:
                    result.memory_peak_mb = 0.0
                
                self.logger.info(
                    f"Text extraction completed: {result.total_characters} chars "
                    f"from {result.page_count} pages in {result.processing_time:.2f}s "
                    f"(chunk size: {current_chunk_size})"
                )
            else:
                result.processing_time = time.time() - start_time
                result.errors.append("All extraction attempts failed")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Text extraction failed with unexpected error: {str(e)}")
            result.success = False
            result.errors.append(f"Unexpected error: {str(e)}")
            result.processing_time = time.time() - start_time
            return result
    
    def _determine_optimal_chunk_size(self, analysis: PDFAnalysisResult, 
                                     requested_chunk_size: Optional[int]) -> int:
        """Determine optimal chunk size based on system resources and PDF characteristics"""
        
        # Start with requested size or analysis recommendation
        if requested_chunk_size is not None:
            base_chunk_size = requested_chunk_size
        else:
            base_chunk_size = analysis.recommended_chunk_size
        
        # Get current system memory state
        try:
            snapshot = self.memory_manager.get_memory_snapshot()
            available_mb = snapshot.available_mb
            
            # Adjust chunk size based on available memory
            if available_mb < 200:  # Low memory system
                optimal_chunk_size = min(base_chunk_size, 1)
            elif available_mb < 500:  # Medium memory system
                optimal_chunk_size = min(base_chunk_size, 3)
            else:  # High memory system
                optimal_chunk_size = base_chunk_size
                
            # Additional adjustments based on PDF characteristics
            if analysis.complexity.value == "complex":
                optimal_chunk_size = max(1, optimal_chunk_size // 2)
            
            if analysis.file_size_mb > 50:  # Large file
                optimal_chunk_size = max(1, optimal_chunk_size // 2)
                
            self.logger.debug(f"Optimal chunk size: {optimal_chunk_size} (base: {base_chunk_size}, available: {available_mb:.0f}MB)")
            return optimal_chunk_size
            
        except Exception as e:
            self.logger.warning(f"Could not determine optimal chunk size: {e}")
            return max(1, base_chunk_size)
    
    def _calculate_conservative_memory_need(self, analysis: PDFAnalysisResult, 
                                          chunk_size: int) -> float:
        """Calculate conservative memory estimate for processing"""
        
        # Base memory per page (MB)
        if analysis.pdf_type == PDFType.TEXT_BASED:
            mb_per_page = 0.5
        else:  # MIXED
            mb_per_page = 1.0  # More conservative for mixed content
        
        # Memory for current chunk
        chunk_memory = chunk_size * mb_per_page
        
        # Add fixed overhead
        fixed_overhead = 30  # MB for libraries, buffers, etc.
        
        # Add safety buffer (smaller than before)
        safety_buffer = chunk_memory * 0.2  # 20% instead of 50%
        
        total_memory = chunk_memory + fixed_overhead + safety_buffer
        
        self.logger.debug(f"Memory estimate: {total_memory:.1f}MB (chunk: {chunk_memory:.1f}MB, overhead: {fixed_overhead}MB, buffer: {safety_buffer:.1f}MB)")
        
        return total_memory
    
    def _check_memory_availability(self, memory_needed: float, 
                                  result: TextExtractionResult) -> bool:
        """Check if enough memory is available for processing"""
        
        try:
            snapshot = self.memory_manager.get_memory_snapshot()
            available_mb = snapshot.available_mb
            
            # Check if we have enough memory with some buffer
            if memory_needed > available_mb * 0.8:  # Use max 80% of available
                result.errors.append(
                    f"Insufficient memory: need {memory_needed:.1f}MB, "
                    f"available {available_mb:.1f}MB"
                )
                result.warnings.append(
                    "Try: 1) Close other applications, "
                    "2) Use --memory-limit with lower value, "
                    "3) Process on system with more RAM"
                )
                return False
            
            # Warn if memory usage will be high
            if memory_needed > available_mb * 0.6:
                result.warnings.append(
                    f"High memory usage expected: {memory_needed:.1f}MB "
                    f"({memory_needed/available_mb*100:.1f}% of available)"
                )
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not check memory availability: {e}")
            return True  # Assume it's OK if we can't check
    
    def _extract_full_document(self, file_path: Path, analysis: PDFAnalysisResult,
                              mode: TextExtractionMode, result: TextExtractionResult) -> TextExtractionResult:
        """Extract text from entire document with better resource management"""
        try:
            doc = fitz.open(str(file_path))
            
            # Register document as managed resource with automatic cleanup
            doc_resource = ManagedResource(
                resource_id=f"pdf_doc_{file_path.stem}_{int(time.time())}",
                resource_type="pdf_document",
                size_mb=analysis.file_size_mb,
                created_at=time.time(),
                last_accessed=time.time(),
                cleanup_callback=lambda: doc.close() if not doc.is_closed else None,
                priority=CleanupPriority.HIGH
            )
            self.memory_manager.register_resource(doc_resource)
            
            try:
                text_parts = []
                links_found = []
                processed_pages = 0
                
                for page_num in range(doc.page_count):
                    try:
                        page = doc[page_num]
                        
                        # Extract text based on mode
                        page_text = self._extract_page_text(page, mode)
                        
                        # Track page details
                        page_info = {
                            'page_number': page_num + 1,
                            'character_count': len(page_text),
                            'has_text': len(page_text.strip()) > 0,
                            'has_images': len(page.get_images()) > 0,
                            'has_links': len(page.get_links()) > 0
                        }
                        
                        result.page_details.append(page_info)
                        
                        # Update counters
                        if page_info['has_text']:
                            result.pages_with_text += 1
                            text_parts.append(page_text)
                        else:
                            result.pages_empty += 1
                        
                        if page_info['has_images']:
                            result.images_detected += len(page.get_images())
                        
                        if page_info['has_links']:
                            page_links = [link.get('uri', '') for link in page.get_links() 
                                        if link.get('uri')]
                            links_found.extend(page_links)
                        
                        processed_pages += 1
                        
                        # Periodic memory check and cleanup
                        if page_num % 50 == 0:  # Every 50 pages
                            self.memory_manager.update_resource_access(doc_resource.resource_id)
                            snapshot = self.memory_manager.get_memory_snapshot()
                            if snapshot.percentage > 75:
                                self.memory_manager.force_garbage_collection()
                    
                    except Exception as e:
                        result.warnings.append(f"Failed to process page {page_num + 1}: {str(e)}")
                        continue
                
                # Combine all text
                result.extracted_text = self._combine_page_texts(text_parts, mode)
                result.total_characters = len(result.extracted_text)
                result.links_extracted = list(set(links_found))
                result.chunks_processed = 1
                result.success = processed_pages > 0
                
                if processed_pages < doc.page_count:
                    result.warnings.append(f"Only processed {processed_pages}/{doc.page_count} pages")
                
            finally:
                # Cleanup resources
                self.memory_manager.unregister_resource(doc_resource.resource_id)
                if not doc.is_closed:
                    doc.close()
            
            return result
            
        except Exception as e:
            result.errors.append(f"Full document extraction failed: {str(e)}")
            return result
    
    def _extract_chunked(self, file_path: Path, analysis: PDFAnalysisResult,
                        mode: TextExtractionMode, chunk_size: int, 
                        result: TextExtractionResult) -> TextExtractionResult:
        """Extract text in chunks with improved error handling and progress tracking"""
        text_parts = []
        links_found = []
        chunk_count = 0
        failed_chunks = 0
        
        total_chunks = (analysis.page_count + chunk_size - 1) // chunk_size
        
        for chunk_start in range(0, analysis.page_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, analysis.page_count)
            chunk_count += 1
            
            try:
                self.logger.debug(f"Processing chunk {chunk_count}/{total_chunks}: pages {chunk_start+1}-{chunk_end}")
                
                # Process chunk with memory management
                chunk_text, chunk_info = self._extract_chunk(
                    file_path, chunk_start, chunk_end, mode
                )
                
                if chunk_text and len(chunk_text.strip()) > 0:
                    text_parts.append(chunk_text)
                
                # Merge chunk information
                result.page_details.extend(chunk_info['page_details'])
                result.pages_with_text += chunk_info['pages_with_text']
                result.pages_empty += chunk_info['pages_empty']
                result.images_detected += chunk_info['images_detected']
                links_found.extend(chunk_info['links_found'])
                
                # Aggressive memory cleanup between chunks
                self.memory_manager.force_garbage_collection()
                
                # Check memory state and warn if getting high
                snapshot = self.memory_manager.get_memory_snapshot()
                if snapshot.percentage > 80:
                    result.warnings.append(f"High memory usage during chunk {chunk_count}: {snapshot.percentage:.1f}%")
                    # Force additional cleanup
                    self.memory_manager._trigger_cleanup(CleanupPriority.MEDIUM)
                
            except Exception as e:
                failed_chunks += 1
                error_msg = f"Chunk {chunk_start+1}-{chunk_end} failed: {str(e)}"
                result.warnings.append(error_msg)
                self.logger.warning(error_msg)
                
                # If too many chunks fail, abort
                if failed_chunks > total_chunks * 0.3:  # More than 30% failed
                    result.errors.append(f"Too many chunks failed ({failed_chunks}/{chunk_count}), aborting")
                    break
                
                continue
        
        # Combine results
        if text_parts:
            result.extracted_text = self._combine_page_texts(text_parts, mode)
            result.total_characters = len(result.extracted_text)
            result.success = True
        else:
            result.errors.append("No text could be extracted from any chunks")
        
        result.links_extracted = list(set(links_found))
        result.chunks_processed = chunk_count - failed_chunks
        
        if failed_chunks > 0:
            result.warnings.append(f"{failed_chunks} chunks failed during processing")
        
        return result
    
    def _extract_chunk(self, file_path: Path, start_page: int, end_page: int,
                      mode: TextExtractionMode) -> Tuple[str, Dict]:
        """Extract text from a specific page range"""
        doc = None
        try:
            doc = fitz.open(str(file_path))
            
            text_parts = []
            chunk_info = {
                'page_details': [],
                'pages_with_text': 0,
                'pages_empty': 0,
                'images_detected': 0,
                'links_found': []
            }
            
            for page_num in range(start_page, end_page):
                if page_num >= doc.page_count:
                    break
                
                page = doc[page_num]
                page_text = self._extract_page_text(page, mode)
                
                # Track page details
                page_info = {
                    'page_number': page_num + 1,
                    'character_count': len(page_text),
                    'has_text': len(page_text.strip()) > 0,
                    'has_images': len(page.get_images()) > 0,
                    'has_links': len(page.get_links()) > 0
                }
                
                chunk_info['page_details'].append(page_info)
                
                if page_info['has_text']:
                    chunk_info['pages_with_text'] += 1
                    text_parts.append(page_text)
                else:
                    chunk_info['pages_empty'] += 1
                
                if page_info['has_images']:
                    chunk_info['images_detected'] += len(page.get_images())
                
                if page_info['has_links']:
                    page_links = [link.get('uri', '') for link in page.get_links() 
                                if link.get('uri')]
                    chunk_info['links_found'].extend(page_links)
            
            chunk_text = self._combine_page_texts(text_parts, mode)
            return chunk_text, chunk_info
            
        finally:
            if doc:
                doc.close()
    
    def _extract_page_text(self, page: fitz.Page, mode: TextExtractionMode) -> str:
        """Extract text from a single page based on mode"""
        try:
            flags = self.extraction_flags.get(mode, fitz.TEXT_PRESERVE_WHITESPACE)
            
            if mode == TextExtractionMode.STRUCTURED:
                # Enhanced extraction with structure preservation
                text = self._extract_structured_text(page)
            else:
                # Standard extraction
                text = page.get_text(flags=flags)
            
            # Clean up text
            return self._clean_text(text)
            
        except Exception as e:
            self.logger.warning(f"Page text extraction failed: {str(e)}")
            return ""
    
    def _extract_structured_text(self, page: fitz.Page) -> str:
        """Extract text with enhanced structure preservation"""
        try:
            # Get text with detailed structure
            text_dict = page.get_text("dict")
            structured_text = []
            
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_text = self._process_text_block(block)
                    if block_text:
                        structured_text.append(block_text)
                elif block.get("type") == 1:  # Image block
                    structured_text.append("[IMAGE]")
            
            return "\n\n".join(structured_text)
            
        except Exception as e:
            self.logger.warning(f"Structured text extraction failed: {str(e)}")
            # Fallback to basic extraction
            return page.get_text()
    
    def _process_text_block(self, block: Dict) -> str:
        """Process a text block with formatting preservation"""
        lines = []
        
        for line in block.get("lines", []):
            line_text_parts = []
            
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    line_text_parts.append(text)
            
            if line_text_parts:
                lines.append(" ".join(line_text_parts))
        
        return "\n".join(lines)
    
    def _combine_page_texts(self, text_parts: List[str], mode: TextExtractionMode) -> str:
        """Combine text from multiple pages"""
        if not text_parts:
            return ""
        
        if mode == TextExtractionMode.LAYOUT:
            # Preserve page breaks
            return "\n\n[PAGE BREAK]\n\n".join(text_parts)
        else:
            # Simple concatenation
            return "\n\n".join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        cleaned = text
        
        # Apply cleanup patterns
        for pattern, replacement, *flags in self.cleanup_patterns:
            flag = flags[0] if flags else 0
            cleaned = re.sub(pattern, replacement, cleaned, flags=flag)
        
        # Remove excessive whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _calculate_confidence(self, result: TextExtractionResult) -> float:
        """Calculate extraction confidence score"""
        if result.page_count == 0:
            return 0.0
        
        # Base confidence from successful page extraction
        page_success_rate = result.pages_with_text / result.page_count
        base_confidence = page_success_rate * 0.8  # Max 0.8 from page success
        
        # Bonus for text density
        if result.total_characters > 0:
            avg_chars_per_page = result.total_characters / result.page_count
            if avg_chars_per_page > 100:  # Reasonable amount of text
                base_confidence += 0.1
            if avg_chars_per_page > 500:  # Good amount of text
                base_confidence += 0.1
        
        # Penalty for warnings/errors
        if result.warnings:
            base_confidence -= len(result.warnings) * 0.05
        if result.errors:
            base_confidence -= len(result.errors) * 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _create_error_result(self, error_message: str) -> TextExtractionResult:
        """Create error result object"""
        return TextExtractionResult(
            success=False,
            extracted_text="",
            page_count=0,
            total_characters=0,
            processing_time=0.0,
            extraction_confidence=0.0,
            pages_with_text=0,
            pages_empty=0,
            tables_detected=0,
            images_detected=0,
            links_extracted=[],
            method_used="text_extractor",
            memory_peak_mb=0.0,
            chunks_processed=0,
            warnings=[],
            errors=[error_message],
            page_details=[]
        )
    
    def extract_text_streaming(self, file_path: Path, analysis: PDFAnalysisResult,
                              chunk_size: int = 5) -> Iterator[str]:
        """Stream text extraction for very large files"""
        try:
            doc = fitz.open(str(file_path))
            
            for chunk_start in range(0, analysis.page_count, chunk_size):
                chunk_end = min(chunk_start + chunk_size, analysis.page_count)
                
                chunk_text_parts = []
                for page_num in range(chunk_start, chunk_end):
                    if page_num >= doc.page_count:
                        break
                    
                    page = doc[page_num]
                    page_text = page.get_text()
                    page_text = self._clean_text(page_text)
                    
                    if page_text.strip():
                        chunk_text_parts.append(page_text)
                
                if chunk_text_parts:
                    yield "\n\n".join(chunk_text_parts)
                
                # Memory cleanup between chunks
                self.memory_manager.force_garbage_collection()
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Streaming extraction failed: {str(e)}")
            yield f"[ERROR: {str(e)}]"


# Convenience functions for direct usage
def extract_text_simple(file_path: Path, analysis: PDFAnalysisResult) -> str:
    """Simple text extraction - returns just the text"""
    extractor = TextExtractor()
    result = extractor.extract_text(file_path, analysis, TextExtractionMode.FAST)
    return result.extracted_text if result.success else ""


def extract_text_with_layout(file_path: Path, analysis: PDFAnalysisResult) -> str:
    """Text extraction with layout preservation"""
    extractor = TextExtractor()
    result = extractor.extract_text(file_path, analysis, TextExtractionMode.LAYOUT)
    return result.extracted_text if result.success else ""


if __name__ == "__main__":
    # Test text extractor
    import sys
    from pdf2text.analyzers.pdf_analyzer import PDFAnalyzer # Changed for packaging
    
    if len(sys.argv) != 2:
        print("Usage: python text_extractor.py <pdf_file>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    print("=== Text Extractor Test ===")
    print(f"File: {file_path}")
    
    # First analyze the PDF
    analyzer = PDFAnalyzer()
    analysis = analyzer.analyze_pdf(file_path)
    
    print(f"PDF Type: {analysis.pdf_type.value}")
    print(f"Pages: {analysis.page_count}")
    print(f"Estimated memory: {analysis.estimated_memory_mb:.1f}MB")
    
    if analysis.pdf_type in [PDFType.TEXT_BASED, PDFType.MIXED]:
        # Extract text
        extractor = TextExtractor()
        result = extractor.extract_text(file_path, analysis, TextExtractionMode.FAST)
        
        print(f"\n=== Extraction Results ===")
        print(f"Success: {result.success}")
        print(f"Characters: {result.total_characters}")
        print(f"Pages with text: {result.pages_with_text}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Confidence: {result.extraction_confidence:.2f}")
        
        if result.extracted_text:
            preview = result.extracted_text[:500] + "..." if len(result.extracted_text) > 500 else result.extracted_text
            print(f"\n=== Text Preview ===")
            print(preview)
        
        if result.warnings:
            print(f"\n=== Warnings ===")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.errors:
            print(f"\n=== Errors ===")
            for error in result.errors:
                print(f"  - {error}")
    else:
        print(f"Text extractor cannot handle {analysis.pdf_type.value} PDFs")
        print("Use OCR extractor for scanned documents")

