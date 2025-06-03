"""
OCR Extractor - Optical Character Recognition for scanned PDFs
Converts PDF pages to images and extracts text using Tesseract OCR
"""

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

from pdf2text.config import get_config
from pdf2text.core.memory_manager import get_memory_manager, ManagedResource, CleanupPriority, MemoryContext
from pdf2text.core.file_manager import get_file_manager
from pdf2text.analyzers.pdf_analyzer import PDFAnalysisResult, PDFType


class OCRQuality(Enum):
    """OCR quality levels"""
    FAST = "fast"           # Quick OCR, lower accuracy
    BALANCED = "balanced"   # Good balance of speed and accuracy
    HIGH = "high"           # Best accuracy, slower processing


class ImagePreprocessing(Enum):
    """Image preprocessing options"""
    NONE = "none"           # No preprocessing
    BASIC = "basic"         # Basic contrast/brightness
    ADVANCED = "advanced"   # Advanced noise reduction, deskewing


@dataclass
class OCRExtractionResult:
    """Results from OCR extraction"""
    success: bool
    extracted_text: str
    page_count: int
    total_characters: int
    processing_time: float
    
    # OCR specific metrics
    average_confidence: float       # Average OCR confidence (0-100)
    pages_processed: int
    pages_failed: int
    low_confidence_pages: List[int] # Pages with confidence < threshold
    
    # Image processing info
    images_processed: int
    preprocessing_applied: str
    ocr_language: str
    
    # Processing details
    method_used: str
    memory_peak_mb: float
    chunks_processed: int
    temp_files_created: int
    
    # Quality breakdown
    high_confidence_pages: int      # Confidence > 80
    medium_confidence_pages: int    # Confidence 60-80
    poor_confidence_pages: int      # Confidence < 60
    
    # Issues encountered
    warnings: List[str]
    errors: List[str]
    
    # Page-by-page results
    page_details: List[Dict]


class OCRExtractor:
    """Advanced OCR text extraction for scanned PDFs"""
    
    def __init__(self):
        self.config = get_config()
        self.memory_manager = get_memory_manager()
        self.file_manager = get_file_manager()
        self.logger = logging.getLogger(__name__)
        
        # OCR settings
        self.ocr_language = self.config.processing.ocr_language
        self.confidence_threshold = 60  # Minimum confidence for reliable text
        
        # Tesseract configuration for different quality levels
        self.tesseract_configs = {
            OCRQuality.FAST: '--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;-+()[]{}\"\'@#$%^&*_=<>/\\|`~',
            OCRQuality.BALANCED: '--oem 1 --psm 3',
            OCRQuality.HIGH: '--oem 1 --psm 1 -c preserve_interword_spaces=1'
        }
        
        # Image processing parameters
        self.image_settings = {
            'dpi': 300,              # DPI for PDF to image conversion
            'format': 'PNG',         # Image format (PNG for better quality)
            'grayscale': True,       # Convert to grayscale for OCR
            'enhance_contrast': 1.2, # Contrast enhancement factor
            'enhance_sharpness': 1.1 # Sharpness enhancement factor
        }
        
        # Validate Tesseract installation
        self._validate_tesseract()
        
        self.logger.info(f"OCR Extractor initialized with language: {self.ocr_language}")
    
    def _validate_tesseract(self):
        """Validate Tesseract OCR installation"""
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
        except Exception as e:
            self.logger.error(f"Tesseract OCR not found: {e}")
            raise RuntimeError("Tesseract OCR is required but not installed")
    
    def extract_text(self, file_path: Path, 
                    analysis: PDFAnalysisResult,
                    quality: OCRQuality = OCRQuality.BALANCED,
                    preprocessing: ImagePreprocessing = ImagePreprocessing.BASIC,
                    chunk_size: Optional[int] = None) -> OCRExtractionResult:
        """
        Main OCR extraction method
        
        Args:
            file_path: Path to PDF file
            analysis: PDF analysis results from Step 2
            quality: OCR quality level
            preprocessing: Image preprocessing level
            chunk_size: Pages per chunk (None = use analysis recommendation)
            
        Returns:
            OCRExtractionResult with extracted text and metadata
        """
        start_time = time.time()
        
        # Validate inputs
        if analysis.pdf_type not in [PDFType.SCANNED, PDFType.MIXED]:
            return self._create_error_result(
                f"OCR extractor is designed for scanned PDFs, got {analysis.pdf_type.value}"
            )
        
        # Determine chunk size (OCR needs smaller chunks due to memory usage)
        if chunk_size is None:
            chunk_size = max(1, analysis.recommended_chunk_size // 2)  # Smaller chunks for OCR
        
        # Initialize result
        result = OCRExtractionResult(
            success=False,
            extracted_text="",
            page_count=analysis.page_count,
            total_characters=0,
            processing_time=0.0,
            average_confidence=0.0,
            pages_processed=0,
            pages_failed=0,
            low_confidence_pages=[],
            images_processed=0,
            preprocessing_applied=preprocessing.value,
            ocr_language=self.ocr_language,
            method_used=f"tesseract_{quality.value}",
            memory_peak_mb=0.0,
            chunks_processed=0,
            temp_files_created=0,
            high_confidence_pages=0,
            medium_confidence_pages=0,
            poor_confidence_pages=0,
            warnings=[],
            errors=[],
            page_details=[]
        )
        
        try:
            # Memory-managed OCR extraction
            estimated_memory = analysis.estimated_memory_mb * 2  # OCR uses more memory
            
            with MemoryContext(self.memory_manager, "ocr_extraction", estimated_memory) as ctx:
                
                # Create temporary directory for image processing
                temp_dir = self.file_manager.create_temp_file(suffix="", prefix="ocr_temp_")
                temp_dir.mkdir(exist_ok=True)
                
                try:
                    if chunk_size >= analysis.page_count:
                        # Process entire document at once
                        result = self._extract_full_document(file_path, analysis, quality, 
                                                           preprocessing, temp_dir, result)
                    else:
                        # Process in chunks
                        result = self._extract_chunked(file_path, analysis, quality, 
                                                     preprocessing, chunk_size, temp_dir, result)
                    
                    # Final processing
                    result.processing_time = time.time() - start_time
                    result.average_confidence = self._calculate_average_confidence(result)
                    result.memory_peak_mb = ctx.memory_manager.get_memory_snapshot().process_mb
                    
                    self.logger.info(
                        f"OCR extraction completed: {result.total_characters} chars "
                        f"from {result.pages_processed}/{result.page_count} pages "
                        f"in {result.processing_time:.2f}s (avg confidence: {result.average_confidence:.1f}%)"
                    )
                    
                finally:
                    # Cleanup temp directory
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                
                return result
                
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            result.success = False
            result.errors.append(str(e))
            result.processing_time = time.time() - start_time
            return result
    
    def _extract_full_document(self, file_path: Path, analysis: PDFAnalysisResult,
                              quality: OCRQuality, preprocessing: ImagePreprocessing,
                              temp_dir: Path, result: OCRExtractionResult) -> OCRExtractionResult:
        """Extract text from entire document using OCR"""
        try:
            # Convert PDF to images
            self.logger.info("Converting PDF to images...")
            images = self._pdf_to_images(file_path, temp_dir)
            result.images_processed = len(images)
            result.temp_files_created = len(images)
            
            if not images:
                result.errors.append("Failed to convert PDF pages to images")
                return result
            
            text_parts = []
            confidences = []
            
            for i, image_path in enumerate(images):
                try:
                    # Process single page
                    page_text, page_confidence, page_info = self._process_single_image(
                        image_path, quality, preprocessing, i + 1
                    )
                    
                    if page_text:
                        text_parts.append(page_text)
                        confidences.append(page_confidence)
                        result.pages_processed += 1
                        
                        # Categorize by confidence
                        if page_confidence >= 80:
                            result.high_confidence_pages += 1
                        elif page_confidence >= 60:
                            result.medium_confidence_pages += 1
                        else:
                            result.poor_confidence_pages += 1
                            result.low_confidence_pages.append(i + 1)
                    else:
                        result.pages_failed += 1
                    
                    result.page_details.append(page_info)
                    
                    # Memory cleanup between pages
                    if i % 5 == 0:  # Every 5 pages
                        self.memory_manager.force_garbage_collection()
                    
                except Exception as e:
                    result.warnings.append(f"Page {i+1} OCR failed: {str(e)}")
                    result.pages_failed += 1
                    continue
            
            # Combine results
            result.extracted_text = "\n\n".join(text_parts)
            result.total_characters = len(result.extracted_text)
            result.chunks_processed = 1
            result.success = result.pages_processed > 0
            
            return result
            
        except Exception as e:
            result.errors.append(f"Full document OCR failed: {str(e)}")
            return result
    
    def _extract_chunked(self, file_path: Path, analysis: PDFAnalysisResult,
                        quality: OCRQuality, preprocessing: ImagePreprocessing,
                        chunk_size: int, temp_dir: Path, 
                        result: OCRExtractionResult) -> OCRExtractionResult:
        """Extract text in chunks for memory efficiency"""
        text_parts = []
        chunk_count = 0
        
        for chunk_start in range(0, analysis.page_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, analysis.page_count)
            
            try:
                self.logger.info(f"Processing OCR chunk: pages {chunk_start+1}-{chunk_end}")
                
                # Convert chunk to images
                chunk_images = self._pdf_to_images_range(file_path, chunk_start, chunk_end, temp_dir)
                result.images_processed += len(chunk_images)
                result.temp_files_created += len(chunk_images)
                
                # Process chunk images
                chunk_text, chunk_info = self._process_image_chunk(
                    chunk_images, quality, preprocessing, chunk_start
                )
                
                if chunk_text:
                    text_parts.append(chunk_text)
                
                # Merge chunk information
                result.page_details.extend(chunk_info['page_details'])
                result.pages_processed += chunk_info['pages_processed']
                result.pages_failed += chunk_info['pages_failed']
                result.high_confidence_pages += chunk_info['high_confidence_pages']
                result.medium_confidence_pages += chunk_info['medium_confidence_pages']
                result.poor_confidence_pages += chunk_info['poor_confidence_pages']
                result.low_confidence_pages.extend(chunk_info['low_confidence_pages'])
                
                chunk_count += 1
                
                # Cleanup chunk images to free memory
                for img_path in chunk_images:
                    try:
                        img_path.unlink()
                    except:
                        pass
                
                # Force memory cleanup
                self.memory_manager.force_garbage_collection()
                
            except Exception as e:
                result.warnings.append(f"Chunk {chunk_start+1}-{chunk_end} failed: {str(e)}")
                continue
        
        # Combine all chunks
        result.extracted_text = "\n\n".join(text_parts)
        result.total_characters = len(result.extracted_text)
        result.chunks_processed = chunk_count
        result.success = chunk_count > 0 and result.pages_processed > 0
        
        return result
    
    def _pdf_to_images(self, file_path: Path, temp_dir: Path) -> List[Path]:
        """Convert entire PDF to images"""
        try:
            images = convert_from_path(
                str(file_path),
                dpi=self.image_settings['dpi'],
                output_folder=str(temp_dir),
                fmt=self.image_settings['format'].lower(),
                grayscale=self.image_settings['grayscale']
            )
            
            image_paths = []
            for i, image in enumerate(images):
                image_path = temp_dir / f"page_{i+1:04d}.png"
                image.save(image_path, self.image_settings['format'])
                image_paths.append(image_path)
            
            self.logger.info(f"Converted {len(image_paths)} pages to images")
            return image_paths
            
        except Exception as e:
            self.logger.error(f"PDF to image conversion failed: {e}")
            return []
    
    def _pdf_to_images_range(self, file_path: Path, start_page: int, end_page: int,
                           temp_dir: Path) -> List[Path]:
        """Convert specific page range to images"""
        try:
            # pdf2image uses 1-based page numbering
            first_page = start_page + 1
            last_page = end_page
            
            images = convert_from_path(
                str(file_path),
                dpi=self.image_settings['dpi'],
                output_folder=str(temp_dir),
                fmt=self.image_settings['format'].lower(),
                grayscale=self.image_settings['grayscale'],
                first_page=first_page,
                last_page=last_page
            )
            
            image_paths = []
            for i, image in enumerate(images):
                page_num = start_page + i + 1
                image_path = temp_dir / f"page_{page_num:04d}.png"
                image.save(image_path, self.image_settings['format'])
                image_paths.append(image_path)
            
            return image_paths
            
        except Exception as e:
            self.logger.error(f"PDF chunk to image conversion failed: {e}")
            return []
    
    def _process_single_image(self, image_path: Path, quality: OCRQuality,
                             preprocessing: ImagePreprocessing, 
                             page_number: int) -> Tuple[str, float, Dict]:
        """Process a single image with OCR"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = self._preprocess_image(image, preprocessing)
            
            # OCR configuration
            config = self.tesseract_configs[quality]
            
            # Extract text with confidence
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                lang=self.ocr_language,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            text_parts = []
            confidences = []
            
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    text_parts.append(text)
                    confidences.append(int(ocr_data['conf'][i]))
            
            # Calculate page text and confidence
            page_text = ' '.join(text_parts)
            page_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Clean text
            page_text = self._clean_ocr_text(page_text)
            
            # Page information
            page_info = {
                'page_number': page_number,
                'character_count': len(page_text),
                'confidence': page_confidence,
                'words_detected': len(text_parts),
                'preprocessing_applied': preprocessing.value,
                'has_text': len(page_text.strip()) > 0
            }
            
            return page_text, page_confidence, page_info
            
        except Exception as e:
            self.logger.warning(f"OCR processing failed for page {page_number}: {e}")
            return "", 0.0, {
                'page_number': page_number,
                'character_count': 0,
                'confidence': 0.0,
                'words_detected': 0,
                'preprocessing_applied': preprocessing.value,
                'has_text': False,
                'error': str(e)
            }
    
    def _process_image_chunk(self, image_paths: List[Path], quality: OCRQuality,
                           preprocessing: ImagePreprocessing, 
                           start_page: int) -> Tuple[str, Dict]:
        """Process a chunk of images"""
        text_parts = []
        chunk_info = {
            'page_details': [],
            'pages_processed': 0,
            'pages_failed': 0,
            'high_confidence_pages': 0,
            'medium_confidence_pages': 0,
            'poor_confidence_pages': 0,
            'low_confidence_pages': []
        }
        
        for i, image_path in enumerate(image_paths):
            page_number = start_page + i + 1
            
            page_text, page_confidence, page_info = self._process_single_image(
                image_path, quality, preprocessing, page_number
            )
            
            if page_text:
                text_parts.append(page_text)
                chunk_info['pages_processed'] += 1
                
                # Categorize by confidence
                if page_confidence >= 80:
                    chunk_info['high_confidence_pages'] += 1
                elif page_confidence >= 60:
                    chunk_info['medium_confidence_pages'] += 1
                else:
                    chunk_info['poor_confidence_pages'] += 1
                    chunk_info['low_confidence_pages'].append(page_number)
            else:
                chunk_info['pages_failed'] += 1
            
            chunk_info['page_details'].append(page_info)
        
        chunk_text = "\n\n".join(text_parts)
        return chunk_text, chunk_info
    
    def _preprocess_image(self, image: Image.Image, 
                         preprocessing: ImagePreprocessing) -> Image.Image:
        """Apply image preprocessing for better OCR results"""
        if preprocessing == ImagePreprocessing.NONE:
            return image
        
        processed = image.copy()
        
        if preprocessing in [ImagePreprocessing.BASIC, ImagePreprocessing.ADVANCED]:
            # Basic enhancements
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(self.image_settings['enhance_contrast'])
            
            enhancer = ImageEnhance.Sharpness(processed)
            processed = enhancer.enhance(self.image_settings['enhance_sharpness'])
        
        if preprocessing == ImagePreprocessing.ADVANCED:
            # Advanced preprocessing
            processed = processed.filter(ImageFilter.MedianFilter(size=3))  # Noise reduction
            
            # Convert to grayscale if not already
            if processed.mode != 'L':
                processed = processed.convert('L')
            
            # Threshold to pure black and white
            threshold = 128
            processed = processed.point(lambda x: 0 if x < threshold else 255, '1')
        
        return processed
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR extracted text"""
        if not text:
            return ""
        
        # OCR-specific cleaning
        cleaned = text
        
        # Fix common OCR errors
        ocr_fixes = [
            (r'\s+', ' '),           # Multiple spaces to single space
            (r'\n\s*\n', '\n'),      # Multiple newlines to single
            (r'[|!1Il]{2,}', '||'),  # Common OCR confusion
            (r'rn', 'm'),            # Common OCR error
            (r'[^\w\s\.,!?:;()\[\]{}\'"@#$%^&*_=+<>/\\|-]', ''),  # Remove weird characters
        ]
        
        for pattern, replacement in ocr_fixes:
            import re
            cleaned = re.sub(pattern, replacement, cleaned)
        
        return cleaned.strip()
    
    def _calculate_average_confidence(self, result: OCRExtractionResult) -> float:
        """Calculate average OCR confidence across all pages"""
        if not result.page_details:
            return 0.0
        
        confidences = [
            page.get('confidence', 0) for page in result.page_details 
            if 'confidence' in page
        ]
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _create_error_result(self, error_message: str) -> OCRExtractionResult:
        """Create error result object"""
        return OCRExtractionResult(
            success=False,
            extracted_text="",
            page_count=0,
            total_characters=0,
            processing_time=0.0,
            average_confidence=0.0,
            pages_processed=0,
            pages_failed=0,
            low_confidence_pages=[],
            images_processed=0,
            preprocessing_applied="none",
            ocr_language=self.ocr_language,
            method_used="ocr_extractor",
            memory_peak_mb=0.0,
            chunks_processed=0,
            temp_files_created=0,
            high_confidence_pages=0,
            medium_confidence_pages=0,
            poor_confidence_pages=0,
            warnings=[],
            errors=[error_message],
            page_details=[]
        )


# Convenience functions
def extract_text_ocr_simple(file_path: Path, analysis: PDFAnalysisResult) -> str:
    """Simple OCR extraction - returns just the text"""
    extractor = OCRExtractor()
    result = extractor.extract_text(file_path, analysis, OCRQuality.BALANCED)
    return result.extracted_text if result.success else ""


def extract_text_ocr_high_quality(file_path: Path, analysis: PDFAnalysisResult) -> str:
    """High quality OCR extraction"""
    extractor = OCRExtractor()
    result = extractor.extract_text(file_path, analysis, OCRQuality.HIGH, 
                                   ImagePreprocessing.ADVANCED)
    return result.extracted_text if result.success else ""


if __name__ == "__main__":
    # Test OCR extractor
    import sys
    from pdf2text.analyzers.pdf_analyzer import PDFAnalyzer # Changed for packaging
    
    if len(sys.argv) != 2:
        print("Usage: python ocr_extractor.py <pdf_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("=== OCR Extractor Test ===")
    print(f"File: {file_path}")
    
    # First analyze the PDF
    analyzer = PDFAnalyzer()
    analysis = analyzer.analyze_pdf(file_path)
    
    print(f"PDF Type: {analysis.pdf_type.value}")
    print(f"Pages: {analysis.page_count}")
    
    if analysis.pdf_type in [PDFType.SCANNED, PDFType.MIXED]:
        # Extract text with OCR
        extractor = OCRExtractor()
        result = extractor.extract_text(file_path, analysis, OCRQuality.BALANCED)
        
        print(f"\n=== OCR Results ===")
        print(f"Success: {result.success}")
        print(f"Characters: {result.total_characters}")
        print(f"Pages processed: {result.pages_processed}/{result.page_count}")
        print(f"Average confidence: {result.average_confidence:.1f}%")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"High confidence pages: {result.high_confidence_pages}")
        print(f"Low confidence pages: {len(result.low_confidence_pages)}")
        
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
        print(f"OCR extractor is designed for scanned PDFs, got {analysis.pdf_type.value}")
        print("Use text extractor for text-based documents")