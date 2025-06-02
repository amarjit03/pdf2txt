"""
Hybrid Extractor - Intelligent combination of text and OCR extraction
Automatically chooses the best method for each page based on content analysis
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from config import get_config
from core.memory_manager import get_memory_manager, MemoryContext
from analyzers.pdf_analyzer import PDFAnalysisResult, PDFType, ContentComplexity
# from .text_extractor import TextExtractor, TextExtractionMode, TextExtractionResult
# from .ocr_extractor import OCRExtractor, OCRQuality, ImagePreprocessing, OCRExtractionResult
from .text_extractors import TextExtractor, TextExtractionMode, TextExtractionResult
from .ocr_extractors import OCRExtractor, OCRQuality, ImagePreprocessing, OCRExtractionResult


class HybridStrategy(Enum):
    """Hybrid extraction strategies"""
    AUTO = "auto"                   # Automatically choose best method per page
    TEXT_FIRST = "text_first"       # Try text extraction first, OCR as fallback
    OCR_FIRST = "ocr_first"         # Try OCR first, text extraction as fallback
    PARALLEL = "parallel"           # Run both methods and choose best result
    QUALITY_FOCUSED = "quality"     # Prioritize highest quality result


@dataclass
class HybridExtractionResult:
    """Results from hybrid extraction combining both methods"""
    success: bool
    extracted_text: str
    page_count: int
    total_characters: int
    processing_time: float
    
    # Method breakdown
    text_extraction_pages: int      # Pages processed with direct text extraction
    ocr_extraction_pages: int       # Pages processed with OCR
    failed_pages: int               # Pages that failed both methods
    
    # Quality metrics
    overall_confidence: float       # Combined confidence score
    text_method_confidence: float   # Average confidence for text-extracted pages
    ocr_method_confidence: float    # Average confidence for OCR pages
    
    # Processing details
    strategy_used: str
    memory_peak_mb: float
    chunks_processed: int
    method_switches: int            # How many times we switched methods
    
    # Detailed results
    text_extraction_result: Optional[TextExtractionResult]
    ocr_extraction_result: Optional[OCRExtractionResult]
    
    # Page-by-page method decisions
    page_methods: List[Dict]        # Which method was used for each page
    
    # Issues encountered
    warnings: List[str]
    errors: List[str]


class HybridExtractor:
    """Intelligent hybrid text extraction combining multiple methods"""
    
    def __init__(self):
        self.config = get_config()
        self.memory_manager = get_memory_manager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-extractors
        self.text_extractor = TextExtractor()
        self.ocr_extractor = OCRExtractor()
        
        # Decision thresholds
        self.text_density_threshold = 0.05      # Minimum text density for direct extraction
        self.min_text_length = 30               # Minimum text length to consider valid
        self.ocr_confidence_threshold = 70      # Minimum OCR confidence to prefer OCR
        self.fallback_confidence_threshold = 50 # Minimum confidence to accept fallback result
        
        self.logger.info("Hybrid Extractor initialized")
    
    def extract_text(self, file_path: Path,
                    analysis: PDFAnalysisResult,
                    strategy: HybridStrategy = HybridStrategy.AUTO,
                    chunk_size: Optional[int] = None) -> HybridExtractionResult:
        """
        Main hybrid extraction method
        
        Args:
            file_path: Path to PDF file
            analysis: PDF analysis results from Step 2
            strategy: Hybrid extraction strategy
            chunk_size: Pages per chunk (None = use analysis recommendation)
            
        Returns:
            HybridExtractionResult with combined extraction results
        """
        start_time = time.time()
        
        # Validate inputs
        if analysis.pdf_type == PDFType.TEXT_BASED:
            self.logger.info("PDF is text-based, using direct text extraction")
            return self._text_extraction_only(file_path, analysis, chunk_size, start_time)
        
        if analysis.pdf_type == PDFType.SCANNED:
            self.logger.info("PDF is scanned, using OCR extraction")
            return self._ocr_extraction_only(file_path, analysis, chunk_size, start_time)
        
        if analysis.pdf_type != PDFType.MIXED:
            return self._create_error_result(
                f"Hybrid extractor cannot handle {analysis.pdf_type.value} PDFs"
            )
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = analysis.recommended_chunk_size
        
        # Initialize result
        result = HybridExtractionResult(
            success=False,
            extracted_text="",
            page_count=analysis.page_count,
            total_characters=0,
            processing_time=0.0,
            text_extraction_pages=0,
            ocr_extraction_pages=0,
            failed_pages=0,
            overall_confidence=0.0,
            text_method_confidence=0.0,
            ocr_method_confidence=0.0,
            strategy_used=strategy.value,
            memory_peak_mb=0.0,
            chunks_processed=0,
            method_switches=0,
            text_extraction_result=None,
            ocr_extraction_result=None,
            page_methods=[],
            warnings=[],
            errors=[]
        )
        
        try:
            # Memory-managed hybrid extraction
            estimated_memory = analysis.estimated_memory_mb * 1.5  # Hybrid uses more memory
            
            with MemoryContext(self.memory_manager, "hybrid_extraction", estimated_memory) as ctx:
                
                if strategy == HybridStrategy.AUTO:
                    result = self._extract_auto_strategy(file_path, analysis, chunk_size, result)
                elif strategy == HybridStrategy.TEXT_FIRST:
                    result = self._extract_text_first(file_path, analysis, chunk_size, result)
                elif strategy == HybridStrategy.OCR_FIRST:
                    result = self._extract_ocr_first(file_path, analysis, chunk_size, result)
                elif strategy == HybridStrategy.PARALLEL:
                    result = self._extract_parallel(file_path, analysis, chunk_size, result)
                elif strategy == HybridStrategy.QUALITY_FOCUSED:
                    result = self._extract_quality_focused(file_path, analysis, chunk_size, result)
                
                # Final processing
                result.processing_time = time.time() - start_time
                result.overall_confidence = self._calculate_overall_confidence(result)
                result.memory_peak_mb = ctx.memory_manager.get_memory_snapshot().process_mb
                
                self.logger.info(
                    f"Hybrid extraction completed: {result.total_characters} chars "
                    f"({result.text_extraction_pages} text + {result.ocr_extraction_pages} OCR pages) "
                    f"in {result.processing_time:.2f}s"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Hybrid extraction failed: {str(e)}")
            result.success = False
            result.errors.append(str(e))
            result.processing_time = time.time() - start_time
            return result
    
    def _extract_auto_strategy(self, file_path: Path, analysis: PDFAnalysisResult,
                              chunk_size: int, result: HybridExtractionResult) -> HybridExtractionResult:
        """Automatic strategy - analyze each page and choose best method"""
        self.logger.info("Using AUTO strategy - analyzing pages individually")
        
        # First, do a quick analysis of all pages to determine optimal approach
        page_analysis = self._analyze_pages_for_method_selection(file_path, analysis)
        
        if not page_analysis:
            result.errors.append("Failed to analyze pages for method selection")
            return result
        
        # Process pages in chunks
        text_parts = []
        chunk_count = 0
        
        for chunk_start in range(0, analysis.page_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, analysis.page_count)
            
            try:
                chunk_text, chunk_info = self._process_chunk_auto(
                    file_path, chunk_start, chunk_end, page_analysis
                )
                
                if chunk_text:
                    text_parts.append(chunk_text)
                
                # Update statistics
                result.text_extraction_pages += chunk_info['text_pages']
                result.ocr_extraction_pages += chunk_info['ocr_pages']
                result.failed_pages += chunk_info['failed_pages']
                result.method_switches += chunk_info['method_switches']
                result.page_methods.extend(chunk_info['page_methods'])
                
                chunk_count += 1
                
                # Memory cleanup between chunks  
                self.memory_manager.force_garbage_collection()
                
            except Exception as e:
                result.warnings.append(f"Chunk {chunk_start+1}-{chunk_end} failed: {str(e)}")
                continue
        
        # Combine results
        result.extracted_text = "\n\n".join(text_parts)
        result.total_characters = len(result.extracted_text)
        result.chunks_processed = chunk_count
        result.success = chunk_count > 0 and result.total_characters > 0
        
        return result
    
    def _analyze_pages_for_method_selection(self, file_path: Path, 
                                          analysis: PDFAnalysisResult) -> List[Dict]:
        """Analyze each page to determine optimal extraction method"""
        import fitz
        
        try:
            doc = fitz.open(str(file_path))
            page_analysis = []
            
            # Sample every 5th page or all pages if document is small
            sample_interval = max(1, analysis.page_count // 20)  # Sample ~20 pages max
            
            for page_num in range(0, analysis.page_count, sample_interval):
                if page_num >= doc.page_count:
                    break
                
                page = doc[page_num]
                
                # Quick text extraction test
                text = page.get_text().strip()
                text_length = len(text)
                
                # Check for images
                images = page.get_images()
                has_images = len(images) > 0
                
                # Calculate text density
                page_area = page.rect.width * page.rect.height
                text_blocks = page.get_text("dict")["blocks"]
                text_area = sum([
                    (block.get("bbox", [0, 0, 0, 0])[2] - block.get("bbox", [0, 0, 0, 0])[0]) *
                    (block.get("bbox", [0, 0, 0, 0])[3] - block.get("bbox", [0, 0, 0, 0])[1])
                    for block in text_blocks if block.get("type") == 0
                ])
                text_density = text_area / page_area if page_area > 0 else 0
                
                # Determine recommended method
                if text_length >= self.min_text_length and text_density >= self.text_density_threshold:
                    recommended_method = "text"
                elif has_images:
                    recommended_method = "ocr"
                else:
                    recommended_method = "text"  # Default fallback
                
                page_info = {
                    'page_number': page_num,
                    'text_length': text_length,
                    'text_density': text_density,
                    'has_images': has_images,
                    'recommended_method': recommended_method
                }
                
                page_analysis.append(page_info)
            
            doc.close()
            
            # Extrapolate analysis to all pages
            full_analysis = []
            for page_num in range(analysis.page_count):
                # Find closest sampled page
                closest_sample = min(page_analysis, key=lambda x: abs(x['page_number'] - page_num))
                
                # Create page info based on closest sample
                page_info = {
                    'page_number': page_num,
                    'recommended_method': closest_sample['recommended_method'],
                    'confidence': 0.8 if page_num in [p['page_number'] for p in page_analysis] else 0.6
                }
                full_analysis.append(page_info)
            
            return full_analysis
            
        except Exception as e:
            self.logger.error(f"Page analysis failed: {e}")
            return []
    
    def _process_chunk_auto(self, file_path: Path, chunk_start: int, chunk_end: int,
                           page_analysis: List[Dict]) -> Tuple[str, Dict]:
        """Process a chunk using auto-selected methods for each page"""
        text_parts = []
        chunk_info = {
            'text_pages': 0,
            'ocr_pages': 0,
            'failed_pages': 0,
            'method_switches': 0,
            'page_methods': []
        }
        
        current_method = None
        
        for page_num in range(chunk_start, chunk_end):
            if page_num >= len(page_analysis):
                continue
            
            page_info = page_analysis[page_num]
            recommended_method = page_info['recommended_method']
            
            # Track method switches
            if current_method and current_method != recommended_method:
                chunk_info['method_switches'] += 1
            current_method = recommended_method
            
            try:
                if recommended_method == 'text':
                    # Try text extraction first
                    page_text = self._extract_single_page_text(file_path, page_num)
                    
                    if page_text and len(page_text.strip()) >= self.min_text_length:
                        text_parts.append(page_text)
                        chunk_info['text_pages'] += 1
                        method_used = 'text'
                        confidence = 0.9
                    else:
                        # Fallback to OCR
                        page_text = self._extract_single_page_ocr(file_path, page_num)
                        if page_text:
                            text_parts.append(page_text)
                            chunk_info['ocr_pages'] += 1
                            method_used = 'ocr_fallback'
                            confidence = 0.7
                        else:
                            chunk_info['failed_pages'] += 1
                            method_used = 'failed'
                            confidence = 0.0
                
                else:  # recommended_method == 'ocr'
                    # Try OCR first
                    page_text = self._extract_single_page_ocr(file_path, page_num)
                    
                    if page_text and len(page_text.strip()) >= self.min_text_length:
                        text_parts.append(page_text)
                        chunk_info['ocr_pages'] += 1
                        method_used = 'ocr'
                        confidence = 0.8
                    else:
                        # Fallback to text extraction
                        page_text = self._extract_single_page_text(file_path, page_num)
                        if page_text:
                            text_parts.append(page_text)
                            chunk_info['text_pages'] += 1
                            method_used = 'text_fallback'
                            confidence = 0.6
                        else:
                            chunk_info['failed_pages'] += 1
                            method_used = 'failed'
                            confidence = 0.0
                
                # Record method decision
                chunk_info['page_methods'].append({
                    'page_number': page_num + 1,
                    'method_used': method_used,
                    'confidence': confidence,
                    'text_length': len(page_text) if 'page_text' in locals() else 0
                })
                
            except Exception as e:
                self.logger.warning(f"Page {page_num + 1} processing failed: {e}")
                chunk_info['failed_pages'] += 1
                chunk_info['page_methods'].append({
                    'page_number': page_num + 1,
                    'method_used': 'error',
                    'confidence': 0.0,
                    'text_length': 0,
                    'error': str(e)
                })
        
        chunk_text = "\n\n".join(text_parts)
        return chunk_text, chunk_info
    
    def _extract_single_page_text(self, file_path: Path, page_num: int) -> str:
        """Extract text from a single page using direct text extraction"""
        import fitz
        
        try:
            doc = fitz.open(str(file_path))
            if page_num >= doc.page_count:
                return ""
            
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            
            # Clean text
            return self._clean_text(text)
            
        except Exception as e:
            self.logger.debug(f"Single page text extraction failed for page {page_num + 1}: {e}")
            return ""
    
    def _extract_single_page_ocr(self, file_path: Path, page_num: int) -> str:
        """Extract text from a single page using OCR"""
        import fitz
        import pytesseract
        from pdf2image import convert_from_path
        
        try:
            # Convert single page to image
            images = convert_from_path(
                str(file_path),
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300
            )
            
            if not images:
                return ""
            
            # OCR the image
            text = pytesseract.image_to_string(
                images[0],
                lang=self.config.processing.ocr_language,
                config='--oem 1 --psm 3'
            )
            
            return self._clean_text(text)
            
        except Exception as e:
            self.logger.debug(f"Single page OCR failed for page {page_num + 1}: {e}")
            return ""
    
    def _extract_text_first(self, file_path: Path, analysis: PDFAnalysisResult,
                           chunk_size: int, result: HybridExtractionResult) -> HybridExtractionResult:
        """Text-first strategy - try text extraction, then OCR for failed pages"""
        self.logger.info("Using TEXT_FIRST strategy")
        
        # First attempt: full text extraction
        text_result = self.text_extractor.extract_text(
            file_path, analysis, TextExtractionMode.FAST, chunk_size
        )
        
        result.text_extraction_result = text_result
        
        if text_result.success and text_result.total_characters > 0:
            # Text extraction worked well
            result.extracted_text = text_result.extracted_text
            result.total_characters = text_result.total_characters
            result.text_extraction_pages = text_result.pages_with_text
            result.failed_pages = text_result.pages_empty
            result.text_method_confidence = text_result.extraction_confidence
            result.success = True
            
            # If some pages were empty, try OCR on those
            if text_result.pages_empty > 0:
                result.warnings.append(f"Attempting OCR on {text_result.pages_empty} pages with no text")
                # Note: This would require more complex page-by-page processing
                # For now, we'll accept the text extraction result
        else:
            # Text extraction failed, fallback to OCR
            result.warnings.append("Text extraction failed, falling back to OCR")
            ocr_result = self.ocr_extractor.extract_text(
                file_path, analysis, OCRQuality.BALANCED, chunk_size=chunk_size
            )
            
            result.ocr_extraction_result = ocr_result
            
            if ocr_result.success:
                result.extracted_text = ocr_result.extracted_text
                result.total_characters = ocr_result.total_characters
                result.ocr_extraction_pages = ocr_result.pages_processed
                result.failed_pages = ocr_result.pages_failed
                result.ocr_method_confidence = ocr_result.average_confidence / 100.0
                result.success = True
            else:
                result.errors.extend(ocr_result.errors)
        
        return result
    
    def _extract_ocr_first(self, file_path: Path, analysis: PDFAnalysisResult,
                          chunk_size: int, result: HybridExtractionResult) -> HybridExtractionResult:
        """OCR-first strategy - try OCR first, then text extraction for failed pages"""
        self.logger.info("Using OCR_FIRST strategy")
        
        # First attempt: OCR extraction
        ocr_result = self.ocr_extractor.extract_text(
            file_path, analysis, OCRQuality.BALANCED, chunk_size=chunk_size
        )
        
        result.ocr_extraction_result = ocr_result
        
        if ocr_result.success and ocr_result.average_confidence >= self.ocr_confidence_threshold:
            # OCR worked well
            result.extracted_text = ocr_result.extracted_text
            result.total_characters = ocr_result.total_characters
            result.ocr_extraction_pages = ocr_result.pages_processed
            result.failed_pages = ocr_result.pages_failed
            result.ocr_method_confidence = ocr_result.average_confidence / 100.0
            result.success = True
        else:
            # OCR confidence too low or failed, try text extraction
            result.warnings.append("OCR confidence low, trying text extraction")
            text_result = self.text_extractor.extract_text(
                file_path, analysis, TextExtractionMode.FAST, chunk_size
            )
            
            result.text_extraction_result = text_result
            
            if text_result.success:
                result.extracted_text = text_result.extracted_text
                result.total_characters = text_result.total_characters
                result.text_extraction_pages = text_result.pages_with_text
                result.failed_pages = text_result.pages_empty
                result.text_method_confidence = text_result.extraction_confidence
                result.success = True
            else:
                # Both methods failed, use best available result
                if ocr_result.total_characters > 0:
                    result.extracted_text = ocr_result.extracted_text
                    result.total_characters = ocr_result.total_characters
                    result.ocr_extraction_pages = ocr_result.pages_processed
                    result.success = True
                    result.warnings.append("Using OCR result despite low confidence")
                else:
                    result.errors.extend(ocr_result.errors)
                    result.errors.extend(text_result.errors)
        
        return result
    
    def _extract_parallel(self, file_path: Path, analysis: PDFAnalysisResult,
                         chunk_size: int, result: HybridExtractionResult) -> HybridExtractionResult:
        """Parallel strategy - run both methods and choose best result"""
        self.logger.info("Using PARALLEL strategy")
        
        # Run both extractions
        text_result = self.text_extractor.extract_text(
            file_path, analysis, TextExtractionMode.FAST, chunk_size
        )
        
        ocr_result = self.ocr_extractor.extract_text(
            file_path, analysis, OCRQuality.BALANCED, chunk_size=chunk_size
        )
        
        result.text_extraction_result = text_result
        result.ocr_extraction_result = ocr_result
        
        # Choose best result based on quality metrics
        text_score = self._calculate_result_quality_score(text_result)
        ocr_score = self._calculate_result_quality_score(ocr_result)
        
        if text_score >= ocr_score:
            # Use text extraction result
            result.extracted_text = text_result.extracted_text
            result.total_characters = text_result.total_characters
            result.text_extraction_pages = text_result.pages_with_text
            result.text_method_confidence = text_result.extraction_confidence
            result.success = text_result.success
            result.warnings.append(f"Chose text extraction (score: {text_score:.2f} vs OCR: {ocr_score:.2f})")
        else:
            # Use OCR result
            result.extracted_text = ocr_result.extracted_text
            result.total_characters = ocr_result.total_characters
            result.ocr_extraction_pages = ocr_result.pages_processed
            result.ocr_method_confidence = ocr_result.average_confidence / 100.0
            result.success = ocr_result.success
            result.warnings.append(f"Chose OCR extraction (score: {ocr_score:.2f} vs text: {text_score:.2f})")
        
        return result
    
    def _extract_quality_focused(self, file_path: Path, analysis: PDFAnalysisResult,
                                chunk_size: int, result: HybridExtractionResult) -> HybridExtractionResult:
        """Quality-focused strategy - prioritize highest quality result regardless of speed"""
        self.logger.info("Using QUALITY_FOCUSED strategy")
        
        # Run high-quality extractions
        text_result = self.text_extractor.extract_text(
            file_path, analysis, TextExtractionMode.STRUCTURED, chunk_size
        )
        
        ocr_result = self.ocr_extractor.extract_text(
            file_path, analysis, OCRQuality.HIGH, ImagePreprocessing.ADVANCED, chunk_size
        )
        
        result.text_extraction_result = text_result
        result.ocr_extraction_result = ocr_result
        
        # Combine results intelligently - use best method for each section
        combined_text = self._combine_best_results(text_result, ocr_result)
        
        result.extracted_text = combined_text
        result.total_characters = len(combined_text)
        result.success = len(combined_text) > 0
        
        # Calculate combined statistics
        if text_result.success:
            result.text_extraction_pages = text_result.pages_with_text
            result.text_method_confidence = text_result.extraction_confidence
        
        if ocr_result.success:
            result.ocr_extraction_pages = ocr_result.pages_processed
            result.ocr_method_confidence = ocr_result.average_confidence / 100.0
        
        return result
    
    def _text_extraction_only(self, file_path: Path, analysis: PDFAnalysisResult,
                             chunk_size: Optional[int], start_time: float) -> HybridExtractionResult:
        """Handle text-based PDFs with text extraction only"""
        text_result = self.text_extractor.extract_text(
            file_path, analysis, TextExtractionMode.FAST, chunk_size
        )
        
        return HybridExtractionResult(
            success=text_result.success,
            extracted_text=text_result.extracted_text,
            page_count=text_result.page_count,
            total_characters=text_result.total_characters,
            processing_time=time.time() - start_time,
            text_extraction_pages=text_result.pages_with_text,
            ocr_extraction_pages=0,
            failed_pages=text_result.pages_empty,
            overall_confidence=text_result.extraction_confidence,
            text_method_confidence=text_result.extraction_confidence,
            ocr_method_confidence=0.0,
            strategy_used="text_only",
            memory_peak_mb=0.0,
            chunks_processed=text_result.chunks_processed,
            method_switches=0,
            text_extraction_result=text_result,
            ocr_extraction_result=None,
            page_methods=[],
            warnings=text_result.warnings,
            errors=text_result.errors
        )
    
    def _ocr_extraction_only(self, file_path: Path, analysis: PDFAnalysisResult,
                            chunk_size: Optional[int], start_time: float) -> HybridExtractionResult:
        """Handle scanned PDFs with OCR extraction only"""
        ocr_result = self.ocr_extractor.extract_text(
            file_path, analysis, OCRQuality.BALANCED, chunk_size=chunk_size
        )
        
        return HybridExtractionResult(
            success=ocr_result.success,
            extracted_text=ocr_result.extracted_text,
            page_count=ocr_result.page_count,
            total_characters=ocr_result.total_characters,
            processing_time=time.time() - start_time,
            text_extraction_pages=0,
            ocr_extraction_pages=ocr_result.pages_processed,
            failed_pages=ocr_result.pages_failed,
            overall_confidence=ocr_result.average_confidence / 100.0,
            text_method_confidence=0.0,
            ocr_method_confidence=ocr_result.average_confidence / 100.0,
            strategy_used="ocr_only",
            memory_peak_mb=0.0,
            chunks_processed=ocr_result.chunks_processed,
            method_switches=0,
            text_extraction_result=None,
            ocr_extraction_result=ocr_result,
            page_methods=[],
            warnings=ocr_result.warnings,
            errors=ocr_result.errors
        )
    
    def _calculate_result_quality_score(self, result) -> float:
        """Calculate quality score for extraction result"""
        if not hasattr(result, 'success') or not result.success:
            return 0.0
        
        score = 0.0
        
        # Base score from character count
        if result.total_characters > 0:
            score += min(0.4, result.total_characters / 10000)  # Up to 0.4 points
        
        # Confidence score
        if hasattr(result, 'extraction_confidence'):
            score += result.extraction_confidence * 0.3  # Up to 0.3 points
        elif hasattr(result, 'average_confidence'):
            score += (result.average_confidence / 100.0) * 0.3  # Up to 0.3 points
        
        # Success rate score
        if hasattr(result, 'pages_with_text') and result.page_count > 0:
            success_rate = result.pages_with_text / result.page_count
            score += success_rate * 0.3  # Up to 0.3 points
        elif hasattr(result, 'pages_processed') and result.page_count > 0:
            success_rate = result.pages_processed / result.page_count
            score += success_rate * 0.3  # Up to 0.3 points
        
        return min(1.0, score)
    
    def _combine_best_results(self, text_result, ocr_result) -> str:
        """Combine text and OCR results intelligently"""
        # Simple combination for now - use the result with more content
        if not text_result.success and not ocr_result.success:
            return ""
        
        if not text_result.success:
            return ocr_result.extracted_text
        
        if not ocr_result.success:
            return text_result.extracted_text
        
        # Both successful - choose based on content length and quality
        text_score = len(text_result.extracted_text) * text_result.extraction_confidence
        ocr_score = len(ocr_result.extracted_text) * (ocr_result.average_confidence / 100.0)
        
        if text_score >= ocr_score:
            return text_result.extracted_text
        else:
            return ocr_result.extracted_text
    
    def _calculate_overall_confidence(self, result: HybridExtractionResult) -> float:
        """Calculate overall confidence score for hybrid result"""
        if not result.success:
            return 0.0
        
        total_pages = result.text_extraction_pages + result.ocr_extraction_pages
        if total_pages == 0:
            return 0.0
        
        # Weighted average of method confidences
        text_weight = result.text_extraction_pages / total_pages
        ocr_weight = result.ocr_extraction_pages / total_pages
        
        confidence = (text_weight * result.text_method_confidence + 
                     ocr_weight * result.ocr_method_confidence)
        
        # Penalty for failed pages
        if result.page_count > 0:
            success_rate = total_pages / result.page_count
            confidence *= success_rate
        
        return min(1.0, confidence)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        import re
        
        # Basic cleaning
        cleaned = text.strip()
        cleaned = re.sub(r'\x00', '', cleaned)        # Remove null characters
        cleaned = re.sub(r'\r\n', '\n', cleaned)      # Normalize line endings
        cleaned = re.sub(r'\r', '\n', cleaned)        # Normalize line endings
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Reduce excessive line breaks
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)  # Reduce excessive spaces
        
        return cleaned
    
    def _create_error_result(self, error_message: str) -> HybridExtractionResult:
        """Create error result object"""
        return HybridExtractionResult(
            success=False,
            extracted_text="",
            page_count=0,
            total_characters=0,
            processing_time=0.0,
            text_extraction_pages=0,
            ocr_extraction_pages=0,
            failed_pages=0,
            overall_confidence=0.0,
            text_method_confidence=0.0,
            ocr_method_confidence=0.0,
            strategy_used="error",
            memory_peak_mb=0.0,
            chunks_processed=0,
            method_switches=0,
            text_extraction_result=None,
            ocr_extraction_result=None,
            page_methods=[],
            warnings=[],
            errors=[error_message]
        )


# Convenience functions
def extract_text_hybrid_auto(file_path: Path, analysis: PDFAnalysisResult) -> str:
    """Simple hybrid extraction with auto strategy"""
    extractor = HybridExtractor()
    result = extractor.extract_text(file_path, analysis, HybridStrategy.AUTO)
    return result.extracted_text if result.success else ""


def extract_text_hybrid_quality(file_path: Path, analysis: PDFAnalysisResult) -> str:
    """High quality hybrid extraction"""
    extractor = HybridExtractor()
    result = extractor.extract_text(file_path, analysis, HybridStrategy.QUALITY_FOCUSED)
    return result.extracted_text if result.success else ""


if __name__ == "__main__":
    # Test hybrid extractor
    import sys
    from analyzers.pdf_analyzer import PDFAnalyzer
    
    if len(sys.argv) != 2:
        print("Usage: python hybrid_extractor.py <pdf_file>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    print("=== Hybrid Extractor Test ===")
    print(f"File: {file_path}")
    
    # First analyze the PDF
    analyzer = PDFAnalyzer()
    analysis = analyzer.analyze_pdf(file_path)
    
    print(f"PDF Type: {analysis.pdf_type.value}")
    print(f"Pages: {analysis.page_count}")
    
    # Extract text with hybrid approach
    extractor = HybridExtractor()
    result = extractor.extract_text(file_path, analysis, HybridStrategy.AUTO)
    
    print(f"\n=== Hybrid Extraction Results ===")
    print(f"Success: {result.success}")
    print(f"Strategy: {result.strategy_used}")
    print(f"Characters: {result.total_characters}")
    print(f"Text extraction pages: {result.text_extraction_pages}")
    print(f"OCR extraction pages: {result.ocr_extraction_pages}")
    print(f"Failed pages: {result.failed_pages}")
    print(f"Overall confidence: {result.overall_confidence:.2f}")
    print(f"Method switches: {result.method_switches}")
    print(f"Processing time: {result.processing_time:.2f}s")
    
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