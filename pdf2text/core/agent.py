"""
Main PDF-to-Text Agent - Central Orchestrator
Coordinates all components into a seamless processing pipeline
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

from pdf2text.config import get_config, ProcessingMode, OutputFormat
from pdf2text.core.memory_manager import get_memory_manager, MemoryContext
from pdf2text.core.file_manager import get_file_manager
from pdf2text.analyzers.pdf_analyzer import PDFAnalyzer, PDFAnalysisResult, PDFType
from pdf2text.analyzers.memory_estimator import MemoryEstimator
# from extractors.text_extractor import TextExtractor, TextExtractionMode # Kept commented
# from extractors.ocr_extractor import OCRExtractor, OCRQuality, ImagePreprocessing # Kept commented
# from extractors.hybrid_extractor import HybridExtractor, HybridStrategy # Kept commented
from pdf2text.extractors.text_extractors import TextExtractor, TextExtractionMode
from pdf2text.extractors.ocr_extractors import OCRExtractor, OCRQuality, ImagePreprocessing
from pdf2text.extractors.hybrid_extractors import HybridExtractor, HybridStrategy


class ProcessingStatus(Enum):
    """Processing status states"""
    INITIALIZED = "initialized"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXTRACTING = "extracting"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Complete processing result with all metadata"""
    # Basic results
    success: bool
    input_file: str
    output_files: Dict[str, Path]
    
    # Processing details
    pdf_type: str
    pages_processed: int
    total_characters: int
    processing_time: float
    
    # Quality metrics
    extraction_confidence: float
    method_used: str
    memory_peak_mb: float
    
    # Detailed breakdown
    analysis_result: Optional[PDFAnalysisResult]
    extraction_result: Optional[Any]
    
    # Issues and warnings
    warnings: List[str]
    errors: List[str]
    
    # Processing statistics
    analysis_time: float = 0.0
    extraction_time: float = 0.0
    saving_time: float = 0.0


class PDFTextAgent:
    """
    Main PDF-to-Text Processing Agent
    
    Orchestrates the complete pipeline:
    1. Analysis (determine PDF type and requirements)
    2. Memory planning (ensure safe processing)
    3. Text extraction (using optimal method)
    4. Output generation (save results in requested formats)
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.config = get_config()
        self.memory_manager = get_memory_manager()
        self.file_manager = get_file_manager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pdf_analyzer = PDFAnalyzer()
        self.memory_estimator = MemoryEstimator()
        self.text_extractor = TextExtractor()
        self.ocr_extractor = OCRExtractor()
        self.hybrid_extractor = HybridExtractor()
        
        # Progress reporting
        self.progress_callback = progress_callback
        self.current_status = ProcessingStatus.INITIALIZED
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'total_pages': 0,
            'total_characters': 0,
            'total_processing_time': 0.0,
            'successful_extractions': 0,
            'failed_extractions': 0
        }
        
        # Start memory monitoring
        self.memory_manager.start_monitoring()
        
        self.logger.info("PDF Text Agent initialized")
    
    def process_file(self, file_path: str, 
                    output_dir: Optional[str] = None,
                    processing_mode: Optional[ProcessingMode] = None,
                    output_format: Optional[OutputFormat] = None) -> ProcessingResult:
        """
        Process a single PDF file - Main entry point
        
        Args:
            file_path: Path to PDF file
            output_dir: Custom output directory (optional)
            processing_mode: Override default processing mode
            output_format: Override default output format
            
        Returns:
            ProcessingResult with complete information
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Initialize result
        result = ProcessingResult(
            success=False,
            input_file=str(file_path),
            output_files={},
            pdf_type="unknown",
            pages_processed=0,
            total_characters=0,
            processing_time=0.0,
            extraction_confidence=0.0,
            method_used="none",
            memory_peak_mb=0.0,
            analysis_result=None,
            extraction_result=None,
            warnings=[],
            errors=[]
        )
        
        try:
            # Override config if specified
            if processing_mode:
                self.config.processing.mode = processing_mode
            if output_format:
                self.config.output.format = output_format
            if output_dir:
                self.config.output.base_output_dir = output_dir
                self.file_manager.ensure_output_directories()
            
            self._update_progress("Starting PDF processing", 0)
            
            # Step 1: Analysis
            self._update_status(ProcessingStatus.ANALYZING)
            analysis_start = time.time()
            
            analysis = self._analyze_pdf(file_path)
            if not self._validate_analysis(analysis, result):
                return result
            
            result.analysis_result = analysis
            result.pdf_type = analysis.pdf_type.value
            result.analysis_time = time.time() - analysis_start
            
            self._update_progress(f"Analysis complete: {analysis.pdf_type.value} PDF, {analysis.page_count} pages", 20)
            
            # Step 2: Memory planning
            self._update_status(ProcessingStatus.PLANNING)
            memory_plan = self._plan_memory_usage(analysis, result)
            if not memory_plan:
                return result
            
            self._update_progress("Processing plan ready", 30)
            
            # Step 3: Text extraction
            self._update_status(ProcessingStatus.EXTRACTING)
            extraction_start = time.time()
            
            extraction_result = self._extract_text(file_path, analysis, result)
            if not extraction_result:
                return result
            
            result.extraction_result = extraction_result
            result.extraction_time = time.time() - extraction_start
            
            # Update result with extraction data
            self._populate_extraction_results(extraction_result, result)
            
            self._update_progress(f"Extraction complete: {result.total_characters} characters", 80)
            
            # Step 4: Save outputs
            self._update_status(ProcessingStatus.SAVING)
            saving_start = time.time()
            
            output_files = self._save_results(file_path, extraction_result, analysis, result)
            result.output_files = output_files
            result.saving_time = time.time() - saving_start
            
            # Final processing
            result.processing_time = time.time() - start_time
            result.memory_peak_mb = self.memory_manager.get_memory_snapshot().process_mb
            result.success = len(output_files) > 0
            
            self._update_status(ProcessingStatus.COMPLETED)
            self._update_progress("Processing complete", 100)
            
            # Update statistics
            self._update_statistics(result)
            
            self.logger.info(f"Successfully processed {file_path.name} in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed for {file_path}: {str(e)}")
            result.errors.append(f"Processing failed: {str(e)}")
            result.processing_time = time.time() - start_time
            self._update_status(ProcessingStatus.FAILED)
            return result
    
    def process_batch(self, input_dir: str, 
                     file_pattern: str = "*.pdf",
                     output_dir: Optional[str] = None,
                     max_files: Optional[int] = None) -> Dict[str, ProcessingResult]:
        """
        Process multiple PDF files in a directory
        
        Args:
            input_dir: Directory containing PDF files
            file_pattern: File pattern to match (default: *.pdf)
            output_dir: Custom output directory
            max_files: Maximum number of files to process
            
        Returns:
            Dictionary mapping filenames to ProcessingResults
        """
        input_path = Path(input_dir)
        results = {}
        
        if not input_path.exists() or not input_path.is_dir():
            self.logger.error(f"Input directory not found: {input_dir}")
            return results
        
        # Find PDF files
        pdf_files = list(input_path.glob(file_pattern))
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_dir}")
            return results
        
        self.logger.info(f"Processing {len(pdf_files)} files in batch mode")
        
        # Process each file
        for i, pdf_file in enumerate(pdf_files):
            try:
                self.logger.info(f"Processing file {i+1}/{len(pdf_files)}: {pdf_file.name}")
                
                result = self.process_file(
                    str(pdf_file),
                    output_dir=output_dir
                )
                
                results[pdf_file.name] = result
                
                # Log progress
                if result.success:
                    self.logger.info(f"✅ {pdf_file.name}: {result.total_characters} characters extracted")
                else:
                    self.logger.error(f"❌ {pdf_file.name}: Processing failed")
                
            except Exception as e:
                self.logger.error(f"Batch processing error for {pdf_file.name}: {e}")
                results[pdf_file.name] = ProcessingResult(
                    success=False,
                    input_file=str(pdf_file),
                    output_files={},
                    pdf_type="error",
                    pages_processed=0,
                    total_characters=0,
                    processing_time=0.0,
                    extraction_confidence=0.0,
                    method_used="none",
                    memory_peak_mb=0.0,
                    analysis_result=None,
                    extraction_result=None,
                    warnings=[],
                    errors=[str(e)]
                )
        
        # Generate batch summary
        self._log_batch_summary(results)
        
        return results
    
    def _analyze_pdf(self, file_path: Path) -> PDFAnalysisResult:
        """Analyze PDF to determine processing strategy"""
        try:
            analysis = self.pdf_analyzer.analyze_pdf(file_path)
            
            self.logger.info(
                f"PDF Analysis: {analysis.pdf_type.value} content, "
                f"{analysis.page_count} pages, "
                f"confidence: {analysis.extraction_confidence:.2f}"
            )
            
            if analysis.warnings:
                for warning in analysis.warnings:
                    self.logger.warning(f"Analysis warning: {warning}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"PDF analysis failed: {e}")
            raise
    
    def _validate_analysis(self, analysis: PDFAnalysisResult, result: ProcessingResult) -> bool:
        """Validate analysis results and check if processing can proceed"""
        if analysis.pdf_type == PDFType.CORRUPTED:
            result.errors.append("PDF file is corrupted or invalid")
            return False
        
        if analysis.pdf_type == PDFType.ENCRYPTED:
            result.errors.append("PDF file is password protected (password handling not yet implemented)")
            return False
        
        if analysis.pdf_type == PDFType.EMPTY:
            result.warnings.append("PDF file contains no pages")
            return False
        
        if analysis.page_count == 0:
            result.errors.append("PDF file has no pages")
            return False
        
        # Check file size limits
        if self.config.is_file_too_large(analysis.file_size_mb):
            result.errors.append(f"File too large: {analysis.file_size_mb:.1f}MB exceeds limit")
            return False
        
        return True
    
    def _plan_memory_usage(self, analysis: PDFAnalysisResult, result: ProcessingResult) -> bool:
        """Plan memory usage and validate system capacity"""
        try:
            memory_estimate = self.memory_estimator.estimate_memory_requirements(analysis)
            
            if not memory_estimate.can_process_file:
                result.errors.append("Insufficient system memory for processing")
                return False
            
            # Add warnings from memory estimator
            result.warnings.extend(memory_estimate.warnings)
            
            self.logger.info(
                f"Memory plan: Peak {memory_estimate.estimated_peak_mb:.1f}MB, "
                f"chunk size: {memory_estimate.safe_chunk_size} pages"
            )
            
            return True
            
        except Exception as e:
            result.errors.append(f"Memory planning failed: {str(e)}")
            return False
    
    def _extract_text(self, file_path: Path, analysis: PDFAnalysisResult, result: ProcessingResult) -> Optional[Any]:
        """Extract text using the appropriate method"""
        try:
            # Choose extraction method based on PDF type and processing mode
            if analysis.pdf_type == PDFType.TEXT_BASED:
                return self._extract_with_text_extractor(file_path, analysis)
            
            elif analysis.pdf_type == PDFType.SCANNED:
                return self._extract_with_ocr_extractor(file_path, analysis)
            
            elif analysis.pdf_type == PDFType.MIXED:
                return self._extract_with_hybrid_extractor(file_path, analysis)
            
            else:
                result.errors.append(f"Cannot process PDF type: {analysis.pdf_type.value}")
                return None
                
        except Exception as e:
            result.errors.append(f"Text extraction failed: {str(e)}")
            return None
    
    def _extract_with_text_extractor(self, file_path: Path, analysis: PDFAnalysisResult):
        """Extract text using direct text extraction"""
        mode = TextExtractionMode.FAST
        
        if self.config.processing.mode == ProcessingMode.QUALITY:
            mode = TextExtractionMode.STRUCTURED
        elif self.config.processing.preserve_layout:
            mode = TextExtractionMode.LAYOUT
        
        return self.text_extractor.extract_text(file_path, analysis, mode)
    
    def _extract_with_ocr_extractor(self, file_path: Path, analysis: PDFAnalysisResult):
        """Extract text using OCR"""
        quality = OCRQuality.BALANCED
        preprocessing = ImagePreprocessing.BASIC
        
        if self.config.processing.mode == ProcessingMode.QUALITY:
            quality = OCRQuality.HIGH
            preprocessing = ImagePreprocessing.ADVANCED
        elif self.config.processing.mode == ProcessingMode.FAST:
            quality = OCRQuality.FAST
            preprocessing = ImagePreprocessing.NONE
        
        return self.ocr_extractor.extract_text(file_path, analysis, quality, preprocessing)
    
    def _extract_with_hybrid_extractor(self, file_path: Path, analysis: PDFAnalysisResult):
        """Extract text using hybrid approach"""
        strategy = HybridStrategy.AUTO
        
        if self.config.processing.mode == ProcessingMode.QUALITY:
            strategy = HybridStrategy.QUALITY_FOCUSED
        elif self.config.processing.mode == ProcessingMode.FAST:
            strategy = HybridStrategy.TEXT_FIRST
        
        return self.hybrid_extractor.extract_text(file_path, analysis, strategy)
    
    def _populate_extraction_results(self, extraction_result: Any, result: ProcessingResult):
        """Populate ProcessingResult with extraction data"""
        if hasattr(extraction_result, 'success') and extraction_result.success:
            result.total_characters = getattr(extraction_result, 'total_characters', 0)
            result.pages_processed = getattr(extraction_result, 'page_count', 0)
            result.extraction_confidence = getattr(extraction_result, 'extraction_confidence', 0.0)
            
            # Handle different result types
            if hasattr(extraction_result, 'method_used'):
                result.method_used = extraction_result.method_used
            elif hasattr(extraction_result, 'strategy_used'):
                result.method_used = extraction_result.strategy_used
            
            # Collect warnings and errors
            if hasattr(extraction_result, 'warnings'):
                result.warnings.extend(extraction_result.warnings)
            if hasattr(extraction_result, 'errors'):
                result.errors.extend(extraction_result.errors)
        else:
            result.errors.append("Text extraction failed")
    
    def _save_results(self, file_path: Path, extraction_result: Any, 
                     analysis: PDFAnalysisResult, result: ProcessingResult) -> Dict[str, Path]:
        """Save extraction results in requested formats"""
        output_files = {}
        
        try:
            extracted_text = getattr(extraction_result, 'extracted_text', '')
            
            if not extracted_text:
                result.warnings.append("No text was extracted to save")
                return output_files
            
            # Prepare metadata
            metadata = self._prepare_metadata(analysis, extraction_result, result)
            
            # Save text file
            if self.config.output.format in [OutputFormat.TEXT_ONLY, OutputFormat.BOTH]:
                try:
                    text_path = self.file_manager.save_text_result(
                        file_path.name, extracted_text, metadata
                    )
                    output_files['text'] = text_path
                    self.logger.info(f"Saved text file: {text_path}")
                except Exception as e:
                    result.warnings.append(f"Failed to save text file: {str(e)}")
            
            # Save JSON file
            if self.config.output.format in [OutputFormat.JSON_ONLY, OutputFormat.BOTH]:
                try:
                    json_data = self._prepare_json_output(extraction_result, analysis, metadata)
                    json_path = self.file_manager.save_json_result(file_path.name, json_data)
                    output_files['json'] = json_path
                    self.logger.info(f"Saved JSON file: {json_path}")
                except Exception as e:
                    result.warnings.append(f"Failed to save JSON file: {str(e)}")
            
            return output_files
            
        except Exception as e:
            result.errors.append(f"Failed to save results: {str(e)}")
            return {}
    
    def _prepare_metadata(self, analysis: PDFAnalysisResult, 
                         extraction_result: Any, result: ProcessingResult) -> Dict:
        """Prepare metadata for output files"""
        return {
            'PDF Type': analysis.pdf_type.value,
            'Pages': analysis.page_count,
            'File Size (MB)': f"{analysis.file_size_mb:.1f}",
            'Processing Time (s)': f"{result.processing_time:.2f}",
            'Extraction Method': result.method_used,
            'Confidence': f"{result.extraction_confidence:.2f}",
            'Characters Extracted': result.total_characters,
            'Memory Peak (MB)': f"{result.memory_peak_mb:.1f}"
        }
    
    def _prepare_json_output(self, extraction_result: Any, 
                           analysis: PDFAnalysisResult, metadata: Dict) -> Dict:
        """Prepare comprehensive JSON output"""
        json_data = {
            'document_info': {
                'pdf_type': analysis.pdf_type.value,
                'page_count': analysis.page_count,
                'file_size_mb': analysis.file_size_mb,
                'complexity': analysis.complexity.value,
                'has_tables': analysis.has_tables,
                'has_images': analysis.has_images,
                'has_forms': analysis.has_forms
            },
            'extraction_details': {
                'method_used': getattr(extraction_result, 'method_used', 'unknown'),
                'processing_time_seconds': getattr(extraction_result, 'processing_time', 0.0),
                'confidence_score': getattr(extraction_result, 'extraction_confidence', 0.0),
                'total_characters': getattr(extraction_result, 'total_characters', 0),
                'chunks_processed': getattr(extraction_result, 'chunks_processed', 0)
            },
            'content': {
                'extracted_text': getattr(extraction_result, 'extracted_text', '')
            },
            'quality_metrics': {},
            'processing_metadata': metadata
        }
        
        # Add method-specific quality metrics
        if hasattr(extraction_result, 'pages_with_text'):
            json_data['quality_metrics']['pages_with_text'] = extraction_result.pages_with_text
            json_data['quality_metrics']['pages_empty'] = extraction_result.pages_empty
        
        if hasattr(extraction_result, 'average_confidence'):
            json_data['quality_metrics']['ocr_confidence'] = extraction_result.average_confidence
            json_data['quality_metrics']['high_confidence_pages'] = getattr(extraction_result, 'high_confidence_pages', 0)
        
        if hasattr(extraction_result, 'text_extraction_pages'):
            json_data['quality_metrics']['text_extraction_pages'] = extraction_result.text_extraction_pages
            json_data['quality_metrics']['ocr_extraction_pages'] = extraction_result.ocr_extraction_pages
        
        return json_data
    
    def _update_status(self, status: ProcessingStatus):
        """Update processing status"""
        self.current_status = status
        self.logger.debug(f"Status: {status.value}")
    
    def _update_progress(self, message: str, percentage: int):
        """Update progress with callback"""
        self.logger.info(f"Progress {percentage}%: {message}")
        
        if self.progress_callback:
            try:
                self.progress_callback(percentage, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def _update_statistics(self, result: ProcessingResult):
        """Update processing statistics"""
        self.stats['files_processed'] += 1
        self.stats['total_pages'] += result.pages_processed
        self.stats['total_characters'] += result.total_characters
        self.stats['total_processing_time'] += result.processing_time
        
        if result.success:
            self.stats['successful_extractions'] += 1
        else:
            self.stats['failed_extractions'] += 1
    
    def _log_batch_summary(self, results: Dict[str, ProcessingResult]):
        """Log summary of batch processing"""
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        total_chars = sum(r.total_characters for r in results.values())
        total_time = sum(r.processing_time for r in results.values())
        
        self.logger.info(f"Batch processing complete:")
        self.logger.info(f"  Files processed: {len(results)}")
        self.logger.info(f"  Successful: {successful}")
        self.logger.info(f"  Failed: {failed}")
        self.logger.info(f"  Total characters: {total_chars:,}")
        self.logger.info(f"  Total time: {total_time:.2f}s")
        self.logger.info(f"  Average time per file: {total_time/len(results):.2f}s")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()
    
    def shutdown(self):
        """Clean shutdown of all components"""
        self.logger.info("PDF Text Agent shutting down")
        
        self.memory_manager.cleanup_and_shutdown()
        self.file_manager.shutdown()
        
        self.logger.info("PDF Text Agent shutdown complete")


# Convenience functions for simple usage
def process_pdf_simple(file_path: str) -> str:
    """Simple PDF processing - returns just the extracted text"""
    agent = PDFTextAgent()
    try:
        result = agent.process_file(file_path)
        return getattr(result.extraction_result, 'extracted_text', '') if result.success else ""
    finally:
        agent.shutdown()


def process_pdf_to_files(file_path: str, output_dir: str = None) -> bool:
    """Process PDF and save to files - returns success status"""
    agent = PDFTextAgent()
    try:
        result = agent.process_file(file_path, output_dir=output_dir)
        return result.success
    finally:
        agent.shutdown()


if __name__ == "__main__":
    # Test the main agent
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python agent.py <pdf_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    def progress_callback(percentage: int, message: str):
        print(f"[{percentage:3d}%] {message}")
    
    print("=== PDF Text Agent Test ===")
    print(f"Processing: {file_path}")
    
    agent = PDFTextAgent(progress_callback=progress_callback)
    
    try:
        result = agent.process_file(file_path)
        
        print(f"\n=== Processing Results ===")
        print(f"Success: {result.success}")
        print(f"PDF Type: {result.pdf_type}")
        print(f"Pages: {result.pages_processed}")
        print(f"Characters: {result.total_characters:,}")
        print(f"Method: {result.method_used}")
        print(f"Confidence: {result.extraction_confidence:.2f}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Memory peak: {result.memory_peak_mb:.1f}MB")
        
        if result.output_files:
            print(f"\n=== Output Files ===")
            for format_type, file_path in result.output_files.items():
                print(f"{format_type}: {file_path}")
        
        if result.warnings:
            print(f"\n=== Warnings ===")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.errors:
            print(f"\n=== Errors ===")
            for error in result.errors:
                print(f"  - {error}")
        
        # Show statistics
        stats = agent.get_statistics()
        print(f"\n=== Agent Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    finally:
        agent.shutdown()