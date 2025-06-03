"""
Memory Estimator - Predicts and manages memory requirements
Prevents memory overruns by calculating optimal processing parameters
"""

import psutil
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import logging
import sys
from pathlib import Path

# Ensure the root directory is on the import path
# sys.path.append(str(Path(__file__).resolve().parent.parent)) # Removed for packaging

from pdf2text.config import get_config
from pdf2text.analyzers.pdf_analyzer import PDFAnalysisResult, PDFType, ContentComplexity


class MemoryStrategy(Enum):
    """Memory management strategies"""
    CONSERVATIVE = "conservative"    # Use 50% of available memory
    BALANCED = "balanced"           # Use 70% of available memory  
    AGGRESSIVE = "aggressive"       # Use 85% of available memory


@dataclass
class MemoryEstimate:
    """Memory usage prediction and optimization parameters"""
    # Current system state
    total_system_memory_mb: float
    available_memory_mb: float
    current_usage_mb: float
    
    # Processing estimates
    estimated_peak_mb: float        # Peak memory during processing
    safe_chunk_size: int           # Pages per chunk to stay within limits
    max_concurrent_chunks: int     # How many chunks can be processed simultaneously
    
    # Safety margins
    buffer_mb: float               # Memory buffer to leave free  
    emergency_threshold_mb: float  # Trigger emergency cleanup at this level
    
    # Strategy recommendations
    recommended_strategy: MemoryStrategy
    can_process_file: bool
    processing_mode: str           # "streaming", "chunked", or "full"
    
    # Warnings and limits
    warnings: list
    memory_pressure: float         # 0.0 = no pressure, 1.0 = critical


class MemoryEstimator:
    """Intelligent memory requirement estimation and optimization"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Memory estimation constants (MB per page)
        self.memory_constants = {
            PDFType.TEXT_BASED: {
                ContentComplexity.SIMPLE: 0.3,
                ContentComplexity.MODERATE: 0.6,
                ContentComplexity.COMPLEX: 1.0
            },
            PDFType.SCANNED: {
                ContentComplexity.SIMPLE: 3.0,
                ContentComplexity.MODERATE: 5.0,
                ContentComplexity.COMPLEX: 8.0
            },
            PDFType.MIXED: {
                ContentComplexity.SIMPLE: 1.5,
                ContentComplexity.MODERATE: 2.5,
                ContentComplexity.COMPLEX: 4.0
            }
        }
        
        # Processing overhead (additional memory needed)
        self.processing_overhead = {
            "text_extraction": 1.2,     # 20% overhead
            "ocr_processing": 2.0,      # 100% overhead (image + text)
            "hybrid_processing": 1.5     # 50% overhead
        }
    
    def estimate_memory_requirements(self, analysis: PDFAnalysisResult) -> MemoryEstimate:
        """
        Calculate memory requirements and optimal processing parameters
        
        Args:
            analysis: PDF analysis results
            
        Returns:
            MemoryEstimate with processing recommendations
        """
        # Get current system memory state
        memory_info = psutil.virtual_memory()
        system_total_mb = memory_info.total / (1024 * 1024)
        system_available_mb = memory_info.available / (1024 * 1024)
        system_used_mb = system_total_mb - system_available_mb
        
        # Calculate base memory requirement per page
        base_memory_per_page = self._get_base_memory_per_page(analysis)
        
        # Calculate processing overhead
        processing_type = self._determine_processing_type(analysis)
        overhead_multiplier = self.processing_overhead.get(processing_type, 1.2)
        
        # Estimate peak memory for different chunk sizes
        chunk_estimates = self._calculate_chunk_estimates(
            analysis, base_memory_per_page, overhead_multiplier
        )
        
        # Determine optimal strategy
        optimal_chunk_size, processing_mode = self._find_optimal_strategy(
            chunk_estimates, system_available_mb
        )
        
        # Calculate final estimates
        estimated_peak_mb = chunk_estimates.get(optimal_chunk_size, {}).get('peak_memory', 0)
        
        # Determine memory strategy
        memory_pressure = system_used_mb / system_total_mb
        if memory_pressure < 0.5:
            strategy = MemoryStrategy.AGGRESSIVE
        elif memory_pressure < 0.7:
            strategy = MemoryStrategy.BALANCED
        else:
            strategy = MemoryStrategy.CONSERVATIVE
        
        # Calculate safety margins
        buffer_mb = max(50, system_total_mb * 0.1)  # At least 50MB or 10% of total
        emergency_threshold = system_total_mb * 0.9
        
        # Check if processing is feasible
        can_process = estimated_peak_mb + buffer_mb < system_available_mb
        
        # Generate warnings
        warnings = self._generate_warnings(
            analysis, estimated_peak_mb, system_available_mb, memory_pressure
        )
        
        return MemoryEstimate(
            total_system_memory_mb=system_total_mb,
            available_memory_mb=system_available_mb,
            current_usage_mb=system_used_mb,
            estimated_peak_mb=estimated_peak_mb,
            safe_chunk_size=optimal_chunk_size,
            max_concurrent_chunks=1,  # Keep simple for now
            buffer_mb=buffer_mb,
            emergency_threshold_mb=emergency_threshold,
            recommended_strategy=strategy,
            can_process_file=can_process,
            processing_mode=processing_mode,
            warnings=warnings,
            memory_pressure=memory_pressure
        )
    
    def _get_base_memory_per_page(self, analysis: PDFAnalysisResult) -> float:
        """Calculate base memory requirement per page"""
        pdf_type = analysis.pdf_type
        complexity = analysis.complexity
        
        # Get base constant
        if pdf_type in self.memory_constants:
            base_memory = self.memory_constants[pdf_type].get(complexity, 1.0)
        else:
            base_memory = 1.0  # Default fallback
        
        # Adjust for special characteristics
        if analysis.has_images:
            base_memory *= 1.5
        if analysis.has_tables:
            base_memory *= 1.2
        if analysis.has_forms:
            base_memory *= 1.1
        
        # Adjust for text density (low density = complex layout = more memory)
        if analysis.text_density_avg < 0.1:
            base_memory *= 1.3
        
        return base_memory
    
    def _determine_processing_type(self, analysis: PDFAnalysisResult) -> str:
        """Determine which processing type will be used"""
        if analysis.pdf_type == PDFType.TEXT_BASED:
            return "text_extraction"
        elif analysis.pdf_type == PDFType.SCANNED:
            return "ocr_processing"
        else:
            return "hybrid_processing"
    
    def _calculate_chunk_estimates(self, analysis: PDFAnalysisResult, 
                                base_memory_per_page: float, 
                                overhead_multiplier: float) -> Dict[int, Dict]:
        """Calculate memory estimates for different chunk sizes"""
        estimates = {}
        
        # Test different chunk sizes
        chunk_sizes = [1, 2, 3, 5, 10, 15, 20]
        
        for chunk_size in chunk_sizes:
            if chunk_size > analysis.page_count:
                continue
            
            # Base memory for chunk
            chunk_memory = chunk_size * base_memory_per_page
            
            # Add processing overhead
            peak_memory = chunk_memory * overhead_multiplier
            
            # Add fixed overhead (libraries, buffers, etc.)
            fixed_overhead = 30  # MB
            total_peak = peak_memory + fixed_overhead
            
            estimates[chunk_size] = {
                'chunk_memory': chunk_memory,
                'peak_memory': total_peak,
                'processing_time_estimate': chunk_size * 0.5,  # rough estimate
                'efficiency_score': chunk_size / total_peak  # pages per MB
            }
        
        return estimates
    
    def _find_optimal_strategy(self, chunk_estimates: Dict, 
                            available_memory_mb: float) -> Tuple[int, str]:
        """Find optimal chunk size and processing mode"""
        config_limit = self.config.memory.max_memory_mb
        effective_limit = min(available_memory_mb * 0.8, config_limit)
        
        # Find largest chunk size that fits in memory
        suitable_chunks = [
            (size, data) for size, data in chunk_estimates.items()
            if data['peak_memory'] <= effective_limit
        ]
        
        if not suitable_chunks:
            # Even 1 page is too much - use streaming mode
            return 1, "streaming"
        
        # Choose chunk size with best efficiency that fits
        best_chunk = max(suitable_chunks, key=lambda x: x[1]['efficiency_score'])
        chunk_size = best_chunk[0]
        
        # Determine processing mode
        if chunk_size >= 10:
            mode = "full"
        elif chunk_size >= 3:
            mode = "chunked"
        else:
            mode = "streaming"
        
        return chunk_size, mode
    
    def _generate_warnings(self, analysis: PDFAnalysisResult, 
                        estimated_peak_mb: float,
                        available_memory_mb: float,
                        memory_pressure: float) -> list:
        """Generate memory-related warnings"""
        warnings = []
        
        # High memory usage warning
        if estimated_peak_mb > available_memory_mb * 0.7:
            warnings.append(
                f"High memory usage expected: {estimated_peak_mb:.1f}MB "
                f"({estimated_peak_mb/available_memory_mb*100:.1f}% of available)"
            )
        
        # System memory pressure
        if memory_pressure > 0.8:
            warnings.append(
                f"System memory pressure high ({memory_pressure*100:.1f}%), "
                "consider closing other applications"
            )
        
        # Large file warning
        if analysis.file_size_mb > 100:
            warnings.append(
                f"Large file ({analysis.file_size_mb:.1f}MB) - "
                "processing may take significant time"
            )
        
        # OCR memory warning
        if analysis.pdf_type in [PDFType.SCANNED, PDFType.MIXED]:
            warnings.append(
                "OCR processing requires additional memory - "
                "consider reducing chunk size if memory issues occur"
            )
        
        # Complex content warning
        if analysis.complexity == ContentComplexity.COMPLEX:
            warnings.append(
                "Complex document structure detected - "
                "memory usage may be higher than estimated"
            )
        
        return warnings
    
    def monitor_current_usage(self) -> Dict:
        """Get current memory usage statistics"""
        memory_info = psutil.virtual_memory()
        
        return {
            'total_mb': memory_info.total / (1024 * 1024),
            'available_mb': memory_info.available / (1024 * 1024),
            'used_mb': memory_info.used / (1024 * 1024),
            'free_mb': memory_info.free / (1024 * 1024),
            'percentage': memory_info.percent,
            'pressure_level': 'high' if memory_info.percent > 80 else 
                            'medium' if memory_info.percent > 60 else 'low'
        }
    
    def check_memory_health(self) -> Tuple[bool, str]:
        """Quick health check - can we proceed with processing?"""
        memory_info = psutil.virtual_memory()
        available_mb = memory_info.available / (1024 * 1024)
        
        min_required_mb = 100  # Minimum memory needed for any processing
        
        if available_mb < min_required_mb:
            return False, f"Insufficient memory: {available_mb:.1f}MB available, need at least {min_required_mb}MB"
        
        if memory_info.percent > 90:
            return False, f"System memory critically low: {memory_info.percent:.1f}% used"
        
        return True, "Memory OK"
    
    def suggest_optimization(self, analysis: PDFAnalysisResult, 
                        estimate: MemoryEstimate) -> Dict:
        """Suggest optimizations for better memory usage"""
        suggestions = {
            'chunk_size': estimate.safe_chunk_size,
            'processing_tips': [],
            'system_tips': []
        }
        
        # Processing optimization suggestions
        if analysis.pdf_type == PDFType.MIXED:
            suggestions['processing_tips'].append(
                "Consider pre-filtering to separate text and scanned pages"
            )
        
        if analysis.has_images and not analysis.pdf_type == PDFType.SCANNED:
            suggestions['processing_tips'].append(
                "Consider skipping image extraction to save memory"
            )
        
        if estimate.estimated_peak_mb > estimate.available_memory_mb * 0.6:
            suggestions['processing_tips'].append(
                f"Reduce chunk size to {max(1, estimate.safe_chunk_size // 2)} pages"
            )
        
        # System optimization suggestions
        if estimate.memory_pressure > 0.7:
            suggestions['system_tips'].append(
                "Close other applications to free up memory"
            )
        
        if estimate.available_memory_mb < 200:
            suggestions['system_tips'].append(
                "Consider processing on a system with more RAM"
            )
        
        return suggestions


def quick_memory_check() -> Dict:
    """Quick memory status check (used by main.py)"""
    try:
        memory_info = psutil.virtual_memory()
        return {
            'available_mb': memory_info.available / (1024 * 1024),
            'percentage_used': memory_info.percent,
            'status': 'ok' if memory_info.percent < 80 else 'warning' if memory_info.percent < 90 else 'critical'
        }
    except Exception as e:
        return {
            'available_mb': 0,
            'percentage_used': 100,
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    # Test memory estimator
    estimator = MemoryEstimator()
    
    print("=== Memory Status ===")
    status = estimator.monitor_current_usage()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    health_ok, health_msg = estimator.check_memory_health()
    print(f"\nMemory Health: {'OK' if health_ok else 'WARNING'}")
    print(f"Message: {health_msg}")