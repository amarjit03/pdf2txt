"""
Configuration managment for the pdf2text 
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict,List, Optional
from enum import Enum

class OutputFormat(Enum):
    TEXT_ONLY = "text"
    JSON_ONLY = "json"
    BOTH = "both"

class ProcessingMode(Enum):
    FAST = "fast"           # Speed over quality
    BALANCED = "balanced"   # Balance speed and quality
    QUALITY = "quality"     # Quality over speed

@dataclass
class MemoryConfig:
    """Memory management settings"""
    max_memory_mb: int = 200              # Maximum memory to use
    chunk_size_small: int = 10            # Pages per chunk for small files
    chunk_size_medium: int = 5            # Pages per chunk for medium files  
    chunk_size_large: int = 2             # Pages per chunk for large files
    cleanup_threshold: float = 0.8        # Trigger cleanup at 80% memory usage
    force_gc_threshold: float = 0.9       # Force garbage collection at 90%

@dataclass
class ProcessingConfig:
    """Processing behavior settings"""
    mode: ProcessingMode = ProcessingMode.BALANCED
    ocr_language: str = "eng"             # Tesseract language code
    preserve_layout: bool = False         # Keep original text layout
    extract_images: bool = False          # Extract images from PDF
    timeout_seconds: int = 300            # Max processing time (5 minutes)
    retry_attempts: int = 2               # Retry failed operations


@dataclass
class OutputConfig:
    """Output format and location settings"""
    format: OutputFormat = OutputFormat.BOTH
    base_output_dir: str = "./output"
    create_subdirs: bool = True           # Create organized subdirectories
    include_metadata: bool = True         # Add processing metadata
    compress_large_outputs: bool = False  # Compress files > 10MB
    
    # Filename patterns
    text_suffix: str = "_extracted.txt"
    json_suffix: str = "_extracted.json"
    failed_suffix: str = "_FAILED.txt"
    
@dataclass
class QualityConfig:
    """Quality and confidence settings"""
    min_confidence_threshold: float = 0.6    # Minimum acceptable OCR confidence
    text_extraction_fallback: bool = True    # Try text extraction if OCR fails
    ocr_fallback: bool = True                # Try OCR if text extraction fails
    partial_results_acceptable: bool = True   # Accept partial extraction results

class Config:
    """Main configration class"""
    
    def __init__(self):
        # Load enviroments variables and variables
        self.memory = MemoryConfig(
            max_memory_mb=int(os.getenv('PDF_AGENT_MAX_MEMORY', 200)),
            chunk_size_small=int(os.getenv('PDF_AGENT_CHUNK_SMALL', 10)),
            chunk_size_medium=int(os.getenv('PDF_AGENT_CHUNK_MEDIUM', 5)),
            chunk_size_large=int(os.getenv('PDF_AGENT_CHUNK_LARGE', 2))
        )
        
        self.processing = ProcessingConfig(
            mode=ProcessingMode(os.getenv('PDF_AGENT_MODE', 'balanced')),
            ocr_language=os.getenv('PDF_AGENT_OCR_LANG', 'eng'),
            timeout_seconds=int(os.getenv('PDF_AGENT_TIMEOUT', 300))
        )
        
        self.output = OutputConfig(
            format=OutputFormat(os.getenv('PDF_AGENT_OUTPUT', 'both')),
            base_output_dir=os.getenv('PDF_AGENT_OUTPUT_DIR', './output')
        )
        
        self.quality = QualityConfig()
        
        # File size threshold in MBs
        self.file_size_thresholds = {
            'small': 5,      # < 5MB = small file
            'medium': 25,    # 5-25MB = medium file  
            'large': 100,    # 25-100MB = large file
            'max': 500       # > 500MB = reject
        }
        
        # Supported file extensions
        self.supported_extensions = ['.pdf']
        
        # OCR preprocessing settings
        self.ocr_preprocessing = {
            'enhance_contrast': True,
            'remove_noise': True,
            'deskew': False,        # Can be slow
            'scale_factor': 2.0     # Upscale images for better OCR
        }
        
        # Logging configuration
        self.logging = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': './output/logs/pdf_agent.log',
            'max_log_size_mb': 10,
            'backup_count': 3
        }

        # Initialize output directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary output directories"""
        base_path = Path(self.output.base_output_dir)
        
        directories = [
            base_path,
            base_path / 'text_files',
            base_path / 'json_files', 
            base_path / 'logs',
            base_path / 'failed'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_chunk_size(self, file_size_mb: float) -> int:
        """Determine optimal chunk size based on file size"""
        if file_size_mb < self.file_size_thresholds['small']:
            return self.memory.chunk_size_small
        elif file_size_mb < self.file_size_thresholds['medium']:
            return self.memory.chunk_size_medium
        else:
            return self.memory.chunk_size_large
    
    def get_file_category(self, file_size_mb: float) -> str:
        """Categorize file size"""
        if file_size_mb < self.file_size_thresholds['small']:
            return 'small'
        elif file_size_mb < self.file_size_thresholds['medium']:
            return 'medium'
        elif file_size_mb < self.file_size_thresholds['large']:
            return 'large'
        elif file_size_mb < self.file_size_thresholds['max']:
            return 'very_large'
        else:
            return 'too_large'
    
    def is_file_too_large(self, file_size_mb: float) -> bool:
        """Check if file exceeds maximum allowed size"""
        return file_size_mb > self.file_size_thresholds['max']
    
    def get_output_paths(self, input_filename: str) -> Dict[str, Path]:
        """Generate output file paths for given input filename"""
        base_path = Path(self.output.base_output_dir)
        stem = Path(input_filename).stem
        
        paths = {
            'text': base_path / 'text_files' / f"{stem}{self.output.text_suffix}",
            'json': base_path / 'json_files' / f"{stem}{self.output.json_suffix}",
            'failed': base_path / 'failed' / f"{stem}{self.output.failed_suffix}",
            'log': base_path / 'logs' / f"{stem}_processing.log"
        }
        
        return paths
        
    def get_processing_strategy(self, pdf_type: str, file_size_mb: float) -> Dict:
        """Get optimal processing strategy based on PDF type and size"""
        strategy = {
            'method': 'auto',
            'chunk_size': self.get_chunk_size(file_size_mb),
            'use_ocr': False,
            'quality_over_speed': False
        }
        
        # Adjust based on PDF type
        if pdf_type == 'scanned':
            strategy['method'] = 'ocr'
            strategy['use_ocr'] = True
        elif pdf_type == 'text':
            strategy['method'] = 'direct'
        elif pdf_type == 'mixed':
            strategy['method'] = 'hybrid'
            strategy['use_ocr'] = True
        
        # Adjust based on processing mode
        if self.processing.mode == ProcessingMode.QUALITY:
            strategy['quality_over_speed'] = True
            strategy['chunk_size'] = max(1, strategy['chunk_size'] // 2)  # Smaller chunks for quality
        elif self.processing.mode == ProcessingMode.FAST:
            strategy['chunk_size'] = min(20, strategy['chunk_size'] * 2)  # Larger chunks for speed
        
        return strategy
        
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check memory settings
        if self.memory.max_memory_mb < 50:
            issues.append("Maximum memory too low (minimum 50MB recommended)")
        
        # Check chunk sizes
        if self.memory.chunk_size_large > self.memory.chunk_size_small:
            issues.append("Large chunk size should not exceed small chunk size")
        
        # Check output directory
        try:
            Path(self.output.base_output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory: {e}")
        
        # Check timeout
        if self.processing.timeout_seconds < 30:
            issues.append("Timeout too short (minimum 30 seconds recommended)")
        
        return issues


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs):
    """Update configuration values"""
    global config
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration key '{key}'")


def print_config_summary():
    """Print a summary of current configuration"""
    print("=== PDF-to-Text Agent Configuration ===")
    print(f"Max Memory: {config.memory.max_memory_mb}MB")
    print(f"Processing Mode: {config.processing.mode.value}")
    print(f"Output Format: {config.output.format.value}")  
    print(f"Output Directory: {config.output.base_output_dir}")
    print(f"OCR Language: {config.processing.ocr_language}")
    print(f"Timeout: {config.processing.timeout_seconds}s")
    print("=" * 40)

if __name__ == "__main__":
    # Test configuration
    print_config_summary()
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!")
