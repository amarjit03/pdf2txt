"""
Logger Setup - Comprehensive Logging System
Provides structured logging for all PDF agent components
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import time
from datetime import datetime

from pdf2text.config import get_config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m'    # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for file logging with additional context"""
    
    def format(self, record):
        # Add structured information
        if not hasattr(record, 'component'):
            record.component = record.name.split('.')[-1] if '.' in record.name else record.name
        
        if not hasattr(record, 'memory_mb'):
            try:
                import psutil
                record.memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            except:
                record.memory_mb = 0.0
        
        return super().format(record)


class PDFAgentLogger:
    """Centralized logging management for PDF Agent"""
    
    def __init__(self):
        self.config = get_config()
        self.loggers = {}
        self.log_dir = Path(self.config.output.base_output_dir) / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        self._setup_root_logger()
        
        # Setup component loggers
        self._setup_component_loggers()
    
    def _setup_root_logger(self):
        """Setup the root logger with console and file handlers"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with detailed logging
        log_file = self.log_dir / f"pdf_agent_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = StructuredFormatter(
            '%(asctime)s | %(levelname)-8s | %(component)-12s | '
            'MEM:%(memory_mb)5.1fMB | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error-only handler for critical issues
        error_file = self.log_dir / f"pdf_agent_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    def _setup_component_loggers(self):
        """Setup specialized loggers for different components"""
        components = [
            'core.agent',
            'core.memory_manager',
            'core.file_manager',
            'analyzers.pdf_analyzer',
            'analyzers.memory_estimator',
            'extractors.text_extractor',
            'extractors.ocr_extractor',
            'extractors.hybrid_extractor',
            'config',
            'main'
        ]
        
        for component in components:
            logger = logging.getLogger(component)
            self.loggers[component] = logger
            
            # Add component-specific context
            self._add_component_context(logger, component)
    
    def _add_component_context(self, logger, component_name):
        """Add component-specific context to logger"""
        # Create a custom LoggerAdapter for component context
        class ComponentAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                return f"[{component_name.split('.')[-1].upper()}] {msg}", kwargs
        
        # Replace logger methods with adapter methods
        adapter = ComponentAdapter(logger, {})
        return adapter
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific component"""
        return logging.getLogger(name)
    
    def log_processing_start(self, filename: str, file_size_mb: float):
        """Log the start of PDF processing"""
        logger = self.get_logger('core.agent')
        logger.info(f"Starting processing: {filename} ({file_size_mb:.1f}MB)")
    
    def log_processing_complete(self, filename: str, success: bool, 
                              processing_time: float, characters: int):
        """Log completion of PDF processing"""
        logger = self.get_logger('core.agent')
        if success:
            logger.info(
                f"Completed processing: {filename} | "
                f"Time: {processing_time:.2f}s | "
                f"Characters: {characters:,}"
            )
        else:
            logger.error(f"Failed processing: {filename} after {processing_time:.2f}s")
    
    def log_memory_state(self, state: str, usage_mb: float, available_mb: float):
        """Log memory state changes"""
        logger = self.get_logger('core.memory_manager')
        logger.info(f"Memory state: {state} | Used: {usage_mb:.1f}MB | Available: {available_mb:.1f}MB")
    
    def log_extraction_method(self, method: str, pages: int, confidence: float):
        """Log extraction method selection"""
        logger = self.get_logger('extractors')
        logger.info(f"Extraction method: {method} | Pages: {pages} | Confidence: {confidence:.2f}")
    
    def log_batch_summary(self, total_files: int, successful: int, 
                         failed: int, total_time: float):
        """Log batch processing summary"""
        logger = self.get_logger('main')
        logger.info(
            f"Batch complete: {total_files} files | "
            f"Success: {successful} | Failed: {failed} | "
            f"Total time: {total_time:.2f}s"
        )
    
    def log_performance_metrics(self, component: str, operation: str, 
                               duration: float, memory_delta: float = 0.0):
        """Log performance metrics for analysis"""
        logger = self.get_logger(f'performance.{component}')
        logger.info(
            f"PERF | {operation} | Duration: {duration:.3f}s | "
            f"Memory delta: {memory_delta:+.1f}MB"
        )
    
    def create_processing_log(self, filename: str) -> logging.Logger:
        """Create a dedicated logger for a specific file processing"""
        log_name = f"processing.{Path(filename).stem}"
        logger = logging.getLogger(log_name)
        
        # Create file-specific log file
        log_file = self.log_dir / f"{Path(filename).stem}_processing.log"
        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        return logger
    
    def cleanup_old_logs(self, max_age_days: int = 30):
        """Clean up old log files"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        cleaned = 0
        
        for log_file in self.log_dir.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    cleaned += 1
            except Exception as e:
                logging.error(f"Failed to cleanup log file {log_file}: {e}")
        
        if cleaned > 0:
            logging.info(f"Cleaned up {cleaned} old log files")


# Global logger instance
_pdf_logger: Optional[PDFAgentLogger] = None


def setup_logging(log_level: str = "INFO") -> PDFAgentLogger:
    """Setup logging system - call this once at application start"""
    global _pdf_logger
    
    if _pdf_logger is None:
        _pdf_logger = PDFAgentLogger()
        
        # Set log level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)
        
        logging.info("PDF Agent logging system initialized")
    
    return _pdf_logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance"""
    if _pdf_logger is None:
        setup_logging()
    
    if name:
        return logging.getLogger(name)
    else:
        return logging.getLogger(__name__)


def log_function_call(func):
    """Decorator to log function calls with timing"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            logger.debug(f"Calling {func.__name__} with args={len(args)}, kwargs={len(kwargs)}")
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {duration:.3f}s: {str(e)}")
            raise
    
    return wrapper


def log_memory_usage(operation_name: str):
    """Context manager to log memory usage during an operation"""
    class MemoryLogger:
        def __init__(self, name):
            self.name = name
            self.logger = get_logger('memory')
            self.start_memory = 0.0
        
        def __enter__(self):
            try:
                import psutil
                self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                self.logger.debug(f"Starting {self.name} | Memory: {self.start_memory:.1f}MB")
            except:
                pass
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                import psutil
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                delta = end_memory - self.start_memory
                self.logger.debug(
                    f"Completed {self.name} | Memory: {end_memory:.1f}MB | "
                    f"Delta: {delta:+.1f}MB"
                )
            except:
                pass
    
    return MemoryLogger(operation_name)


if __name__ == "__main__":
    # Test logging system
    print("Testing PDF Agent Logging System")
    
    # Setup logging
    pdf_logger = setup_logging("DEBUG")
    
    # Test different loggers
    main_logger = get_logger('main')
    agent_logger = get_logger('core.agent')
    memory_logger = get_logger('core.memory_manager')
    
    # Test log messages
    main_logger.info("Testing main logger")
    agent_logger.info("Testing agent logger")
    memory_logger.warning("Testing memory manager logger")
    
    # Test structured logging
    pdf_logger.log_processing_start("test.pdf", 15.5)
    pdf_logger.log_memory_state("WARNING", 180.5, 320.0)
    pdf_logger.log_extraction_method("hybrid_auto", 25, 0.89)
    pdf_logger.log_processing_complete("test.pdf", True, 45.2, 28500)
    
    # Test performance logging
    with log_memory_usage("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    # Test function decorator
    @log_function_call
    def test_function(x, y=10):
        time.sleep(0.05)
        return x + y
    
    result = test_function(5, y=15)
    
    print("Logging test complete. Check log files in ./output/logs/")