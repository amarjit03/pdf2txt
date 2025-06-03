"""
File Manager - Safe File Operations with Memory Integration
Handles all file I/O operations with memory awareness and cleanup
"""

import os
import shutil
import tempfile
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator
from dataclasses import dataclass
from enum import Enum
import logging
import time
import threading
from contextlib import contextmanager

import sys

# Ensure the root directory is on the import path
# sys.path.append(str(Path(__file__).resolve().parent.parent)) # Removed for packaging
from pdf2text.config import get_config
from pdf2text.core.memory_manager import get_memory_manager, ManagedResource, CleanupPriority


class FileType(Enum):
    """Supported file types"""
    PDF = "pdf"
    TEXT = "txt"
    JSON = "json"
    IMAGE = "image"
    TEMP = "temp"


class FileOperation(Enum):
    """File operation types for logging"""
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"


@dataclass
class FileHandle:
    """Managed file handle with automatic cleanup"""
    file_path: Path
    file_type: FileType
    size_mb: float
    created_at: float
    last_accessed: float
    is_temporary: bool = False
    auto_cleanup: bool = True
    resource_id: Optional[str] = None


class FileManager:
    """Memory-aware file operations manager"""
    
    def __init__(self):
        self.config = get_config()
        self.memory_manager = get_memory_manager()
        self.logger = logging.getLogger(__name__)
        
        # File tracking
        self.open_files: Dict[str, FileHandle] = {}
        self.temp_files: List[Path] = []
        self.file_locks: Dict[str, threading.Lock] = {}
        
        # Output directories
        self.output_base = Path(self.config.output.base_output_dir)
        self.ensure_output_directories()
        
        # File operation stats
        self.stats = {
            'files_read': 0,
            'files_written': 0,
            'temp_files_created': 0,
            'files_cleaned': 0,
            'total_bytes_read': 0,
            'total_bytes_written': 0
        }
        
        self.logger.info("File Manager initialized")
    
    def ensure_output_directories(self):
        """Ensure all required output directories exist"""
        directories = [
            self.output_base,
            self.output_base / 'text_files',
            self.output_base / 'json_files',
            self.output_base / 'logs',
            self.output_base / 'failed',
            self.output_base / 'temp'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")
    
    def get_output_path(self, input_filename: str, output_type: str) -> Path:
        """Get appropriate output path for a file"""
        stem = Path(input_filename).stem
        
        path_map = {
            'text': self.output_base / 'text_files' / f"{stem}_extracted.txt",
            'json': self.output_base / 'json_files' / f"{stem}_extracted.json", 
            'failed': self.output_base / 'failed' / f"{stem}_FAILED.txt",
            'log': self.output_base / 'logs' / f"{stem}_processing.log",
            'temp': self.output_base / 'temp' / f"{stem}_temp"
        }
        
        return path_map.get(output_type, self.output_base / f"{stem}_{output_type}")
    
    def read_file_safe(self, file_path: Path, max_size_mb: float = 100) -> str:
        """Safely read a file with memory checks"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Check file size
        if file_size_mb > max_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB")
        
        # Check available memory
        if not self.memory_manager.check_can_allocate(file_size_mb * 2):  # 2x for buffer
            raise MemoryError(f"Insufficient memory to read {file_size_mb:.1f}MB file")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.stats['files_read'] += 1
            self.stats['total_bytes_read'] += len(content.encode('utf-8'))
            
            self.logger.debug(f"Read file: {file_path} ({file_size_mb:.1f}MB)")
            return content
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    self.logger.warning(f"File read with {encoding} encoding: {file_path}")
                    return content
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Could not decode file with any supported encoding: {file_path}")
    
    def write_file_safe(self, file_path: Path, content: str, 
                       backup_existing: bool = True) -> bool:
        """Safely write content to file with memory checks"""
        content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
        
        # Check memory availability
        if not self.memory_manager.check_can_allocate(content_size_mb):
            self.logger.error(f"Insufficient memory to write {content_size_mb:.1f}MB file")
            return False
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file if requested
        backup_path = None
        if backup_existing and file_path.exists():
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            try:
                shutil.copy2(file_path, backup_path)
            except Exception as e:
                self.logger.warning(f"Could not create backup: {e}")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.stats['files_written'] += 1
            self.stats['total_bytes_written'] += len(content.encode('utf-8'))
            
            # Remove backup if write was successful
            if backup_path and backup_path.exists():
                backup_path.unlink()
            
            self.logger.debug(f"Wrote file: {file_path} ({content_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write file {file_path}: {e}")
            
            # Restore backup if write failed
            if backup_path and backup_path.exists():
                try:
                    shutil.move(backup_path, file_path)
                    self.logger.info("Restored backup after write failure")
                except Exception as restore_error:
                    self.logger.error(f"Could not restore backup: {restore_error}")
            
            return False
    
    def write_json_safe(self, file_path: Path, data: Dict, 
                       indent: int = 2) -> bool:
        """Safely write JSON data to file"""
        try:
            json_content = json.dumps(data, indent=indent, ensure_ascii=False)
            return self.write_file_safe(file_path, json_content)
        except Exception as e:
            self.logger.error(f"Failed to serialize JSON for {file_path}: {e}")
            return False
    
    def read_json_safe(self, file_path: Path) -> Optional[Dict]:
        """Safely read JSON file"""
        try:
            content = self.read_file_safe(file_path)
            return json.loads(content)
        except Exception as e:
            self.logger.error(f"Failed to read JSON from {file_path}: {e}")
            return None
    
    def create_temp_file(self, suffix: str = "", prefix: str = "pdf_agent_") -> Path:
        """Create a managed temporary file"""
        temp_dir = self.output_base / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        # Create unique filename
        timestamp = str(int(time.time() * 1000))
        temp_filename = f"{prefix}{timestamp}{suffix}"
        temp_path = temp_dir / temp_filename
        
        # Track temporary file
        self.temp_files.append(temp_path)
        
        # Register with memory manager for cleanup
        temp_resource = ManagedResource(
            resource_id=f"temp_file_{timestamp}",
            resource_type="temp_file",
            size_mb=0.0,  # Will be updated when file is written
            created_at=time.time(),
            last_accessed=time.time(),
            cleanup_callback=lambda: self._cleanup_temp_file(temp_path),
            priority=CleanupPriority.LOW
        )
        
        self.memory_manager.register_resource(temp_resource)
        
        self.stats['temp_files_created'] += 1
        self.logger.debug(f"Created temp file: {temp_path}")
        
        return temp_path
    
    def _cleanup_temp_file(self, temp_path: Path):
        """Clean up a specific temporary file"""
        try:
            if temp_path.exists():
                temp_path.unlink()
                self.logger.debug(f"Cleaned temp file: {temp_path}")
            
            if temp_path in self.temp_files:
                self.temp_files.remove(temp_path)
                
            self.stats['files_cleaned'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp file {temp_path}: {e}")
    
    def cleanup_old_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary files"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned = 0
        
        for temp_path in self.temp_files[:]:  # Copy list to avoid modification during iteration
            try:
                if temp_path.exists():
                    stat = temp_path.stat()
                    if stat.st_ctime < cutoff_time:
                        temp_path.unlink()
                        self.temp_files.remove(temp_path)
                        cleaned += 1
                else:
                    # File already gone, remove from tracking
                    self.temp_files.remove(temp_path)
                    
            except Exception as e:
                self.logger.error(f"Error cleaning temp file {temp_path}: {e}")
        
        if cleaned > 0:
            self.logger.info(f"Cleaned {cleaned} old temporary files")
            self.stats['files_cleaned'] += cleaned
    
    def save_text_result(self, input_filename: str, extracted_text: str, 
                        metadata: Optional[Dict] = None) -> Path:
        """Save extracted text with optional metadata"""
        output_path = self.get_output_path(input_filename, 'text')
        
        # Prepare content
        content_parts = []
        
        if metadata and self.config.output.include_metadata:
            content_parts.append("=== PDF EXTRACTION RESULTS ===")
            content_parts.append(f"Source: {input_filename}")
            content_parts.append(f"Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            for key, value in metadata.items():
                content_parts.append(f"{key}: {value}")
            
            content_parts.append("\n=== EXTRACTED TEXT ===")
        
        content_parts.append(extracted_text)
        
        if metadata and self.config.output.include_metadata:
            content_parts.append(f"\n=== PROCESSING NOTES ===")
            content_parts.append(f"Text length: {len(extracted_text)} characters")
            content_parts.append(f"Extraction completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        full_content = "\n".join(content_parts)
        
        if self.write_file_safe(output_path, full_content):
            self.logger.info(f"Saved text result: {output_path}")
            return output_path
        else:
            raise IOError(f"Failed to save text result to {output_path}")
    
    def save_json_result(self, input_filename: str, extraction_data: Dict) -> Path:
        """Save complete extraction results as JSON"""
        output_path = self.get_output_path(input_filename, 'json')
        
        # Add file metadata
        json_data = {
            'extraction_metadata': {
                'source_file': input_filename,
                'processed_date': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'agent_version': '1.0.0',
                'output_format': 'json'
            },
            **extraction_data
        }
        
        if self.write_json_safe(output_path, json_data):
            self.logger.info(f"Saved JSON result: {output_path}")
            return output_path
        else:
            raise IOError(f"Failed to save JSON result to {output_path}")
    
    def save_error_report(self, input_filename: str, error_info: Dict) -> Path:
        """Save error report for failed processing"""
        output_path = self.get_output_path(input_filename, 'failed')
        
        error_content = [
            "=== PDF PROCESSING FAILED ===",
            f"File: {input_filename}",
            f"Failed at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=== ERROR DETAILS ==="
        ]
        
        for key, value in error_info.items():
            if isinstance(value, list):
                error_content.append(f"{key}:")
                for item in value:
                    error_content.append(f"  - {item}")
            else:
                error_content.append(f"{key}: {value}")
        
        error_content.extend([
            "",
            "=== TROUBLESHOOTING SUGGESTIONS ===",
            "1. Check if the PDF file is corrupted",
            "2. Verify the file is not password protected", 
            "3. Try using a different PDF processing tool",
            "4. Contact support if the issue persists"
        ])
        
        full_content = "\n".join(error_content)
        
        if self.write_file_safe(output_path, full_content):
            self.logger.info(f"Saved error report: {output_path}")
            return output_path
        else:
            raise IOError(f"Failed to save error report to {output_path}")
    
    def stream_write_large_file(self, file_path: Path, content_iterator: Iterator[str],
                               chunk_size: int = 1024 * 1024) -> bool:
        """Write large content using streaming to avoid memory issues"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                bytes_written = 0
                
                for chunk in content_iterator:
                    f.write(chunk)
                    bytes_written += len(chunk.encode('utf-8'))
                    
                    # Periodic memory check for very large files
                    if bytes_written > chunk_size:
                        if not self.memory_manager.check_can_allocate(10):  # Need some buffer
                            self.memory_manager._trigger_cleanup(CleanupPriority.MEDIUM)
                        bytes_written = 0
            
            self.stats['files_written'] += 1
            self.logger.info(f"Streamed large file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stream write {file_path}: {e}")
            return False
    
    def get_file_stats(self) -> Dict:
        """Get file operation statistics"""
        return {
            **self.stats,
            'open_files': len(self.open_files),
            'temp_files': len(self.temp_files),
            'output_directories': [
                str(self.output_base / subdir) 
                for subdir in ['text_files', 'json_files', 'logs', 'failed']
            ]
        }
    
    def cleanup_all_temp_files(self):
        """Clean up all temporary files"""
        cleaned = 0
        for temp_path in self.temp_files[:]:
            try:
                if temp_path.exists():
                    temp_path.unlink()
                    cleaned += 1
                self.temp_files.remove(temp_path)
            except Exception as e:
                self.logger.error(f"Error cleaning temp file {temp_path}: {e}")
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} temporary files")
            self.stats['files_cleaned'] += cleaned
    
    @contextmanager
    def managed_temp_file(self, suffix: str = "", prefix: str = "pdf_agent_"):
        """Context manager for automatic temp file cleanup"""
        temp_path = self.create_temp_file(suffix, prefix)
        try:
            yield temp_path
        finally:
            self._cleanup_temp_file(temp_path)
    
    def shutdown(self):
        """Clean shutdown - cleanup all managed files"""
        self.logger.info("File Manager shutting down")
        
        # Cleanup temp files
        self.cleanup_all_temp_files()
        
        # Close any open file handles
        for file_id in list(self.open_files.keys()):
            try:
                # File handles would be closed here if we were tracking them
                pass
            except Exception as e:
                self.logger.error(f"Error closing file {file_id}: {e}")
        
        self.logger.info("File Manager shutdown complete")


# Global file manager instance
_file_manager: Optional[FileManager] = None

def get_file_manager() -> FileManager:
    """Get the global file manager instance"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager


if __name__ == "__main__":
    # Test file manager
    import tempfile
    
    manager = FileManager()
    
    print("=== File Manager Test ===")
    
    # Test temp file creation
    with manager.managed_temp_file(suffix=".txt") as temp_path:
        print(f"Created temp file: {temp_path}")
        
        # Test writing
        test_content = "This is test content for the PDF agent.\n" * 100
        success = manager.write_file_safe(temp_path, test_content)
        print(f"Write success: {success}")
        
        # Test reading
        if success:
            read_content = manager.read_file_safe(temp_path)
            print(f"Read {len(read_content)} characters")
    
    # Test JSON operations
    test_data = {
        'test': 'data',
        'numbers': [1, 2, 3],
        'nested': {'key': 'value'}
    }
    
    json_path = manager.get_output_path("test.pdf", "json")
    json_success = manager.write_json_safe(json_path, test_data)
    print(f"JSON write success: {json_success}")
    
    if json_success:
        read_data = manager.read_json_safe(json_path)
        print(f"JSON read success: {read_data is not None}")
        json_path.unlink()  # Cleanup
    
    # Show stats
    stats = manager.get_file_stats()
    print("\nFile Manager Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    manager.cleanup_old_temp_files(max_age_hours=0)  # Clean all temp files
    manager.shutdown()