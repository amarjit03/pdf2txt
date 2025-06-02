#!/usr/bin/env python3
"""
PDF-to-Text Agent - Main Entry Point
Simple command-line interface for converting PDFs to text

Usage:
    python main.py input.pdf                    # Process single file with defaults
    python main.py input.pdf --output json      # Output JSON only
    python main.py input.pdf --mode quality     # High quality processing
    python main.py --batch folder/              # Process all PDFs in folder
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Import our configuration
from config import get_config, print_config_summary, OutputFormat, ProcessingMode

# We'll import these as we create them in next steps
# from core.agent import PDFTextAgent
# from utils.logger import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PDF-to-Text Agent - Convert PDFs to readable text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf                     # Basic conversion
  python main.py scan.pdf --output both          # Save as text and JSON
  python main.py large.pdf --mode fast           # Fast processing
  python main.py contract.pdf --mode quality     # High quality OCR
  python main.py --batch ./pdfs/                 # Process all PDFs in folder
  python main.py --config                        # Show current configuration
        """
    )
    
    # Main input argument
    parser.add_argument(
        'input', 
        nargs='?',
        help='PDF file to process or directory for batch processing'
    )
    
    # Processing options
    parser.add_argument(
        '--mode', 
        choices=['fast', 'balanced', 'quality'],
        default='balanced',
        help='Processing mode: fast (speed), balanced (default), quality (accuracy)'
    )
    
    parser.add_argument(
        '--output',
        choices=['text', 'json', 'both'], 
        default='both',
        help='Output format: text (.txt), json (.json), or both (default)'
    )
    
    parser.add_argument(
        '--language',
        default='eng',
        help='OCR language code (default: eng for English)'
    )
    
    # Batch processing
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all PDFs in the specified directory'
    )
    
    # Configuration and info
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=int,
        help='Maximum memory to use in MB (overrides config)'
    )
    
    return parser.parse_args()


def validate_input(input_path: str, is_batch: bool) -> bool:
    """Validate input path and type"""
    path = Path(input_path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return False
    
    if is_batch:
        if not path.is_dir():
            print(f"Error: Batch mode requires a directory, got: {input_path}")
            return False
    else:
        if not path.is_file():
            print(f"Error: Expected a file, got: {input_path}")
            return False
        
        if path.suffix.lower() != '.pdf':
            print(f"Error: File must be a PDF, got: {path.suffix}")
            return False
    
    return True


def find_pdf_files(directory: Path) -> List[Path]:
    """Find all PDF files in directory"""
    pdf_files = []
    
    for file_path in directory.rglob('*.pdf'):
        if file_path.is_file():
            pdf_files.append(file_path)
    
    pdf_files.sort()  # Process in alphabetical order
    return pdf_files


def process_single_file(file_path: Path, config) -> bool:
    """Process a single PDF file using the complete agent"""
    from core.agent import PDFTextAgent
    
    print(f"\n📄 Processing: {file_path.name}")
    print(f"   Size: {file_path.stat().st_size / (1024*1024):.1f} MB")
    
    def progress_callback(percentage: int, message: str):
        # Create progress bar
        bar_length = 20
        filled_length = int(bar_length * percentage // 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   [{bar}] {percentage:3d}% - {message}")
    
    agent = PDFTextAgent(progress_callback=progress_callback)
    
    try:
        result = agent.process_file(str(file_path))
        
        if result.success:
            print(f"   ✅ Success! Processed in {result.processing_time:.1f} seconds")
            print(f"   📊 Method: {result.method_used}")
            print(f"   📊 Confidence: {result.extraction_confidence:.2f}")
            print(f"   📊 Characters: {result.total_characters:,}")
            print(f"   📊 Memory peak: {result.memory_peak_mb:.1f}MB")
            
            # Show output files
            if result.output_files:
                for format_type, file_path in result.output_files.items():
                    if format_type == 'text':
                        print(f"   📝 Text saved: {file_path}")
                    elif format_type == 'json':
                        print(f"   📊 JSON saved: {file_path}")
            
            # Show warnings if any
            if result.warnings:
                print(f"   ⚠️  Warnings:")
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    print(f"      - {warning}")
                if len(result.warnings) > 3:
                    print(f"      ... and {len(result.warnings) - 3} more")
            
            return True
        else:
            print(f"   ❌ Failed after {result.processing_time:.1f} seconds")
            if result.errors:
                print(f"   💥 Errors:")
                for error in result.errors[:2]:  # Show first 2 errors
                    print(f"      - {error}")
            
            return False
        
    except Exception as e:
        print(f"   💥 Unexpected error: {str(e)}")
        return False
    
    finally:
        agent.shutdown()


def process_batch(directory: Path, config) -> tuple:
    """Process all PDF files in directory using the complete agent"""
    from core.agent import PDFTextAgent
    
    pdf_files = find_pdf_files(directory)
    
    if not pdf_files:
        print(f"No PDF files found in: {directory}")
        return 0, 0
    
    print(f"\n📁 Batch Processing: Found {len(pdf_files)} PDF files")
    print("=" * 60)
    
    # Initialize agent for batch processing
    agent = PDFTextAgent()
    
    try:
        # Use the agent's batch processing method
        results = agent.process_batch(str(directory))
        
        # Count results
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        
        # Detailed results display
        print(f"\n📊 Detailed Results:")
        print("-" * 60)
        
        for filename, result in results.items():
            if result.success:
                print(f"✅ {filename}")
                print(f"   📊 {result.total_characters:,} characters | "
                      f"{result.method_used} | "
                      f"confidence: {result.extraction_confidence:.2f} | "
                      f"{result.processing_time:.1f}s")
                
                if result.output_files:
                    for format_type in result.output_files:
                        print(f"   📄 {format_type} file saved")
                
                if result.warnings:
                    print(f"   ⚠️  {len(result.warnings)} warnings")
            else:
                print(f"❌ {filename}")
                if result.errors:
                    error_preview = result.errors[0][:80] + "..." if len(result.errors[0]) > 80 else result.errors[0]
                    print(f"   💥 {error_preview}")
            print()
        
        return successful, failed
    
    finally:
        agent.shutdown()


def main():
    """Main application entry point"""
    args = parse_arguments()
    config = get_config()
    
    # Handle configuration display
    if args.config:
        print_config_summary()
        return 0
    
    # Validate input is provided
    if not args.input:
        print("Error: No input file or directory specified")
        print("Use --help for usage information")
        return 1
    
    # Update configuration based on arguments
    if args.memory_limit:
        config.memory.max_memory_mb = args.memory_limit
    
    config.processing.mode = ProcessingMode(args.mode)
    config.output.format = OutputFormat(args.output)
    config.processing.ocr_language = args.language
    
    # Validate input
    if not validate_input(args.input, args.batch):
        return 1
    
    # Show configuration summary
    if args.verbose:
        print_config_summary()
    
    print("🚀 PDF-to-Text Agent Starting...")
    print(f"   Mode: {config.processing.mode.value}")
    print(f"   Output: {config.output.format.value}")
    print(f"   Memory limit: {config.memory.max_memory_mb}MB")
    
    # Process files
    start_time = time.time()
    
    if args.batch:
        successful, failed = process_batch(Path(args.input), config)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"📊 Batch Processing Complete!")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Total time: {total_time:.1f} seconds")
        
        return 0 if failed == 0 else 1
        
    else:
        success = process_single_file(Path(args.input), config)
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total processing time: {total_time:.1f} seconds")
        
        return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)