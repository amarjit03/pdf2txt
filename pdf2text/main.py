import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Import our configuration
from config import get_config, print_config_summary, OutputFormat, ProcessingMode

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PDF-to-Text Agent - Convert PDFs to readable text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
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
    """validate input path and type"""
    path = Path(input_path)

    if not path.exists():
        print("ERROR : unable to get file path {input_path} ")
        return False
    
    if is_batch:
        if not path.is_dir():
            print(f"Error: Batch mode requires a directory, got: {input_path}")
            return False
    else:
        if not path.is_file():
            print("ERROR : expected a file got : {input_path}")
            return False
        
        if path.suffix.lower() != '.pdf':
            print(f"Error: File must be a PDF, got: {path.suffix}")
            return False
    return True

def find_pdf_files(directory: Path) -> List[Path]:
    pdf_files = []

    for file_path in directory.rglob('*pdf'):
        if file_path.is_file():
            pdf_files.append(file_path)
    
    pdf_files.sort(key=lambda x:x)
    return pdf_files

def process_single_file(file_path: Path, config) -> bool:
    """process a single pdf file"""
    print(f"\n📄 Processing: {file_path.name}")
    print(f"   Size: {file_path.stat().st_size / (1024*1024):.1f} MB")

    start_time = time.time()
    try:
        # This is where we'll call our agent (to be implemented in next steps)
        print("   Status: Analyzing PDF type...")
        time.sleep(2)  # Placeholder for actual processing
        
        print("   Status: Extracting text...")
        time.sleep(2)  # Placeholder for actual processing
        
        print("   Status: Saving results...")
        time.sleep(2)  # Placeholder for actual processing
        
        processing_time = time.time() - start_time
        print(f"   ✅ Success! Processed in {processing_time:.1f} seconds")
        
        # Show output file locations
        output_paths = config.get_output_paths(file_path.name)
        if config.output.format in [OutputFormat.TEXT_ONLY, OutputFormat.BOTH]:
            print(f"   📝 Text saved: {output_paths['text']}")
        if config.output.format in [OutputFormat.JSON_ONLY, OutputFormat.BOTH]:
            print(f"   📊 JSON saved: {output_paths['json']}")
        
        return True
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   ❌ Failed after {processing_time:.1f} seconds")
        print(f"   Error: {str(e)}")
        return False


def process_batch(directory: Path, config) -> tuple:
    """Process all PDF files in directory"""
    pdf_files = find_pdf_files(directory)
    
    if not pdf_files:
        print(f"No PDF files found in: {directory}")
        return 0, 0
    
    print(f"\n📁 Batch Processing: Found {len(pdf_files)} PDF files")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}]", end=" ")
        
        if process_single_file(pdf_file, config):
            successful += 1
        else:
            failed += 1
    
    return successful, failed

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



