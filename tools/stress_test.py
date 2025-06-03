import argparse
import gc
import logging
import psutil
import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path to ensure pdf2text modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pdf2text.core.agent import PDFTextAgent
except ImportError:
    print("Error: Could not import PDFTextAgent. Ensure pdf2text is installed or PROJECT_ROOT is correct.")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(process)d - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_stress_test(pdf_dir_path: Path, num_iterations: int, base_output_dir: Optional[Path]):
    """
    Runs the stress test for PDF text extraction.
    """
    process = psutil.Process()
    initial_mem = process.memory_info().rss
    logger.info(f"Initial process memory: {initial_mem / 1024**2:.2f} MB")

    if not any(pdf_dir_path.glob("*.pdf")):
        logger.warning(f"No PDF files found in directory: {pdf_dir_path}. Stress test may not be effective.")

    for i in range(num_iterations):
        iteration_num = i + 1
        logger.info(f"Starting iteration {iteration_num}/{num_iterations}")
        mem_before_iteration = process.memory_info().rss
        logger.info(f"Memory before iteration {iteration_num}: {mem_before_iteration / 1024**2:.2f} MB")

        output_dir_for_iteration: Path
        temp_dir_context = None # For managing TemporaryDirectory lifecycle

        if base_output_dir:
            output_dir_for_iteration = base_output_dir / f"iteration_{iteration_num}"
            output_dir_for_iteration.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using output directory: {output_dir_for_iteration}")
        else:
            temp_dir_context = tempfile.TemporaryDirectory(prefix=f"stress_test_iter_{iteration_num}_")
            output_dir_for_iteration = Path(temp_dir_context.name)
            logger.info(f"Using temporary output directory: {output_dir_for_iteration}")

        try:
            # Initialize PDFTextAgent inside the loop for full lifecycle testing
            agent = PDFTextAgent()

            logger.info(f"Processing batch from '{pdf_dir_path}' to '{output_dir_for_iteration}'")
            # process_batch expects string paths
            results = agent.process_batch(
                input_dir=str(pdf_dir_path),
                output_dir=str(output_dir_for_iteration)
            )

            logger.info(f"Iteration {iteration_num} processing complete. Results: {results}")

            # Explicitly shut down the agent to trigger cleanup
            logger.info(f"Shutting down agent for iteration {iteration_num}")
            agent.shutdown()
            logger.info(f"Agent for iteration {iteration_num} shut down.")

        except Exception as e:
            logger.error(f"Error during iteration {iteration_num}: {e}", exc_info=True)
        finally:
            # Force garbage collection
            gc.collect()
            logger.debug(f"Garbage collection forced after iteration {iteration_num}.")

            mem_after_iteration = process.memory_info().rss
            iteration_diff = mem_after_iteration - mem_before_iteration
            logger.info(
                f"Iteration {iteration_num}: "
                f"Memory Before: {mem_before_iteration / 1024**2:.2f}MB, "
                f"Memory After: {mem_after_iteration / 1024**2:.2f}MB, "
                f"Diff: {iteration_diff / 1024**2:.2f}MB"
            )

            if temp_dir_context:
                try:
                    temp_dir_context.cleanup()
                    logger.info(f"Temporary directory {output_dir_for_iteration} cleaned up.")
                except Exception as e:
                    logger.error(f"Failed to cleanup temporary directory {output_dir_for_iteration}: {e}")

        # Optional: Add a small delay between iterations if needed, e.g., time.sleep(1)
        # time.sleep(1)

    final_mem = process.memory_info().rss
    overall_diff = final_mem - initial_mem
    logger.info(
        f"\nStress Test Complete. Overall Memory Change: "
        f"Initial: {initial_mem / 1024**2:.2f}MB, "
        f"Final: {final_mem / 1024**2:.2f}MB, "
        f"Diff: {overall_diff / 1024**2:.2f}MB"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test script for PDF text extraction application.")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        required=True,
        help="Path to the directory containing PDF files for testing."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations to run the test. Default is 5."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base directory for output files. If not provided, a temporary directory will be created for each iteration."
    )

    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        logger.error(f"Error: PDF directory '{pdf_dir}' does not exist or is not a directory.")
        sys.exit(1)

    num_iterations = args.iterations
    if num_iterations <= 0:
        logger.error("Error: Number of iterations must be a positive integer.")
        sys.exit(1)

    base_output_path = Path(args.output_dir) if args.output_dir else None
    if base_output_path:
        base_output_path.mkdir(parents=True, exist_ok=True)


    logger.info(f"Starting stress test with {num_iterations} iterations.")
    logger.info(f"PDF Source Directory: {pdf_dir.resolve()}")
    if base_output_path:
        logger.info(f"Base Output Directory: {base_output_path.resolve()}")
    else:
        logger.info("Using temporary directories for iteration outputs (will be cleaned up).")

    try:
        run_stress_test(pdf_dir, num_iterations, base_output_path)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred during the stress test: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Stress test script finished.")
        # Ensure all handlers are flushed, especially if using file logging in a more complex setup
        logging.shutdown()
```
