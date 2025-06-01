from pathlib import Path

file_path = Path("analyzers/sample.pdf")

print(f"Exists: {file_path.exists()}")
print(f"Is file: {file_path.is_file()}")
print(f"Suffix: {file_path.suffix}")
print(f"Size (bytes): {file_path.stat().st_size if file_path.exists() else 0}")
