import os
import shutil
from pathlib import Path

def cleanup_project():
    """Remove unnecessary files and directories from the project."""
    project_root = Path(__file__).parent.parent
    
    # Directories to clean
    dirs_to_clean = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ipynb_checkpoints",
        "debug_output",  # Remove debug outputs
        "models",        # Will be recreated during training
    ]
    
    # File patterns to remove
    file_patterns = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".coverage",
        "coverage.xml",
        "*.log",
        "*.bak",
        "*.tmp"
    ]
    
    print("Cleaning up project...")
    
    # Remove directories
    for dir_name in dirs_to_clean:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"Removing directory: {dir_path}")
            shutil.rmtree(dir_path, ignore_errors=True)
    
    # Remove files matching patterns
    for pattern in file_patterns:
        for file_path in project_root.rglob(pattern):
            try:
                if file_path.is_file():
                    print(f"Removing file: {file_path}")
                    file_path.unlink()
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    # Remove empty directories
    for dirpath, dirnames, _ in os.walk(project_root, topdown=False):
        for dirname in dirnames:
            full_path = Path(dirpath) / dirname
            try:
                if full_path.is_dir() and not any(full_path.iterdir()):
                    print(f"Removing empty directory: {full_path}")
                    full_path.rmdir()
            except Exception as e:
                print(f"Error removing directory {full_path}: {e}")
    
    print("\nCleanup complete!")
    print("\nProject structure now contains:")
    for item in sorted(project_root.glob('*')):
        if item.is_dir():
            print(f"- {item.name}/")
        else:
            print(f"- {item.name}")

if __name__ == "__main__":
    cleanup_project()
