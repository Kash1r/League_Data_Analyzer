import os
from pathlib import Path

def inspect_file(file_path: Path):
    """Inspect the contents of a file and print information about it."""
    print(f"\nInspecting file: {file_path.name}")
    print("=" * 80)
    
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'utf-16']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                print(f"\nSuccessfully read with {encoding} encoding")
                print("First 200 characters:")
                print("-" * 40)
                print(content[:200])
                print("-" * 40)
                print("\nFile structure:")
                print("-" * 40)
                for i, line in enumerate(content.split('\n')[:20]):
                    print(f"{i+1:2d}: {line.strip()}")
                print("-" * 40)
                return True
                
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding")
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {str(e)}")
            continue
    
    print("\nFailed to read file with any encoding")
    return False

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Find the first match file
    match_files = list(data_dir.glob('objectives_*.csv'))
    
    if not match_files:
        print(f"No match files found in {data_dir}")
    else:
        # Inspect the first file
        inspect_file(match_files[0])
