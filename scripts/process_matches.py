import os
import re
import csv
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure pandas to display more columns in the console output
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
import os
from pathlib import Path

# Use absolute path to ensure reliability
BASE_DIR = Path(__file__).parent.parent  # Points to League_Data_Analyzer directory
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'processed_data'
OUTPUT_FILE = OUTPUT_DIR / 'match_features_15min.csv'

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_gold_line(line: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse a line from the gold summary section.
    
    Example formats:
    - 1,"2,500","2,500",+0
    - 2,"2,994","3,011",-17
    """
    try:
        # Clean the line and split by comma, handling quoted values
        reader = csv.reader([line.strip()])
        parts = next(reader)
        
        # Clean each part
        parts = [p.strip('"\' ') for p in parts if p.strip()]
        
        if len(parts) < 4:
            return None
            
        # Parse minute (first column)
        minute = int(parts[0])
        
        # Parse team gold values (handle quoted numbers with commas)
        team_100_gold = int(parts[1].replace(',', ''))
        team_200_gold = int(parts[2].replace(',', ''))
        
        # Parse or calculate gold difference
        if len(parts) > 3 and parts[3].strip() and parts[3].strip() != '+0':
            gold_diff_str = parts[3].replace('+', '').replace(',', '').strip()
            gold_diff = int(gold_diff_str) if gold_diff_str else 0
        else:
            gold_diff = team_100_gold - team_200_gold
            
        return minute, team_100_gold, team_200_gold, gold_diff
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing gold line '{line}': {str(e)}")
        return None

def clean_value(value: str) -> str:
    """Clean up a value by removing leading/trailing spaces and quotes."""
    if not value:
        return ''
    # Remove leading/trailing whitespace and quotes
    value = value.strip().strip('"\'')
    # Remove leading comma if present
    if value.startswith(','):
        value = value[1:].strip()
    return value

def parse_metadata_line(line: str) -> Tuple[str, str]:
    """Parse a metadata line and return (key, value)."""
    # Handle lines with multiple colons (e.g., timestamps)
    if line.count(':') > 1 and 'Gold Summary' not in line and 'Objective Timeline' not in line:
        # Split only on the first colon
        parts = line.split(':', 1)
    else:
        parts = line.split(':', 1)
        
    if len(parts) == 2:
        key = parts[0].strip().lower().replace(' ', '_')
        value = clean_value(parts[1])
        return key, value
    return None, None

def parse_objective_line(line: str) -> Optional[Dict]:
    """Parse a line from the objective timeline."""
    parts = line.strip().split(',')
    if len(parts) < 3:
        return None
    
    time_str = parts[0]
    team = parts[1]
    objective = parts[2]
    details = ','.join(parts[3:]) if len(parts) > 3 else ""
    
    # Convert MM:SS to seconds
    minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    
    return {
        'time': time_str,
        'minutes': minutes,
        'team': team,
        'objective': objective,
        'details': details
    }

def process_match_file(file_path: Path, debug: bool = False) -> Optional[Dict]:
    """Process a single match file and return a dictionary of features.
    
    Args:
        file_path: Path to the match file to process
        debug: If True, print debug information
        
    Returns:
        dict: Processed match data or None if processing failed
    """
    if debug:
        print(f"\nProcessing file: {file_path.name}")
    
    match_data = {
        'match_id': None,
        'game_duration': 0,
        'queue': None,
        'game_mode': None,
        'winner': None,
        'surrendered': None,
        'gold_diff': {},
        'objectives': []
    }
    
    try:
        # Try reading with different encodings
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if debug:
                    print(f"  Successfully read file with {encoding} encoding")
                    print("  First 15 lines:")
                    for i, line in enumerate(lines[:15]):
                        print(f"    {i+1:2d}: {line}")
                
                # Process metadata (first few lines)
                for line in lines[:10]:  # Only check first 10 lines for metadata
                    if 'Match ID:' in line:
                        match_data['match_id'] = line.split(':', 1)[1].strip()
                    elif 'Game Duration:' in line:
                        duration_str = line.split(':', 1)[1].strip().replace(',', '')
                        match_data['game_duration'] = int(float(duration_str)) if duration_str else 0
                    elif 'Queue:' in line:
                        match_data['queue'] = line.split(':', 1)[1].strip()
                    elif 'Game Mode:' in line:
                        match_data['game_mode'] = line.split(':', 1)[1].strip()
                    elif 'Winner:' in line:
                        match_data['winner'] = line.split(':', 1)[1].strip()
                    elif 'Surrendered:' in line:
                        match_data['surrendered'] = line.split(':', 1)[1].strip()
                
                # Find the gold summary section
                gold_start = -1
                objective_start = -1
                
                for i, line in enumerate(lines):
                    if 'Gold Summary' in line or ('Minute' in line and 'Gold' in line):
                        gold_start = i + 1
                        break
                
                # Find the objective timeline section
                for i, line in enumerate(lines):
                    if 'Objective Timeline' in line or ('Time' in line and 'Objective' in line):
                        objective_start = i + 1
                        break
                
                # Parse gold data if found
                if gold_start > 0:
                    for line in lines[gold_start:]:
                        if not line.strip() or not line[0].isdigit():
                            continue
                        
                        gold_data = parse_gold_line(line)
                        if gold_data:
                            minute, team_100_gold, team_200_gold, gold_diff = gold_data
                            match_data['gold_diff'][minute] = gold_diff
                
                # Parse objective data if found
                if objective_start > 0:
                    for line in lines[objective_start:]:
                        if not line.strip() or ('Time' in line and 'Objective' in line):
                            continue
                        
                        # Simple parsing for objectives (can be enhanced)
                        parts = line.split(',', 3)
                        if len(parts) >= 3:
                            time_str = parts[0].strip()
                            team = parts[1].strip()
                            objective = parts[2].strip()
                            details = parts[3].strip() if len(parts) > 3 else ''
                            
                            # Parse time to minutes (format: "MM:SS" or "H:MM:SS")
                            time_parts = time_str.split(':')
                            if len(time_parts) == 2:  # MM:SS
                                minutes = int(time_parts[0])
                            elif len(time_parts) == 3:  # H:MM:SS
                                minutes = int(time_parts[0]) * 60 + int(time_parts[1])
                            else:
                                continue  # Skip if time format is invalid
                                
                            # Only include objectives in the first 15 minutes
                            if minutes <= 15:
                                match_data['objectives'].append({
                                    'time': minutes,
                                    'team': team,
                                    'objective': objective,
                                    'details': details
                                })
                
                # Check if we got the basic data
                if not match_data['match_id'] or not match_data['winner']:
                    if debug:
                        print("  Warning: Missing required fields in metadata")
                    return None
                
                if debug:
                    print(f"  Successfully processed match {match_data['match_id']}")
                    print(f"  Game duration: {match_data['game_duration']} seconds")
                    print(f"  Winner: {match_data['winner']}")
                    print(f"  Gold diffs (minutes 1-5): {[match_data['gold_diff'].get(m, 'N/A') for m in range(1, 6)]}")
                    print(f"  Objectives (first 5): {match_data['objectives'][:5]}")
                
                return match_data
                
            except UnicodeDecodeError:
                if debug:
                    print(f"  Failed to read with {encoding} encoding")
                continue
            except Exception as e:
                if debug:
                    print(f"  Error processing with {encoding}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # If we get here, all encodings failed
        if debug:
            print("  Failed to read file with any encoding")
        return None
        
    except Exception as e:
        if debug:
            print(f"  Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
        return None

def extract_features(match_data: Dict) -> Dict:
    """Extract features from match data."""
    if not match_data or not match_data.get('match_id'):
        return None
        
    features = {}
    
    # Basic match info
    features['match_id'] = match_data.get('match_id', '')
    features['game_duration'] = match_data.get('game_duration', 0)
    features['queue'] = match_data.get('queue', '')
    features['game_mode'] = match_data.get('game_mode', '')
    features['surrendered'] = 1 if str(match_data.get('surrendered', '')).lower() == 'yes' else 0
    
    # Winner (1 for Team 100 win, 0 for Team 200 win)
    winner = match_data.get('winner', '')
    features['team_100_wins'] = 1 if winner == 'Team 100' else (0 if winner == 'Team 200' else -1)
    
    # Gold differential features
    gold_diff = match_data.get('gold_diff', {})
    for minute in range(1, 16):  # First 15 minutes
        features[f'gold_diff_{minute}'] = gold_diff.get(minute, 0)
    
    # Objective features (first 15 minutes)
    objectives = match_data.get('objectives', [])
    
    # Initialize objective counters
    obj_features = {
        'team_100_towers': 0,
        'team_200_towers': 0,
        'team_100_dragons': 0,
        'team_200_dragons': 0,
        'team_100_heralds': 0,
        'team_200_heralds': 0,
        'team_100_barons': 0,
        'team_200_barons': 0,
        'first_blood': -1,  # 1 if Team 100, 0 if Team 200, -1 if none
        'first_tower': -1,  # 1 if Team 100, 0 if Team 200, -1 if none
        'first_dragon': -1, # 1 if Team 100, 0 if Team 200, -1 if none
    }
    
    try:
        # Basic match info
        features['match_id'] = match_data.get('match_id', '')
        features['game_duration'] = match_data.get('game_duration', 0)
        features['queue'] = match_data.get('queue', '')
        features['game_mode'] = match_data.get('game_mode', '')
        
        # Determine the winner (1 if team 100 wins, 0 otherwise)
        winner = match_data.get('winner', '')
        if '100' in str(winner) or 'blue' in str(winner).lower():
            features['team_100_wins'] = 1
        elif '200' in str(winner) or 'red' in str(winner).lower():
            features['team_100_wins'] = 0
        else:
            # Skip if we can't determine the winner
            return None
        
        # Surrendered flag
        surrendered = match_data.get('surrendered', '')
        features['surrendered'] = 1 if str(surrendered).lower() == 'true' else 0
        
        # Gold difference features (up to 15 minutes)
        gold_diffs = match_data.get('gold_diff', {})
        for minute in range(1, 16):
            features[f'gold_diff_{minute}'] = gold_diffs.get(minute, 0)
        
        # Objective features (up to 15 minutes)
        objectives = match_data.get('objectives', [])
        obj_features = {
            'first_blood': 0,          # 1 if team 100 got first blood, -1 if team 200 did
            'first_tower': 0,          # 1 if team 100 got first tower, -1 if team 200 did
            'first_dragon': 0,         # 1 if team 100 got first dragon, -1 if team 200 did
            'first_herald': 0,         # 1 if team 100 got first herald, -1 if team 200 did
            'towers_taken': 0,         # Net towers taken (team 100 - team 200)
            'dragons_taken': 0,        # Net dragons taken (team 100 - team 200)
            'heralds_taken': 0,        # Net heralds taken (team 100 - team 200)
            'barons_taken': 0,         # Net barons taken (team 100 - team 200)
            'inhibitors_taken': 0,     # Net inhibitors taken (team 100 - team 200)
            'objectives_count': 0      # Total number of objectives in first 15 minutes
        }
        
        for obj in objectives:
            if obj['time'] > 15:  # Only consider objectives in first 15 minutes
                continue
                
            obj_type = obj['objective'].lower()
            team = obj['team']
            is_team_100 = team == '100' or 'blue' in str(team).lower()
            
            # Determine the value to add (positive for team 100, negative for team 200)
            value = 1 if is_team_100 else -1
            
            # Categorize the objective
            if 'blood' in obj_type:
                if obj_features['first_blood'] == 0:  # Only set first blood once
                    obj_features['first_blood'] = value
            elif 'tower' in obj_type:
                obj_features['towers_taken'] += value
                if obj_features['first_tower'] == 0:  # First tower
                    obj_features['first_tower'] = value
            elif 'dragon' in obj_type:
                obj_features['dragons_taken'] += value
                if obj_features['first_dragon'] == 0:  # First dragon
                    obj_features['first_dragon'] = value
            elif 'herald' in obj_type or 'rift' in obj_type:
                obj_features['heralds_taken'] += value
                if obj_features['first_herald'] == 0:  # First herald
                    obj_features['first_herald'] = value
            elif 'baron' in obj_type:
                obj_features['barons_taken'] += value
            elif 'inhibitor' in obj_type:
                obj_features['inhibitors_taken'] += value
            
            # Increment total objectives count
            obj_features['objectives_count'] += 1
        
        # Add objective features to the main features dict
        features.update(obj_features)
        
        # Add some derived features
        features['gold_diff_15'] = features.get('gold_diff_15', 0)
        features['gold_diff_10'] = features.get('gold_diff_10', 0)
        features['gold_diff_5'] = features.get('gold_diff_5', 0)
        
        # Calculate gold advantage at different game stages
        features['gold_lead_early'] = features.get('gold_diff_5', 0)
        features['gold_lead_mid'] = features.get('gold_diff_10', 0) - features.get('gold_diff_5', 0)
        features['gold_lead_late'] = features.get('gold_diff_15', 0) - features.get('gold_diff_10', 0)
        
        # Calculate objective advantage
        features['objective_advantage'] = (
            obj_features['towers_taken'] * 5 +  # Towers are important
            obj_features['dragons_taken'] * 3 +  # Dragons give stacking buffs
            obj_features['heralds_taken'] * 4 +  # Heralds help take towers
            obj_features['barons_taken'] * 10 +  # Baron is very impactful
            obj_features['inhibitors_taken'] * 8  # Inhibitors create pressure
        )
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the DataFrame before saving.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Cleaned and validated DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Ensure all column names are strings and clean them
    df.columns = [str(col).strip() for col in df.columns]
    
    # Ensure all string columns are properly encoded and cleaned
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip("\"' ").str.replace('\n', ' ').replace('\r', '')
    
    # Convert numeric columns, coercing errors to NaN
    numeric_cols = [col for col in df.columns if col not in ['match_id', 'queue', 'game_mode']]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Ensure required columns exist
    required_columns = ['match_id', 'team_100_wins']
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"Required column '{col}' not found in the processed data")
            if col == 'team_100_wins':
                df[col] = 0  # Add default value if missing
    
    # Drop duplicate rows based on match_id if it exists
    if 'match_id' in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=['match_id'])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} duplicate matches")
    
    return df

def save_to_csv(df: pd.DataFrame, file_path: Path) -> bool:
    """Save DataFrame to CSV with proper error handling and validation.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Validate and clean the DataFrame
        df = validate_dataframe(df)
        if df.empty:
            logger.error("Cannot save empty DataFrame")
            return False
        
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with explicit settings for consistent output
        df.to_csv(
            file_path,
            index=False,
            encoding='utf-8',
            lineterminator='\n',
            quoting=csv.QUOTE_NONNUMERIC,
            float_format='%.2f',
            errors='replace'
        )
        
        # Verify the file was created and is not empty
        if not file_path.exists() or os.path.getsize(file_path) == 0:
            raise IOError("Failed to write CSV file or file is empty")
            
        logger.info(f"Successfully saved {len(df)} records to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {str(e)}", exc_info=True)
        return False

def process_all_matches(debug: bool = True) -> pd.DataFrame:
    """Process all match files and return a DataFrame with features.
    
    Args:
        debug: If True, print debug information
        
    Returns:
        pd.DataFrame: DataFrame containing extracted features
    """
    global DATA_DIR
    all_features = []
    processed_count = 0
    error_count = 0
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for files in: {DATA_DIR}")
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"Error: Directory does not exist: {DATA_DIR}")
        return pd.DataFrame()
    
    # Find all match files
    match_files = sorted(list(DATA_DIR.glob('objectives_*.csv')))
    
    if not match_files:
        print(f"No match files found in {DATA_DIR}. Directory contents:")
        try:
            print("\n".join(os.listdir(DATA_DIR)))
        except Exception as e:
            print(f"Error listing directory: {e}")
        return pd.DataFrame()
    
    print(f"Found {len(match_files)} match files to process...")
    
    # Create a test directory for debug output
    debug_dir = Path('debug_output')
    debug_dir.mkdir(exist_ok=True)
    
    # Process files in batches for better progress tracking
    batch_size = 10
    for i, file_path in enumerate(match_files, 1):
        try:
            # Process the match file with debug info for the first few files
            debug_file = debug and (i <= 5 or processed_count < 3)
            
            # Save the first few files for inspection
            if debug_file:
                debug_file_path = debug_dir / f"raw_{file_path.name}"
                with open(file_path, 'r', encoding='utf-8', errors='replace') as src, \
                     open(debug_file_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
                print(f"Saved debug copy to: {debug_file_path}")
            
            match_data = process_match_file(file_path, debug=debug_file)
            
            if not match_data:
                if debug_file:
                    print(f"  - No valid match data found in {file_path.name}")
                error_count += 1
                if i % batch_size == 0 or i == len(match_files):
                    print(f"Processed {i}/{len(match_files)} files ({processed_count} valid, {error_count} errors)...")
                continue
                
            # Extract features
            features = extract_features(match_data)
            if debug_file:
                print(f"  - Extracted features: {features}")
                
            if features and 'team_100_wins' in features and features['team_100_wins'] in [0, 1]:
                all_features.append(features)
                processed_count += 1
                if debug_file:
                    print(f"  - Successfully processed {file_path.name}")
                    
                    # Save the processed data for debugging
                    debug_processed_path = debug_dir / f"processed_{file_path.stem}.json"
                    import json
                    with open(debug_processed_path, 'w', encoding='utf-8') as f:
                        json.dump(features, f, indent=2)
                    print(f"  - Saved processed data to: {debug_processed_path}")
            else:
                if debug_file:
                    print(f"  - Invalid features or winner in {file_path.name}")
                    if features:
                        print(f"     Features: {features}")
                    else:
                        print("     No features extracted")
                error_count += 1
            
            if i % batch_size == 0 or i == len(match_files):
                print(f"Processed {i}/{len(match_files)} files ({processed_count} valid, {error_count} errors)...")
                
        except Exception as e:
            error_count += 1
            if debug or i <= 5:
                print(f"  - Error processing {file_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Save the error file for debugging
                error_file_path = debug_dir / f"error_{file_path.name}"
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as src, \
                         open(error_file_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                    print(f"  - Saved error file to: {error_file_path}")
                except Exception as save_error:
                    print(f"  - Could not save error file: {str(save_error)}")
            
            if i % batch_size == 0 or i == len(match_files):
                print(f"Processed {i}/{len(match_files)} files ({processed_count} valid, {error_count} errors)...")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} matches")
    print(f"Files with errors: {error_count}")
    
    if not all_features:
        print("Warning: No valid match data was processed!")
        return pd.DataFrame()
    
    # Create DataFrame and ensure proper data types
    df = pd.DataFrame(all_features)
    
    # Convert numeric columns to appropriate types
    numeric_cols = [col for col in df.columns if col not in ['match_id', 'queue', 'game_mode']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with missing target variable
    df = df.dropna(subset=['team_100_wins'])
    
    print(f"\nFinal dataset shape: {df.shape}")
    if not df.empty:
        print("\nSample of the processed data:")
        print(df.head())
    
    return df

def setup_logging():
    """Set up logging configuration."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('match_processor.log')
            ]
        )
        return logging.getLogger(__name__)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return None

if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    if logger is None:
        print("Failed to set up logging. Continuing without logging.")
    
    try:
        logger.info("Starting match data processing...")
        print("Starting match data processing...")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        
        # Ensure output directory exists
        output_dir = Path("processed_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "match_features_15min.csv"
        
        # Process all matches
        df = process_all_matches(debug=True)
        
        if df is not None and not df.empty:
            # Save to CSV using our robust function
            if save_to_csv(df, output_file):
                # Print summary
                print("\n=== Processing Complete ===")
                print(f"Total matches processed: {len(df)}")
                print(f"Output file: {output_file.absolute()}")
                print("\nSample of processed data:")
                print(df.head().to_string())
                
                # Log summary
                logger.info(f"Successfully processed {len(df)} matches. Results saved to {output_file}")
            else:
                logger.error("Failed to save processed data to CSV")
                print("\nERROR: Failed to save processed data to CSV. Check the logs for details.")
        else:
            error_msg = "No valid match data was processed. Check the input files and debug output."
            print(f"\nERROR: {error_msg}")
            logger.error(error_msg)
            
            # Check if any debug files were created
            debug_dir = Path('debug_output')
            if debug_dir.exists():
                debug_files = list(debug_dir.glob('*'))
                if debug_files:
                    print("\nDebug files found in debug_output/ directory:")
                    for f in debug_files[:5]:  # Show first 5 debug files
                        print(f"  - {f.name}")
                    if len(debug_files) > 5:
                        print(f"  ... and {len(debug_files) - 5} more")
                    
                    # Show content of first debug file
                    try:
                        first_debug = debug_files[0]
                        print(f"\nContent of {first_debug.name}:")
                        with open(first_debug, 'r', encoding='utf-8', errors='replace') as f:
                            for _ in range(10):  # Show first 10 lines
                                line = f.readline().strip()
                                if not line:
                                    break
                                print(f"  {line}")
                    except Exception as e:
                        print(f"  Could not read debug file: {str(e)}")
            
            # Try different encodings for the first file if needed
            try:
                test_files = list(DATA_DIR.glob('objectives_*.csv'))
                if test_files:
                    test_file = test_files[0]
                    print(f"\nTesting file: {test_file.name}")
                    
                    encodings = ['utf-8', 'latin1', 'cp1252', 'utf-16']
                    for encoding in encodings:
                        try:
                            with open(test_file, 'r', encoding=encoding) as f:
                                print(f"\nSample content with {encoding}:")
                                for _ in range(5):
                                    line = f.readline()
                                    if not line:
                                        break
                                    print(f"  {line.strip()}")
                        except UnicodeDecodeError:
                            print(f"  Failed to read with {encoding}")
                        except Exception as e:
                            print(f"  Error with {encoding}: {str(e)}")
            except Exception as e:
                print(f"Error during encoding test: {str(e)}")
    
    except Exception as e:
        error_msg = f"Fatal error during processing: {str(e)}"
        print(f"\nCRITICAL ERROR: {error_msg}")
        logger.critical(error_msg, exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nProcessing complete. Check the log file for details.")
