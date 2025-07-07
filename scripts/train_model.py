import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "processed_data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Set display options for better readability
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

def load_and_clean_data():
    """Load and clean the processed match data."""
    # Try to load the processed data file
    data_file = DATA_DIR / "match_features_15min.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
        
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file, encoding='utf-8')
    
    # If the file is empty, raise an error
    if df.empty:
        raise ValueError(f"The data file is empty: {data_file}")
    
    print(f"Successfully loaded {len(df)} rows")
    
    # Clean column names
    df.columns = [col.strip(", \"'") for col in df.columns]
    
    # Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].notna().any():  # Only process if column contains non-null values
            df[col] = df[col].astype(str).str.strip(", \"'")
    
    # Convert numeric columns (exclude non-numeric columns)
    non_numeric_cols = ['match_id', 'queue', 'game_mode']
    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
    
    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip(", \"'"), errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert column '{col}' to numeric: {str(e)}")
    
    # Convert team_100_wins to int (it's our target variable)
    if 'team_100_wins' in df.columns:
        df['team_100_wins'] = pd.to_numeric(df['team_100_wins'], errors='coerce').fillna(0).astype(int)
    
    # Drop rows with missing target values
    initial_count = len(df)
    df = df.dropna(subset=['team_100_wins'])
    if len(df) < initial_count:
        print(f"Dropped {initial_count - len(df)} rows with missing target values")
    
    print(f"Final dataset contains {len(df)} matches")
    
    # Ensure we have the required columns
    required_columns = ['team_100_wins']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data")
    
    return df

def create_features(df):
    """Create additional features from the existing ones."""
    # Gold difference features
    for minute in range(2, 16):
        df[f'gold_diff_trend_{minute}'] = df[f'gold_diff_{minute}'] - df[f'gold_diff_{minute-1}']
    
    # Early game gold advantage (first 5 minutes)
    df['early_gold_adv'] = df[[f'gold_diff_{i}' for i in range(1, 6)]].mean(axis=1)
    
    # Mid game gold advantage (minutes 6-10)
    df['mid_gold_adv'] = df[[f'gold_diff_{i}' for i in range(6, 11)]].mean(axis=1)
    
    # Late game gold advantage (minutes 11-15)
    df['late_gold_adv'] = df[[f'gold_diff_{i}' for i in range(11, 16)]].mean(axis=1)
    
    return df

def explore_data(df):
    """Perform exploratory data analysis on the dataset."""
    print("\n=== Dataset Overview ===")
    print(f"Number of matches: {len(df)}")
    print(f"Win rate (Team 100): {df['team_100_wins'].mean():.2f}")
    
    # Plot correlation matrix for key features
    plt.figure(figsize=(12, 10))
    corr_cols = ['team_100_wins', 'gold_diff_5', 'gold_diff_10', 'gold_diff_15', 
                'towers_taken', 'dragons_taken', 'heralds_taken']
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'feature_correlation.png')
    plt.close()
    
    # Plot feature importance
    X = df.drop(['match_id', 'team_100_wins', 'queue', 'game_mode'], axis=1, errors='ignore')
    y = df['team_100_wins']
    
    # Train a simple model to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importances.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'feature_importance.png')
    plt.close()
    
    return importances

def train_and_evaluate_models(df):
    """Train and evaluate machine learning models."""
    # Prepare features and target
    X = df.drop(['match_id', 'team_100_wins', 'queue', 'game_mode'], axis=1, errors='ignore')
    y = df['team_100_wins']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        
        # Train the model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            X_eval = X_test_scaled
        else:
            model.fit(X_train, y_train)
            X_eval = X_test
        
        # Make predictions
        y_pred = model.predict(X_eval)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Save the model
        model_file = MODEL_DIR / f'{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_file)
        print(f"Model saved to {model_file}")
        
        # Save results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report
        }
    
    return results

def main():
    """Main function to run the model training pipeline."""
    print("Starting model training pipeline...")
    
    try:
        # Step 1: Load and clean data
        print("\nStep 1: Loading and cleaning data...")
        df = load_and_clean_data()
        
        # Basic data validation
        if df.empty:
            raise ValueError("No data loaded. Please check the input files.")
            
        print(f"\nData loaded successfully with {len(df)} matches")
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        # Step 2: Create additional features
        print("\nStep 2: Creating additional features...")
        df = create_features(df)
        
        # Step 3: Explore the data
        print("\nStep 3: Exploring the data...")
        try:
            importances = explore_data(df)
            if importances is not None and not importances.empty:
                print("\nTop 10 most important features:")
                print(importances.head(10).to_string())
        except Exception as e:
            print(f"Warning: Could not generate feature importances: {str(e)}")
        
        # Step 4: Train and evaluate models
        print("\nStep 4: Training and evaluating models...")
        results = train_and_evaluate_models(df)
        
        # Save the processed dataset with new features
        processed_file = DATA_DIR / 'processed_matches_with_features.csv'
        df.to_csv(processed_file, index=False)
        
        print("\n=== Model Training Complete ===")
        print(f"- Models and artifacts saved to: {MODEL_DIR.absolute()}")
        print(f"- Processed dataset saved to: {processed_file}")
        
    except Exception as e:
        print(f"\n=== Error in model training pipeline ===")
        print(f"Error: {str(e)}")
        print("\nPlease check the input data and try again.")
        raise

if __name__ == "__main__":
    main()
