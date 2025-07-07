# League of Legends Match Outcome Predictor

A machine learning project to predict League of Legends match outcomes using 15-minute game data.

## Project Structure

```
League_Data_Analyzer/
├── data/                   # Raw match data files
├── models/                 # Trained models and scalers
├── processed_data/         # Processed datasets and visualizations
├── scripts/                # Python scripts
│   ├── process_matches.py  # Process raw match data
│   └── train_model.py      # Train and evaluate models
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Getting Started

### 1. Set up the environment

First, create and activate a virtual environment:

```bash
# On Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Unix/Linux
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Prepare the Data

Place your raw match data files in the `data/` directory with the naming format:
```
data/objectives_<REGION>_<MATCH_ID>.csv
```

### 3. Process the Data

Run the data processing script:
```bash
python scripts/process_matches.py
```

This will:
- Process all match files in the `data/` directory
- Extract relevant features
- Save processed data to `processed_data/match_features_15min.csv`

### 4. Train the Models

Train the prediction models:
```bash
python scripts/train_model.py
```

This will:
- Load and clean the processed data
- Perform exploratory data analysis
- Train and evaluate machine learning models
- Save trained models to the `models/` directory
- Generate visualizations in the `processed_data/` directory

## Data

### Input Data Format

Raw match data files should be placed in the `data/` directory with the following naming convention:
- `objectives_<REGION>_<MATCH_ID>.csv`

Each file should contain match data including:
- Match metadata (ID, duration, queue type, etc.)
- Gold summary (minute-by-minute gold for each team)
- Objective timeline (towers, dragons, heralds, etc.)

### Output Data

Processed data will be saved in the `processed_data/` directory:
- `match_features_15min.csv`: Extracted features for model training
- `processed_matches_with_features.csv`: Processed data with additional engineered features
- Visualizations (feature correlation, importance, etc.)

## Model Performance

The project includes two machine learning models:
1. **Logistic Regression** - A simple linear model for binary classification
2. **Random Forest** - An ensemble model that often provides better performance

### Performance Metrics
- **Accuracy**: ~75.6% for both models
- **Precision/Recall**: ~76% for both classes

### Key Features
Top predictive features identified:
1. Late game gold advantage
2. Gold difference at 15 minutes
3. Game duration
4. Gold difference at 14 minutes
5. Late game gold advantage (engineered feature)

## Output Files
- `processed_data/`
  - `match_features_15min.csv`: Processed match data with features
  - `processed_matches_with_features.csv`: Dataset with additional engineered features
  - `feature_importance.png`: Visualization of feature importance
  - `feature_correlation.png`: Heatmap of feature correlations
- `models/`
  - `logistic_regression_model.pkl`: Trained logistic regression model
  - `random_forest_model.pkl`: Trained random forest model
  - `scaler.pkl`: Feature scaler used for preprocessing

## Cleaning Up
To clean up generated files:
```bash
# Remove processed data and models
rm -rf processed_data/*
rm -rf models/*

# Remove debug outputs
rm -rf debug_output/*
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
