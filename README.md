# League of Legends Match Outcome Predictor

This project analyzes League of Legends match data to predict match outcomes based on in-game statistics up to the 15-minute mark.

## Project Structure

```
League_Data_Analyzer/
├── data/                    # Raw match data files (CSV format)
├── processed_data/          # Processed and cleaned data files
├── models/                  # Trained model files
├── scripts/                 # Python scripts for data processing and modeling
│   ├── process_matches.py   # Script to process raw match data
│   ├── train_model.py       # Script to train and evaluate models
│   └── inspect_file.py      # Utility for inspecting data files
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## Setup

1. **Prerequisites**
   - Python 3.8+
   - Required Python packages (install using `pip install -r requirements.txt`):
     - pandas
     - numpy
     - scikit-learn
     - matplotlib
     - seaborn
     - joblib

2. **Installation**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd League_Data_Analyzer
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## Usage

### 1. Process Raw Match Data

Run the following command to process the raw match data files:

```bash
python scripts/process_matches.py
```

This will:
- Process all match files in the `data/` directory
- Extract relevant features
- Save the processed data to `processed_data/match_features_15min.csv`

### 2. Train and Evaluate Models

After processing the data, train the prediction models:

```bash
python scripts/train_model.py
```

This will:
- Load and clean the processed data
- Perform exploratory data analysis
- Train and evaluate machine learning models
- Save the trained models to the `models/` directory
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

## Model

The project includes two machine learning models:
1. **Logistic Regression** - A simple linear model for binary classification
2. **Random Forest** - An ensemble model that often provides better performance

Model performance metrics and visualizations are displayed during training and saved in the output directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
