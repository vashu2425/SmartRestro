# Unified Kitchen Management System

This project combines two modules:
1. Demand Waste Predictor
2. Smart Kitchen Sales Forecaster

## Project Structure
```
unified_project/
├── main.py                 # Main entry point
├── requirements.txt        # Project dependencies
├── README.md              # This file
├── src/
│   ├── demand_waste/      # Demand waste prediction module
│   └── smart_kitchen/     # Smart kitchen sales module
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── config/               # Configuration files
└── models/              # Saved models
```

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run either module using the main.py script:

1. For Demand Waste Predictor:
   ```bash
   python main.py demand
   ```

2. For Smart Kitchen Sales:
   ```bash
   python main.py sales
   ```

## Module Descriptions

### Demand Waste Predictor
- Predicts waste based on inventory data
- Uses machine learning to optimize inventory management
- Outputs predictions to data/processed/waste_prediction_results.csv

### Smart Kitchen Sales
- Forecasts sales using XGBoost and Prophet models
- Provides accuracy metrics and future predictions
- Uses configuration from config/config.yaml 