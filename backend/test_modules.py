#!/usr/bin/env python3
"""
Test script to verify both modules in the unified project work correctly.
"""

import os
import sys
from pathlib import Path

def test_demand_waste_module():
    """Test the demand waste predictor module"""
    print("\n=== Testing Demand Waste Predictor Module ===")
    try:
        # Import the module
        from src.demand_waste.data_preprocessor import load_inventory_data
        from src.demand_waste.waste_predictor import predict_waste
        
        # Check if data file exists
        input_path = "data/raw/inventory_dataset.csv"
        if not os.path.exists(input_path):
            print(f"Error: Data file not found at {input_path}")
            return False
            
        # Load data
        print(f"Loading data from: {input_path}")
        df = load_inventory_data(input_path)
        print(f"Loaded {len(df)} rows of data")
        
        # Make predictions
        predictions = predict_waste(df)
        print(f"Generated {len(predictions)} predictions")
        
        return True
    except Exception as e:
        print(f"Error testing Demand Waste Predictor: {str(e)}")
        return False

def test_smart_kitchen_module():
    """Test the smart kitchen sales module"""
    print("\n=== Testing Smart Kitchen Sales Module ===")
    try:
        # Import the module
        from src.smart_kitchen.data_preprocessor import DataPreprocessor
        from src.smart_kitchen.sales_forecaster_xgboost import SalesForecaster
        
        # Check if data file exists
        data_path = "data/raw/sales_data.csv"
        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            return False
            
        # Check if config file exists
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            return False
            
        # Load data
        print(f"Loading data from: {data_path}")
        preprocessor = DataPreprocessor(data_path)
        processed_data = preprocessor.preprocess()
        print(f"Processed {len(processed_data)} rows of data")
        
        # Initialize forecaster
        forecaster = SalesForecaster(config_path)
        print("Initialized SalesForecaster")
        
        return True
    except Exception as e:
        print(f"Error testing Smart Kitchen Sales: {str(e)}")
        return False

def main():
    """Run tests for both modules"""
    print("Testing Unified Kitchen Management System")
    print("=======================================")
    
    # Test demand waste module
    demand_waste_ok = test_demand_waste_module()
    
    # Test smart kitchen module
    smart_kitchen_ok = test_smart_kitchen_module()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Demand Waste Predictor: {'✓ PASS' if demand_waste_ok else '✗ FAIL'}")
    print(f"Smart Kitchen Sales: {'✓ PASS' if smart_kitchen_ok else '✗ FAIL'}")
    
    # Return success if both modules passed
    return demand_waste_ok and smart_kitchen_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 