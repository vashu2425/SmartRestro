import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yaml
import json
from pathlib import Path
import tempfile
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uvicorn
import cv2
import time  # Add this import for task tracking
import asyncio
import traceback

# Import your modules
from src.demand_waste.data_preprocessor import load_inventory_data
from src.demand_waste.waste_predictor import predict_waste
from src.smart_kitchen.data_preprocessor import DataPreprocessor
from src.smart_kitchen.sales_forecaster_prophet import SalesForecaster
from src.menu_optimization.recipe_recommender import RecipeRecommender
from src.menu_optimization.recipe_generator import RecipeGenerator
from src.menu_optimization.cost_optimizer import CostOptimizer
from src.food_spoilage_detection.food_spoilage_detection import FoodSpoilageDetector
from src.inventory_tracking.inventory_tracking import InventoryTracker
from src.inventory_tracking.stock_detection import StockDetector
from src.vision_analyis.food_waste_classification import FoodWasteClassifier
from src.vision_analyis.waste_heatmap import WasteHeatmapGenerator
from src.vision_analyis.PlDashboard import RestaurantWasteTracker

# No global initialization of objects here - only initialize when needed

# Helper function to delete files
def delete_file(file_path: str) -> None:
    """Delete a file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {str(e)}")

# Helper function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Create FastAPI app
app = FastAPI(
    title="Kitchen Management API",
    description="API for kitchen management operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8081", "http://127.0.0.1:8081", "http://192.168.1.26:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create temporary directory for uploaded files
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Load config
config_path = "config/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Models for request/response
class DemandWasteRequest(BaseModel):
    input_path: Optional[str] = "data/raw/inventory_data.csv"

class SalesForecastRequest(BaseModel):
    days_ahead: Optional[int] = 7

class RecipeRequest(BaseModel):
    current_date: Optional[str] = None

class SpoilageDetectionRequest(BaseModel):
    image_path: Optional[str] = None

class InventoryDetectionRequest(BaseModel):
    image_path: Optional[str] = None

class WasteClassificationRequest(BaseModel):
    image_path: Optional[str] = None

class WasteHeatmapRequest(BaseModel):
    image_path: Optional[str] = None

# Helper functions
def get_temp_file_path(extension: str) -> str:
    """Generate a temporary file path with the given extension"""
    return str(TEMP_DIR / f"{uuid.uuid4()}.{extension}")

# Task tracking dictionary to store task status
task_tracker = {}

# API endpoints
@app.get("/")
async def read_root():
    """Root endpoint that provides links to the main app and API tester"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kitchen Management API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            .card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .button {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Kitchen Management API Server</h1>
        
        <div class="card">
            <h2>Main Application</h2>
            <p>Access the main frontend application.</p>
            <a href="http://localhost:8081" class="button">Go to Main Application</a>
        </div>
        
        <div class="card">
            <h2>API Tester</h2>
            <p>Test the API endpoints directly in your browser.</p>
            <a href="/static/api_tester.html" class="button">Go to API Tester</a>
        </div>
        
        <div class="card">
            <h2>API Documentation</h2>
            <p>View the auto-generated FastAPI documentation.</p>
            <a href="/docs" class="button">View API Documentation</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# Simple test endpoint to check API functionality
@app.get("/api/test")
async def test_api():
    """Simple test endpoint to check if the API is working"""
    return {
        "status": "success",
        "message": "API is working correctly!",
        "endpoints": [
            {"path": "/api/demand-waste-prediction", "method": "GET/POST"},
            {"path": "/api/sales-forecasting", "method": "GET"},
            {"path": "/api/recipe-recommendation", "method": "GET"},
            {"path": "/api/recipe-generation", "method": "GET"},
            {"path": "/api/cost-optimization", "method": "GET"},
            {"path": "/api/spoilage-detection", "method": "GET/POST"},
            {"path": "/api/waste-classification", "method": "GET/POST"},
            {"path": "/api/inventory-tracking", "method": "GET/POST"},
            {"path": "/api/stock-detection", "method": "GET/POST"},
            {"path": "/api/waste-heatmap", "method": "GET/POST"},
            {"path": "/api/dashboard", "method": "GET"}
        ]
    }

@app.post("/api/demand-waste-prediction")
@app.get("/api/demand-waste-prediction")
async def run_demand_waste():
    """Run the demand waste prediction module"""
    try:
        print("\n=== Running Demand Waste Predictor Module ===")
        
        # Use default path from config
        input_path = config.get('data', {}).get('waste_inventory_path', "data/raw/inventory_data.csv")
        print(f"Loading data from: {input_path}")
        
        df = load_inventory_data(input_path)
        predictions = predict_waste(df)
        
        # Convert predictions to JSON-serializable format
        result = predictions.to_dict(orient='records')
        
        return JSONResponse(content={
            "status": "success",
            "predictions": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Demand Waste Predictor: {str(e)}")

@app.get("/api/sales-forecasting")
async def run_sales_forecast():
    """Run the smart kitchen sales forecasting module"""
    try:
        print("\n=== Running Smart Kitchen Sales Module ===")
        
        # Load config
        print(f"Loading config from: {config_path}")
        
        # Preprocess data
        processed_data = pd.read_csv(config['data']['raw_path'])
        processed_data['date'] = pd.to_datetime(processed_data['date'])

        # Train/test split
        train_cutoff = pd.to_datetime('2024-12-01')
        train_data = processed_data[processed_data['date'] <= train_cutoff]
        test_data = processed_data[processed_data['date'] > train_cutoff]
        
        # Train and evaluate model
        forecaster = SalesForecaster(config_path)
        # Check if models exist
        model_files = [f for f in os.listdir(config['model']['path']) if f.endswith('_model.json')]
        if len(model_files) < len(processed_data['item'].unique()):
            print("Training models...")
            forecaster.train(train_data)
        else:
            print("Loading existing models...")
            forecaster.load_models()
        accuracy = forecaster.evaluate(test_data)
        
        # Generate future predictions
        future_data = generate_future_data(processed_data, 7)  # Default to 7 days
        future_preds = forecaster.predict(future_data)
        
        # Convert to JSON-serializable format
        accuracy_dict = accuracy.to_dict()
        future_preds_dict = future_preds.to_dict(orient='records')
        
        return JSONResponse(content={
            "status": "success",
            "accuracy": accuracy_dict,
            "future_predictions": future_preds_dict
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Smart Kitchen Sales: {str(e)}")

@app.post("/api/sales-forecasting")
async def upload_sales_data(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None)
):
    """Upload sales data for forecasting"""
    try:
        print("\n=== Processing Uploaded Sales Data ===")
        
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Save uploaded file
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['csv', 'xls', 'xlsx']:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload CSV or Excel files.")
        
        temp_file_path = get_temp_file_path(file_extension)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add cleanup task
        background_tasks.add_task(os.remove, temp_file_path)
        
        # Process the uploaded file
        if file_extension == 'csv':
            processed_data = pd.read_csv(temp_file_path)
        else:
            processed_data = pd.read_excel(temp_file_path)
        
        # Ensure required columns exist
        required_columns = ['date', 'item', 'quantity']
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        if missing_columns:
            raise HTTPException(status_code=400, 
                                detail=f"Missing required columns: {', '.join(missing_columns)}. File must include date, item, and quantity columns.")
        
        # Convert date to datetime
        processed_data['date'] = pd.to_datetime(processed_data['date'])
        
        # Train/test split
        train_cutoff = processed_data['date'].max() - timedelta(days=7)  # Use last week as test data
        train_data = processed_data[processed_data['date'] <= train_cutoff]
        test_data = processed_data[processed_data['date'] > train_cutoff]
        
        if len(test_data) == 0:
            # If no test data, use last 10% of data as test
            train_size = int(len(processed_data) * 0.9)
            train_data = processed_data.iloc[:train_size]
            test_data = processed_data.iloc[train_size:]
        
        # Train and evaluate model
        forecaster = SalesForecaster(config_path)
        print("Training models on uploaded data...")
        forecaster.train(train_data)
        accuracy = forecaster.evaluate(test_data)
        
        # Generate future predictions
        future_data = generate_future_data(processed_data, 7)  # Default to 7 days
        future_preds = forecaster.predict(future_data)
        
        # Convert to JSON-serializable format
        accuracy_dict = accuracy.to_dict()
        future_preds_dict = future_preds.to_dict(orient='records')
        
        return JSONResponse(content={
            "status": "success",
            "message": "Sales data processed successfully",
            "accuracy": accuracy_dict,
            "future_predictions": future_preds_dict
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sales data: {str(e)}")

def generate_future_data(historical_data, days_ahead):
    """Generate future data for sales forecasting"""
    last_date = historical_data['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
    future_data = []
    
    for date in future_dates:
        for item in historical_data['item'].unique():
            item_data = historical_data[historical_data['item'] == item].tail(7)
            last_row = item_data.iloc[-1]
            lag_7 = item_data['quantity'].iloc[0] if len(item_data) >= 7 else 0
            future_data.append({
                'date': date,
                'item': item,
                'day_of_week': date.dayofweek,
                'month': date.month,
                'is_weekend': 1 if date.dayofweek in [5, 6] else 0,
                'lag_1': last_row['quantity'],
                'lag_7': lag_7
            })
    return pd.DataFrame(future_data)

@app.get("/api/recipe-recommendation")
async def run_recipe_recommender():
    """Run the recipe recommender module"""
    try:
        print("\n=== Running Recipe Recommender Module ===")
        
        # Initialize recommender
        recommender = RecipeRecommender(config_path)
        
        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get recipe recommendation
        recommended_recipe = recommender.suggest_daily_special(current_date)
        
        # If we have a valid recipe name, add the ingredients list
        if recommended_recipe["recipe_name"] != "No suitable special found":
            # Get the recipe details from the CSV file
            recipe_name = recommended_recipe["recipe_name"]
            recipes_df = pd.read_csv(recommender.config['data']['recipe_path'])
            recipe_data = recipes_df[recipes_df['recipe_name'] == recipe_name]
            
            if not recipe_data.empty:
                # Format ingredients as a list of strings
                ingredients_list = []
                for _, row in recipe_data.iterrows():
                    ingredient_text = f"{row['quantity']} {row['unit']} {row['ingredient']}"
                    ingredients_list.append(ingredient_text)
                
                # Add ingredients to the recommendation
                recommended_recipe["ingredients"] = ingredients_list
        
        return JSONResponse(content={
            "status": "success",
            "current_date": current_date,
            "recommended_recipe": recommended_recipe
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Recipe Recommender: {str(e)}")

@app.get("/api/recipe-generation")
async def run_recipe_generator():
    """Run the recipe generator module"""
    try:
        print("\n=== Running Recipe Generator Module ===")
        
        # Initialize generator
        generator = RecipeGenerator(config_path)
        
        # Generate recipes
        recipes = generator.generate_recipes()
        
        return JSONResponse(content={
            "status": "success",
            "recipes": recipes
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Recipe Generator: {str(e)}")

@app.get("/api/cost-optimization")
async def run_cost_optimizer():
    """Run the cost optimizer module"""
    try:
        print("\n=== Running Cost Optimizer Module ===")
        
        # Initialize optimizer
        optimizer = CostOptimizer(config_path)
        
        # Optimize costs
        optimized_costs = optimizer.optimize_costs()
        
        # Convert NumPy types to Python native types
        optimized_costs = convert_numpy_types(optimized_costs)
        
        return JSONResponse(content={
            "status": "success",
            "optimized_costs": optimized_costs
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Cost Optimizer: {str(e)}")

@app.get("/api/spoilage-detection")
async def run_spoilage_detection_get():
    """Run the food spoilage detection module with GET request"""
    try:
        print("\n=== Running Food Spoilage Detection Module (GET) ===")
        
        # Initialize detector
        detector = FoodSpoilageDetector(config_path)
        
        # Use a sample image
        sample_images = detector.get_sample_images()
        if sample_images:
            image_path = os.path.join(detector.config['data']['raw_spoilage_image_path'], sample_images[0])
        else:
            raise HTTPException(status_code=400, detail="No sample images found")
        
        # Detect spoilage
        result = detector.detect_spoilage(image_path)
        
        return JSONResponse(content={
            "status": "success",
            "image_path": image_path,
            "result": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Food Spoilage Detection: {str(e)}")

@app.post("/api/spoilage-detection")
async def run_spoilage_detection_post(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Run the food spoilage detection module with POST request and file upload"""
    try:
        print("\n=== Running Food Spoilage Detection Module (POST) ===")
        
        # Initialize detector
        detector = FoodSpoilageDetector(config_path)
        
        # Save uploaded file
        file_extension = file.filename.split('.')[-1].lower()
        temp_file_path = get_temp_file_path(file_extension)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add cleanup task
        background_tasks.add_task(os.remove, temp_file_path)
        
        # Use the temporary file
        image_path = temp_file_path
        
        # Detect spoilage
        result = detector.detect_spoilage(image_path)
        
        return JSONResponse(content={
            "status": "success",
            "image_path": image_path,
            "result": result
        })
    except Exception as e:
        print(f"Error in food spoilage detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Food Spoilage Detection: {str(e)}")


@app.get("/api/waste-classification")
async def run_waste_classification_get():
    """Run the waste classification module with GET request"""
    try:
        print("\n=== Running Waste Classification Module ===")
        
        # Initialize classifier
        classifier = FoodWasteClassifier(config_path)
        
        # Use the first sample image
        sample_images = classifier.config['data']['sample_waste_images']
        if not sample_images:
            raise HTTPException(status_code=400, detail="No sample images found in configuration")
            
        image_path = os.path.join(classifier.config['data']['raw_waste_image_path'], sample_images[6])
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail=f"Sample image not found at path: {image_path}")
            
        print(f"Using sample image: {image_path}")
        
        # Classify waste
        result = classifier.detect_food_waste(image_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to classify waste - no result returned")
        
        # Convert NumPy types to Python native types
        result = convert_numpy_types(result)
        
        return JSONResponse(content={
            "status": "success",
            "classification": result,
            "image_path": image_path
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in waste classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Waste Classification: {str(e)}")

@app.get("/api/inventory-tracking")
async def inventory_tracking():
    """Run inventory tracking on images."""
    try:
        tracker = InventoryTracker(config_path)
        print("tracker")
        results = tracker.detect_inventory()
        print(results)
        return {
            "status": "success",
            "message": "Inventory tracking completed successfully",
            "results": results,
            "output_path": tracker.annotated_image_path
        }
    except Exception as e:
        print(f"Error in inventory tracking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Inventory Tracking: {str(e)}")

@app.post("/api/inventory-tracking")
async def inventory_tracking_post(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Run inventory tracking on uploaded image."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file to temporary location
        temp_file_path = get_temp_file_path(".jpg")
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        
        try:
            # Initialize tracker with the uploaded image
            tracker = InventoryTracker(config_path)
            tracker.input_image_path = temp_file_path
            
            # Process the image
            results = tracker.detect_inventory()
            
            # Create output directory if it doesn't exist
            output_dir = Path("data/output/detection_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the original filename for the output
            original_filename = file.filename
            output_filename = f"detected_{original_filename}"
            output_path = output_dir / output_filename
            
            # Save the annotated image to the output directory
            if tracker.annotated_image_path:
                # Ensure the output directory exists
                output_dir.mkdir(parents=True, exist_ok=True)
                # Save the annotated image
                cv2.imwrite(str(output_path), tracker.annotated_image)
            
            # Create static directory if it doesn't exist
            static_dir = Path("static")
            static_dir.mkdir(exist_ok=True)
            
            # Copy the processed image to static directory
            static_path = static_dir / output_filename
            shutil.copy2(output_path, static_path)
            
            # Clean up temporary file
            background_tasks.add_task(delete_file, temp_file_path)
            
            # Convert results to JSON-serializable format
            serializable_results = convert_numpy_types(results)
            
            return JSONResponse(content={
                "status": "success",
                "message": "Inventory tracking completed successfully",
                "results": serializable_results,
                "output_path": f"/static/{output_filename}"
            })
        except Exception as e:
            # Clean up temporary file in case of processing error
            background_tasks.add_task(delete_file, temp_file_path)
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in inventory tracking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Inventory Tracking: {str(e)}")

@app.get("/api/stock-detection")
async def stock_detection():
    """Run stock detection on video."""
    try:
        print("stock detection")
        # Load config
        print(f"Loading config from: {config_path}")
        
        # Initialize detector with config
        detector = StockDetector(config_path=config_path)
        print("detector")
        results = detector.detect_stock()
        print(results)
        return {
            "status": "success",
            "message": "Stock detection completed successfully",
            "results": results,
            "video_url": f"/static/output_video.mp4"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stock-detection")
async def upload_stock_detection_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a video for stock detection"""
    try:
        print("Uploading video for stock detection")
        
        # Generate a unique filename for the uploaded video
        file_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())  # Generate task ID for tracking
        video_filename = f"uploaded_video_{file_id}.mp4"
        video_path = os.path.join(config['data']['video_image_path'].rsplit('/', 1)[0], video_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Save the uploaded file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update config with the new video path
        config['data']['video_image_path'] = video_path
        
        # Setup output video path
        output_filename = f"output_video_{file_id}.mp4"
        output_path = os.path.join("static", output_filename)
        results_filename = f"detection_results_{file_id}.json"
        results_path = os.path.join("static", results_filename)
        config['data']['output_video_image_path'] = output_path
        
        # Initialize task in the tracker
        task_tracker[task_id] = {
            "status": "processing",
            "progress": 0,
            "start_time": time.time(),
            "result": None,
            "error": None,
            "file_id": file_id
        }
        
        # Run detection in the background
        async def process_video():
            try:
                # Update task status to "processing"
                task_tracker[task_id]["status"] = "processing"
                
                # Create the StockDetector instance with the updated config
                detector = StockDetector(config_path=config_path)
                detector.video_path = video_path  # Explicitly set video path
                
                # Set output video path in the detector - use full path for processing
                detector.output_video_path = os.path.join(os.getcwd(), output_path)
                
                # Ensure the static directory exists
                os.makedirs("static", exist_ok=True)
                
                # Simulate progress updates while processing
                for progress in range(10, 90, 10):
                    # Simulate processing work
                    await asyncio.sleep(1)
                    # Update progress
                    task_tracker[task_id]["progress"] = progress
                
                # Run actual detection
                results = detector.detect_stock()
                print(f"Detection results: {results}")
                
                # Save results to a JSON file
                with open(results_path, "w") as f:
                    json.dump(results, f)
                
                # Update progress to reflect video processing is done
                task_tracker[task_id]["progress"] = 90
                
                # Define the expected paths
                expected_output_path = detector.output_video_path
                default_output_path = "data/output/stock_prediction/output_video.mp4"
                relative_output_path = output_path
                
                # Check if the detector output video exists
                if os.path.exists(expected_output_path):
                    print(f"Output video exists at: {expected_output_path}")
                    # Make sure we copy the file to the static directory if it's not already there
                    if expected_output_path != os.path.join(os.getcwd(), relative_output_path):
                        print(f"Copying from {expected_output_path} to {relative_output_path}")
                        shutil.copy2(expected_output_path, relative_output_path)
                elif os.path.exists(default_output_path):
                    print(f"Copying video from default path {default_output_path} to {relative_output_path}")
                    shutil.copy2(default_output_path, relative_output_path)
                else:
                    # Try to find any output video files that might have been created
                    output_dir = "data/output/stock_prediction"
                    if os.path.exists(output_dir):
                        output_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
                        if output_files:
                            # Use the most recently modified file
                            output_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                            found_path = os.path.join(output_dir, output_files[0])
                            print(f"Found alternative output video at {found_path}, copying to {relative_output_path}")
                            shutil.copy2(found_path, relative_output_path)
                        else:
                            error_msg = f"Could not find any output video in {output_dir}"
                            print(f"Warning: {error_msg}")
                            raise Exception(error_msg)
                    else:
                        error_msg = f"Could not find output video at {expected_output_path} or {default_output_path}"
                        print(f"Warning: {error_msg}")
                        raise Exception(error_msg)
                
                # Verify the output file in static directory actually exists
                if not os.path.exists(relative_output_path):
                    error_msg = f"Failed to copy output video to {relative_output_path}"
                    print(f"Error: {error_msg}")
                    raise Exception(error_msg)
                
                # Set the video URL (relative to the server root)
                base_url = ""  # Empty for relative URLs
                video_url = f"{base_url}/static/{output_filename}"
                results_url = f"{base_url}/static/{results_filename}"
                
                # Final progress update
                task_tracker[task_id]["progress"] = 100
                
                # Mark task as completed with results and video URL
                task_tracker[task_id]["status"] = "completed"
                
                task_tracker[task_id]["result"] = {
                    "results": results,
                    "video_url": video_url,
                    "results_url": results_url,
                    "timestamp": datetime.now().isoformat()
                }
                task_tracker[task_id]["end_time"] = time.time()
                
                # Log successful completion
                print(f"Task {task_id} completed successfully. Video available at: {video_url}")
                
            except Exception as e:
                # Mark task as failed
                task_tracker[task_id]["status"] = "failed"
                task_tracker[task_id]["error"] = str(e)
                task_tracker[task_id]["end_time"] = time.time()
                
                # Log the full error with traceback
                print(f"Error in background processing for task {task_id}: {str(e)}")
                print(f"Error details: {traceback.format_exc()}")
                
                # Try to clean up any temporary files
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        print(f"Cleaned up temporary video file: {video_path}")
                except Exception as cleanup_error:
                    print(f"Error during cleanup: {str(cleanup_error)}")
        
        # Start the background task
        background_tasks.add_task(process_video)
        
        return {
            "status": "success",
            "message": "Video uploaded and processing started",
            "task_id": task_id  # Return the task_id to the client
        }
    except Exception as e:
        print(f"Error uploading video: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading video: {str(e)}")

# Add task status endpoint
@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a task by its ID"""
    if task_id not in task_tracker:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_tracker[task_id]
    
    # Calculate elapsed time
    elapsed_time = time.time() - task["start_time"]
    
    response = {
        "status": task["status"],
        "progress": task["progress"],
        "elapsed_time": round(elapsed_time, 2)
    }
    
    # Include additional information based on status
    if task["status"] == "completed":
        response["result"] = task["result"]
    elif task["status"] == "failed":
        response["error"] = task["error"]
    
    return response

@app.post("/api/waste-classification")
async def run_waste_classification_post(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None)
):
    """Run the waste classification module with POST request"""
    try:
        print("\n=== Running Waste Classification Module ===")
        
        # Initialize classifier
        classifier = FoodWasteClassifier(config_path)
        
        # Handle image input
        if file:
            # Save uploaded file
            file_extension = file.filename.split('.')[-1].lower()
            temp_file_path = get_temp_file_path(file_extension)
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Add cleanup task
            background_tasks.add_task(os.remove, temp_file_path)
            
            # Use the temporary file
            image_path = temp_file_path
        else:
            # Use the first sample image if no image is provided
            sample_images = classifier.config['data']['sample_waste_images']
            if sample_images:
                image_path = os.path.join(classifier.config['data']['raw_waste_image_path'], sample_images[2])
                print(f"Using sample image: {image_path}")
            else:
                raise HTTPException(status_code=400, detail="No sample images found")
        
        # Classify waste
        result = classifier.detect_food_waste(image_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to classify waste - no result returned")
        
        # Convert NumPy types to Python native types
        result = convert_numpy_types(result)
        
        return JSONResponse(content={
            "status": "success",
            "classification": result,
            "image_path": image_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Waste Classification: {str(e)}")

@app.get("/api/waste-heatmap")
async def run_waste_heatmap_get():
    """Run the waste heatmap generation module with GET request"""
    try:
        print("\n=== Running Waste Heatmap Generation Module ===")
        
        # Initialize generator
        generator = WasteHeatmapGenerator(config_path)
        
        # Use the first sample image
        sample_images = generator.config['data']['sample_waste_heatmap_images']
        if not sample_images:
            raise HTTPException(status_code=400, detail="No sample images found in configuration")
            
        image_path = os.path.join(generator.config['data']['raw_waste_heatmap_path'], sample_images[2])
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail=f"Sample image not found at path: {image_path}")
            
        print(f"Using sample image: {image_path}")
        
        # Generate heatmap
        heatmap_path, detections_path = generator.create_waste_heatmap(image_path)
        if not heatmap_path or not detections_path:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap or detections")
            
        if not os.path.exists(heatmap_path) or not os.path.exists(detections_path):
            raise HTTPException(status_code=500, detail="Generated files not found")
        
        # Convert paths to relative URLs
        heatmap_url = f"/static/{os.path.basename(heatmap_path)}"
        detections_url = f"/static/{os.path.basename(detections_path)}"
        
        # Create static directory if it doesn't exist
        os.makedirs("static", exist_ok=True)
        
        # Copy files to static directory
        try:
            shutil.copy2(heatmap_path, os.path.join("static", os.path.basename(heatmap_path)))
            shutil.copy2(detections_path, os.path.join("static", os.path.basename(detections_path)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to copy files to static directory: {str(e)}")
        
        return JSONResponse(content={
            "status": "success",
            "heatmap_url": heatmap_url,
            "detections_url": detections_url,
            "image_path": image_path
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in waste heatmap generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Waste Heatmap Generation: {str(e)}")

@app.post("/api/waste-heatmap")
async def run_waste_heatmap_post(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None)
):
    """Run the waste heatmap generation module with POST request"""
    try:
        print("\n=== Running Waste Heatmap Generation Module ===")
        
        # Initialize generator
        generator = WasteHeatmapGenerator(config_path)
        
        # Handle image input
        if file:
            # Save uploaded file
            file_extension = file.filename.split('.')[-1].lower()
            temp_file_path = get_temp_file_path(file_extension)
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Add cleanup task
            background_tasks.add_task(os.remove, temp_file_path)
            
            # Use the temporary file
            image_path = temp_file_path
        else:
            # Use the first sample image if no image is provided
            sample_images = generator.config['data']['sample_waste_heatmap_images']
            if sample_images:
                image_path = os.path.join(generator.config['data']['raw_waste_heatmap_path'], sample_images[2])
                print(f"Using sample image: {image_path}")
            else:
                raise HTTPException(status_code=400, detail="No sample images found")
        
        # Generate heatmap
        heatmap_path, detections_path = generator.create_waste_heatmap(image_path)
        if not heatmap_path or not detections_path:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap or detections")
            
        if not os.path.exists(heatmap_path) or not os.path.exists(detections_path):
            raise HTTPException(status_code=500, detail="Generated files not found")
        
        # Convert paths to relative URLs
        heatmap_url = f"/static/{os.path.basename(heatmap_path)}"
        detections_url = f"/static/{os.path.basename(detections_path)}"
        
        # Create static directory if it doesn't exist
        os.makedirs("static", exist_ok=True)
        
        # Copy files to static directory
        try:
            shutil.copy2(heatmap_path, os.path.join("static", os.path.basename(heatmap_path)))
            shutil.copy2(detections_path, os.path.join("static", os.path.basename(detections_path)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to copy files to static directory: {str(e)}")
        
        return JSONResponse(content={
            "status": "success",
            "heatmap_url": heatmap_url,
            "detections_url": detections_url,
            "image_path": image_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Waste Heatmap Generation: {str(e)}")

@app.get("/api/dashboard")
async def run_dashboard():
    """Run the dashboard module and return the analysis results"""
    try:
        print("\n=== Running Dashboard Module ===")
        
        
        output_dashboard_path = config['data']['output_dashboard_path']
        
        # Initialize dashboard
        tracker = RestaurantWasteTracker(config_path)
        
        
        tracker.save_data_to_csv(output_dashboard_path)
        tracker.generate_chart_data_csvs(output_dashboard_path)
        
        # Generate summary data
        dashboard = tracker.generate_profit_loss_dashboard()
        time_analysis = tracker.generate_time_based_analysis()
        
        # Convert NumPy types to Python native types
        dashboard = convert_numpy_types(dashboard)
        time_analysis = {
            'weekly': convert_numpy_types(time_analysis['weekly'].to_dict('records')),
            'monthly': convert_numpy_types(time_analysis['monthly'].to_dict('records')),
            'quarterly': convert_numpy_types(time_analysis['quarterly'].to_dict('records'))
        }
        
        return JSONResponse(content={
            "status": "success",
            # "dashboard": dashboard,
            # "time_analysis": time_analysis,
            "output_path": output_dashboard_path
        })
    except Exception as e:
        print(f"Error in Dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Dashboard: {str(e)}")

# Run the app with uvicorn
if __name__ == "__main__":
    
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 