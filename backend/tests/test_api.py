import os
import time
import uuid
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the FastAPI app
from backend.api import app, task_tracker

client = TestClient(app)

def test_task_status():
    """Test the task status endpoint by creating a mock task and checking its status"""
    # Create a mock task_id
    task_id = str(uuid.uuid4())
    
    # Mock data for testing
    test_start_time = time.time()
    test_task_data = {
        "status": "processing",
        "progress": 50,
        "start_time": test_start_time,
        "result": None,
        "error": None,
        "file_id": str(uuid.uuid4())
    }
    
    # Add the mock task to the task_tracker dictionary in the API
    task_tracker[task_id] = test_task_data
    
    # Test the task status endpoint
    response = client.get(f"/api/task-status/{task_id}")
    
    # Verify the response
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert response.json()["progress"] == 50
    assert "elapsed_time" in response.json()
    
    # Test task not found scenario
    non_existent_task_id = str(uuid.uuid4())
    response = client.get(f"/api/task-status/{non_existent_task_id}")
    assert response.status_code == 404
    
    # Update task to completed state
    task_tracker[task_id]["status"] = "completed"
    task_tracker[task_id]["progress"] = 100
    task_tracker[task_id]["result"] = {
        "results": {"test_item": 1},
        "video_url": "/static/test_video.mp4",
        "results_url": "/static/test_results.json"
    }
    task_tracker[task_id]["end_time"] = time.time()
    
    # Test the task status endpoint for a completed task
    response = client.get(f"/api/task-status/{task_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["progress"] == 100
    assert "result" in response.json()
    assert response.json()["result"]["video_url"] == "/static/test_video.mp4"
    
    # Update task to failed state
    task_tracker[task_id]["status"] = "failed"
    task_tracker[task_id]["error"] = "Test error message"
    task_tracker[task_id]["end_time"] = time.time()
    
    # Test the task status endpoint for a failed task
    response = client.get(f"/api/task-status/{task_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "error" in response.json()
    assert response.json()["error"] == "Test error message"
    
    # Clean up
    del task_tracker[task_id]

@pytest.mark.skip(reason="Requires actual video file for upload testing")
def test_upload_stock_detection_video():
    """Test the video upload endpoint with a mock video file"""
    # Create a mock file for upload
    test_video_path = os.path.join(os.path.dirname(__file__), "test_video.mp4")
    
    # Skip if test video doesn't exist
    if not os.path.exists(test_video_path):
        pytest.skip(f"Test video file not found at {test_video_path}")
    
    # Mock StockDetector to avoid actual processing
    with patch("backend.api.StockDetector") as mock_detector_class:
        # Setup mock detector instance
        mock_detector = MagicMock()
        mock_detector.detect_stock.return_value = {"test_item": 1}
        mock_detector.output_video_path = "static/test_output_video.mp4"
        mock_detector_class.return_value = mock_detector
        
        # Create test directory and file
        os.makedirs("static", exist_ok=True)
        with open("static/test_output_video.mp4", "w") as f:
            f.write("dummy content")
        
        # Test file upload
        with open(test_video_path, "rb") as f:
            response = client.post(
                "/api/stock-detection",
                files={"file": ("test_video.mp4", f, "video/mp4")}
            )
        
        # Verify response
        assert response.status_code == 200
        assert "task_id" in response.json()
        assert response.json()["status"] == "success"
        
        # Get the task_id from response
        task_id = response.json()["task_id"]
        
        # Wait for background processing
        time.sleep(1)
        
        # Check task status
        response = client.get(f"/api/task-status/{task_id}")
        assert response.status_code == 200
        
        # Cleanup
        if os.path.exists("static/test_output_video.mp4"):
            os.remove("static/test_output_video.mp4") 