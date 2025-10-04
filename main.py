import cv2
from ultralytics import YOLO
import numpy as np
import typing 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.services.cv_service import ComputerVisionService



# App and app configurations
app = FastAPI()
cv_service = ComputerVisionService()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def health_check():
    return {"Hello":"World"}

@app.post('/api/detection/start')
async def start_detection():
    result = cv_service.activate_detection()
    return result

@app.post('/api/detection/stop')
async def stop_detection():
    result = cv_service.destroy_detection()
    return result

@app.get('/api/detection/status')
async def get_detection_status():
    status = cv_service.get_status()
    return status




