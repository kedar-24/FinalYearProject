import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Project Configuration using Pydantic Settings.
    Loads from environment variables or .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App Settings
    APP_TITLE: str = "Crop Disease Diagnosis System"
    DEBUG: bool = False
    PORT: int = 7860

    # Model Paths — Updated for YOLO11m
    EFFICIENTNET_PATH: str = os.getenv("EFFICIENTNET_PATH", "models/best_efficientnet.pth")
    YOLO_PATH: str = os.getenv("YOLO_PATH", "models/best_yolov8m.pt")
    YOLO_FALLBACK_MODEL: str = "yolo8m.pt"

    # Inference Settings
    INPUT_SIZE: int = 224
    CONFIDENCE_THRESHOLD: float = 0.25

    # ── EfficientNet Classification Classes (14) ──
    # Removed Cassava temporarily because the Kaggle checkpoint only has 14 classes.
    CLASS_NAMES: List[str] = [
        # Corn (4 classes)
        'Corn_Cercospora_Leaf_Spot',
        'Corn_Common_Rust',
        'Corn_Healthy',
        'Corn_Northern_Leaf_Blight',
        # Tomato (10 classes)
        'Tomato_Bacterial_Spot',
        'Tomato_Early_Blight',
        'Tomato_Healthy',
        'Tomato_Late_Blight',
        'Tomato_Leaf_Mold',
        'Tomato_Mosaic_Virus',
        'Tomato_Septoria_Leaf_Spot',
        'Tomato_Spider_Mites',
        'Tomato_Target_Spot',
        'Tomato_Yellow_Leaf_Curl_Virus',
    ]

    NUM_CLASSES: int = 14


# Global settings instance
settings = Settings()
