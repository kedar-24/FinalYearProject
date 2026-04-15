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

    # Model Paths — Updated for YOLOv8m
    EFFICIENTNET_PATH: str = os.getenv("EFFICIENTNET_PATH", "models/best_efficientnet.pth")
    YOLO_PATH: str = os.getenv("YOLO_PATH", "models/best_yolo8m.pt")
    YOLO_FALLBACK_MODEL: str = "yolov8m.pt"

    # Inference Settings
    INPUT_SIZE: int = 224
    CONFIDENCE_THRESHOLD: float = 0.25

    # ── EfficientNet Classification Classes (17) ──
    CLASS_NAMES: List[str] = [
        'Cassava_Bacterial_Blight',
        'Cassava_Brown_Spot',
        'Cassava_Green_Mite',
        'Cassava_Healthy',
        'Cassava_Mosaic',
        'Maize_Fall_Armyworm',
        'Maize_Grasshoper',
        'Maize_Healthy',
        'Maize_Leaf_Beetle',
        'Maize_Leaf_Blight',
        'Maize_Leaf_Spot',
        'Maize_Streak_Virus',
        'Tomato_Healthy',
        'Tomato_Leaf_Blight',
        'Tomato_Leaf_Curl',
        'Tomato_Septoria_Leaf_Spot',
        'Tomato_Verticulium_Wilt',
    ]

    NUM_CLASSES: int = 17


# Global settings instance
settings = Settings()
