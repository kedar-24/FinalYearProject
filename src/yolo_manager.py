from ultralytics import YOLO
import os
from .logger import logger

class YOLOManager:
    """
    Manages YOLOv8m training and inference with robust logging.
    """
    def __init__(self, model_path: str = 'yolov8m.pt'):
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded from {model_path}")
        except Exception as e:
            logger.critical(f"Could not load YOLO model: {e}")
            raise

    def train(self, data_yaml_path: str, epochs: int = 50, imgsz: int = 640, device: str = 'cuda'):
        """
        Launches YOLOv11m training.
        """
        logger.info(f"Starting YOLO training for {epochs} epochs on {device}...")
        try:
            results = self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                name='yolov8m_plant_disease',
                verbose=True
            )
            logger.info("YOLO training completed successfully.")
            return results
        except Exception as e:
            logger.error(f"YOLO training failed: {e}")
            raise

    def predict(self, image_path_or_array):
        """
        Runs inference on a single image.
        """
        try:
            return self.model.predict(image_path_or_array, save=False, conf=0.25)
        except Exception as e:
            logger.error(f"YOLO prediction failed: {e}")
            raise

    def validate(self):
        """
        Runs validation and returns metrics.
        """
        try:
            metrics = self.model.val()
            logger.info(f"mAP@50-95: {metrics.box.map}")
            return metrics
        except Exception as e:
            logger.error(f"YOLO validation failed: {e}")
            raise
