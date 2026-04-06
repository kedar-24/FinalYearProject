"""
Verify the entire inference pipeline works end-to-end.
Run this after downloading trained weights from Kaggle.

Usage:
    python verify_setup.py
    python verify_setup.py --image path/to/test_image.jpg
"""
import sys
import os
import argparse
import torch
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))


def check_imports():
    """Verify all modules import correctly."""
    print("1️⃣  Checking imports...")
    try:
        from src.config import settings
        from src.model import FLEfficientNet, MultiClassFocalLoss
        from src.dataset import IMAGENET_MEAN, IMAGENET_STD
        from src.logger import logger
        from src.yolo_manager import YOLOManager
        from src.inference_engine import DiseaseInferenceEngine
        print(f"   ✅ All modules imported successfully")
        print(f"   ✅ Config loaded: {settings.APP_TITLE}")
        print(f"   ✅ Classes: {len(settings.CLASS_NAMES)} ({settings.CLASS_NAMES[0]} ... {settings.CLASS_NAMES[-1]})")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def check_models():
    """Verify model instantiation and forward pass."""
    print("\n2️⃣  Checking model architecture...")
    from src.model import FLEfficientNet, MultiClassFocalLoss
    from src.config import settings

    num_classes = len(settings.CLASS_NAMES)

    # EfficientNet
    model = FLEfficientNet(num_classes=num_classes, pretrained=False)
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    assert output.shape == (1, num_classes), f"Expected (1, {num_classes}), got {output.shape}"
    print(f"   ✅ EfficientNet forward pass OK → output shape: {output.shape}")

    # Focal Loss
    criterion = MultiClassFocalLoss()
    target = torch.tensor([0])
    loss = criterion(output, target)
    print(f"   ✅ Focal Loss OK → loss value: {loss.item():.4f}")

    return True


def check_weights():
    """Check if trained weights exist."""
    print("\n3️⃣  Checking trained weights...")
    from src.config import settings

    yolo_exists = os.path.exists(settings.YOLO_PATH)
    eff_exists = os.path.exists(settings.EFFICIENTNET_PATH)

    print(f"   {'✅' if yolo_exists else '⚠️'} YOLO weights: {settings.YOLO_PATH} {'(found)' if yolo_exists else '(not found — will use fallback)'}")
    print(f"   {'✅' if eff_exists else '⚠️'} EfficientNet weights: {settings.EFFICIENTNET_PATH} {'(found)' if eff_exists else '(not found — using pretrained backbone)'}")

    return True


def check_inference(image_path=None):
    """Run a full inference pipeline test."""
    print("\n4️⃣  Testing full inference pipeline...")
    from src.inference_engine import DiseaseInferenceEngine

    engine = DiseaseInferenceEngine()

    if image_path and os.path.exists(image_path):
        import cv2
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"   📸 Using provided image: {image_path}")
    else:
        # Create a dummy image (random noise)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print("   📸 Using dummy random image (provide --image for real test)")

    annotated, classification = engine.predict(image)

    print(f"   ✅ Detection output shape: {annotated.shape}")
    print(f"   ✅ Classification results:")
    for cls_name, prob in classification.items():
        print(f"      • {cls_name}: {prob:.2%}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify project setup")
    parser.add_argument("--image", type=str, help="Path to a test image", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("🔍 Crop Disease Diagnosis — System Verification")
    print("=" * 60)

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = []
    results.append(("Imports", check_imports()))
    results.append(("Models", check_models()))
    results.append(("Weights", check_weights()))
    results.append(("Inference", check_inference(args.image)))

    print("\n" + "=" * 60)
    print("📋 Summary:")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} — {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! System is ready.")
        print("   Run: python app.py")
    else:
        print("\n⚠️  Some checks failed. Review the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
