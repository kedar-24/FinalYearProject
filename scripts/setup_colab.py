# --- PASTE THIS INTO YOUR GOOGLE COLAB CELL ---
import os
import sys
import subprocess

def run_command(command):
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True)

print("🚀 Setting up Colab Production Environment...")

# 1. Mount Drive
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
    print("✅ Drive Mounted!")
else:
    print("✅ Drive already mounted.")

# 2. Project Path Setup
PROJECT_PATH = '/content/drive/MyDrive/FinalProject'
if os.path.exists(PROJECT_PATH):
    os.chdir(PROJECT_PATH)
    print(f"📂 Project Root: {os.getcwd()}")
    sys.path.append(PROJECT_PATH)
else:
    print(f"⚠️ Warning: {PROJECT_PATH} not found. Ensure your folder name is 'FinalProject'.")

# 3. Install Production Dependencies
print("\n--- Installing Dependencies ---")
if os.path.exists('requirements.txt'):
    run_command("pip install -r requirements.txt")
    run_command("pip install roboflow") # In case you need your custom dataset
else:
    run_command("pip install ultralytics torch torchvision gradio pydantic-settings python-dotenv")

# 4. Success Message & Next Steps
print("\n🎉 Colab Environment Ready!")
print("-" * 50)
print("NEXT STEPS:")
print("1. If using Roboflow, download your dataset in a new cell.")
print("2. Run Detection (YOLO):")
print("   from src.yolo_manager import YOLOManager")
print("   manager = YOLOManager()")
print("   manager.train(data_yaml_path='path/to/data.yaml')")
print("\n3. Run Classification (EfficientNet):")
print("   !python -m src.train 'your_data_folder'")
print("-" * 50)
