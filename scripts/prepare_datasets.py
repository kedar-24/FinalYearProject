import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import yaml

# --- CONFIGURATION ---
PLANTDOC_PATH = "PlantDoc-Dataset" # Assume unzipped here
PLANTDISEASE_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" # Common kaggle path
OUTPUT_YOLO_DIR = "datasets/PlantDoc_YOLO"
OUTPUT_CLS_DIR = "datasets/PlantDisease_CLS"

def convert_xml_to_yolo(xml_file, img_width, img_height, class_mapping):
    """Parses XML and returns YOLO format lines."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_lines = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            class_mapping[class_name] = len(class_mapping)
        
        class_id = class_mapping[class_name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Normalize
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
    return yolo_lines

def prepare_plantdoc_yolo(input_dir, output_dir):
    print(f"Preparing PlantDoc in {output_dir}...")
    
    # Create dirs
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # 1. Scan and Index Files
    xml_map = {}
    txt_map = {}
    img_map = {}
    ext_counts = {}

    print(f"Scanning {input_dir} (recursive)...")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            
            basename = os.path.splitext(file)[0]
            full_path = os.path.join(root, file)
            
            if ext == '.xml':
                xml_map[basename] = full_path
            elif ext == '.txt':
                txt_map[basename] = full_path
            elif ext in ['.jpg', '.jpeg', '.png']:
                img_map[basename] = full_path

    print(f"File scan results: {ext_counts}")

    # 2. Determine Mode (XML vs TXT)
    use_mode = None
    if len(xml_map) > 0:
        use_mode = 'xml'
        print(f"Detected {len(xml_map)} XML files. Using XML -> YOLO conversion.")
    elif len(txt_map) > 0:
        use_mode = 'txt'
        print(f"No XMLs, but detected {len(txt_map)} TXT files. Assuming existing YOLO format.")
    else:
        raise ValueError(f"No annotations found (0 XML, 0 TXT) in {input_dir}. Found extensions: {ext_counts}")

    # 3. Match Pairs
    all_pairs = [] # (img_path, label_path)
    
    for basename, img_path in img_map.items():
        if use_mode == 'xml' and basename in xml_map:
            all_pairs.append((img_path, xml_map[basename]))
        elif use_mode == 'txt' and basename in txt_map:
            all_pairs.append((img_path, txt_map[basename]))

    print(f"Matched {len(all_pairs)} valid Image-{use_mode.upper()} pairs.")
    
    if len(all_pairs) == 0:
        raise ValueError("Images found, labels found, but NO matching basenames! Check filenames.")

    # 4. Process
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    class_mapping = {} # Only populated if we read XMLs. If TXT, we assume pre-defined classes.
    # If TXT mode, we might need to read classes.txt if it exists, or infer from data.yaml if provided
    # For now, if TXT, we assume the dataset is ready-to-go and we just move files.
    
    def process_split(pairs, split_name):
        for img_path, label_path in tqdm(pairs, desc=f"Processing {split_name}"):
            try:
                # Copy Image
                dest_img_path = os.path.join(output_dir, 'images', split_name, os.path.basename(img_path))
                shutil.copy(img_path, dest_img_path)
                
                # Handle Label
                dest_label_path = os.path.join(output_dir, 'labels', split_name, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
                
                if use_mode == 'xml':
                    # Convert
                     with Image.open(img_path) as img:
                        w, h = img.size
                     yolo_lines = convert_xml_to_yolo(label_path, w, h, class_mapping)
                     if yolo_lines:
                         with open(dest_label_path, 'w') as f:
                             f.write('\n'.join(yolo_lines))
                else: 
                    # Copy existing TXT
                    shutil.copy(label_path, dest_label_path)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    process_split(train_pairs, 'train')
    process_split(val_pairs, 'val')
    
    # Create data.yaml
    # If using XML, we built class_mapping.
    # If using TXT, we don't know class names unless we find a classes.txt.
    # We will try to find a classes.txt or assume generic names.
    
    final_names = {}
    if use_mode == 'xml':
        final_names = {v: k for k, v in class_mapping.items()}
    else:
        # Try to find classes.txt
        # If not, warn user they might need to update data.yaml manually
        print("Warning: Using existing TXT files. Class names are not inferred automatically.")
        print("Please check if your dataset provided a 'classes.txt' or 'data.yaml' and update the generated yaml.")
        # Create dummy classes to prevent errors
        final_names = {0: 'class_0', 1: 'class_1'} 

    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': final_names
    }
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)
        
    print(f"Preparation complete. Config saved to {os.path.join(output_dir, 'data.yaml')}")


def prepare_plantdisease_cls(input_dir, output_dir):
    print(f"Preparing PlantDisease for Classification in {output_dir}...")
    
    # Input dir likely has 'train' and 'valid' already, or just classes
    # If it has 'train' and 'valid', copy stricture.
    # If just formatted as class folders, split them.
    
    if os.path.exists(os.path.join(input_dir, 'train')):
        print("Detected 'train' folder. copying structure (symlink/copy)...")
        # For simplicity, we just point the user to use the input_dir directly if it's already split
        # specific to 'PlantDisease' kaggle dataset (emmarex), it usually has train/valid directories.
        print(f"Dataset seems already split. You can use '{input_dir}' directly in src/train.py")
        return

    # If flat structure
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for cls in tqdm(classes, desc="Processing Classes"):
        cls_dir = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        split = int(len(images) * 0.8)
        train_imgs = images[:split]
        val_imgs = images[split:]
        
        os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', cls), exist_ok=True)
        
        for img in train_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(output_dir, 'train', cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(output_dir, 'val', cls, img))

    print("Classification preparation complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plantdoc', help='Path to PlantDoc dataset root')
    parser.add_argument('--plantdisease', help='Path to PlantDisease dataset root')
    args = parser.parse_args()
    
    if args.plantdoc:
        prepare_plantdoc_yolo(args.plantdoc, OUTPUT_YOLO_DIR)
        
    if args.plantdisease:
        prepare_plantdisease_cls(args.plantdisease, OUTPUT_CLS_DIR)

    if not args.plantdoc and not args.plantdisease:
        print("Please provide --plantdoc or --plantdisease paths.")
        print("Example: python prepare_datasets.py --plantdoc /path/to/PlantDoc --plantdisease /path/to/PlantDisease")
