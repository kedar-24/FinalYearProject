import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import yaml

# --- CONSTANTS ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- EFFICIENTNET DATALOADER (CLASSIFICATION) ---

class PlantVillageClassifierDataset(Dataset):
    """
    Custom Dataset for PlantVillage Classification.
    Assumes structure: root_dir/class_name/image.jpg
    """
    def __init__(self, root_dir, split='train', img_size=224):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Use ImageFolder to handle class indexing, assumes the structure is robust
        # If specific splits are not pre-separated, this might need logic to split files.
        # Here we assume root_dir points to 'train' or 'val' folders if split is specified,
        # or we can split manually. For simplicity in this demo, we assume root_dir contains class folders.
        # If split is 'train', we apply heavy augmentation.
        
        target_dir = os.path.join(root_dir, split) if os.path.exists(os.path.join(root_dir, split)) else root_dir
        
        # Define Transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # "Brightness/Contrast jitter"
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            
        self.dataset = datasets.ImageFolder(root=target_dir, transform=self.transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def get_classification_dataloader(root_dir, split='train', batch_size=32, num_workers=2):
    dataset = PlantVillageClassifierDataset(root_dir, split=split)
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# --- YOLO CONFIG SETUP (DETECTION) ---

def create_yolo_yaml(data_path, class_names, output_path='data.yaml'):
    """
    Creates the data.yaml file required for YOLOv8 training.
    """
    data = {
        'path': os.path.abspath(data_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f)
    print(f"YOLO data config saved to {output_path}")

