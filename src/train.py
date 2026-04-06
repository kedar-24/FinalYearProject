import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from .model import FLEfficientNet, MultiClassFocalLoss
from .dataset import get_classification_dataloader
from .config import settings
from .logger import logger
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import argparse
import os

class EfficientNetTrainer:
    """
    Handles training and validation of the EfficientNet classifier.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', epochs=20, lr=1e-3):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        
        # Focal Loss (gamma=2, alpha=0.25)
        self.criterion = MultiClassFocalLoss(gamma=2, alpha=0.25)
        
        # AdamW Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # OneCycleLR Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=lr, 
            steps_per_epoch=len(train_loader), 
            epochs=epochs
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': running_loss / (total/labels.size(0)), 
                'Acc': 100 * correct / total
            })

        return running_loss / len(self.train_loader), 100 * correct / total

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': running_loss / (total/labels.size(0)), 
                    'Acc': 100 * correct / total
                })
        
        val_acc = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return running_loss / len(self.val_loader), val_acc, f1

    def run(self):
        best_acc = 0.0
        logger.info(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc, f1 = self.validate(epoch)
            
            logger.info(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, F1={f1:.4f}"
            )
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = settings.EFFICIENTNET_PATH
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best model saved to {save_path} (Acc: {best_acc:.2f}%)")
                
        return best_acc

def main():
    parser = argparse.ArgumentParser(description="Crop Disease Classification Trainer")
    parser.add_argument("data_dir", type=str, help="Path to classification dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory '{args.data_dir}' not found.")
        return

    train_loader = get_classification_dataloader(args.data_dir, split='train', batch_size=args.batch_size)
    val_loader = get_classification_dataloader(args.data_dir, split='val', batch_size=args.batch_size)
    
    num_classes = len(train_loader.dataset.classes)
    logger.info(f"Found {num_classes} classes in the dataset.")

    from .model import get_efficientnet_model
    model = get_efficientnet_model(num_classes=num_classes)
    
    trainer = EfficientNetTrainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=args.epochs,
        lr=args.lr
    )
    trainer.run()

if __name__ == "__main__":
    main()
