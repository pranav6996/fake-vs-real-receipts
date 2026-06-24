

import os
import random
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split

CLEAN_REAL_DIR = "/content/drive/MyDrive/CLEAN_DATASET/real"
CLEAN_AI_DIR   = "/content/drive/MyDrive/CLEAN_DATASET/ai"
OUTPUT_DIR     = "/content/drive/MyDrive/MODEL_OUTPUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE   = 224 # Swin strictly requires 224x224
BATCH_SIZE = 16
EPOCHS     = 20
LR         = 5e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths, labels = [], []
for f in os.listdir(CLEAN_REAL_DIR):
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        paths.append(os.path.join(CLEAN_REAL_DIR, f)); labels.append(0)
for f in os.listdir(CLEAN_AI_DIR):
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        paths.append(os.path.join(CLEAN_AI_DIR, f)); labels.append(1)

# Class Weight Calculation
weight_real = len(labels) / (2.0 * labels.count(0))
weight_ai   = len(labels) / (2.0 * labels.count(1))
class_weights = torch.tensor([weight_real, weight_ai], dtype=torch.float).to(DEVICE)

train_p, temp_p, train_l, temp_l = train_test_split(paths, labels, test_size=0.2, stratify=labels)
val_p, test_p, val_l, test_l     = train_test_split(temp_p, temp_l, test_size=0.5, stratify=temp_l)

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomRotation(degrees=45, fill=255),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), 
    T.GaussianBlur(kernel_size=5),                              
    T.ToTensor(),                                                
    T.RandomErasing(p=0.3, scale=(0.02, 0.1)),                   
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), 
    T.ToTensor(), 
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ReceiptDataset(Dataset):
    def __init__(self, p, l, t): self.p, self.l, self.t = p, l, t
    def __len__(self): return len(self.p)
    def __getitem__(self, i): 
        return self.t(Image.open(self.p[i]).convert("RGB")), torch.tensor(self.l[i])

train_loader = DataLoader(ReceiptDataset(train_p, train_l, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(ReceiptDataset(val_p, val_l, val_transform), batch_size=BATCH_SIZE, num_workers=2)

print(f"✅ Data Ready: {len(train_p)} Train | {len(val_p)} Val | {len(test_p)} Test")

# --- 3. CORE TRAINING ENGINE ---
def train_architecture(model_name):
    print(f"\n{'='*50}\n INITIALIZING: {model_name}\n{'='*50}")
    model = timm.create_model(model_name, pretrained=True, num_classes=2).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler    = GradScaler()
    
    best_acc = 0.0
    save_path = os.path.join(OUTPUT_DIR, f"{model_name}_best_weights.pth")

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == lbls).sum().item()
            total += lbls.size(0)
            
        t_acc = 100 * correct / total
        
        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                outputs = model(imgs.to(DEVICE))
                v_correct += (outputs.argmax(1) == lbls.to(DEVICE)).sum().item()
                v_total += lbls.size(0)
                
        v_acc = 100 * v_correct / v_total
        
        status = ""
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), save_path)
            status = "SAVED"
            
        print(f"Epoch {epoch:02d} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | {status}")
        
    print(f"{model_name} Training Complete. Peak Val Acc: {best_acc:.2f}%")

train_architecture('resnet50')
train_architecture('swin_tiny_patch4_window7_224')