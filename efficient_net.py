

import os
import re
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

import timm
warnings.filterwarnings('ignore')

REAL_FOLDER  = "/content/drive/MyDrive/cleaned_real"      
AI_FOLDER    = "/content/drive/MyDrive/cleaned_ai"        
OUTPUT_DIR   = "/content/drive/MyDrive/fake-vs-real-receipts/efficientnet"



MODEL_NAME   = 'EfficientNet_B3'
TIMM_NAME    = 'efficientnet_b3'
IMAGE_SIZE   = 300             
BATCH_SIZE   = 32
EPOCHS       = 25
LR           = 1e-4
PATIENCE     = 5
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
SEED         = 42
NUM_WORKERS  = 2            
NUM_CLASSES  = 2
CLASS_NAMES  = ['Real', 'AI Generated']
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED    = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CM_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices_per_epoch")
os.makedirs(CM_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  MODEL        :  {MODEL_NAME}")
print(f"  DEVICE       :  {DEVICE}")
print(f"  IMAGE SIZE   :  {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"  BATCH SIZE   :  {BATCH_SIZE}")
print(f"  EPOCHS       :  {EPOCHS}")
print(f"  OUTPUT DIR   :  {OUTPUT_DIR}")

def get_group_id(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[_\-]aug\d*$', '', name, flags=re.IGNORECASE)
    return name
def load_image_paths(real_folder, ai_folder):
    if not os.path.exists(real_folder):
        raise FileNotFoundError(f"Real folder not found:\n  {real_folder}")
    if not os.path.exists(ai_folder):
        raise FileNotFoundError(f"AI folder not found:\n  {ai_folder}")

    records = []

    for f in sorted(os.listdir(real_folder)):
        if f.lower().endswith(SUPPORTED):
            records.append({
                'path'    : os.path.join(real_folder, f),
                'label'   : 0,
                'group_id': 'real_' + get_group_id(f)
            })

    for f in sorted(os.listdir(ai_folder)):
        if f.lower().endswith(SUPPORTED):
            records.append({
                'path'    : os.path.join(ai_folder, f),
                'label'   : 1,
                'group_id': 'ai_' + get_group_id(f)
            })

    df = pd.DataFrame(records)
    n_real = (df['label'] == 0).sum()
    n_ai   = (df['label'] == 1).sum()

    print(f"  Dataset Loaded!")
    print(f"  Real images  :  {n_real}")
    print(f"  AI images    :  {n_ai}")
    print(f"  Total        :  {len(df)}")
    return df


def compute_class_weights(df):
    n_total = len(df)
    n_real  = (df['label'] == 0).sum()
    n_ai    = (df['label'] == 1).sum()
    w_real  = n_total / (2 * n_real)
    w_ai    = n_total / (2 * n_ai)
    weights = torch.tensor([w_real, w_ai], dtype=torch.float).to(DEVICE)
    print(f"\n  Class Weights → Real: {w_real:.4f} | AI: {w_ai:.4f}")
    return weights


def split_dataset_by_group(df):
    groups = df.groupby('group_id')['label'].first().reset_index()

    train_val_groups, test_groups = train_test_split(
        groups, test_size=TEST_SPLIT,
        random_state=SEED, stratify=groups['label']
    )
    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    train_groups, val_groups = train_test_split(
        train_val_groups, test_size=val_ratio,
        random_state=SEED, stratify=train_val_groups['label']
    )

    train_ids = set(train_groups['group_id'])
    val_ids   = set(val_groups['group_id'])
    test_ids  = set(test_groups['group_id'])

    # Hard leakage check
    assert len(train_ids & val_ids)  == 0, "LEAKAGE: train/val overlap!"
    assert len(train_ids & test_ids) == 0, "LEAKAGE: train/test overlap!"
    assert len(val_ids   & test_ids) == 0, "LEAKAGE: val/test overlap!"

    train_df = df[df['group_id'].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df['group_id'].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df['group_id'].isin(test_ids)].reset_index(drop=True)

    total = len(df)
    print(f"\n  Group-Safe Split (zero leakage verified)")
    print(f"  Train  :  {len(train_df):4d} images ({len(train_df)/total*100:.1f}%)")
    print(f"  Val    :  {len(val_df):4d} images ({len(val_df)/total*100:.1f}%)")
    print(f"  Test   :  {len(test_df):4d} images ({len(test_df)/total*100:.1f}%)")

    return train_df, val_df, test_df


class ReceiptDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths     = df['path'].tolist()
        self.labels    = df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(255,255,255))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(is_train=True):
    if is_train:
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # Anti-shortcut augmentations
            T.RandomRotation(degrees=45, fill=255),        
            T.ColorJitter(brightness=0.5, contrast=0.5,
                          saturation=0.5),                 
            T.GaussianBlur(kernel_size=5),                 
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.RandomErasing(p=0.3, scale=(0.02, 0.1)),     
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def build_dataloaders(train_df, val_df, test_df):
    train_ds = ReceiptDataset(train_df, get_transforms(True))
    val_ds   = ReceiptDataset(val_df,   get_transforms(False))
    test_ds  = ReceiptDataset(test_df,  get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader



def plot_confusion_matrix(all_labels, all_preds, epoch,
                          val_acc, val_loss, save_dir):
    """
    Saves a clean, annotated confusion matrix for a given epoch.
    Also shows True Positives, True Negatives, FP, FN breakdown.
    """
    cm = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1])

    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=1, linecolor='white',
        annot_kws={"size": 22, "weight": "bold"},
        ax=ax1
    )
    ax1.set_title(
        f'Confusion Matrix — Epoch {epoch}\n'
        f'Val Acc: {val_acc*100:.2f}%  |  Val Loss: {val_loss:.4f}',
        fontsize=13, fontweight='bold', pad=12
    )
    ax1.set_xlabel('Predicted Label', fontsize=11)
    ax1.set_ylabel('True Label', fontsize=11)

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    total       = tn + fp + fn + tp
    precision   = tp / (tp + fp + 1e-8) * 100
    recall      = tp / (tp + fn + 1e-8) * 100
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8) * 100

    stats = [
        ("True Positives  (TP)", f"{tp}  — AI correctly identified as AI"),
        ("True Negatives  (TN)", f"{tn}  — Real correctly identified as Real"),
        ("False Positives (FP)", f"{fp}  — Real wrongly predicted as AI"),
        ("False Negatives (FN)", f"{fn}  — AI wrongly predicted as Real"),
        ("", ""),
        ("Precision",  f"{precision:.2f}%"),
        ("Recall",     f"{recall:.2f}%"),
        ("F1 Score",   f"{f1:.2f}%"),
        ("Specificity",f"{specificity:.2f}%"),
        ("Total Samples", f"{total}"),
    ]

    y = 0.95
    for label, value in stats:
        if label == "":
            y -= 0.04
            continue
        color = '#CC0000' if 'False' in label else '#1F3864'
        ax2.text(0.0, y, label,
                 transform=ax2.transAxes,
                 fontsize=9.5, fontweight='bold', color=color, va='top')
        ax2.text(0.45, y, value,
                 transform=ax2.transAxes,
                 fontsize=9, color='#333333', va='top')
        y -= 0.09

    plt.suptitle(
        f'{MODEL_NAME} — Training Progress',
        fontsize=14, fontweight='bold', color='#1F3864', y=1.01
    )
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'confusion_matrix_epoch_{epoch:02d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Confusion matrix saved → epoch_{epoch:02d}.png")
    return tn, fp, fn, tp



def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def validate_with_preds(model, loader, criterion):
    """Returns loss, accuracy AND raw predictions for confusion matrix"""
    model.eval()
    total_loss     = 0.0
    all_preds      = []
    all_labels_out = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_out.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels_out, all_preds)
    return avg_loss, accuracy, all_labels_out, all_preds

def evaluate_test(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            with autocast():
                outputs = model(images)
                probs   = torch.softmax(outputs, dim=1)[:, 1]
                preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc       = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc   = roc_auc_score(all_labels, all_probs)

    print(f"\n  {'='*50}")
    print(f"  FINAL TEST RESULTS — {MODEL_NAME}")
    print(f"  {'='*50}")
    print(f"  Accuracy   :  {acc*100:.2f}%")
    print(f"  Precision  :  {precision*100:.2f}%")
    print(f"  Recall     :  {recall*100:.2f}%")
    print(f"  F1 Score   :  {f1*100:.2f}%")
    print(f"  ROC-AUC    :  {roc_auc:.4f}")
    print(f"  {'='*50}")

    plot_confusion_matrix(
        all_labels, all_preds,
        epoch='FINAL_TEST',
        val_acc=acc,
        val_loss=0.0,
        save_dir=OUTPUT_DIR
    )

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {MODEL_NAME}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_roc_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return {
        'Model'     : MODEL_NAME,
        'Accuracy'  : round(acc*100, 2),
        'Precision' : round(precision*100, 2),
        'Recall'    : round(recall*100, 2),
        'F1 Score'  : round(f1*100, 2),
        'ROC-AUC'   : round(roc_auc, 4),
    }



def plot_training_curves(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='steelblue', lw=2)
    axes[0].plot(epochs_range, history['val_loss'],   label='Val Loss',   color='coral',     lw=2)
    axes[0].set_title(f'{MODEL_NAME} — Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, [a*100 for a in history['train_acc']], label='Train Acc', color='steelblue', lw=2)
    axes[1].plot(epochs_range, [a*100 for a in history['val_acc']],   label='Val Acc',   color='coral',     lw=2)
    axes[1].set_title(f'{MODEL_NAME} — Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'{MODEL_NAME} Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":

    print("Loading dataset...")
    df = load_image_paths(REAL_FOLDER, AI_FOLDER)

    class_weights = compute_class_weights(df)

    print("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset_by_group(df)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df
    )

    print(f"\nCreating {MODEL_NAME}...")
    model = timm.create_model(TIMM_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {total_params:.1f}M")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    best_path  = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best.pth')
    ckpt_path  = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_checkpoint.pth')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : [],
        'epoch_tp'  : [], 'epoch_tn': [],
        'epoch_fp'  : [], 'epoch_fn': [],
    }
    best_val_acc, patience_count, start_epoch = 0.0, 0, 1

    if os.path.exists(ckpt_path):
        print(f"\n  Checkpoint found! Resuming {MODEL_NAME}...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_acc   = ckpt['best_val_acc']
        patience_count = ckpt['patience_count']
        history        = ckpt['history']
        print(f"  Resumed from epoch {ckpt['epoch']} | Best: {best_val_acc*100:.2f}%")
    else:
        print(f"\n  Starting {MODEL_NAME} fresh!")

    print(f"\n  {'Epoch':<8}{'TrainLoss':<12}{'TrainAcc':<12}{'ValLoss':<12}{'ValAcc':<12}{'Status'}")
    print(f"  {'-'*68}")

    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        val_loss, val_acc, val_labels, val_preds = validate_with_preds(
            model, val_loader, criterion
        )

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc, patience_count = val_acc, 0
            torch.save(model.state_dict(), best_path)
            status = '[BEST]'
        else:
            patience_count += 1
            status = f'({patience_count}/{PATIENCE})'

        print(f"  {epoch:<8}{train_loss:<12.4f}{train_acc*100:<12.2f}{val_loss:<12.4f}{val_acc*100:<12.2f}{status}")

        tn, fp, fn, tp = plot_confusion_matrix(
            val_labels, val_preds,
            epoch=epoch,
            val_acc=val_acc,
            val_loss=val_loss,
            save_dir=CM_DIR
        )
        history['epoch_tp'].append(int(tp))
        history['epoch_tn'].append(int(tn))
        history['epoch_fp'].append(int(fp))
        history['epoch_fn'].append(int(fn))

        torch.save({
            'epoch'          : epoch,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state'   : scaler.state_dict(),
            'best_val_acc'   : best_val_acc,
            'patience_count' : patience_count,
            'history'        : history,
        }, ckpt_path)

        if patience_count >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training time : {elapsed/60:.1f} minutes")
    print(f"  Best Val Acc  : {best_val_acc*100:.2f}%")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    plot_training_curves(history)

    print("\nRunning final test evaluation...")
    final_metrics = evaluate_test(model, test_loader)

    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_results.csv'), index=False
    )
    print(f"\n  Results saved → {OUTPUT_DIR}")
    print(f"\n  DONE! All confusion matrices saved in:")
    print(f"  {CM_DIR}")