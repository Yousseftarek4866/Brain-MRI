"""
CAI3105 — Brain Tumor MRI Classification
train.py  →  trains both approaches and saves model weights

SMART SKIP: if all model files already exist, skips training entirely.
Run with --force to retrain from scratch:  python train.py --force
"""

import os, sys, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib, json

from utils.data_utils  import get_transforms, get_loaders
from utils.train_utils import train_e2e, extract_features, get_test_predictions, evaluate_metrics

# ── Paths — always relative to THIS file so it works from any directory ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def p(filename):
    """Return absolute path inside models/ folder."""
    return os.path.join(MODELS_DIR, filename)

# ── Config ────────────────────────────────────────────────────────────────
TRAIN_DIR  = r'F:\My Files\sem6\deep learning\dataset\Training'
TEST_DIR   = r'F:\My Files\sem6\deep learning\dataset\Testing'
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
VAL_SPLIT  = 0.2
SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Files that must exist to consider training complete ───────────────────
REQUIRED_FILES = [
    p('resnet_e2e.pth'),
    p('resnet_backbone.pth'),
    p('svm_model.pkl'),
    p('scaler.pkl'),
    p('results.json'),
]

def all_models_exist():
    return all(os.path.exists(f) for f in REQUIRED_FILES)

def print_banner(text):
    print(f"\n{'═'*60}")
    print(f"  {text}")
    print(f"{'═'*60}")

# ══════════════════════════════════════════════════════════════
def main():
    force = '--force' in sys.argv

    # ── SKIP CHECK ────────────────────────────────────────────
    if all_models_exist() and not force:
        print_banner("✓  All models already trained and saved!")
        print(f"\n  Models folder: {MODELS_DIR}\n")
        for f in REQUIRED_FILES:
            size = os.path.getsize(f) / (1024*1024)
            print(f"  ✓  {os.path.basename(f):30s}  ({size:.1f} MB)")
        print("\n  To retrain from scratch run:  python train.py --force")
        print("  To launch the GUI run:        python gui/app.py")
        return

    if force:
        print("  --force flag detected. Retraining from scratch...")

    print_banner(f"Starting Training  |  Device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_loaders(
        TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, SEED
    )
    NUM_CLASSES = len(class_names)
    print(f"  Classes ({NUM_CLASSES}): {class_names}")

    # ══════════════════════════════════════════════════════════
    # APPROACH 1 — ResNet50 Feature Extractor + SVM
    # ══════════════════════════════════════════════════════════
    print_banner("APPROACH 1: ResNet50 Feature Extractor + SVM")

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone = backbone.to(DEVICE)

    t0 = time.time()
    print("  Extracting features...")
    X_train, y_train = extract_features(backbone, train_loader, DEVICE)
    X_val,   y_val   = extract_features(backbone, val_loader,   DEVICE)
    X_test,  y_test  = extract_features(backbone, test_loader,  DEVICE)

    X_fit    = np.concatenate([X_train, X_val])
    y_fit    = np.concatenate([y_train, y_val])
    scaler   = StandardScaler()
    X_fit_s  = scaler.fit_transform(X_fit)
    X_test_s = scaler.transform(X_test)

    print("  Training SVM (linear kernel)...")
    svm = SVC(kernel='linear', C=1.0, probability=True, random_state=SEED)
    svm.fit(X_fit_s, y_fit)
    t1 = time.time()

    y_pred_svm = svm.predict(X_test_s)
    m1 = evaluate_metrics(y_test, y_pred_svm, class_names, "Approach-1 ResNet50+SVM")
    m1['time'] = round(t1 - t0, 1)

    joblib.dump(svm,    p('svm_model.pkl'))
    joblib.dump(scaler, p('scaler.pkl'))
    torch.save(backbone.state_dict(), p('resnet_backbone.pth'))
    print(f"\n  ✓ Saved: svm_model.pkl, scaler.pkl, resnet_backbone.pth")
    print(f"  Approach-1 done in {m1['time']}s  |  Accuracy: {m1['accuracy']:.4f}")

    # ══════════════════════════════════════════════════════════
    # APPROACH 2 — End-to-End ResNet50
    # ══════════════════════════════════════════════════════════
    print_banner("APPROACH 2: End-to-End ResNet50")

    resnet_e2e = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for name, param in resnet_e2e.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    resnet_e2e.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, NUM_CLASSES))
    resnet_e2e = resnet_e2e.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda prm: prm.requires_grad, resnet_e2e.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    t0 = time.time()
    resnet_e2e, hist_r = train_e2e(
        resnet_e2e, train_loader, val_loader,
        criterion, optimizer, scheduler, EPOCHS, DEVICE, "ResNet50 E2E"
    )
    t1 = time.time()

    y_true, y_pred = get_test_predictions(resnet_e2e, test_loader, DEVICE)
    m2 = evaluate_metrics(y_true, y_pred, class_names, "Approach-2 E2E ResNet50")
    m2['time'] = round(t1 - t0, 1)

    torch.save(resnet_e2e.state_dict(), p('resnet_e2e.pth'))
    print(f"\n  ✓ Saved: resnet_e2e.pth")
    print(f"  Approach-2 done in {m2['time']}s  |  Accuracy: {m2['accuracy']:.4f}")

    # ══════════════════════════════════════════════════════════
    # BONUS — VGG16 End-to-End
    # ══════════════════════════════════════════════════════════
    print_banner("BONUS: VGG16 End-to-End")

    vgg_e2e = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for param in vgg_e2e.features.parameters():
        param.requires_grad = False
    vgg_e2e.classifier[6] = nn.Sequential(nn.Dropout(0.4), nn.Linear(4096, NUM_CLASSES))
    vgg_e2e = vgg_e2e.to(DEVICE)

    opt_v = optim.Adam(vgg_e2e.classifier.parameters(), lr=LR)
    sch_v = optim.lr_scheduler.StepLR(opt_v, step_size=5, gamma=0.5)

    t0 = time.time()
    vgg_e2e, hist_v = train_e2e(
        vgg_e2e, train_loader, val_loader,
        criterion, opt_v, sch_v, EPOCHS, DEVICE, "VGG16 E2E"
    )
    t1 = time.time()

    y_true_v, y_pred_v = get_test_predictions(vgg_e2e, test_loader, DEVICE)
    m3 = evaluate_metrics(y_true_v, y_pred_v, class_names, "Bonus VGG16")
    m3['time'] = round(t1 - t0, 1)

    torch.save(vgg_e2e.state_dict(), p('vgg16_e2e.pth'))
    print(f"\n  ✓ Saved: vgg16_e2e.pth")
    print(f"  VGG16 done in {m3['time']}s  |  Accuracy: {m3['accuracy']:.4f}")

    # ══════════════════════════════════════════════════════════
    # SAVE RESULTS JSON for GUI
    # ══════════════════════════════════════════════════════════
    results = {
        'class_names':    class_names,
        'approach1':      {k: v for k, v in m1.items() if k != 'history'},
        'approach2':      {k: v for k, v in m2.items() if k != 'history'},
        'bonus_vgg':      {k: v for k, v in m3.items() if k != 'history'},
        'history_resnet': hist_r,
        'history_vgg':    hist_v,
    }
    with open(p('results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print_banner("✓  ALL DONE")
    print(f"  Models saved to: {MODELS_DIR}\n")
    for fname in REQUIRED_FILES:
        if os.path.exists(fname):
            size = os.path.getsize(fname) / (1024*1024)
            print(f"  ✓  {os.path.basename(fname):30s}  ({size:.1f} MB)")
    print(f"\n  Summary:")
    print(f"    Approach-1  ResNet50+SVM  →  Accuracy: {m1['accuracy']:.4f}")
    print(f"    Approach-2  ResNet50 E2E  →  Accuracy: {m2['accuracy']:.4f}")
    print(f"    Bonus       VGG16 E2E     →  Accuracy: {m3['accuracy']:.4f}")
    print(f"\n  Now run:  python gui/app.py")


if __name__ == '__main__':
    main()
