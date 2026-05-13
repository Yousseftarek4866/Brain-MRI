"""utils/train_utils.py — training helpers"""
import copy
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)


def extract_features(model, loader, device):
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc='Extracting features', leave=False):
            imgs = imgs.to(device)
            out  = model(imgs)
            feats.append(out.cpu().numpy())
            labs.extend(lbls.numpy())
    return np.concatenate(feats), np.array(labs)


def train_e2e(model, train_loader, val_loader, criterion, optimizer,
              scheduler, epochs, device, name="Model"):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        rl, correct, total = 0.0, 0, 0
        for imgs, lbls in tqdm(train_loader, desc=f'[{name}] Epoch {epoch+1}/{epochs}', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            rl      += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == lbls).sum().item()
            total   += imgs.size(0)
        t_loss, t_acc = rl / total, correct / total

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out  = model(imgs)
                loss = criterion(out, lbls)
                vl  += loss.item() * imgs.size(0)
                vc  += (out.argmax(1) == lbls).sum().item()
                vt  += imgs.size(0)
        v_loss, v_acc = vl / vt, vc / vt
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        print(f"  [{name}] Epoch {epoch+1:2d}/{epochs}  "
              f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    print(f"  ✓ Best val accuracy: {best_val_acc:.4f}")
    return model, history


def get_test_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            p    = model(imgs).argmax(1).cpu().numpy()
            preds.extend(p)
            labels.extend(lbls.numpy())
    return np.array(labels), np.array(preds)


def evaluate_metrics(y_true, y_pred, class_names, title=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n── {title} ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
