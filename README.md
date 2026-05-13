# Brain Tumor MRI Classification
### CAI3105 Deep Learning — Prof. Nashwa El-Bendary

---

## Project Structure

```
brain_tumor_project/
│
├── dataset/                  ← PUT YOUR DATASET HERE
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── models/                   ← saved after training
│   ├── resnet_e2e.pth
│   ├── resnet_backbone.pth
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   └── results.json
│
├── utils/
│   ├── data_utils.py
│   └── train_utils.py
│
├── gui/
│   └── app.py                ← Doctor GUI
│
├── train.py                  ← Main training script
├── requirements.txt
└── README.md
```

---

## How to Run (Step by Step)

### Step 1 — Install Python
Download Python 3.10+ from https://python.org

### Step 2 — Open in VS Code
1. Open VS Code
2. File → Open Folder → select `brain_tumor_project`
3. You'll see all files in the sidebar on the left

### Step 3 — Install dependencies
Open the VS Code terminal (Ctrl + ` ) and run:
```bash
pip install -r requirements.txt
```

### Step 4 — Add your dataset
Copy your extracted dataset folders into `dataset/`:
```
dataset/Training/glioma/      ← MRI images
dataset/Training/meningioma/
dataset/Training/notumor/
dataset/Training/pituitary/
dataset/Testing/...
```
(These are the same folders from /content/final_dataset/ in Colab)

### Step 5 — Train the models
```bash
python train.py
```
This takes 30–60 minutes on CPU, ~10 min on GPU.
Saves all model files to `models/`

### Step 6 — Launch the Doctor GUI
```bash
python gui/app.py
```

---

## GUI Features
- Upload any MRI image (JPG/PNG)
- Get instant diagnosis with confidence score
- Choose between ResNet50 E2E, ResNet50+SVM, or Both
- See class probability bars
- Generate and save a clinical report
- View model performance charts and learning curves
- Compare all approaches side by side
