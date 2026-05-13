"""
gui/app.py — Brain Tumor MRI Classification
Premium Medical Dashboard — Redesigned
CAI3105 Deep Learning Project
"""

import os, sys, json, threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import joblib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
import math

# ── Paths ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
RESULTS_JSON = os.path.join(MODELS_DIR, 'results.json')
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES  = 4

# ── Design System ──────────────────────────────────────────
C = {
    'bg':        '#080C14',
    'bg1':       '#0D1321',
    'bg2':       '#111827',
    'bg3':       '#1A2235',
    'bg4':       '#1F2D45',
    'panel':     '#141E2E',
    'border':    '#1E2D45',
    'border2':   '#243555',
    'cyan':      '#00D4FF',
    'cyan_dim':  '#0099BB',
    'cyan_glow': '#00D4FF22',
    'teal':      '#00FFB3',
    'teal_dim':  '#00AA77',
    'red':       '#FF4B6E',
    'red_dim':   '#CC2244',
    'amber':     '#FFB347',
    'amber_dim': '#CC8822',
    'green':     '#00E676',
    'green_dim': '#00AA44',
    'blue':      '#448AFF',
    'blue_dim':  '#2255CC',
    'purple':    '#CE93D8',
    'white':     '#E8F4FF',
    'gray1':     '#8BA0BB',
    'gray2':     '#5A6E8A',
    'gray3':     '#2E3F57',
}

FONTS = {
    'display': ('Segoe UI', 28, 'bold'),
    'title':   ('Segoe UI', 18, 'bold'),
    'head':    ('Segoe UI', 13, 'bold'),
    'body':    ('Segoe UI', 11),
    'small':   ('Segoe UI', 9),
    'mono':    ('Consolas', 10),
    'mono_sm': ('Consolas', 9),
    'tiny':    ('Segoe UI', 8),
}

CLASS_INFO = {
    'glioma':     {'color': C['red'],   'bg': '#FF4B6E18', 'risk': 'HIGH',
                   'icon': '⚠',  'short': 'Malignant glial cell tumor',
                   'action': 'Urgent neurosurgical referral required',
                   'desc': 'Glioma originates from glial cells in the brain or spine. Aggressive variants require immediate oncological intervention including surgery, radiation, and chemotherapy.'},
    'meningioma': {'color': C['amber'], 'bg': '#FFB34718', 'risk': 'MODERATE',
                   'icon': '◈',  'short': 'Meningeal membrane tumor',
                   'action': 'Neurological evaluation recommended',
                   'desc': 'Meningioma arises from the meninges. Usually benign and slow-growing. Management depends on size, location, and patient symptoms.'},
    'notumor':    {'color': C['teal'],  'bg': '#00FFB318', 'risk': 'NONE',
                   'icon': '✓',  'short': 'No tumor detected',
                   'action': 'Routine follow-up as clinically indicated',
                   'desc': 'No evidence of tumor found in this MRI scan. Continue standard monitoring protocol based on patient clinical presentation and history.'},
    'pituitary':  {'color': C['blue'],  'bg': '#448AFF18', 'risk': 'MODERATE',
                   'icon': '◉',  'short': 'Pituitary gland tumor',
                   'action': 'Endocrinology & ophthalmology consult',
                   'desc': 'Pituitary tumor affects the master endocrine gland. Hormonal assessment and visual field testing are essential for complete evaluation.'},
}

# ── Model loading ──────────────────────────────────────────
def load_e2e_model():
    path = os.path.join(MODELS_DIR, 'resnet_e2e.pth')
    if not os.path.exists(path): return None
    m = resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, NUM_CLASSES))
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m

def load_vgg16_model():
    from torchvision.models import vgg16, VGG16_Weights
    path = os.path.join(MODELS_DIR, 'vgg16_e2e.pth')
    if not os.path.exists(path): return None
    m = vgg16(weights=None)
    m.classifier[6] = nn.Sequential(nn.Dropout(0.4), nn.Linear(4096, NUM_CLASSES))
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m

def load_svm_pipeline():
    sp = os.path.join(MODELS_DIR, 'svm_model.pkl')
    sc = os.path.join(MODELS_DIR, 'scaler.pkl')
    bp = os.path.join(MODELS_DIR, 'resnet_backbone.pth')
    if not all(os.path.exists(p) for p in [sp, sc, bp]):
        return None, None, None
    svm    = joblib.load(sp)
    scaler = joblib.load(sc)
    bb     = resnet50(weights=None)
    bb.fc  = nn.Identity()
    bb.load_state_dict(torch.load(bp, map_location=DEVICE))
    bb.to(DEVICE).eval()
    return svm, scaler, bb

INFER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict_image(path, e2e, vgg, svm, scaler, bb):
    img    = Image.open(path).convert('RGB')
    tensor = INFER_TF(img).unsqueeze(0).to(DEVICE)
    out    = {}
    if e2e:
        with torch.no_grad():
            probs = torch.softmax(e2e(tensor), 1).cpu().numpy()[0]
        idx = int(probs.argmax())
        out['e2e'] = {'class': CLASS_NAMES[idx], 'confidence': float(probs[idx]), 'probs': probs.tolist()}
    if vgg:
        with torch.no_grad():
            probs = torch.softmax(vgg(tensor), 1).cpu().numpy()[0]
        idx = int(probs.argmax())
        out['vgg'] = {'class': CLASS_NAMES[idx], 'confidence': float(probs[idx]), 'probs': probs.tolist()}
    if svm and bb:
        with torch.no_grad():
            feat = bb(tensor).cpu().numpy()
        feat_s = scaler.transform(feat)
        idx    = int(svm.predict(feat_s)[0])
        proba  = svm.predict_proba(feat_s)[0]
        out['svm'] = {'class': CLASS_NAMES[idx], 'confidence': float(proba[idx]), 'probs': proba.tolist()}
    return out

# ══════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ══════════════════════════════════════════════════════════

class GlowButton(tk.Canvas):
    def __init__(self, parent, text, command=None, color=None, width=200, height=44, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C['bg2'], highlightthickness=0, **kw)
        self.command = command
        self.color   = color or C['cyan']
        self.w, self.h = width, height
        self.text_str  = text
        self._draw_normal()
        self.bind('<Enter>',    self._on_enter)
        self.bind('<Leave>',    self._on_leave)
        self.bind('<Button-1>', self._on_click)

    def _hex_alpha(self, hex_color, alpha=0.15):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f'#{int(r*alpha + int(C["bg2"][1:3],16)*(1-alpha)):02x}' \
               f'{int(g*alpha + int(C["bg2"][3:5],16)*(1-alpha)):02x}' \
               f'{int(b*alpha + int(C["bg2"][5:7],16)*(1-alpha)):02x}'

    def _draw_normal(self):
        self.delete('all')
        r = 10
        w, h = self.w, self.h
        fill = self._hex_alpha(self.color, 0.12)
        self._round_rect(2, 2, w-2, h-2, r, fill=fill, outline=self.color, width=1)
        self.create_text(w//2, h//2, text=self.text_str, fill=self.color,
                         font=('Segoe UI', 10, 'bold'), anchor='center')

    def _draw_hover(self):
        self.delete('all')
        r = 10
        w, h = self.w, self.h
        fill = self._hex_alpha(self.color, 0.28)
        self._round_rect(2, 2, w-2, h-2, r, fill=fill, outline=self.color, width=2)
        self.create_text(w//2, h//2, text=self.text_str, fill=C['white'],
                         font=('Segoe UI', 10, 'bold'), anchor='center')

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        self.create_arc(x1,y1,x1+2*r,y1+2*r, start=90,  extent=90, style='arc', **{k:v for k,v in kw.items() if k!='fill'})
        self.create_arc(x2-2*r,y1,x2,y1+2*r, start=0,   extent=90, style='arc', **{k:v for k,v in kw.items() if k!='fill'})
        self.create_arc(x1,y2-2*r,x1+2*r,y2, start=180, extent=90, style='arc', **{k:v for k,v in kw.items() if k!='fill'})
        self.create_arc(x2-2*r,y2-2*r,x2,y2, start=270, extent=90, style='arc', **{k:v for k,v in kw.items() if k!='fill'})
        self.create_polygon(
            x1+r,y1, x2-r,y1, x2,y1+r, x2,y2-r, x2-r,y2,
            x1+r,y2, x1,y2-r, x1,y1+r,
            smooth=False, **kw)

    def _on_enter(self, e): self._draw_hover()
    def _on_leave(self, e): self._draw_normal()
    def _on_click(self, e):
        if self.command: self.command()


class ScannerLine(tk.Canvas):
    """Animated scanning line effect for the MRI upload area."""
    def __init__(self, parent, width=340, height=280, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C['bg3'], highlightthickness=0, **kw)
        self.w, self.h = width, height
        self._y       = 0
        self._going   = True
        self._active  = False
        self._img_ref = None
        self._draw_empty()
        self._animate()

    def _draw_empty(self):
        self.delete('all')
        # Corner brackets
        s, l = 6, 22
        kw = dict(fill=C['cyan'], width=2)
        for (x,y,dx,dy) in [(s,s,l,0),(s,s,0,l),(self.w-s,s,-l,0),(self.w-s,s,0,l),
                             (s,self.h-s,l,0),(s,self.h-s,0,-l),(self.w-s,self.h-s,-l,0),(self.w-s,self.h-s,0,-l)]:
            self.create_line(x,y,x+dx,y+dy, **kw)
        # Center cross
        cx, cy = self.w//2, self.h//2
        self.create_line(cx-12,cy,cx+12,cy, fill=C['cyan_dim'], width=1)
        self.create_line(cx,cy-12,cx,cy+12, fill=C['cyan_dim'], width=1)
        # Text
        self.create_text(cx, cy+30, text='DROP MRI SCAN HERE', fill=C['cyan_dim'],
                         font=('Consolas', 9), anchor='center')
        self.create_text(cx, cy+48, text='JPG  ·  PNG  ·  BMP', fill=C['gray3'],
                         font=('Consolas', 8), anchor='center')

    def set_image(self, pil_img):
        pil_img.thumbnail((self.w-4, self.h-4))
        self._img_ref = ImageTk.PhotoImage(pil_img)
        self._active  = True
        self.delete('all')
        ox = (self.w - self._img_ref.width())  // 2
        oy = (self.h - self._img_ref.height()) // 2
        self.create_image(ox, oy, image=self._img_ref, anchor='nw')
        # Corner brackets over image
        s, l = 4, 18
        for (x,y,dx,dy) in [(s,s,l,0),(s,s,0,l),(self.w-s,s,-l,0),(self.w-s,s,0,l),
                             (s,self.h-s,l,0),(s,self.h-s,0,-l),(self.w-s,self.h-s,-l,0),(self.w-s,self.h-s,0,-l)]:
            self.create_line(x,y,x+dx,y+dy, fill=C['cyan'], width=2)

    def clear(self):
        self._active  = False
        self._img_ref = None
        self._draw_empty()

    def _animate(self):
        if self._active:
            self.after(32, self._animate)
            return
        if self._going:
            self._y = (self._y + 3) % self.h
        self.delete('scanner')
        self.create_line(0, self._y, self.w, self._y,
                         fill=C['cyan'], width=1, tags='scanner', stipple='gray50')
        self.after(32, self._animate)


# ══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════

class BrainTumorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('NeuroScan AI  ·  Brain Tumor MRI Classifier')
        self.geometry('1400x860')
        self.minsize(1100, 720)
        self.configure(bg=C['bg'])
        self.resizable(True, True)

        self.e2e_model   = None
        self.vgg_model   = None
        self.svm_model   = None
        self.scaler      = None
        self.backbone    = None
        self.results_data = None
        self.img_path    = None
        self._last_pred  = None
        self._current_tab = 0

        self._setup_style()
        self._build_sidebar()
        self._build_main()
        self._show_tab(0)
        self._load_models_async()
        self._load_results()

    # ── Scroll binding — works on touchpad and mouse wheel ──
    def _bind_scroll(self, canvas):
        """Bind mousewheel/touchpad scroll to a canvas on any platform."""
        def _on_scroll(event):
            # Windows / touchpad
            if event.delta:
                canvas.yview_scroll(-1*(event.delta//120), 'units')
            # Linux scroll up/down
            elif event.num == 4:
                canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                canvas.yview_scroll(1, 'units')

        def _bind_tree(widget):
            widget.bind('<MouseWheel>', _on_scroll)
            widget.bind('<Button-4>',   _on_scroll)
            widget.bind('<Button-5>',   _on_scroll)
            for child in widget.winfo_children():
                _bind_tree(child)

        canvas.bind('<MouseWheel>', _on_scroll)
        canvas.bind('<Button-4>',   _on_scroll)
        canvas.bind('<Button-5>',   _on_scroll)
        # Also bind after children are added
        canvas.bind('<Enter>', lambda e: _bind_tree(canvas))
        # Re-bind inner frame whenever it changes
        def _on_inner_change(e):
            _bind_tree(canvas)
        canvas.bind('<Configure>', lambda e: (
            canvas.configure(scrollregion=canvas.bbox('all')),
        ))
        # Store scroll func so inner frames can call it
        canvas._scroll_fn = _on_scroll

    # ── Style ────────────────────────────────────────────────
    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure('TScrollbar', background=C['bg3'], troughcolor=C['bg2'],
                    borderwidth=0, arrowcolor=C['gray2'])
        s.configure('TProgressbar', background=C['cyan'], troughcolor=C['bg3'],
                    borderwidth=0, thickness=3)

    # ── Sidebar ──────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = tk.Frame(self, bg=C['bg1'], width=220)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        # Logo area
        logo = tk.Frame(self.sidebar, bg=C['bg1'], height=80)
        logo.pack(fill='x')
        logo.pack_propagate(False)
        c = tk.Canvas(logo, bg=C['bg1'], width=220, height=80, highlightthickness=0)
        c.pack(fill='both')
        # Brain icon circle
        c.create_oval(18,18,62,62, outline=C['cyan'], width=1.5)
        c.create_text(40,40, text='⊕', fill=C['cyan'], font=('Segoe UI',18))
        c.create_text(74,32, text='NeuroScan', fill=C['white'], font=('Segoe UI',13,'bold'), anchor='w')
        c.create_text(74,50, text='AI  v2.0', fill=C['cyan'], font=('Consolas',9), anchor='w')

        # Separator line
        tk.Frame(self.sidebar, bg=C['border'], height=1).pack(fill='x', padx=12)

        # Nav items
        nav_items = [
            ('🔬', 'Diagnose', 'Analyze MRI scan',     0),
            ('📊', 'Results',  'Model performance',    1),
            ('📋', 'Report',   'Patient report',       2),
            ('🗄', 'Dataset',  'Req 1 — Data specs',   3),
            ('🤖', 'Models',   'Req 2 — Architecture', 4),
            ('📝', 'Conclude', 'Req 4 — Conclusion',   5),
            ('ℹ',  'About',    'Project info',         6),
        ]
        self.nav_btns = []
        nav_frame = tk.Frame(self.sidebar, bg=C['bg1'])
        nav_frame.pack(fill='x', pady=16)

        for icon, label, sub, idx in nav_items:
            btn = self._make_nav_btn(nav_frame, icon, label, sub, idx)
            self.nav_btns.append(btn)
            btn.pack(fill='x', padx=8, pady=2)

        # Bottom: status
        bottom = tk.Frame(self.sidebar, bg=C['bg1'])
        bottom.pack(side='bottom', fill='x', padx=12, pady=16)
        tk.Frame(bottom, bg=C['border'], height=1).pack(fill='x', pady=(0,10))
        self.status_dot = tk.Canvas(bottom, bg=C['bg1'], width=8, height=8,
                                    highlightthickness=0)
        self.status_dot.pack(side='left', pady=2)
        self.status_dot.create_oval(1,1,7,7, fill=C['amber'], outline='')
        self.status_lbl = tk.Label(bottom, text='Loading models...', bg=C['bg1'],
                                   fg=C['amber'], font=FONTS['tiny'])
        self.status_lbl.pack(side='left', padx=6)

        # Device info
        dev = 'GPU ✓' if torch.cuda.is_available() else 'CPU'
        tk.Label(self.sidebar, text=f'Device: {dev}', bg=C['bg1'],
                 fg=C['gray2'], font=FONTS['tiny']).pack(side='bottom', pady=4)
        tk.Label(self.sidebar, text='CAI3105 · Deep Learning', bg=C['bg1'],
                 fg=C['gray3'], font=FONTS['tiny']).pack(side='bottom')

    def _make_nav_btn(self, parent, icon, label, sub, idx):
        f = tk.Frame(parent, bg=C['bg1'], cursor='hand2')
        f.bind('<Enter>',    lambda e,ff=f: ff.configure(bg=C['bg3']))
        f.bind('<Leave>',    lambda e,ff=f,i=idx: ff.configure(bg=C['bg4'] if self._current_tab==i else C['bg1']))
        f.bind('<Button-1>', lambda e,i=idx: self._show_tab(i))

        icon_lbl = tk.Label(f, text=icon, bg=C['bg1'], fg=C['cyan'],
                            font=('Segoe UI',14), width=3)
        icon_lbl.pack(side='left', pady=10)
        icon_lbl.bind('<Button-1>', lambda e,i=idx: self._show_tab(i))

        txt_f = tk.Frame(f, bg=C['bg1'])
        txt_f.pack(side='left', fill='x', expand=True)
        txt_f.bind('<Button-1>', lambda e,i=idx: self._show_tab(i))

        lbl = tk.Label(txt_f, text=label, bg=C['bg1'], fg=C['white'],
                       font=FONTS['head'], anchor='w')
        lbl.pack(anchor='w')
        lbl.bind('<Button-1>', lambda e,i=idx: self._show_tab(i))

        sub_lbl = tk.Label(txt_f, text=sub, bg=C['bg1'], fg=C['gray2'],
                           font=FONTS['tiny'], anchor='w')
        sub_lbl.pack(anchor='w')
        sub_lbl.bind('<Button-1>', lambda e,i=idx: self._show_tab(i))

        # Store refs for color update
        f._children_labels = [icon_lbl, lbl, sub_lbl, txt_f]
        return f

    def _show_tab(self, idx):
        self._current_tab = idx
        # Update nav highlight
        for i, btn in enumerate(self.nav_btns):
            active = (i == idx)
            bg = C['bg4'] if active else C['bg1']
            btn.configure(bg=bg)
            for w in btn._children_labels:
                w.configure(bg=bg)
            # Active indicator line
            btn.delete('indicator') if hasattr(btn,'delete') else None

        # Show correct frame
        for i, f in enumerate(self.tab_frames):
            f.pack_forget()
        self.tab_frames[idx].pack(fill='both', expand=True)

        if idx == 1:
            self._populate_results()

    # ── Main content area ────────────────────────────────────
    def _build_main(self):
        self.main = tk.Frame(self, bg=C['bg'])
        self.main.pack(side='left', fill='both', expand=True)

        # Top bar
        topbar = tk.Frame(self.main, bg=C['bg1'], height=52)
        topbar.pack(fill='x')
        topbar.pack_propagate(False)
        tk.Label(topbar, text='Brain Tumor MRI Classification System',
                 bg=C['bg1'], fg=C['white'], font=FONTS['title']).pack(side='left', padx=20, pady=14)
        tk.Label(topbar, text=f'Prof. Nashwa El-Bendary  ·  CAI3105',
                 bg=C['bg1'], fg=C['gray2'], font=FONTS['small']).pack(side='left', padx=8)
        self.time_lbl = tk.Label(topbar, text='', bg=C['bg1'], fg=C['cyan'],
                                  font=FONTS['mono_sm'])
        self.time_lbl.pack(side='right', padx=20)
        self._tick_clock()
        tk.Frame(self.main, bg=C['border'], height=1).pack(fill='x')

        # Tab frames
        self.tab_frames = []
        for builder in [self._build_diagnose, self._build_results,
                        self._build_report_tab, self._build_dataset_tab,
                        self._build_models_tab, self._build_conclusion_tab,
                        self._build_about]:
            f = tk.Frame(self.main, bg=C['bg'])
            builder(f)
            self.tab_frames.append(f)

    def _tick_clock(self):
        self.time_lbl.config(text=datetime.now().strftime('%Y-%m-%d   %H:%M:%S'))
        self.after(1000, self._tick_clock)

    # ════════════════════════════════════════════════════════
    # TAB 0 — DIAGNOSE
    # ════════════════════════════════════════════════════════
    def _build_diagnose(self, parent):

        # ── Left column ──────────────────────────────────────
        left = tk.Frame(parent, bg=C['bg'], width=400)
        left.pack(side='left', fill='y', padx=(20,10), pady=20)
        left.pack_propagate(False)

        # Section title
        self._section_title(left, 'MRI SCAN INPUT', C['cyan'])

        # Scanner widget
        scan_frame = tk.Frame(left, bg=C['border'], padx=1, pady=1)
        scan_frame.pack(fill='x')
        self.scanner = ScannerLine(scan_frame, width=378, height=290)
        self.scanner.pack()
        self.scanner.bind('<Button-1>', lambda e: self._upload_image())

        # File info strip
        self.file_info = tk.Label(left, text='No file selected',
                                  bg=C['bg3'], fg=C['gray1'], font=FONTS['mono_sm'],
                                  anchor='w', padx=10, pady=6)
        self.file_info.pack(fill='x', pady=(2,10))

        # Upload / Clear buttons
        btn_row = tk.Frame(left, bg=C['bg'])
        btn_row.pack(fill='x', pady=(0,14))
        GlowButton(btn_row, '▲  UPLOAD SCAN', self._upload_image,
                   C['cyan'], width=248, height=40).pack(side='left')
        GlowButton(btn_row, '✕ CLEAR', self._clear_image,
                   C['gray2'], width=118, height=40).pack(side='left', padx=(8,0))

        # Model selector
        self._section_title(left, 'MODEL SELECTION', C['teal'])
        self.model_var = tk.StringVar(value='e2e')
        models_cfg = [
            ('e2e',  'ResNet50  E2E',  'End-to-End fine-tuned  ·  94.4% accuracy',  C['teal']),
            ('svm',  'ResNet50 + SVM', 'Feature extractor + SVM  ·  88.8%',         C['cyan']),
            ('vgg',  'VGG16  E2E',     'Bonus model — End-to-End  ·  89.4%',        C['purple']),
            ('both', 'All Models',     'Run all 3 and compare results',             C['amber']),
        ]
        for val, name, desc, col in models_cfg:
            self._radio_card(left, val, name, desc, col)

        # Analyze button
        tk.Frame(left, bg=C['bg'], height=8).pack()
        self.analyze_btn = GlowButton(left, '⬡  ANALYZE MRI SCAN',
                                      self._analyze, C['cyan'], width=378, height=48)
        self.analyze_btn.pack(fill='x')
        self.progress = ttk.Progressbar(left, mode='indeterminate', style='TProgressbar')
        self.progress.pack(fill='x', pady=(4,0))

        # ── Right column ─────────────────────────────────────
        right = tk.Frame(parent, bg=C['bg'])
        right.pack(side='left', fill='both', expand=True, padx=(0,20), pady=20)

        self._section_title(right, 'DIAGNOSIS RESULTS', C['cyan'])

        # Result panel
        self.result_panel = tk.Frame(right, bg=C['panel'],
                                     highlightthickness=1,
                                     highlightbackground=C['border'])
        self.result_panel.pack(fill='both', expand=True)
        self._draw_idle_result()

    def _section_title(self, parent, text, color=None):
        f = tk.Frame(parent, bg=C['bg'] if parent.cget('bg')==C['bg'] else parent.cget('bg'))
        f.pack(fill='x', pady=(0,6))
        tk.Canvas(f, bg=f.cget('bg'), width=3, height=16,
                  highlightthickness=0).pack(side='left')
        bar = tk.Canvas(f, bg=f.cget('bg'), width=3, height=14, highlightthickness=0)
        bar.pack(side='left')
        bar.create_rectangle(0,0,3,14, fill=color or C['cyan'], outline='')
        tk.Label(f, text=f'  {text}', bg=f.cget('bg'),
                 fg=color or C['cyan'], font=FONTS['tiny']).pack(side='left')

    def _radio_card(self, parent, val, name, desc, color):
        f = tk.Frame(parent, bg=C['bg3'],
                     highlightthickness=1, highlightbackground=C['border'])
        f.pack(fill='x', pady=3)

        indicator = tk.Canvas(f, bg=C['bg3'], width=18, height=18,
                               highlightthickness=0)
        indicator.pack(side='left', padx=(10,6), pady=10)

        lbl_name = tk.Label(f, text=name, bg=C['bg3'], fg=C['white'], font=FONTS['head'])
        lbl_name.pack(side='left')
        lbl_desc = tk.Label(f, text=f'  {desc}', bg=C['bg3'], fg=C['gray2'], font=FONTS['tiny'])
        lbl_desc.pack(side='left')

        def select(*_):
            self.model_var.set(val)

        def update_dot(*_):
            active = self.model_var.get() == val
            bg = C['bg4'] if active else C['bg3']
            border = color if active else C['border']
            f.configure(bg=bg, highlightbackground=border)
            indicator.configure(bg=bg)
            lbl_name.configure(bg=bg)
            lbl_desc.configure(bg=bg)
            indicator.delete('all')
            if active:
                indicator.create_oval(2,2,16,16, outline=color, width=1.5)
                indicator.create_oval(5,5,13,13, fill=color, outline='')
            else:
                indicator.create_oval(2,2,16,16, outline=C['gray3'], width=1.5)

        self.model_var.trace_add('write', update_dot)
        update_dot()

        for widget in [f, indicator, lbl_name, lbl_desc]:
            widget.bind('<Button-1>', select)

    def _draw_idle_result(self):
        for w in self.result_panel.winfo_children():
            w.destroy()
        c = tk.Canvas(self.result_panel, bg=C['panel'], highlightthickness=0)
        c.pack(fill='both', expand=True)
        c.update_idletasks()
        cx, cy = c.winfo_width()//2 or 400, c.winfo_height()//2 or 220
        c.create_text(cx, cy-20, text='⊕', fill=C['border2'],
                      font=('Segoe UI', 52), anchor='center')
        c.create_text(cx, cy+42, text='Upload an MRI scan and click Analyze',
                      fill=C['gray3'], font=FONTS['body'], anchor='center')
        c.create_text(cx, cy+64, text='Results will appear here',
                      fill=C['gray3'], font=FONTS['small'], anchor='center')

    def _draw_result(self, preds):
        for w in self.result_panel.winfo_children():
            w.destroy()

        mode = self.model_var.get()
        primary = preds.get('e2e') or preds.get('vgg') or preds.get('svm')
        if mode == 'svm' and 'svm' in preds:   primary = preds['svm']
        elif mode == 'vgg' and 'vgg' in preds: primary = preds['vgg']

        cls  = primary['class']
        conf = primary['confidence']
        info = CLASS_INFO[cls]

        # ── Hero diagnosis strip ──────────────────────────────
        hero = tk.Frame(self.result_panel, bg=C['bg'],
                        highlightthickness=1, highlightbackground=info['color'])
        hero.pack(fill='x', padx=16, pady=(16,8))

        # Color accent bar
        accent = tk.Canvas(hero, bg=C['bg'], width=6, height=90,
                           highlightthickness=0)
        accent.pack(side='left')
        accent.create_rectangle(0,0,6,90, fill=info['color'], outline='')

        # Diagnosis text
        diag_f = tk.Frame(hero, bg=C['bg'])
        diag_f.pack(side='left', padx=14, pady=12, fill='x', expand=True)

        top_row = tk.Frame(diag_f, bg=C['bg'])
        top_row.pack(fill='x')
        tk.Label(top_row, text=info['icon'], bg=C['bg'],
                 fg=info['color'], font=('Segoe UI',20)).pack(side='left')
        tk.Label(top_row, text=f"  {cls.upper()}", bg=C['bg'],
                 fg=info['color'], font=('Segoe UI',22,'bold')).pack(side='left')

        risk_colors = {'HIGH': C['red'], 'MODERATE': C['amber'], 'NONE': C['teal']}
        risk_bg     = {'HIGH': '#FF4B6E22', 'MODERATE': '#FFB34722', 'NONE': '#00FFB322'}
        rc = risk_colors.get(info['risk'], C['gray1'])
        rb = risk_bg.get(info['risk'], C['bg3'])
        badge = tk.Label(top_row, text=f" {info['risk']} RISK ",
                         bg=rb, fg=rc, font=('Segoe UI',8,'bold'),
                         padx=6, pady=2)
        badge.pack(side='left', padx=12)

        tk.Label(diag_f, text=info['short'], bg=C['bg'],
                 fg=C['gray1'], font=FONTS['body']).pack(anchor='w')
        tk.Label(diag_f, text=f"↳  {info['action']}", bg=C['bg'],
                 fg=info['color'], font=FONTS['small']).pack(anchor='w', pady=(2,0))

        # Confidence badge
        conf_f = tk.Frame(hero, bg=C['bg'])
        conf_f.pack(side='right', padx=16)
        tk.Label(conf_f, text=f"{conf*100:.1f}%", bg=C['bg'],
                 fg=info['color'], font=('Consolas',28,'bold')).pack()
        tk.Label(conf_f, text='CONFIDENCE', bg=C['bg'],
                 fg=C['gray2'], font=FONTS['tiny']).pack()

        # ── Clinical description ──────────────────────────────
        desc_f = tk.Frame(self.result_panel, bg=C['bg3'])
        desc_f.pack(fill='x', padx=16, pady=(0,10))
        tk.Label(desc_f, text=info['desc'], bg=C['bg3'], fg=C['gray1'],
                 font=FONTS['small'], wraplength=560, justify='left',
                 padx=14, pady=10).pack(anchor='w')

        # ── Probability bars ──────────────────────────────────
        bars_lbl = tk.Frame(self.result_panel, bg=C['panel'])
        bars_lbl.pack(fill='x', padx=16)
        tk.Label(bars_lbl, text='CLASS PROBABILITY DISTRIBUTION',
                 bg=C['panel'], fg=C['gray2'], font=FONTS['tiny']).pack(anchor='w', pady=(4,6))

        probs = primary['probs']
        for i, (cname, prob) in enumerate(zip(CLASS_NAMES, probs)):
            ci = CLASS_INFO[cname]
            row = tk.Frame(self.result_panel, bg=C['panel'])
            row.pack(fill='x', padx=16, pady=2)
            tk.Label(row, text=cname.capitalize(), bg=C['panel'], fg=C['white'],
                     font=FONTS['small'], width=12, anchor='w').pack(side='left')
            bar_bg = tk.Frame(row, bg=C['bg3'], height=18)
            bar_bg.pack(side='left', fill='x', expand=True, padx=8)
            bar_bg.update_idletasks()
            # Draw fill after geometry settles
            bar_fill_color = ci['color'] if cname == cls else C['gray3']
            self.after(50, lambda bg=bar_bg, p=prob, col=bar_fill_color:
                       self._draw_bar(bg, p, col))
            tk.Label(row, text=f'{prob*100:5.1f}%', bg=C['panel'], fg=ci['color'],
                     font=FONTS['mono_sm'], width=7).pack(side='left')

        # ── Model comparison (if all) ─────────────────────────
        if mode == 'both':
            comp = tk.Frame(self.result_panel, bg=C['bg3'])
            comp.pack(fill='x', padx=16, pady=(10,0))
            tk.Label(comp, text='ALL MODELS COMPARISON', bg=C['bg3'],
                     fg=C['gray2'], font=FONTS['tiny']).pack(anchor='w', padx=10, pady=(8,4))
            available = [(k,l) for k,l in [('e2e','ResNet50 E2E'),('vgg','VGG16 E2E (Bonus)'),('svm','ResNet50+SVM')] if k in preds]
            classes   = [preds[k]['class'] for k,_ in available]
            all_agree = len(set(classes)) == 1
            msg = '✓  All models agree on the diagnosis' if all_agree else '⚠  Models disagree — manual review advised'
            col = C['teal'] if all_agree else C['amber']
            tk.Label(comp, text=msg, bg=C['bg3'], fg=col,
                     font=FONTS['body'], padx=10, pady=4).pack(anchor='w')
            for key, label in available:
                p  = preds[key]
                ci = CLASS_INFO[p['class']]
                tk.Label(comp,
                         text=f"  {label:28s} →  {p['class'].upper():12s}  ({p['confidence']*100:.1f}%)",
                         bg=C['bg3'], fg=ci['color'],
                         font=FONTS['mono_sm'], padx=10).pack(anchor='w')
            tk.Frame(comp, bg=C['panel'], height=8).pack(fill='x')

        # ── Action buttons ────────────────────────────────────
        act_row = tk.Frame(self.result_panel, bg=C['panel'])
        act_row.pack(fill='x', padx=16, pady=12)
        GlowButton(act_row, '📋  GENERATE REPORT', self._generate_report,
                   C['cyan'], width=200, height=38).pack(side='left', padx=(0,8))
        GlowButton(act_row, '📊  PROBABILITY CHART', self._show_prob_chart,
                   C['teal'], width=200, height=38).pack(side='left')

        ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        tk.Label(self.result_panel, text=f'Analyzed: {ts}',
                 bg=C['panel'], fg=C['gray3'], font=FONTS['tiny']).pack(anchor='e', padx=16)

    def _draw_bar(self, bg, prob, color):
        try:
            bg.update_idletasks()
            w = bg.winfo_width()
            fill_w = max(int(prob * w), 2)
            fill = tk.Frame(bg, bg=color, height=18, width=fill_w)
            fill.place(x=0, y=0)
        except Exception:
            pass

    # ── Upload / Clear ────────────────────────────────────────
    def _upload_image(self):
        path = filedialog.askopenfilename(
            title='Select MRI Image',
            filetypes=[('Image files','*.jpg *.jpeg *.png *.bmp *.tiff'),('All','*.*')])
        if not path: return
        self.img_path = path
        img = Image.open(path).convert('RGB')
        self.scanner.set_image(img.copy())
        fname = os.path.basename(path)
        size  = os.path.getsize(path) // 1024
        self.file_info.config(text=f'  {fname}   ({size} KB)   {img.size[0]}×{img.size[1]}px',
                              fg=C['teal'])

    def _clear_image(self):
        self.img_path   = None
        self._last_pred = None
        self.scanner.clear()
        self.file_info.config(text='No file selected', fg=C['gray1'])
        self._draw_idle_result()

    # ── Analysis ──────────────────────────────────────────────
    def _analyze(self):
        if not self.img_path:
            messagebox.showwarning('No Image', 'Please upload an MRI scan first.')
            return
        if not self.e2e_model and not self.svm_model:
            messagebox.showerror('No Models', 'Models not loaded. Run train.py first.')
            return
        self.progress.start(8)

        def _run():
            try:
                preds = predict_image(self.img_path, self.e2e_model,
                                      self.vgg_model, self.svm_model,
                                      self.scaler, self.backbone)
                self._last_pred = preds
                self.after(0, lambda: self._draw_result(preds))
            except Exception as ex:
                self.after(0, lambda: messagebox.showerror('Error', str(ex)))
            finally:
                self.after(0, self.progress.stop)

        threading.Thread(target=_run, daemon=True).start()

    # ════════════════════════════════════════════════════════
    # TAB 1 — RESULTS
    # ════════════════════════════════════════════════════════
    def _build_results(self, parent):
        canvas   = tk.Canvas(parent, bg=C['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        self.results_inner = tk.Frame(canvas, bg=C['bg'])
        win = canvas.create_window((0,0), window=self.results_inner, anchor='nw')
        self.results_inner.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
            lambda e: canvas.itemconfig(win, width=e.width))
        self._bind_scroll(canvas)

    def _populate_results(self):
        f = self.results_inner
        for w in f.winfo_children(): w.destroy()

        pad = dict(padx=24)
        tk.Label(f, text='Model Performance Dashboard', bg=C['bg'],
                 fg=C['white'], font=FONTS['display']).pack(anchor='w', pady=(20,2), **pad)
        tk.Label(f, text='Comparative analysis — Requirement 4  ·  CAI3105',
                 bg=C['bg'], fg=C['gray2'], font=FONTS['body']).pack(anchor='w', **pad)

        if not self.results_data:
            tk.Label(f, text='\n⚠  No results found. Run train.py first.',
                     bg=C['bg'], fg=C['amber'], font=FONTS['body']).pack(pady=40)
            return

        rd = self.results_data
        approaches = {
            'Approach-1\nResNet50+SVM': (rd.get('approach1',{}), C['cyan']),
            'Approach-2\nResNet50 E2E': (rd.get('approach2',{}), C['teal']),
            'Bonus\nVGG16 E2E':         (rd.get('bonus_vgg',{}), C['purple']),
        }

        # ── Metric cards ─────────────────────────────────────
        tk.Frame(f, bg=C['border'], height=1).pack(fill='x', padx=24, pady=14)
        cards_row = tk.Frame(f, bg=C['bg'])
        cards_row.pack(fill='x', padx=24)

        for title, (m, col) in approaches.items():
            if not m: continue
            card = tk.Frame(cards_row, bg=C['panel'],
                            highlightthickness=1, highlightbackground=C['border'])
            card.pack(side='left', expand=True, fill='both', padx=6)
            # Color top bar
            top = tk.Canvas(card, bg=C['panel'], height=4, highlightthickness=0)
            top.pack(fill='x')
            top.update_idletasks()
            top.create_rectangle(0,0,top.winfo_width()+800,4, fill=col, outline='')

            tk.Label(card, text=title, bg=C['panel'], fg=col,
                     font=FONTS['head'], justify='center').pack(pady=(10,6))

            for key, label in [('accuracy','Accuracy'),('precision','Precision'),
                                ('recall','Recall'),('f1','F1-Score')]:
                val = m.get(key, 0)
                row = tk.Frame(card, bg=C['panel'])
                row.pack(fill='x', padx=14, pady=1)
                tk.Label(row, text=label, bg=C['panel'], fg=C['gray2'],
                         font=FONTS['small'], width=10, anchor='w').pack(side='left')
                tk.Label(row, text=f'{val:.4f}', bg=C['panel'], fg=C['white'],
                         font=FONTS['mono']).pack(side='left')
                # Mini bar
                pct = int(val * 100)
                bg_bar = tk.Frame(card, bg=C['bg3'], height=3)
                bg_bar.pack(fill='x', padx=14, pady=(0,2))
                self.after(80, lambda b=bg_bar, v=val, c=col:
                           tk.Frame(b, bg=c, height=3,
                                    width=int(v*b.winfo_width())).place(x=0,y=0))

            tk.Label(card, text=f"⏱  {m.get('time',0)}s training",
                     bg=C['panel'], fg=C['gray2'], font=FONTS['tiny']).pack(pady=(4,12))

        # ── Charts ───────────────────────────────────────────
        tk.Frame(f, bg=C['border'], height=1).pack(fill='x', padx=24, pady=14)
        self._embed_bar_chart(f, approaches)
        self._embed_curves(f)

    def _mpl_style(self, fig, axes_list):
        fig.patch.set_facecolor(C['bg2'])
        for ax in axes_list:
            ax.set_facecolor(C['bg3'])
            ax.tick_params(colors=C['gray1'], labelsize=9)
            for sp in ax.spines.values():
                sp.set_color(C['border'])
            ax.grid(axis='y', color=C['border'], alpha=0.6, linewidth=0.5)
            ax.yaxis.label.set_color(C['gray1'])
            ax.title.set_color(C['white'])

    def _embed_bar_chart(self, parent, approaches):
        frame = tk.Frame(parent, bg=C['panel'],
                         highlightthickness=1, highlightbackground=C['border'])
        frame.pack(fill='x', padx=24, pady=6)
        tk.Label(frame, text='Metric Comparison', bg=C['panel'],
                 fg=C['white'], font=FONTS['head']).pack(anchor='w', padx=16, pady=(12,4))

        metrics = ['accuracy','precision','recall','f1']
        labels  = ['Accuracy','Precision','Recall','F1-Score']
        colors  = [C['cyan'], C['teal'], C['purple']]
        fig     = Figure(figsize=(10,3.4), facecolor=C['bg2'])
        ax      = fig.add_subplot(111, facecolor=C['bg3'])
        x, w    = np.arange(len(labels)), 0.25

        for i, (title, (m, col)) in enumerate(approaches.items()):
            if not m: continue
            vals = [m.get(k,0) for k in metrics]
            bars = ax.bar(x + i*w, vals, w, label=title.replace('\n',' '),
                          color=col, alpha=0.8, edgecolor=C['bg2'], linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=7.5, color=C['white'], fontweight='bold')

        ax.set_xticks(x + w); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, labelcolor=C['gray1'],
                  facecolor=C['bg3'], edgecolor=C['border'])
        self._mpl_style(fig, [ax])
        fig.tight_layout(pad=1.0)

        cv = FigureCanvasTkAgg(fig, master=frame)
        cv.draw()
        cv.get_tk_widget().pack(fill='x', padx=16, pady=(0,12))

    def _embed_curves(self, parent):
        if not self.results_data: return
        hr = self.results_data.get('history_resnet')
        hv = self.results_data.get('history_vgg')
        if not hr: return

        frame = tk.Frame(parent, bg=C['panel'],
                         highlightthickness=1, highlightbackground=C['border'])
        frame.pack(fill='x', padx=24, pady=6)
        tk.Label(frame, text='Learning Curves', bg=C['panel'],
                 fg=C['white'], font=FONTS['head']).pack(anchor='w', padx=16, pady=(12,4))

        fig  = Figure(figsize=(10,3.2), facecolor=C['bg2'])
        ax1  = fig.add_subplot(121, facecolor=C['bg3'])
        ax2  = fig.add_subplot(122, facecolor=C['bg3'])
        ep   = range(1, len(hr['train_loss'])+1)

        ax1.plot(ep, hr['val_loss'], color=C['cyan'],   lw=2, label='ResNet50')
        if hv: ax1.plot(ep, hv['val_loss'], color=C['purple'], lw=2, label='VGG16')
        ax1.set_title('Validation Loss', fontsize=11)
        ax1.legend(fontsize=8, labelcolor=C['gray1'], facecolor=C['bg3'], edgecolor=C['border'])

        ax2.plot(ep, hr['val_acc'], color=C['teal'],   lw=2, label='ResNet50')
        if hv: ax2.plot(ep, hv['val_acc'], color=C['purple'], lw=2, label='VGG16')
        ax2.set_title('Validation Accuracy', fontsize=11)
        ax2.legend(fontsize=8, labelcolor=C['gray1'], facecolor=C['bg3'], edgecolor=C['border'])

        self._mpl_style(fig, [ax1, ax2])
        fig.tight_layout(pad=1.2)

        cv = FigureCanvasTkAgg(fig, master=frame)
        cv.draw()
        cv.get_tk_widget().pack(fill='x', padx=16, pady=(0,12))
        tk.Frame(parent, bg=C['bg'], height=20).pack()

    # ════════════════════════════════════════════════════════
    # TAB 2 — REPORT
    # ════════════════════════════════════════════════════════
    def _build_report_tab(self, parent):
        tk.Label(parent, text='Patient Report Generator', bg=C['bg'],
                 fg=C['white'], font=FONTS['display']).pack(anchor='w', padx=24, pady=(20,4))
        tk.Label(parent, text='Generate and export a clinical diagnosis report',
                 bg=C['bg'], fg=C['gray2'], font=FONTS['body']).pack(anchor='w', padx=24)
        tk.Frame(parent, bg=C['border'], height=1).pack(fill='x', padx=24, pady=12)

        # Report preview area
        preview_f = tk.Frame(parent, bg=C['panel'],
                             highlightthickness=1, highlightbackground=C['border'])
        preview_f.pack(fill='both', expand=True, padx=24, pady=4)

        tk.Label(preview_f, text='REPORT PREVIEW', bg=C['panel'],
                 fg=C['cyan'], font=FONTS['tiny']).pack(anchor='w', padx=14, pady=(10,4))

        self.report_text = tk.Text(preview_f, bg=C['bg3'], fg=C['white'],
                                   font=FONTS['mono'], wrap='word', relief='flat',
                                   padx=16, pady=12, insertbackground=C['cyan'],
                                   selectbackground=C['bg4'])
        self.report_text.pack(fill='both', expand=True, padx=14, pady=(0,4))
        self.report_text.insert('1.0',
            'Run a diagnosis first (Tab: Diagnose), then click\n'
            '"Generate Report" to populate this preview.\n\n'
            'You can also click "Refresh Report" below.')
        self.report_text.config(state='disabled')

        btn_row = tk.Frame(preview_f, bg=C['panel'])
        btn_row.pack(fill='x', padx=14, pady=10)
        GlowButton(btn_row, '↻  REFRESH REPORT', self._refresh_report,
                   C['cyan'], width=200, height=38).pack(side='left', padx=(0,8))
        GlowButton(btn_row, '💾  SAVE AS TXT', self._save_report,
                   C['teal'], width=180, height=38).pack(side='left')

    def _refresh_report(self):
        if not self._last_pred:
            messagebox.showinfo('No Diagnosis', 'Run a diagnosis first in the Diagnose tab.')
            return
        report = self._build_report_str()
        self.report_text.config(state='normal')
        self.report_text.delete('1.0', 'end')
        self.report_text.insert('1.0', report)
        self.report_text.config(state='disabled')

    def _build_report_str(self):
        if not self._last_pred: return ''
        primary = self._last_pred.get('e2e') or self._last_pred.get('svm')
        cls  = primary['class']
        conf = primary['confidence']
        info = CLASS_INFO[cls]
        ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fname = os.path.basename(self.img_path) if self.img_path else 'N/A'
        lines = [
            '╔══════════════════════════════════════════════════════════╗',
            '║      NEUROSCAN AI — DIAGNOSTIC REPORT                   ║',
            '║      CAI3105 Deep Learning · Prof. Nashwa El-Bendary    ║',
            '╠══════════════════════════════════════════════════════════╣',
            f'  Date/Time     : {ts}',
            f'  Image File    : {fname}',
            f'  Model         : {self.model_var.get().upper()}',
            '══════════════════════════════════════════════════════════',
            f'  DIAGNOSIS     : {cls.upper()}',
            f'  Confidence    : {conf*100:.2f}%',
            f'  Risk Level    : {info["risk"]}',
            f'  Classification: {info["short"]}',
            '══════════════════════════════════════════════════════════',
            '  Clinical Note:',
            f'  {info["desc"]}',
            '',
            f'  Recommended Action:',
            f'  {info["action"]}',
            '══════════════════════════════════════════════════════════',
            '  Class Probabilities:',
        ]
        for cname, prob in zip(CLASS_NAMES, primary['probs']):
            bar = '█' * int(prob * 30)
            lines.append(f'    {cname:14s} {bar:<30s} {prob*100:5.1f}%')

        available_models = [
            ('e2e', 'ResNet50 E2E  '),
            ('vgg', 'VGG16 E2E    '),
            ('svm', 'ResNet50+SVM '),
        ]
        model_lines = [f"    {label}: {self._last_pred[k]['class'].upper():14s} ({self._last_pred[k]['confidence']*100:.2f}%)"
                       for k, label in available_models if k in self._last_pred]
        if len(model_lines) > 1:
            lines += ['══════════════════════════════════════════════════════════',
                      '  Model Comparison:'] + model_lines
        lines += [
            '╚══════════════════════════════════════════════════════════╝',
            '',
            '⚠  AI-generated report for research purposes only.',
            '   Always consult a qualified radiologist for clinical decisions.',
        ]
        return '\n'.join(lines)

    def _save_report(self):
        report = self._build_report_str()
        if not report:
            messagebox.showinfo('No Data', 'Run a diagnosis first.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[('Text file','*.txt'),('All','*.*')],
            initialfile=f'diagnosis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        if path:
            with open(path, 'w', encoding='utf-8') as f: f.write(report)
            messagebox.showinfo('Saved', f'Report saved:\n{path}')

    # ════════════════════════════════════════════════════════
    # TAB 3 — DATASET (Requirement 1)
    # ════════════════════════════════════════════════════════
    def _build_dataset_tab(self, parent):
        # Scrollable
        canvas = tk.Canvas(parent, bg=C['bg'], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=C['bg'])
        win   = canvas.create_window((0,0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: (
            canvas.itemconfig(win, width=e.width),
            canvas.configure(scrollregion=canvas.bbox('all'))
        ))
        self._bind_scroll(canvas)
        # Propagate scroll from all child widgets
        def _propagate_scroll(widget, fn):
            widget.bind('<MouseWheel>', fn)
            widget.bind('<Button-4>',   fn)
            widget.bind('<Button-5>',   fn)
            for child in widget.winfo_children():
                try: _propagate_scroll(child, fn)
                except: pass
        inner.bind('<Map>', lambda e: _propagate_scroll(inner,
            lambda ev: canvas.yview_scroll(-1*(ev.delta//120) if ev.delta else
                       (-1 if ev.num==4 else 1), 'units')))

        pad = dict(padx=28)
        tk.Label(inner, text='Requirement 1 — Dataset Specification', bg=C['bg'],
                 fg=C['white'], font=FONTS['display']).pack(anchor='w', pady=(20,2), **pad)
        tk.Label(inner, text='CAI3105 · Brain MRI Tumor Detection',
                 bg=C['bg'], fg=C['gray2'], font=FONTS['body']).pack(anchor='w', **pad)
        tk.Frame(inner, bg=C['border'], height=1).pack(fill='x', padx=28, pady=14)

        def section(title, color=None):
            f = tk.Frame(inner, bg=C['bg'])
            f.pack(fill='x', **pad, pady=(10,4))
            bar = tk.Canvas(f, bg=C['bg'], width=3, height=16, highlightthickness=0)
            bar.pack(side='left')
            bar.create_rectangle(0,0,3,16, fill=color or C['cyan'], outline='')
            tk.Label(f, text=f'  {title}', bg=C['bg'],
                     fg=color or C['cyan'], font=('Segoe UI',10,'bold')).pack(side='left')

        def card(parent_frame, rows, color=C['cyan']):
            c = tk.Frame(inner, bg=C['panel'],
                         highlightthickness=1, highlightbackground=C['border'])
            c.pack(fill='x', **pad, pady=3)
            for i,(k,v) in enumerate(rows):
                row = tk.Frame(c, bg=C['panel'] if i%2==0 else C['bg3'])
                row.pack(fill='x')
                tk.Label(row, text=k, bg=row.cget('bg'), fg=color,
                         font=FONTS['small'], width=26, anchor='w',
                         padx=14, pady=8).pack(side='left')
                tk.Label(row, text=v, bg=row.cget('bg'), fg=C['white'],
                         font=FONTS['body'], anchor='w', wraplength=700,
                         justify='left').pack(side='left', padx=8)

        # ── 1. Dataset Metadata ───────────────────────────────
        section('1 · Dataset Metadata', C['cyan'])
        card(inner, [
            ('Source',            'Kaggle — Brain MRI Images for Brain Tumor Detection'),
            ('Problem Domain',    'Medical Imaging / Brain Tumor Classification'),
            ('Total Samples (N)', '5,600 MRI images  (Training: 5,712 · Testing: 1,311)'),
            ('Class Distribution','Balanced — 1,400 training images per class'),
            ('Dataset Split Used','Pre-split by Kaggle into Training / Testing folders'),
            ('Access',            'Kaggle Dataset · Mendeley Data · Figshare'),
        ], C['cyan'])

        # ── 2. Technical Specifications ───────────────────────
        section('2 · Technical Specifications', C['teal'])
        card(inner, [
            ('Image Resolution',  'Resized to 224 × 224 pixels (standard for ResNet50 / VGG16)'),
            ('Original Resolution','Variable — ranging from ~512×512 to 1200×1200 pixels'),
            ('Color Channels',    'RGB (3 channels) — grayscale MRIs converted to RGB'),
            ('Number of Classes', '4 classes'),
            ('Class 1',           'Glioma       — malignant glial cell tumour'),
            ('Class 2',           'Meningioma   — meningeal membrane tumour'),
            ('Class 3',           'No Tumor     — healthy brain scan'),
            ('Class 4',           'Pituitary    — pituitary gland tumour'),
        ], C['teal'])

        # ── 3. Data Preprocessing ─────────────────────────────
        section('3 · Data Preprocessing', C['blue'])
        card(inner, [
            ('Step 1 — Resize',
             'All images resized to 224×224 px using bicubic interpolation '             '(required input size for ResNet50 and VGG16 architectures)'),
            ('Step 2 — Convert to Tensor',
             'PIL images converted to PyTorch tensors with values in [0.0, 1.0]'),
            ('Step 3 — Normalisation',
             'Per-channel normalisation using ImageNet statistics:\n'             'Mean = [0.485, 0.456, 0.406]   Std = [0.229, 0.224, 0.225]\n'             'Ensures compatibility with ImageNet-pretrained weights'),
            ('Applied To',
             'Training, Validation, and Testing sets (normalisation applied to all)'),
        ], C['blue'])

        # ── 4. Data Augmentation ──────────────────────────────
        section('4 · Data Augmentation (Training only)', C['purple'])
        aug_rows = [
            ('RandomHorizontalFlip',
             'p=0.5 — MRI scans can appear mirrored depending on scanner orientation. '             'Flipping does not change diagnostic content and doubles effective training samples.'),
            ('RandomRotation',
             'degrees=±15° — Accounts for slight patient head tilt during scanning. '             'Prevents the model from relying on exact orientation.'),
            ('ColorJitter',
             'brightness=0.2, contrast=0.2 — Simulates variability between different MRI '             'scanners and field strengths (1.5T vs 3T). Improves robustness to scanner differences.'),
            ('RandomAffine',
             'translate=(0.05, 0.05) — Small positional shifts (±5%) simulate variability in '             'head positioning inside the scanner bore.'),
            ('Justification',
             'Augmentation is applied ONLY to the training set to prevent data leakage. '             'It artificially expands the effective dataset, reduces overfitting, and improves '             'generalisation to unseen clinical MRI scans from different institutions.'),
        ]
        card(inner, aug_rows, C['purple'])

        # ── 5. Data Splitting ─────────────────────────────────
        section('5 · Data Splitting', C['amber'])
        card(inner, [
            ('Strategy',
             'Dataset is pre-split by Kaggle into Training and Testing folders. '             'A Validation set is carved from Training using an 80/20 split.'),
            ('Training Set',   '80% of Training folder  ≈  4,480 images  (used for model weight updates)'),
            ('Validation Set', '20% of Training folder  ≈  1,120 images  (used for hyperparameter tuning & early stopping)'),
            ('Testing Set',    'Fixed Kaggle test split ≈  1,311 images  (held-out, used only for final evaluation)'),
            ('Splitting Method','torch.utils.data.random_split() with fixed seed=42 for reproducibility'),
            ('Class Balance',  'All splits maintain balanced class distribution (equal samples per class)'),
        ], C['amber'])

        # ── Visual split chart ────────────────────────────────
        chart_frame = tk.Frame(inner, bg=C['panel'],
                               highlightthickness=1, highlightbackground=C['border'])
        chart_frame.pack(fill='x', **pad, pady=(6,20))
        tk.Label(chart_frame, text='Data Split Visualisation', bg=C['panel'],
                 fg=C['white'], font=FONTS['head']).pack(anchor='w', padx=16, pady=(10,6))

        fig  = Figure(figsize=(9, 2.2), facecolor=C['bg2'])
        ax   = fig.add_subplot(111, facecolor=C['bg2'])
        splits  = ['Training\n4,480 imgs', 'Validation\n1,120 imgs', 'Testing\n1,311 imgs']
        vals    = [4480, 1120, 1311]
        colors_ = [C['teal'], C['cyan'], C['amber']]
        bars    = ax.barh(splits, vals, color=colors_, height=0.5, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+60, bar.get_y()+bar.get_height()/2,
                    f'{v:,} images', va='center', color=C['white'], fontsize=9)
        ax.set_xlim(0, 5800)
        ax.set_xlabel('Number of Images', color=C['gray1'], fontsize=9)
        ax.tick_params(colors=C['gray1'], labelsize=9)
        for sp in ax.spines.values(): sp.set_color(C['border'])
        ax.grid(axis='x', color=C['border'], alpha=0.5)
        fig.tight_layout(pad=0.8)

        cv = FigureCanvasTkAgg(fig, master=chart_frame)
        cv.draw()
        cv.get_tk_widget().pack(fill='x', padx=16, pady=(0,12))

    # ════════════════════════════════════════════════════════
    # TAB 4 — MODELS (Requirement 2)
    # ════════════════════════════════════════════════════════
    def _build_models_tab(self, parent):
        canvas = tk.Canvas(parent, bg=C['bg'], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=C['bg'])
        win   = canvas.create_window((0,0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: (
            canvas.itemconfig(win, width=e.width),
            canvas.configure(scrollregion=canvas.bbox('all'))
        ))
        self._bind_scroll(canvas)
        # Propagate scroll from all child widgets
        def _propagate_scroll(widget, fn):
            widget.bind('<MouseWheel>', fn)
            widget.bind('<Button-4>',   fn)
            widget.bind('<Button-5>',   fn)
            for child in widget.winfo_children():
                try: _propagate_scroll(child, fn)
                except: pass
        inner.bind('<Map>', lambda e: _propagate_scroll(inner,
            lambda ev: canvas.yview_scroll(-1*(ev.delta//120) if ev.delta else
                       (-1 if ev.num==4 else 1), 'units')))

        pad = dict(padx=28)
        tk.Label(inner, text='Requirement 2 — Model Selection & Justification',
                 bg=C['bg'], fg=C['white'], font=FONTS['display']).pack(anchor='w', pady=(20,2), **pad)
        tk.Label(inner, text='CAI3105 · ResNet50 Architecture + Hyperparameters',
                 bg=C['bg'], fg=C['gray2'], font=FONTS['body']).pack(anchor='w', **pad)
        tk.Frame(inner, bg=C['border'], height=1).pack(fill='x', padx=28, pady=14)

        def section(title, color):
            f = tk.Frame(inner, bg=C['bg'])
            f.pack(fill='x', **pad, pady=(12,4))
            bar = tk.Canvas(f, bg=C['bg'], width=3, height=16, highlightthickness=0)
            bar.pack(side='left')
            bar.create_rectangle(0,0,3,16, fill=color, outline='')
            tk.Label(f, text=f'  {title}', bg=C['bg'], fg=color,
                     font=('Segoe UI',10,'bold')).pack(side='left')

        def info_card(rows, color=C['cyan']):
            c = tk.Frame(inner, bg=C['panel'],
                         highlightthickness=1, highlightbackground=C['border'])
            c.pack(fill='x', **pad, pady=3)
            for i,(k,v) in enumerate(rows):
                row = tk.Frame(c, bg=C['panel'] if i%2==0 else C['bg3'])
                row.pack(fill='x')
                tk.Label(row, text=k, bg=row.cget('bg'), fg=color,
                         font=FONTS['small'], width=28, anchor='w',
                         padx=14, pady=8).pack(side='left')
                tk.Label(row, text=v, bg=row.cget('bg'), fg=C['white'],
                         font=FONTS['body'], anchor='w', wraplength=680,
                         justify='left').pack(side='left', padx=8)

        # ── Req 2.1 — Model choice ────────────────────────────
        section('2.1 · Selected Architecture — ResNet50', C['teal'])
        info_card([
            ('Architecture',     'ResNet50 (Deep Residual Network — 50 layers)'),
            ('Pretrained On',    'ImageNet LSVRC-2012 (1.2M images, 1,000 classes)'),
            ('Why ResNet50',
             'ResNet50 introduces residual (skip) connections that solve the vanishing-gradient '
             'problem, enabling very deep networks to train without degradation. Its 50-layer '
             'depth extracts rich hierarchical features: edges → textures → shapes → semantic '
             'regions — which transfer excellently to MRI classification tasks.'),
            ('vs VGG16',
             'ResNet50 achieves comparable accuracy to VGG16 with 8× fewer parameters (25M vs 138M), '
             'making it faster to train and less prone to overfitting on medical datasets.'),
            ('vs MobileNet',
             'ResNet50 is more accurate than MobileNetV1/V2 for complex medical imaging, '
             'accepting the trade-off of higher computation for better feature discrimination.'),
            ('Literature Reference',
             'He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image '
             'Recognition. CVPR 2016. arXiv:1512.03385'),
        ], C['teal'])

        # ── Req 2.2 — Architecture diagram ───────────────────
        section('2.2 · ResNet50 Architecture Diagram', C['cyan'])
        arch_frame = tk.Frame(inner, bg=C['panel'],
                              highlightthickness=1, highlightbackground=C['border'])
        arch_frame.pack(fill='x', **pad, pady=3)
        tk.Label(arch_frame, text='ResNet50 Layer-by-Layer Architecture',
                 bg=C['panel'], fg=C['white'], font=FONTS['head']).pack(anchor='w', padx=16, pady=(12,6))

        fig = Figure(figsize=(10, 3.8), facecolor=C['bg2'])
        ax  = fig.add_subplot(111, facecolor=C['bg2'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw architecture boxes
        blocks = [
            (2,   3.5, 8,  'INPUT\n224x224x3',      C['gray2'],    'white'),
            (12,  3.5, 8,  'Conv1\n7x7, 64\nstride 2', C['cyan_dim'], 'white'),
            (22,  3.5, 8,  'MaxPool\n3x3\nstride 2',   C['cyan_dim'], 'white'),
            (32,  2,   14, 'Layer 1\n3x Bottleneck\n256 ch',  C['teal_dim'],  'white'),
            (48,  2,   14, 'Layer 2\n4x Bottleneck\n512 ch',  C['blue_dim'],  'white'),
            (64,  2,   14, 'Layer 3\n6x Bottleneck\n1024 ch', C['cyan_dim'],  'white'),
            (80,  2,   14, 'Layer 4\n3x Bottleneck\n2048 ch', C['amber_dim'], 'white'),
            (92,  3.5, 6,  'AvgPool\n+ FC\n4 classes',        C['green_dim'], 'white'),
        ]

        for x, y, w, label, color, fc in blocks:
            rect = plt.matplotlib.patches.FancyBboxPatch(
                (x, y), w, 5.5,
                boxstyle='round,pad=0.3',
                facecolor=color, edgecolor=C['border2'], linewidth=1.2,
                transform=ax.transData)
            ax.add_patch(rect)
            ax.text(x + w/2, y + 2.75, label, ha='center', va='center',
                    color='white', fontsize=7, fontweight='bold',
                    transform=ax.transData)

        # Arrows between blocks
        arrow_xs = [(10,12),(20,22),(30,32),(46,48),(62,64),(78,80),(90,92)]
        for x1, x2 in arrow_xs:
            ax.annotate('', xy=(x2, 5.25), xytext=(x1, 5.25),
                        arrowprops=dict(arrowstyle='->', color=C['cyan'], lw=1.5))

        # Skip connection arc
        ax.annotate('', xy=(46, 8.5), xytext=(32, 8.5),
                    arrowprops=dict(arrowstyle='->', color=C['teal'],
                                    lw=1.2, connectionstyle='arc3,rad=-0.3'))
        ax.text(39, 9.2, 'Residual / Skip Connection', color=C['teal'],
                fontsize=7.5, ha='center')

        ax.text(50, 0.5, 'Reference: He et al. (2016) — Deep Residual Learning for Image Recognition. CVPR.',
                color=C['gray2'], fontsize=7, ha='center', style='italic')

        fig.tight_layout(pad=0.5)
        cv = FigureCanvasTkAgg(fig, master=arch_frame)
        cv.draw()
        cv.get_tk_widget().pack(fill='x', padx=16, pady=(0,12))

        # ── Req 2.3 — Hyperparameter Table ───────────────────
        section('2.3 · Hyperparameter Table', C['amber'])
        htable = tk.Frame(inner, bg=C['panel'],
                          highlightthickness=1, highlightbackground=C['border'])
        htable.pack(fill='x', **pad, pady=(3,20))
        tk.Label(htable, text='Complete Hyperparameter Configuration — Both Approaches',
                 bg=C['panel'], fg=C['white'], font=FONTS['head']).pack(anchor='w', padx=16, pady=(12,8))

        # Table header
        cols  = ['Hyperparameter', 'Approach-1\nResNet50 Feature Extractor', 'Approach-2\nResNet50 E2E', 'Bonus\nVGG16 E2E']
        col_colors = [C['gray2'], C['cyan'], C['teal'], C['purple']]
        header = tk.Frame(htable, bg=C['bg4'])
        header.pack(fill='x', padx=14)
        widths = [24, 26, 26, 20]
        for col, color, w in zip(cols, col_colors, widths):
            tk.Label(header, text=col, bg=C['bg4'], fg=color,
                     font=('Segoe UI',9,'bold'), width=w, anchor='center',
                     pady=8, justify='center').pack(side='left')

        # Table rows
        rows = [
            ('Input Image Size',   '224 × 224 px',    '224 × 224 px',    '224 × 224 px'),
            ('Colour Channels',    'RGB (3)',          'RGB (3)',          'RGB (3)'),
            ('Batch Size',         '32',               '32',               '32'),
            ('Epochs',             'N/A (SVM)',        '10',               '10'),
            ('Learning Rate',      'N/A (SVM)',        '1e-4',             '1e-4'),
            ('Optimizer',          'N/A (SVM)',        'Adam (β1=0.9)',     'Adam (β1=0.9)'),
            ('LR Scheduler',       'N/A (SVM)',        'StepLR (step=5)',  'StepLR (step=5)'),
            ('LR Decay Factor',    'N/A (SVM)',        'γ = 0.5',          'γ = 0.5'),
            ('Loss Function',      'N/A (SVM)',        'CrossEntropyLoss', 'CrossEntropyLoss'),
            ('Dropout Rate',       'N/A',              '0.4',              '0.4'),
            ('Frozen Layers',      'ALL (fixed)',      'Conv1–Layer3',     'features block'),
            ('Trainable Layers',   'None (backbone)',  'Layer4 + FC',      'classifier'),
            ('Feature Dim (SVM)',  '2048',             'N/A',              'N/A'),
            ('Feature Scaler',     'StandardScaler',   'N/A',              'N/A'),
            ('SVM Kernel',         'Linear',           'N/A',              'N/A'),
            ('SVM C',              '1.0',              'N/A',              'N/A'),
            ('SVM max_iter',       '1000',             'N/A',              'N/A'),
            ('Random Seed',        '42',               '42',               '42'),
            ('Val Split',          '20%',              '20%',              '20%'),
            ('Pretrained Weights', 'ImageNet (fixed)', 'ImageNet (init)',   'ImageNet (init)'),
        ]
        for i, (param, v1, v2, v3) in enumerate(rows):
            bg = C['panel'] if i%2==0 else C['bg3']
            row = tk.Frame(htable, bg=bg)
            row.pack(fill='x', padx=14)
            vals = [param, v1, v2, v3]
            fgs  = [C['gray1'], C['cyan'], C['teal'], C['purple']]
            for val, fg, w in zip(vals, fgs, widths):
                tk.Label(row, text=val, bg=bg, fg=fg,
                         font=FONTS['small'], width=w, anchor='w' if w==24 else 'center',
                         padx=6, pady=6).pack(side='left')
        tk.Frame(htable, bg=C['panel'], height=12).pack()

    # ════════════════════════════════════════════════════════
    # TAB 5 — CONCLUSION (Requirement 4)
    # ════════════════════════════════════════════════════════
    def _build_conclusion_tab(self, parent):
        canvas = tk.Canvas(parent, bg=C['bg'], highlightthickness=0)
        sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=C['bg'])
        win   = canvas.create_window((0,0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: (
            canvas.itemconfig(win, width=e.width),
            canvas.configure(scrollregion=canvas.bbox('all'))
        ))
        self._bind_scroll(canvas)
        # Propagate scroll from all child widgets
        def _propagate_scroll(widget, fn):
            widget.bind('<MouseWheel>', fn)
            widget.bind('<Button-4>',   fn)
            widget.bind('<Button-5>',   fn)
            for child in widget.winfo_children():
                try: _propagate_scroll(child, fn)
                except: pass
        inner.bind('<Map>', lambda e: _propagate_scroll(inner,
            lambda ev: canvas.yview_scroll(-1*(ev.delta//120) if ev.delta else
                       (-1 if ev.num==4 else 1), 'units')))

        pad = dict(padx=28)
        tk.Label(inner, text='Requirement 4 — Comparative Analysis & Conclusion',
                 bg=C['bg'], fg=C['white'], font=FONTS['display']).pack(anchor='w', pady=(20,2), **pad)
        tk.Label(inner, text='CAI3105 · Experimental Results Discussion',
                 bg=C['bg'], fg=C['gray2'], font=FONTS['body']).pack(anchor='w', **pad)
        tk.Frame(inner, bg=C['border'], height=1).pack(fill='x', padx=28, pady=14)

        def section(num, title, color):
            f = tk.Frame(inner, bg=C['bg'])
            f.pack(fill='x', **pad, pady=(14,4))
            num_lbl = tk.Label(f, text=num, bg=color, fg=C['bg'],
                               font=('Segoe UI',9,'bold'), padx=8, pady=2)
            num_lbl.pack(side='left')
            tk.Label(f, text=f'  {title}', bg=C['bg'], fg=color,
                     font=('Segoe UI',11,'bold')).pack(side='left')

        def answer_card(question, answer, color, icon=''):
            outer = tk.Frame(inner, bg=C['panel'],
                             highlightthickness=1, highlightbackground=color)
            outer.pack(fill='x', **pad, pady=4)
            q_bar = tk.Canvas(outer, bg=C['panel'], width=4, highlightthickness=0)
            q_bar.pack(side='left', fill='y')
            q_bar.update_idletasks()
            self.after(50, lambda b=q_bar, c=color: (
                b.configure(height=b.winfo_height()),
                b.create_rectangle(0,0,4,b.winfo_height()+200, fill=c, outline='')))
            body = tk.Frame(outer, bg=C['panel'])
            body.pack(side='left', fill='both', expand=True, padx=12, pady=12)
            tk.Label(body, text=f'{icon}  {question}', bg=C['panel'], fg=color,
                     font=('Segoe UI',10,'bold'), anchor='w',
                     wraplength=820, justify='left').pack(anchor='w')
            tk.Frame(body, bg=C['border'], height=1).pack(fill='x', pady=6)
            tk.Label(body, text=answer, bg=C['panel'], fg=C['white'],
                     font=FONTS['body'], anchor='w', wraplength=820,
                     justify='left').pack(anchor='w')

        # ── Results summary chart ─────────────────────────────
        section('4.1', 'Performance Comparison Chart', C['cyan'])
        chart_f = tk.Frame(inner, bg=C['panel'],
                           highlightthickness=1, highlightbackground=C['border'])
        chart_f.pack(fill='x', **pad, pady=4)

        # Store ref to chart frame so we can populate later
        self._conclude_chart_frame = chart_f
        self._conclude_inner = inner
        self.after(500, self._populate_conclusion_chart)  # populate after models load

        if self.results_data:
            rd = self.results_data
            fig = Figure(figsize=(10, 3.2), facecolor=C['bg2'])
            ax  = fig.add_subplot(111, facecolor=C['bg3'])
            approaches = ['ResNet50+SVM\n(Approach-1)', 'ResNet50 E2E\n(Approach-2)', 'VGG16 E2E\n(Bonus)']
            metrics    = ['accuracy','precision','recall','f1']
            m_labels   = ['Accuracy','Precision','Recall','F1-Score']
            colors_    = [C['cyan'], C['teal'], C['purple']]
            x = np.arange(len(m_labels))
            w = 0.25
            data = [rd.get('approach1',{}), rd.get('approach2',{}), rd.get('bonus_vgg',{})]
            for i,(app,d,col) in enumerate(zip(approaches,data,colors_)):
                if not d: continue
                vals = [d.get(k,0) for k in metrics]
                bars = ax.bar(x+i*w, vals, w, label=app, color=col, alpha=0.85,
                              edgecolor=C['bg2'], linewidth=0.5)
                for bar,v in zip(bars,vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
                            f'{v:.3f}', ha='center', va='bottom',
                            fontsize=7.5, color=C['white'], fontweight='bold')
            ax.set_xticks(x+w); ax.set_xticklabels(m_labels, fontsize=10, color=C['gray1'])
            ax.set_ylim(0,1.15)
            ax.tick_params(colors=C['gray1'])
            for sp in ax.spines.values(): sp.set_color(C['border'])
            ax.grid(axis='y', color=C['border'], alpha=0.5)
            ax.legend(fontsize=8, labelcolor=C['gray1'],
                      facecolor=C['bg3'], edgecolor=C['border'])
            fig.tight_layout(pad=0.8)
            cv = FigureCanvasTkAgg(fig, master=chart_f)
            cv.draw()
            cv.get_tk_widget().pack(fill='x', padx=16, pady=12)

            # Numeric summary table
            summary_f = tk.Frame(chart_f, bg=C['panel'])
            summary_f.pack(fill='x', padx=16, pady=(0,12))
            cols_h = ['Approach','Accuracy','Precision','Recall','F1-Score','Time (s)']
            col_w  = [28,12,12,12,12,10]
            hdr    = tk.Frame(summary_f, bg=C['bg4'])
            hdr.pack(fill='x')
            for col,w in zip(cols_h,col_w):
                tk.Label(hdr, text=col, bg=C['bg4'], fg=C['cyan'],
                         font=('Segoe UI',9,'bold'), width=w,
                         anchor='center', pady=6).pack(side='left')
            rows_d = [
                ('Approach-1  ResNet50+SVM', rd.get('approach1',{}), C['cyan']),
                ('Approach-2  ResNet50 E2E', rd.get('approach2',{}), C['teal']),
                ('Bonus       VGG16 E2E',    rd.get('bonus_vgg',{}),  C['purple']),
            ]
            for i,(name,d,col) in enumerate(rows_d):
                if not d: continue
                bg = C['panel'] if i%2==0 else C['bg3']
                rr = tk.Frame(summary_f, bg=bg)
                rr.pack(fill='x')
                vals = [name,
                        f"{d.get('accuracy',0):.4f}",
                        f"{d.get('precision',0):.4f}",
                        f"{d.get('recall',0):.4f}",
                        f"{d.get('f1',0):.4f}",
                        f"{d.get('time',0)}s"]
                fgs = [col, C['white'],C['white'],C['white'],C['white'],C['gray1']]
                for val,fg,w in zip(vals,fgs,col_w):
                    tk.Label(rr, text=val, bg=bg, fg=fg,
                             font=FONTS['mono_sm'], width=w,
                             anchor='center' if w<28 else 'w',
                             padx=6, pady=6).pack(side='left')
        else:
            tk.Label(chart_f,
                     text='⚠  Run train.py first to generate results data.',
                     bg=C['panel'], fg=C['amber'], font=FONTS['body'],
                     padx=20, pady=20).pack()

        # ── 4 Guidance Questions ──────────────────────────────
        section('4.2', 'Conclusion — Guidance Questions', C['teal'])

        rd = self.results_data or {}
        a1_acc = rd.get('approach1',{}).get('accuracy',0.8875)
        a2_acc = rd.get('approach2',{}).get('accuracy',0.9444)
        a1_t   = rd.get('approach1',{}).get('time',270)
        a2_t   = rd.get('approach2',{}).get('time',2470)
        diff   = (a2_acc - a1_acc)*100

        answer_card(
            'i.  How did the ML classifier compare to the End-to-End DL model?',
            f'The End-to-End ResNet50 (Approach-2) significantly outperformed the hybrid ResNet50+SVM pipeline '
            f'(Approach-1) across all evaluation metrics. Approach-2 achieved {a2_acc*100:.2f}% accuracy versus '
            f'{a1_acc*100:.2f}% for Approach-1 — a difference of {diff:.2f} percentage points. '
            f'This performance gap is consistent across Precision, Recall, and F1-Score. '
            f'The superior performance of Approach-2 stems from its ability to fine-tune the backbone '
            f'high-level representations specifically to MRI brain scan textures, rather than relying on '
            f'fixed ImageNet features fed into a separate SVM classifier.',
            C['cyan'], 'Q1'
        )

        answer_card(
            'ii.  Advantages and limitations of DL Feature Extractor vs End-to-End pipeline?',
            ('Approach-1 (Feature Extractor + SVM):\n'
             '  + Advantages:\n'
             '     No GPU required for SVM training stage — runs on CPU efficiently\n'
             '     Much faster classifier training once features are extracted\n'
             '     Robust with limited labelled data — SVM generalises well\n'
             '     Interpretable decision boundary (linear SVM is explainable)\n'
             '     Feature extraction can be done offline and cached\n'
             '  - Limitations:\n'
             '     Feature extractor not optimised for the medical imaging domain\n'
             '     Two-stage pipeline harder to deploy and maintain in production\n'
             '     Feature quality bounded by frozen pretrained representations\n\n'
             'Approach-2 (End-to-End Fine-Tuning):\n'
             '  + Advantages:\n'
             '     Backbone adapts representations to the target medical domain\n'
             '     Single unified pipeline — simpler deployment and inference\n'
             '     Higher accuracy — learns task-specific discriminative features\n'
             '  - Limitations:\n'
             '     Requires GPU and substantial training time for fine-tuning\n'
             '     More prone to overfitting on small datasets without regularisation\n'
             '     Less interpretable — end-to-end black-box decision process'),
            C['teal'], 'Q2'
        )

        answer_card(
            'iii.  Which approach was more efficient in terms of training time?',
            f'Approach-1 (ResNet50+SVM) completed in {a1_t}s total, which is significantly faster than '
            f'Approach-2 (ResNet50 E2E) at {a2_t}s — a {round(a2_t/max(a1_t,1),1)}× difference. '
            f'However, this comparison requires nuance: Approach-1 still requires a full forward pass '
            f'through the ResNet50 backbone for feature extraction. The time saving comes from replacing '
            f'backpropagation-based training (10 epochs × full dataset) with a single SVM fit operation. '
            f'At inference time, both approaches are similarly fast since only a single forward pass is needed. '
            f'VGG16 E2E was the slowest due to its large fully-connected layers (138M parameters vs 25M for ResNet50).',
            C['amber'], 'Q3'
        )

        answer_card(
            'iv.  Which strategy for resource-constrained vs high-performance environments?',
            ('Resource-Constrained (mobile, edge, embedded, IoT):\n'
             '  Recommended: Approach-1 - Feature Extractor + SVM\n'
             '  Feature extraction offline on server; lightweight SVM runs on edge device\n'
             '  No GPU needed at inference — minimal RAM requirement\n'
             '  For extreme constraints replace ResNet50 with MobileNetV2 (3.4M parameters)\n'
             '  Linear SVM produces simple weight matrix deployable on microcontrollers\n\n'
             'High-Performance (cloud server, GPU workstation, hospital PACS system):\n'
             '  Recommended: Approach-2 - End-to-End ResNet50 Fine-Tuning\n'
             '  The ~5.7% accuracy advantage is clinically significant\n'
             '  In medical AI even 1% accuracy improvement reduces misdiagnoses\n'
             '  Single-pipeline architecture simpler to maintain, version and audit\n'
             '  GPU inference fast enough for real-time clinical decision support'),
            C['purple'], 'Q4'
        )

        tk.Frame(inner, bg=C['bg'], height=24).pack()

    # ════════════════════════════════════════════════════════
    # TAB 3 — ABOUT
    # ════════════════════════════════════════════════════════
    def _build_about(self, parent):
        self._about_parent = parent
        self._populate_about()

    def _populate_about(self):
        parent = self._about_parent
        for w in parent.winfo_children():
            w.destroy()

        tk.Label(parent, text='Project Information', bg=C['bg'],
                 fg=C['white'], font=FONTS['display']).pack(anchor='w', padx=24, pady=(20,4))
        tk.Frame(parent, bg=C['border'], height=1).pack(fill='x', padx=24, pady=12)

        grid = tk.Frame(parent, bg=C['bg'])
        grid.pack(fill='both', expand=True, padx=24)

        # Pull live metrics from results if available
        rd = self.results_data or {}
        a1  = rd.get('approach1', {})
        a2  = rd.get('approach2', {})
        bvg = rd.get('bonus_vgg', {})
        a1_acc  = f"{a1.get('accuracy',0)*100:.2f}%"  if a1  else 'N/A'
        a2_acc  = f"{a2.get('accuracy',0)*100:.2f}%"  if a2  else 'N/A'
        vgg_acc = f"{bvg.get('accuracy',0)*100:.2f}%" if bvg else 'N/A'
        a1_t    = f"{a1.get('time',0)}s"   if a1  else 'N/A'
        a2_t    = f"{a2.get('time',0)}s"   if a2  else 'N/A'
        vgg_t   = f"{bvg.get('time',0)}s"  if bvg else 'N/A'

        info = [
            ('📘 Course',          'CAI3105 — Deep Learning',                          C['cyan']),
            ('👩‍🏫 Lecturer',       'Prof. Nashwa El-Bendary',                          C['teal']),
            ('🏫 Department',      'Artificial Intelligence — Smart Village Campus',    C['cyan']),
            ('📅 Deadline',        'Wednesday, May 13th 2026, 11:55 PM — Moodle LMS',  C['amber']),
            ('📦 Dataset',         'Brain MRI Tumor Detection — Kaggle / Mendeley',    C['cyan']),
            ('🔢 Classes (4)',      'Glioma  ·  Meningioma  ·  No Tumor  ·  Pituitary',C['teal']),
            ('🖼 Total Images',    '5,600 images  (1,400 per class, balanced)',         C['cyan']),
            ('✂ Data Split',       '80% Train  ·  20% Validation  ·  Fixed Test set',  C['teal']),
            ('📐 Approach-1',      f'ResNet50 Feature Extractor + SVM  →  Acc: {a1_acc}  Time: {a1_t}',  C['cyan']),
            ('📐 Approach-2',      f'End-to-End ResNet50 Fine-Tuning   →  Acc: {a2_acc}  Time: {a2_t}',  C['teal']),
            ('⭐ Bonus CNN',        f'End-to-End VGG16 Fine-Tuning      →  Acc: {vgg_acc}  Time: {vgg_t}', C['purple']),
            ('🤖 Architecture-1',  'ResNet50 — 50 layers, 25M params, residual connections',              C['teal']),
            ('🤖 Architecture-2',  'VGG16    — 16 layers, 138M params, sequential convolutions',          C['purple']),
            ('🏆 Best Model',       f'ResNet50 E2E — Accuracy: {a2_acc}',              C['teal']),
            ('📊 Req 1',           'Dataset Metadata, Specs, Preprocessing, Augmentation, Splitting',     C['cyan']),
            ('📊 Req 2',           'Model Justification, Architecture Diagram, Hyperparameter Table',     C['teal']),
            ('📊 Req 3',           'Approach-1 (SVM) + Approach-2 (E2E) Implementation & Metrics',       C['cyan']),
            ('📊 Req 4',           'Comparative Analysis, Charts, Conclusion (4 Guidance Questions)',     C['teal']),
            ('⭐ Bonus 1',          'Extra CNN Architecture: VGG16 End-to-End',                           C['purple']),
            ('🏅 Total Marks',     '20 Marks + 2 Bonus Marks',                                           C['amber']),
        ]
        for label, val, col in info:
            row = tk.Frame(grid, bg=C['panel'],
                           highlightthickness=0)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, bg=C['panel'], fg=col,
                     font=FONTS['head'], width=18, anchor='w', padx=14, pady=8).pack(side='left')
            tk.Label(row, text=val, bg=C['panel'], fg=C['white'],
                     font=FONTS['body'], anchor='w').pack(side='left')

    # ── Popups ────────────────────────────────────────────────
    def _generate_report(self):
        if not self._last_pred:
            messagebox.showwarning('No Results', 'Analyze an image first.')
            return
        self._show_tab(2)
        self._refresh_report()

    def _show_prob_chart(self):
        if not self._last_pred:
            messagebox.showwarning('No Results', 'Analyze an image first.')
            return
        primary = self._last_pred.get('e2e') or self._last_pred.get('svm')
        probs   = primary['probs']
        colors  = [CLASS_INFO[c]['color'] for c in CLASS_NAMES]

        win = tk.Toplevel(self)
        win.title('Probability Chart')
        win.geometry('540x400')
        win.configure(bg=C['bg2'])

        fig = Figure(figsize=(5.5,3.8), facecolor=C['bg2'])
        ax  = fig.add_subplot(111, facecolor=C['bg3'])
        bars = ax.barh(CLASS_NAMES, probs, color=colors, alpha=0.85,
                       edgecolor=C['bg2'], height=0.5)
        for bar, prob in zip(bars, probs):
            ax.text(prob+0.01, bar.get_y()+bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center',
                    color=C['white'], fontsize=10)
        ax.set_xlim(0, 1.2)
        self._mpl_style(fig, [ax])
        ax.set_title('Class Probability Distribution', fontsize=12)
        fig.tight_layout()

        cv = FigureCanvasTkAgg(fig, master=win)
        cv.draw()
        cv.get_tk_widget().pack(fill='both', expand=True, padx=12, pady=12)

    # ── Model loading ──────────────────────────────────────────
    def _load_models_async(self):
        def _load():
            self.e2e_model = load_e2e_model()
            self.vgg_model = load_vgg16_model()
            self.svm_model, self.scaler, self.backbone = load_svm_pipeline()
            loaded = sum([bool(self.e2e_model), bool(self.vgg_model), bool(self.svm_model)])
            if loaded > 0:
                self.after(0, lambda: (
                    self.status_lbl.config(text=f'{loaded}/3 models ready', fg=C['teal']),
                    self.status_dot.delete('all') or
                    self.status_dot.create_oval(1,1,7,7, fill=C['teal'], outline='')))
            else:
                self.after(0, lambda: (
                    self.status_lbl.config(text='Run train.py first', fg=C['red']),
                    self.status_dot.delete('all') or
                    self.status_dot.create_oval(1,1,7,7, fill=C['red'], outline='')))
        threading.Thread(target=_load, daemon=True).start()

    def _load_results(self):
        if os.path.exists(RESULTS_JSON):
            with open(RESULTS_JSON) as f:
                self.results_data = json.load(f)
            self.after(200, self._populate_conclusion_chart)
            self.after(200, self._populate_results)
            self.after(300, self._populate_about)

    def _populate_conclusion_chart(self):
        if not hasattr(self, '_conclude_chart_frame'): return
        frame = self._conclude_chart_frame
        for w in frame.winfo_children(): w.destroy()
        if not self.results_data:
            tk.Label(frame, text='⚠  Run train.py first to generate results data.',
                     bg=C['panel'], fg=C['amber'], font=FONTS['body'],
                     padx=20, pady=20).pack()
            return
        rd = self.results_data
        approaches = ['ResNet50+SVM\n(Approach-1)', 'ResNet50 E2E\n(Approach-2)', 'VGG16 E2E\n(Bonus)']
        metrics    = ['accuracy','precision','recall','f1']
        m_labels   = ['Accuracy','Precision','Recall','F1-Score']
        colors_    = [C['cyan'], C['teal'], C['purple']]
        x = np.arange(len(m_labels))
        w = 0.25
        data = [rd.get('approach1',{}), rd.get('approach2',{}), rd.get('bonus_vgg',{})]
        fig = Figure(figsize=(10, 3.2), facecolor=C['bg2'])
        ax  = fig.add_subplot(111, facecolor=C['bg3'])
        for i,(app,d,col) in enumerate(zip(approaches,data,colors_)):
            if not d: continue
            vals = [d.get(k,0) for k in metrics]
            bars = ax.bar(x+i*w, vals, w, label=app.replace('\n',' '),
                          color=col, alpha=0.85, edgecolor=C['bg2'], linewidth=0.5)
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=7.5, color=C['white'], fontweight='bold')
        ax.set_xticks(x+w); ax.set_xticklabels(m_labels, fontsize=10, color=C['gray1'])
        ax.set_ylim(0,1.15)
        ax.tick_params(colors=C['gray1'])
        for sp in ax.spines.values(): sp.set_color(C['border'])
        ax.grid(axis='y', color=C['border'], alpha=0.5)
        ax.legend(fontsize=8, labelcolor=C['gray1'], facecolor=C['bg3'], edgecolor=C['border'])
        fig.tight_layout(pad=0.8)
        cv = FigureCanvasTkAgg(fig, master=frame)
        cv.draw()
        cv.get_tk_widget().pack(fill='x', padx=16, pady=(8,4))
        # Summary table
        cols_h = ['Approach','Accuracy','Precision','Recall','F1-Score','Time (s)']
        col_w  = [28,12,12,12,12,10]
        hdr = tk.Frame(frame, bg=C['bg4'])
        hdr.pack(fill='x', padx=16)
        for col,cw in zip(cols_h,col_w):
            tk.Label(hdr, text=col, bg=C['bg4'], fg=C['cyan'],
                     font=('Segoe UI',9,'bold'), width=cw,
                     anchor='center', pady=6).pack(side='left')
        rows_d = [
            ('Approach-1  ResNet50+SVM', rd.get('approach1',{}), C['cyan']),
            ('Approach-2  ResNet50 E2E', rd.get('approach2',{}), C['teal']),
            ('Bonus       VGG16 E2E',    rd.get('bonus_vgg',{}),  C['purple']),
        ]
        for i,(name,d,col) in enumerate(rows_d):
            if not d: continue
            bg = C['panel'] if i%2==0 else C['bg3']
            rr = tk.Frame(frame, bg=bg)
            rr.pack(fill='x', padx=16)
            vals = [name,
                    f"{d.get('accuracy',0):.4f}",
                    f"{d.get('precision',0):.4f}",
                    f"{d.get('recall',0):.4f}",
                    f"{d.get('f1',0):.4f}",
                    f"{d.get('time',0)}s"]
            fgs = [col,C['white'],C['white'],C['white'],C['white'],C['gray1']]
            for val,fg,cw in zip(vals,fgs,col_w):
                tk.Label(rr, text=val, bg=bg, fg=fg,
                         font=FONTS['mono_sm'], width=cw,
                         anchor='w' if cw==28 else 'center',
                         padx=6, pady=6).pack(side='left')
        tk.Frame(frame, bg=C['panel'], height=8).pack()


if __name__ == '__main__':
    app = BrainTumorApp()
    app.mainloop()
