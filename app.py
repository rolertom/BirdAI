import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

import librosa
import cv2
from PIL import Image
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

def resolve_path(p):
    p = Path(p)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

def find_first_existing(paths):
    for p in paths:
        rp = resolve_path(p)
        if rp.exists():
            return str(rp)
    return None

# Config
# =========================
st.set_page_config(page_title="Bird Species Detector", layout="centered")
st.title("🦜 Bird Species Detector (Audio)")

DEFAULT_SR = 32000
DEFAULT_DURATION = 5.0
DEFAULT_N_MELS = 128
DEFAULT_IMG_SIZE = 128  

MODEL_CANDIDATES = [
    "models/bird_model.pth",
    "model_effecientnet_final.pth",  
    "model_efficientnet_final.pth",
    "models/model_effecientnet_final.pth",
    "models/model_efficientnet_final.pth",
]

LABEL_CANDIDATES = [
    "models/idx2label.json",
    "idx2label.json",
]

METADATA_CANDIDATES = [
    "metadata_top20.csv",
    "models/metadata_top20.csv",
]


# ========================
# Helpers
# =========================
def load_labels():
    """
    Prioritas:
    1) idx2label.json (idx -> label)
    2) metadata_top20.csv (ambil kolom yang paling masuk akal)
    3) folder dataset_for_training (nama subfolder)
    4) fallback: class_0..class_n
    """
    # 1) idx2label.json
    label_path = find_first_existing(LABEL_CANDIDATES)
    if label_path:
        with open(label_path, "r", encoding="utf-8") as f:
            idx2label = json.load(f)

        items = sorted(((int(k), v) for k, v in idx2label.items()), key=lambda x: x[0])
        labels = [str(v) for _, v in items]
        return labels, f"Loaded labels from {label_path}"

    # 2) metadata csv
    meta_path = find_first_existing(METADATA_CANDIDATES)
    if meta_path:
        df = pd.read_csv(meta_path)

        col = "ebird_code" if "ebird_code" in df.columns else df.columns[0]
        labels = sorted(df[col].dropna().astype(str).unique().tolist())

        return labels, f"Loaded labels from {meta_path} (sorted column: {col})"

    # 3) dataset_for_training folder
    ds_dir = Path("dataset_for_training")
    if ds_dir.exists() and ds_dir.is_dir():
        labels = sorted([p.name for p in ds_dir.iterdir() if p.is_dir()])
        if labels:
            return labels, "Loaded labels from dataset_for_training/ subfolders"

    # 4) fallback
    labels = [f"class_{i}" for i in range(20)]
    return labels, "Labels not found → using fallback class_0..class_19"


def build_model(num_classes: int):
    """
    EfficientNet-B0 dengan classifier diganti sesuai jumlah kelas.
    """
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model_and_labels():
    labels, label_msg = load_labels()
    num_classes = len(labels)

    model_path = find_first_existing(MODEL_CANDIDATES)
    if not model_path:
        raise FileNotFoundError(
            "File model .pth tidak ditemukan. Taruh file di salah satu path ini:\n"
            + "\n".join([f"- {p}" for p in MODEL_CANDIDATES])
        )

    model = build_model(num_classes)
    state = torch.load(model_path, map_location="cpu")

    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
    
        raise RuntimeError(
            f"Gagal load state_dict dari {model_path}.\n"
            f"Biasanya karena arsitektur/num_classes tidak sama dengan saat training.\n\n"
            f"Detail error: {e}"
        )

    model.eval()
    return model, labels, model_path, label_msg         


def audio_to_spectrogram_image(
    audio_bytes: bytes,
    file_suffix: str = ".wav",
    target_sr: int = DEFAULT_SR,
    duration_sec: float = DEFAULT_DURATION,
    n_mels: int = DEFAULT_N_MELS,
    img_size: int = DEFAULT_IMG_SIZE,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    y, _ = librosa.effects.trim(y, top_db=20)

    target_len = int(target_sr * duration_sec)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    melspec = librosa.feature.melspectrogram(
        y=y, sr=target_sr, n_mels=int(n_mels), fmax=14000
    )
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    img = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-9) * 255.0
    img = img.astype(np.uint8)
    img = np.flip(img, axis=0)

    pil_img = Image.fromarray(img).convert("RGB")
    return pil_img


# torchvision transform (ImageNet normalization)
IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def predict(model, labels, pil_img, topk=5):
    x = IMG_TRANSFORM(pil_img).unsqueeze(0)  
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k=k)
    results = [(labels[i], float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]
    return results


# =========================
# UI
# =========================
with st.expander("Settings", expanded=True):
    sr = st.number_input("Sample rate", min_value=8000, max_value=96000, value=DEFAULT_SR, step=1000)
    duration = st.number_input("Durasi (detik)", min_value=1.0, max_value=30.0, value=float(DEFAULT_DURATION), step=0.5)
    n_mels = st.number_input("n_mels", min_value=32, max_value=256, value=DEFAULT_N_MELS, step=16)
    topk = st.slider("Top-K", 1, 10, 5)
    show_spec = st.checkbox("Tampilkan spectrogram", value=True)

st.divider()

try:
    model, labels, model_path, label_msg = load_model_and_labels()
    st.caption(f"Model: `{model_path}`")
    st.caption(f"Labels: {label_msg}")

    with st.expander("DEBUG: Labels order", expanded=False):
        st.write("First 10 labels:", labels[:10])
        st.write("Index houspa:", labels.index("houspa") if "houspa" in labels else "not found")
        st.write("Index daejun:", labels.index("daejun") if "daejun" in labels else "not found")
        st.write("Total classes:", len(labels))
        
    with st.expander("DEBUG: Paths", expanded=False):
        st.write("BASE_DIR:", str(BASE_DIR))
        st.write("label_json found:", find_first_existing(LABEL_CANDIDATES))
        st.write("meta_csv found:", find_first_existing(METADATA_CANDIDATES))
    
except Exception as e:
    st.error(str(e))
    st.stop()

audio_file = st.file_uploader(
    "Upload audio (wav/mp3/flac/ogg/m4a)",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
)

if audio_file:
    st.audio(audio_file)

    audio_bytes = audio_file.read()

    with st.spinner("Membuat spectrogram & prediksi..."):
        try:
            suffix = Path(audio_file.name).suffix.lower()
            if suffix not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                suffix = ".wav"

            pil_img = audio_to_spectrogram_image(
                audio_bytes,
                file_suffix=suffix,
                target_sr=int(sr),
                duration_sec=float(duration),
                n_mels=int(n_mels),
                img_size=DEFAULT_IMG_SIZE,
            )
        except Exception as e:
            st.error(
                "Gagal membaca audio atau membuat spectrogram.\n"
                "Coba upload format WAV (paling aman) atau pastikan file tidak corrupt.\n\n"
                f"Detail: {e}"
            )
            st.stop()

        if show_spec:
            st.image(pil_img, caption="Spectrogram (Mel)", use_container_width=True)

        results = predict(model, labels, pil_img, topk=topk)

    st.subheader("Hasil Prediksi")
    for rank, (name, p) in enumerate(results, start=1):
        st.write(f"{rank}. **{name}** — {p*100:.2f}%")
