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
from PIL import Image
import pandas as pd

# ========================
# App config
# ========================
st.set_page_config(page_title="Bird Species Detector", layout="centered")
st.title("🦜 Bird Species Detector (Audio)")

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SR = 32000
DEFAULT_DURATION = 5.0
DEFAULT_N_MELS = 128
DEFAULT_IMG_SIZE = 128

MODEL_CANDIDATES = [
    "models/bird_model.pth",
    "model_effecientnet_final.pth",       
    "models/model_effecientnet_final.pth",
    "models/model_efficientnet_final.pth",
]

LABEL_JSON_CANDIDATES = [
    "models/idx2label.json",
    "idx2label.json",
]

METADATA_CANDIDATES = [
    "metadata_top20.csv",
    "models/metadata_top20.csv",
]


# =======================
# Helpers
# ========================
def resolve_path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (BASE_DIR / p)


def find_first_existing(paths) -> str | None:
    for p in paths:
        rp = resolve_path(p)
        if rp.exists():
            return str(rp)
    return None


def load_labels_and_names():
    """
    FIX UTAMA:
    - Saat training, class diurut alfabetis berdasarkan FULL NAME (nama folder/full_name),
      bukan berdasarkan ebird_code.
    - Jadi di sini labels harus dibuat dengan order yang sama:
        sort metadata_top20.csv berdasarkan kolom full_name,
        lalu labels = list ebird_code sesuai urutan tersebut.
    - Output ditampilkan pakai FULL NAME (bukan ebird_code).
    """
    meta_path = find_first_existing(METADATA_CANDIDATES)
    if meta_path:
        df = pd.read_csv(meta_path)

        code_col = next((c for c in ["ebird_code", "code", "label"] if c in df.columns), None)
        name_col = next((c for c in ["full_name", "fullName", "species_name", "species", "name"] if c in df.columns), None)

        if not code_col:
            raise ValueError(
                f"metadata CSV ditemukan ({meta_path}) tapi kolom ebird_code tidak ada. "
                f"Kolom tersedia: {list(df.columns)}"
            )
        if not name_col:
            name_col = code_col

        df = df[[code_col, name_col]].dropna()
        df[code_col] = df[code_col].astype(str).str.strip()
        df[name_col] = df[name_col].astype(str).str.strip()

        df = df.drop_duplicates(subset=[code_col], keep="first")

        df = df.sort_values(by=name_col, key=lambda s: s.str.lower(), kind="mergesort")

        labels = df[code_col].tolist()
        code_to_full = dict(zip(df[code_col], df[name_col]))

        return labels, code_to_full, f"metadata_top20.csv (sorted by {name_col})"

    # fallback: idx2label.json (kalau metadata tidak ada)
    label_path = find_first_existing(LABEL_JSON_CANDIDATES)
    if label_path:
        with open(label_path, "r", encoding="utf-8") as f:
            idx2label = json.load(f)
        items = sorted(((int(k), v) for k, v in idx2label.items()), key=lambda x: x[0])
        labels = [str(v) for _, v in items]
        code_to_full = {c: c for c in labels}
        return labels, code_to_full, "idx2label.json"

    labels = [f"class_{i}" for i in range(20)]
    code_to_full = {c: c for c in labels}
    return labels, code_to_full, "fallback class_0..class_19"


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@st.cache_resource
def load_model_bundle():
    labels, code_to_full, label_source = load_labels_and_names()
    num_classes = len(labels)

    model_path = find_first_existing(MODEL_CANDIDATES)
    if not model_path:
        raise FileNotFoundError(
            "File model .pth tidak ditemukan. Taruh file di salah satu path ini:\n"
            + "\n".join([f"- {p}" for p in MODEL_CANDIDATES])
        )

    model = build_model(num_classes)

    try:
        state = torch.load(model_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(
            f"Gagal membaca file model: {model_path}\n"
            f"Detail: {e}\n\n"
            "Kalau muncul 'PytorchStreamReader failed reading zip archive', biasanya file .pth corrupt / salah file."
        )

    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        raise RuntimeError(
            f"Gagal load state_dict dari {model_path}.\n"
            f"Biasanya karena num_classes / arsitektur tidak sama seperti saat training.\n\n"
            f"Detail error: {e}"
        )

    model.eval()
    return model, labels, code_to_full, model_path, label_source


def audio_to_spectrogram_image(
    audio_bytes: bytes,
    file_suffix: str = ".wav",
    target_sr: int = DEFAULT_SR,
    duration_sec: float = DEFAULT_DURATION,
    n_mels: int = DEFAULT_N_MELS,
) -> Image.Image:
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, _sr = librosa.load(tmp_path, sr=target_sr, mono=True)
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

    melspec = librosa.feature.melspectrogram(y=y, sr=target_sr, n_mels=int(n_mels), fmax=14000)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    img = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-9) * 255.0
    img = np.flip(img.astype(np.uint8), axis=0) 

    return Image.fromarray(img).convert("RGB")


IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_topk(model: nn.Module, labels: list[str], pil_img: Image.Image, topk: int = 5):
    x = IMG_TRANSFORM(pil_img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)

    k = min(int(topk), probs.numel())
    vals, idxs = torch.topk(probs, k=k)
    return [(labels[i], float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]


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
    model, labels, code_to_full, model_path, label_source = load_model_bundle()
    st.caption(f"Model: `{Path(model_path).name}`")
    st.caption(f"Label source: {label_source}")
except Exception as e:
    st.error(str(e))
    st.stop()

audio_file = st.file_uploader(
    "Upload audio (wav/mp3/flac/ogg/m4a)",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
)

if audio_file:
    st.audio(audio_file)
    audio_bytes = audio_file.getvalue()

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

        results = predict_topk(model, labels, pil_img, topk=topk)

    st.subheader("Hasil Prediksi")
    for rank, (code, p) in enumerate(results, start=1):
        full_name = code_to_full.get(code, code)
        st.write(f"{rank}. **{full_name}** — {p*100:.2f}%")
