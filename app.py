import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
from PIL import Image


# =========================
# App config
# =========================
st.set_page_config(page_title="Bird Species Detector", layout="centered")
st.title("🦜 Bird Species Detector (Audio)")

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SR = 32000
DEFAULT_DURATION = 5.0
DEFAULT_N_MELS = 128
DEFAULT_IMG_SIZE = 128

MODEL_PATH = BASE_DIR / "model_effecientnet_final.pth"
IDX2LABEL_PATH = BASE_DIR / "idx2label.json"
METADATA_PATH = BASE_DIR / "metadata_top20.csv"


# ========================
# model/ckpt
# =========================
def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _unwrap_checkpoint(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def infer_num_classes_from_state(state: dict) -> int | None:
    for key in ("classifier.1.weight", "classifier.weight", "fc.weight"):
        if key in state and hasattr(state[key], "shape"):
            return int(state[key].shape[0])
    return None


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# =========================
# labels & metadata
# =========================
def load_code_to_fullname() -> dict[str, str]:

    code_to_full: dict[str, str] = {}

    if not METADATA_PATH.exists():
        return code_to_full

    df = pd.read_csv(METADATA_PATH)
    if "ebird_code" not in df.columns:
        return code_to_full

    full_col = "full_name" if "full_name" in df.columns else None

    for _, row in df.iterrows():
        code = str(row["ebird_code"]).strip()
        if not code:
            continue

        full = ""
        if full_col and pd.notna(row[full_col]):
            full = str(row[full_col]).strip()

        if full:
            full = full.replace("_", " ").strip()

        code_to_full[code] = full

    return code_to_full


def load_idx2code_from_json() -> list[str]:

    if not IDX2LABEL_PATH.exists():
        raise FileNotFoundError(f"idx2label.json tidak ketemu: {IDX2LABEL_PATH}")

    with open(IDX2LABEL_PATH, "r", encoding="utf-8") as f:
        idx2label = json.load(f)

    items = sorted(((int(k), str(v).strip()) for k, v in idx2label.items()), key=lambda x: x[0])
    idx2code = [v for _, v in items]

    if not idx2code or any(x == "" for x in idx2code):
        raise RuntimeError("idx2label.json ada yang kosong / format tidak valid.")

    return idx2code


# ========================
# Load bundle (cached)
# =========================
@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model .pth tidak ketemu: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = _unwrap_checkpoint(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint model tidak berbentuk state_dict yang valid.")
    state = _strip_module_prefix(state)

    num_classes = infer_num_classes_from_state(state)
    if num_classes is None:
        raise RuntimeError("Gagal mendeteksi jumlah kelas dari checkpoint (state_dict).")

    idx2code = load_idx2code_from_json()

    # pastikan jumlah kelas model = jumlah label
    if len(idx2code) != num_classes:
        raise RuntimeError(
            f"Jumlah kelas tidak match:\n"
            f"- dari model checkpoint: {num_classes}\n"
            f"- dari idx2label.json: {len(idx2code)}\n\n"
            f"Ini harus sama biar mapping label tidak ngaco."
        )

    code_to_full = load_code_to_fullname()

    model = build_model(num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, idx2code, code_to_full


# =========================
# Audio -> spectrogram image
# =========================
def audio_to_spectrogram_image(
    audio_bytes: bytes,
    file_suffix: str,
    target_sr: int,
    duration_sec: float,
    n_mels: int,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, _ = librosa.load(tmp_path, sr=target_sr, mono=True)
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
    img = img.astype(np.uint8)
    img = np.flip(img, axis=0)

    return Image.fromarray(img).convert("RGB")


IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_topk(model, x_img: Image.Image, topk: int):
    x = IMG_TRANSFORM(x_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(int(topk), probs.numel())
    vals, idxs = torch.topk(probs, k=k)
    return idxs.tolist(), vals.tolist()


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
    model, idx2code, code_to_full = load_model_bundle()
    st.caption(f"Loaded: {MODEL_PATH.name} | classes: {len(idx2code)}")
except Exception as e:
    st.error(str(e))
    st.stop()

audio_file = st.file_uploader("Upload audio (wav/mp3/flac/ogg/m4a)", type=["wav", "mp3", "flac", "ogg", "m4a"])

if audio_file:
    st.audio(audio_file)
    audio_bytes = audio_file.getvalue()

    with st.spinner("Membuat spectrogram & prediksi..."):
        suffix = Path(audio_file.name).suffix.lower()
        if suffix not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            suffix = ".wav"

        try:
            pil_img = audio_to_spectrogram_image(
                audio_bytes=audio_bytes,
                file_suffix=suffix,
                target_sr=int(sr),
                duration_sec=float(duration),
                n_mels=int(n_mels),
            )
        except Exception as e:
            st.error(f"Gagal membaca audio / membuat spectrogram.\n\nDetail: {e}")
            st.stop()

        if show_spec:
            st.image(pil_img, caption="Spectrogram (Mel)", use_container_width=True)

        idxs, probs = predict_topk(model, pil_img, topk=topk)

    st.subheader("Hasil Prediksi")
    for rank, (i, p) in enumerate(zip(idxs, probs), start=1):
        code = idx2code[i]
        full = code_to_full.get(code, "").strip()
        if full:
            display_name = f"{full} ({code})"
        else:
            display_name = code
        st.write(f"{rank}. **{display_name}** — {p*100:.2f}%")
