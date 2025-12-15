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

MODEL_CANDIDATES = [
    "models/bird_model.pth",
    "model_effecientnet_final.pth",
    "model_efficientnet_final.pth",
    "model_effecientnet_final.pth",
    "model_efficientnet_final.pth",
    "models/model_effecientnet_final.pth",
    "models/model_efficientnet_final.pth",
]

IDX2LABEL_CANDIDATES = [
    "models/idx2label.json",
    "idx2label.json",
]

METADATA_CANDIDATES = [
    "metadata_top20.csv",
    "models/metadata_top20.csv",
]


# =========================
# Helpers: paths & IO
# =========================
def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (BASE_DIR / pp)


def find_first_existing(paths: list[str]) -> str | None:
    for p in paths:
        rp = resolve_path(p)
        if rp.exists():
            return str(rp)
    return None


def _strip_module_prefix(state: dict) -> dict:
    # handle DDP-trained checkpoints
    if not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _unwrap_checkpoint(obj):
    # try common wrappers
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def infer_num_classes_from_state(state: dict) -> int | None:
    # EfficientNet-B0 classifier is usually classifier.1
    for key in ("classifier.1.weight", "classifier.weight", "fc.weight"):
        if key in state and hasattr(state[key], "shape"):
            return int(state[key].shape[0])
    # fallback: find any 2D weight with out_features == number of classes
    for k, v in state.items():
        if k.endswith("classifier.1.weight") and hasattr(v, "shape"):
            return int(v.shape[0])
    return None


def load_metadata_maps():
    """Return:
    - code_to_full: ebird_code -> pretty full_name
    """
    meta_path = find_first_existing(METADATA_CANDIDATES)
    code_to_full: dict[str, str] = {}
    if not meta_path:
        return code_to_full

    df = pd.read_csv(meta_path)
    if "ebird_code" not in df.columns:
        return code_to_full

    full_col = "full_name" if "full_name" in df.columns else None
    for _, row in df.iterrows():
        code = str(row["ebird_code"])
        full = str(row[full_col]) if full_col and pd.notna(row[full_col]) else ""
        full = full.strip()
        if full:
            # make it nicer for display
            full = full.replace("_", " ").strip()
        code_to_full[code] = full

    return code_to_full


def load_idx2code(num_classes: int) -> list[str]:
    """
    idx -> ebird_code (HARUS urut sama dengan saat training).
    Prioritas:
      1) idx2label.json (paling aman)
      2) fallback dari metadata_top20.csv: urut berdasarkan full_name (kalau ada), tapi tetap pastikan panjang = num_classes
    """
    json_path = find_first_existing(IDX2LABEL_CANDIDATES)
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # keys bisa string angka
        pairs = sorted(((int(k), str(v)) for k, v in raw.items()), key=lambda x: x[0])
        idx2 = [v for _, v in pairs]

        # pastikan panjang sama dengan num_classes (checkpoint)
        if len(idx2) < num_classes:
            idx2 += [f"class_{i}" for i in range(len(idx2), num_classes)]
        elif len(idx2) > num_classes:
            idx2 = idx2[:num_classes]
        return idx2

    # --- fallback: metadata ---
    meta_path = find_first_existing(METADATA_CANDIDATES)
    if meta_path:
        df = pd.read_csv(meta_path)
        if "ebird_code" in df.columns:
            full_col = "full_name" if "full_name" in df.columns else None
            rows = []
            for _, r in df.iterrows():
                code = str(r["ebird_code"])
                full = ""
                if full_col and pd.notna(r[full_col]):
                    full = str(r[full_col]).strip()
                sort_key = full if full else code
                rows.append((sort_key, code))
            rows.sort(key=lambda x: x[0])
            idx2 = [code for _, code in rows]

            if len(idx2) < num_classes:
                idx2 += [f"class_{i}" for i in range(len(idx2), num_classes)]
            elif len(idx2) > num_classes:
                idx2 = idx2[:num_classes]
            return idx2

    return [f"class_{i}" for i in range(num_classes)]


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@st.cache_resource
def load_model_bundle():
    model_path = find_first_existing(MODEL_CANDIDATES)
    if not model_path:
        raise FileNotFoundError("Model file tidak ditemukan. Pastikan .pth ada di repo (atau folder models/).")

    ckpt = torch.load(model_path, map_location="cpu")
    state = _unwrap_checkpoint(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint model tidak berbentuk state_dict yang valid.")

    state = _strip_module_prefix(state)

    num_classes = infer_num_classes_from_state(state)
    if num_classes is None:
        raise RuntimeError("Gagal mendeteksi jumlah kelas dari checkpoint (state_dict).")

    idx2code = load_idx2code(num_classes)
    code_to_full = load_metadata_maps()

    model = build_model(num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, idx2code, code_to_full, model_path


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
    model, idx2code, code_to_full, model_path = load_model_bundle()
    st.caption(f"Model: `{Path(model_path).name}` | classes: {len(idx2code)}")
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
        code = idx2code[i] if 0 <= i < len(idx2code) else f"class_{i}"
        full = code_to_full.get(code, "")
        display_name = full if full else code
        # kalau full_name ada, tampilkan (ebird_code) biar jelas
        if full:
            display_name = f"{full} ({code})"
        st.write(f"{rank}. **{display_name}** — {p*100:.2f}%")
