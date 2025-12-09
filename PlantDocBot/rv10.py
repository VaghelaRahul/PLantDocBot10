# rv10.py  — PlantDocBot (global deployable version)
# - Downloads big models from Google Drive (once) into /models
# - Uses TF BERT (TFAutoModelForSequenceClassification)
# - Uses sentence-transformers to refine to professional disease names

import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import tensorflow as tf

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import gdown

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Plant Health Assistant", page_icon="🌿", layout="wide")
st.title("🌿 Plant Health Assistant")
st.write("AI-powered plant disease prediction using **Image + Text** with treatment recommendations.")

# ---------------------------------------------------------
# PATHS (RELATIVE TO THIS FILE) + GOOGLE DRIVE IDS
# ---------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()

# Small files that live directly in the repo
LABEL_ENCODER_FILE = APP_DIR / "label_encoder_merged.joblib"
IMAGE_LABEL_ENCODER_FILE = APP_DIR / "image_label_encoder.joblib"
DISEASE_TO_MERGED_CSV = APP_DIR / "disease_to_merged.csv"
DISPLAY_MAP_FILE = APP_DIR / "display_name_by_merged.csv"  # optional but useful

# Big files we download into /models (NOT committed to GitHub)
MODELS_BASE = APP_DIR / "models"
MODELS_BASE.mkdir(exist_ok=True)

TEXT_MODEL_DIR = MODELS_BASE / "Final_Text_Disease_Model"
IMAGE_MODEL_FILE = MODELS_BASE / "plant_disease_model_fixed.keras"
REMEDY_CSV_PATH = MODELS_BASE / "plant_disease_dataset_clean_lemmatized03.csv"

# ---- Your Google Drive IDs (from the links you gave) ----
GDRIVE_IMAGE_ID = "1oFVBpl-Q81kryI6h-TC94Wt7rcMd8AXn"
GDRIVE_TEXT_FOLDER_ID = "1L_yTMpvW5xFSKUFHQN-_mz5t2K9fjxnG"
GDRIVE_CSV_ID = "1hVEoCo-EecTqFdVWTP7c-nqSGl1iV3e6"

# ---------------------------------------------------------
# HELPERS TO DOWNLOAD FROM GOOGLE DRIVE (RUN ONCE)
# ---------------------------------------------------------
@st.cache_resource
def ensure_image_model_path() -> str:
    """Download plant_disease_model_fixed.keras from Drive if missing."""
    if not IMAGE_MODEL_FILE.exists() or IMAGE_MODEL_FILE.stat().st_size == 0:
        st.write("📥 Downloading image CNN model from Google Drive...")
        gdown.download(
            id=GDRIVE_IMAGE_ID,
            output=str(IMAGE_MODEL_FILE),
            quiet=False,
        )
    return str(IMAGE_MODEL_FILE)


@st.cache_resource
def ensure_text_model_dir() -> str:
    """Download Final_Text_Disease_Model folder from Drive if missing."""
    config_path = TEXT_MODEL_DIR / "config.json"
    if not config_path.exists():
        st.write("📥 Downloading text BERT model from Google Drive...")
        TEXT_MODEL_DIR.mkdir(exist_ok=True)
        gdown.download_folder(
            id=GDRIVE_TEXT_FOLDER_ID,
            output=str(TEXT_MODEL_DIR),
            quiet=False,
            use_cookies=False,
        )
    return str(TEXT_MODEL_DIR)


@st.cache_resource
def ensure_remedy_csv_path() -> str:
    """Download plant_disease_dataset_clean_lemmatized03.csv from Drive if missing."""
    if not REMEDY_CSV_PATH.exists() or REMEDY_CSV_PATH.stat().st_size == 0:
        st.write("📥 Downloading remedy CSV from Google Drive...")
        gdown.download(
            id=GDRIVE_CSV_ID,
            output=str(REMEDY_CSV_PATH),
            quiet=False,
        )
    return str(REMEDY_CSV_PATH)

# ---------------------------------------------------------
# COARSE NORMALIZATION (group-level, for fallback/remedies)
# ---------------------------------------------------------
def normalize_disease(d: str) -> str:
    d = "" if d is None else str(d).lower()
    if "healthy" in d:
        return "healthy"
    if "bacterial" in d or "spot" in d:
        return "bacterial"
    if "early blight" in d or "late blight" in d or "blight" in d:
        return "blight"
    if "powdery" in d or "mildew" in d or "mold" in d:
        return "mildew"
    if "virus" in d or "mosaic" in d:
        return "viral"
    if "rot" in d or "canker" in d:
        return "rot_mold"
    if "scab" in d:
        return "scab"
    if "curl" in d:
        return "curl"
    return "other"

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_nlp_model():
    text_model_path = ensure_text_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    # IMPORTANT: TF version, not PyTorch
    model = TFAutoModelForSequenceClassification.from_pretrained(
        text_model_path,
        from_pt=False,
    )
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    return tokenizer, model, label_encoder


@st.cache_resource
def load_image_model():
    image_model_path = ensure_image_model_path()
    return tf.keras.models.load_model(
        image_model_path,
        compile=False,
        safe_mode=False,
    )


@st.cache_resource
def load_image_label_encoder():
    enc = joblib.load(IMAGE_LABEL_ENCODER_FILE)
    if isinstance(enc, dict):
        # dict: class_name -> idx  => we want idx -> class_name
        return {v: k for k, v in enc.items()}
    # sklearn LabelEncoder
    return {i: cls for i, cls in enumerate(enc.classes_)}

# ---------------------------------------------------------
# REMEDY MAPS (coarse + exact merged_label)
# ---------------------------------------------------------
@st.cache_resource
def load_remedy_maps():
    csv_path = ensure_remedy_csv_path()
    df = pd.read_csv(csv_path)

    # ensure merged_label exists
    if "merged_label" not in df.columns:
        if DISEASE_TO_MERGED_CSV.exists():
            mapdf = pd.read_csv(DISEASE_TO_MERGED_CSV)
            disease_to_merged = dict(
                zip(
                    mapdf["Disease"].astype(str).str.lower(),
                    mapdf["merged_label"].astype(str).str.lower(),
                )
            )
            df["merged_label"] = (
                df.get("Disease", "")
                .astype(str)
                .str.lower()
                .map(disease_to_merged)
            )
            df["merged_label"] = df["merged_label"].fillna(
                df.get("Disease", "").astype(str).str.lower()
            )
        else:
            if "Disease" in df.columns:
                df["merged_label"] = df["Disease"].astype(str).str.lower()
            else:
                df["merged_label"] = "other"

    df["merged_label"] = (
        df["merged_label"]
        .astype(str)
        .str.strip()
        .replace("", "other")
        .fillna("other")
        .str.lower()
    )
    if "Remedy" not in df.columns:
        df["Remedy"] = pd.NA

    df["merged_norm"] = df["merged_label"].apply(normalize_disease)

    def top_rem(series):
        s = series.dropna().astype(str).str.strip()
        return s.value_counts().index[0] if len(s) else None

    # merged_label -> top remedy
    remedy_by_merged = {}
    for name, sub in df.groupby("merged_label")["Remedy"]:
        top = top_rem(sub)
        if top:
            remedy_by_merged[name] = top

    # normalized group -> top remedy
    remedy_by_norm = {}
    for name, sub in df.groupby("merged_norm")["Remedy"]:
        top = top_rem(sub)
        if top:
            remedy_by_norm[name] = top

    remedy_by_norm.setdefault("healthy", "No treatment needed.")
    remedy_by_norm.setdefault("other", "No remedy available.")

    return remedy_by_merged, remedy_by_norm

# ---------------------------------------------------------
# OPTIONAL DISPLAY MAP (merged_label -> pretty name)
# ---------------------------------------------------------
@st.cache_resource
def load_display_map():
    if DISPLAY_MAP_FILE.exists():
        try:
            dm = pd.read_csv(DISPLAY_MAP_FILE)
            return dict(
                zip(
                    dm["merged_label"].astype(str).str.lower(),
                    dm["suggested_display_name"].astype(str),
                )
            )
        except Exception:
            return {}
    return {}

# ---------------------------------------------------------
# DISEASE-LEVEL REFINER USING SENTENCE-TRANSFORMERS
# ---------------------------------------------------------
@st.cache_resource
def load_disease_refiner():
    """
    Build an index of diseases (fine-grained) using your CSV.
    Each disease_key gets:
      - display_name (pretty disease string from CSV)
      - merged_label (coarse group)
      - rep_text (representative symptom text)
      - top_remedy (disease-specific remedy, if available)
      - embedding (for semantic similarity)
    """
    csv_path = ensure_remedy_csv_path()
    df = pd.read_csv(csv_path)

    if "Disease" not in df.columns:
        return None

    # ensure merged_label exists, same logic as load_remedy_maps
    if "merged_label" not in df.columns:
        if DISEASE_TO_MERGED_CSV.exists():
            mapdf = pd.read_csv(DISEASE_TO_MERGED_CSV)
            disease_to_merged = dict(
                zip(
                    mapdf["Disease"].astype(str).str.lower(),
                    mapdf["merged_label"].astype(str).str.lower(),
                )
            )
            df["merged_label"] = (
                df["Disease"].astype(str).str.lower().map(disease_to_merged)
            )
            df["merged_label"] = df["merged_label"].fillna(
                df["Disease"].astype(str).str.lower()
            )
        else:
            df["merged_label"] = df["Disease"].astype(str).str.lower()

    df["merged_label"] = (
        df["merged_label"]
        .astype(str)
        .str.strip()
        .replace("", "other")
        .fillna("other")
        .str.lower()
    )

    df["Disease_raw"] = df["Disease"].astype(str)
    df["disease_key"] = df["Disease_raw"].str.strip().str.lower()

    # Build one representative record per disease_key
    records = []
    for disease_key, sub in df.groupby("disease_key"):
        # Coarse group for this disease
        if not sub["merged_label"].isna().all():
            merged_group = sub["merged_label"].value_counts().index[0]
        else:
            merged_group = "other"

        # Pretty name: most common Disease string
        display_name = sub["Disease_raw"].value_counts().index[0]

        # Representative symptom text
        text_pieces = []
        if "Symptoms" in sub.columns:
            text_pieces = sub["Symptoms"].dropna().astype(str).unique().tolist()
        elif "clean_text" in sub.columns:
            text_pieces = sub["clean_text"].dropna().astype(str).unique().tolist()

        if not text_pieces:
            text_pieces = [display_name]

        rep_text = " ".join(text_pieces[:3])

        # Disease-specific top remedy (if exists)
        if "Remedy" in sub.columns:
            remedies = sub["Remedy"].dropna().astype(str).str.strip()
            top_rem = remedies.value_counts().index[0] if len(remedies) else None
        else:
            top_rem = None

        records.append(
            {
                "disease_key": disease_key,
                "display_name": display_name,
                "merged_label": merged_group,
                "rep_text": rep_text,
                "top_remedy": top_rem,
            }
        )

    if not records:
        return None

    index_df = pd.DataFrame(records)

    # SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(
        index_df["rep_text"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    return {
        "model": model,
        "index_df": index_df,
        "embs": embs,
    }


def refine_disease_with_embeddings(symptoms_text: str, coarse_merged: str, min_sim: float = 0.35):
    """
    Use sentence-transformer to refine from coarse merged_label to a specific Disease.
    Returns (display_name, disease_key, disease_remedy, similarity) or (None, None, None, None).
    """
    refiner = load_disease_refiner()
    if refiner is None:
        return None, None, None, None

    model = refiner["model"]
    index_df = refiner["index_df"]
    embs = refiner["embs"]

    coarse_key = str(coarse_merged).strip().lower()
    # restrict candidate diseases to same coarse group
    candidate_indices = index_df.index[index_df["merged_label"] == coarse_key].tolist()

    if not candidate_indices:
        # try fallback by normalized group
        norm_key = normalize_disease(coarse_key)
        candidate_indices = index_df.index[
            index_df["merged_label"].apply(normalize_disease) == norm_key
        ].tolist()

    if not candidate_indices:
        return None, None, None, None

    # encode the user text
    query_emb = model.encode(
        symptoms_text,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    # pick candidate embeddings
    candidate_embs = embs[candidate_indices]

    sims = util.cos_sim(query_emb, candidate_embs)[0]  # 1D tensor
    best_pos = int(torch.argmax(sims).item())
    best_sim = float(sims[best_pos])

    if best_sim < min_sim:
        return None, None, None, None

    best_global_idx = candidate_indices[best_pos]
    row = index_df.loc[best_global_idx]

    return (
        row["display_name"],
        row["disease_key"],
        row["top_remedy"],
        best_sim,
    )

# ---------------------------------------------------------
# LOAD ALL RESOURCES (calls cached functions)
# ---------------------------------------------------------
tokenizer, nlp_model, label_encoder = load_nlp_model()
cnn_model = load_image_model()
img_idx_to_class = load_image_label_encoder()
remedy_by_merged, remedy_by_norm = load_remedy_maps()
display_name_by_merged = load_display_map()
# disease_refiner is loaded lazily via load_disease_refiner()

# ---------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------
def split_plant_and_disease(class_name):
    if not isinstance(class_name, str):
        class_name = str(class_name)
    c = class_name.replace("___", "_").replace("__", "_")
    parts = c.split("_")
    plant = parts[0].title() if parts else "Unknown"
    disease = " ".join(parts[1:]).replace("_", " ").title() if len(parts) > 1 else "Unknown"
    return plant, disease

# ---------------------------------------------------------
# PREDICT TEXT (WITH REFINER)
# ---------------------------------------------------------
def predict_text(symptoms: str):
    # 1) BERT coarse classification
    enc = tokenizer(
        symptoms,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="tf",
    )
    outputs = nlp_model(enc)
    pred = int(tf.argmax(outputs.logits, axis=1).numpy()[0])
    merged_raw = label_encoder.inverse_transform([pred])[0]
    merged_key = str(merged_raw).lower().strip()

    # base display (if refinement fails)
    base_display = display_name_by_merged.get(
        merged_key, merged_key.replace("_", " ").title()
    )

    # 2) Refine to specific Disease
    refined_name, disease_key, disease_remedy, sim = refine_disease_with_embeddings(
        symptoms, merged_key
    )

    if refined_name is not None:
        final_display = refined_name
        final_label_key = disease_key  # more specific
        if disease_remedy:
            remedy = disease_remedy
        else:
            remedy = (
                remedy_by_merged.get(merged_key)
                or remedy_by_norm.get(normalize_disease(merged_key), "No remedy available.")
            )
        refined_from = merged_key
        similarity = sim
    else:
        # fallback: use coarse label only
        final_display = base_display
        final_label_key = merged_key
        remedy = (
            remedy_by_merged.get(merged_key)
            or remedy_by_norm.get(normalize_disease(merged_key), "No remedy available.")
        )
        refined_from = None
        similarity = None

    confidence = float(tf.nn.softmax(outputs.logits)[0][pred])

    return {
        "display_name": final_display,
        "final_key": final_label_key,
        "coarse_label": merged_key,
        "refined_from": refined_from,
        "confidence": confidence,
        "similarity": similarity,
        "remedy": remedy,
    }

# ---------------------------------------------------------
# PREDICT IMAGE (OPTIONALLY USE REFINER)
# ---------------------------------------------------------
def predict_image(image_file):
    img = Image.open(image_file).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)

    preds = cnn_model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    class_name = img_idx_to_class.get(idx, str(idx))
    plant, disease_full = split_plant_and_disease(class_name)

    # Try to map to exact Disease in CSV using refiner index
    refiner = load_disease_refiner()
    if refiner is not None:
        index_df = refiner["index_df"]
        disease_key = disease_full.lower().strip()
        mask = index_df["disease_key"] == disease_key
        if mask.any():
            row = index_df[mask].iloc[0]
            display_name = row["display_name"]
            disease_remedy = row["top_remedy"]
            merged_label = row["merged_label"]
            remedy = (
                disease_remedy
                or remedy_by_merged.get(merged_label)
                or remedy_by_norm.get(normalize_disease(merged_label), "No remedy available.")
            )
        else:
            merged_candidate = disease_full.lower().strip().replace(" ", "_")
            remedy = (
                remedy_by_merged.get(merged_candidate)
                or remedy_by_norm.get(normalize_disease(merged_candidate), "No remedy available.")
            )
            display_name = disease_full
            disease_key = merged_candidate
            merged_label = normalize_disease(merged_candidate)
    else:
        merged_candidate = disease_full.lower().strip().replace(" ", "_")
        remedy = (
            remedy_by_merged.get(merged_candidate)
            or remedy_by_norm.get(normalize_disease(merged_candidate), "No remedy available.")
        )
        display_name = disease_full
        disease_key = merged_candidate
        merged_label = normalize_disease(merged_candidate)

    return {
        "plant": plant,
        "disease_display": display_name,
        "disease_full": disease_full,
        "internal_key": disease_key,
        "merged_label": merged_label,
        "confidence": confidence,
        "remedy": remedy,
    }

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.header("📸 Image-based Prediction")
uploaded_img = st.file_uploader("Upload leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_img:
    res_img = predict_image(uploaded_img)

    st.subheader(f"🌿 Detected Disease: **{res_img['plant']} – {res_img['disease_display']}**")
    st.write(f"Model class: `{res_img['disease_full']}`")
    st.write(f"Internal key: `{res_img['internal_key']}`  | merged group: `{res_img['merged_label']}`")
    st.write(f"Confidence: **{res_img['confidence']*100:.2f}%**")
    st.subheader("🧪 Recommended Treatment:")
    st.write(res_img["remedy"])

st.header("📝 Text-based Prediction")
text_input = st.text_area("Describe the symptoms here...")

if st.button("Analyze Text"):
    if text_input.strip():
        res_txt = predict_text(text_input)

        st.subheader(f"🌿 Detected Disease: **{res_txt['display_name']}**")
        st.write(f"Coarse group predicted by BERT: `{res_txt['coarse_label']}`")
        if res_txt["refined_from"]:
            st.write(
                f"Refined from coarse label `{res_txt['refined_from']}` "
                f"using symptom similarity (score ≈ {res_txt['similarity']:.2f})."
            )
        st.write(f"Internal key: `{res_txt['final_key']}`")
        st.write(f"Confidence (model softmax): **{res_txt['confidence']*100:.2f}%**")

        st.subheader("🧪 Recommended Treatment:")
        st.write(res_txt["remedy"])
    else:
        st.warning("Please enter symptoms to analyze.")
