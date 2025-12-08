# rv10_pro.py — PlantDocBot with Google Drive download + professional UI

import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# semantic refiner
from sentence_transformers import SentenceTransformer, util
import torch

# Google Drive downloader
import gdown

# ---------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ---------------------------------------------------------
st.set_page_config(page_title="Plant Health Assistant", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    .app-title {
        font-size: 28px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .app-subtitle {
        color: #9ca3af;
        margin-bottom: 1.2rem;
    }
    .card {
        background: #0f1720;
        color: #e6eef8;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.55);
    }
    .card h4 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 18px;
        font-weight: 600;
        color: #e6eef8;
    }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    .tag-healthy {
        background: rgba(16, 185, 129, 0.18);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .tag-disease {
        background: rgba(248, 113, 113, 0.16);
        color: #fecaca;
        border: 1px solid rgba(248, 113, 113, 0.45);
    }
    .muted {
        color: #9ca3af;
        font-size: 13px;
    }
    .section-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b7280;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='app-title'>🌿 Plant Health Assistant</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='app-subtitle'>AI-powered plant health insights from leaf images and symptom descriptions.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# PATHS (LOCAL) + GOOGLE DRIVE IDS
# ---------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()

# small files you commit to GitHub (should live directly in repo)
SMALL_BASE = APP_DIR

LABEL_ENCODER_FILE = SMALL_BASE / "label_encoder_merged.joblib"
IMAGE_LABEL_ENCODER_FILE = SMALL_BASE / "image_label_encoder.joblib"
DISEASE_TO_MERGED_CSV = SMALL_BASE / "disease_to_merged.csv"
DISPLAY_MAP_FILE = SMALL_BASE / "display_name_by_merged.csv"  # optional but useful

# big files we download into /models (not in GitHub)
MODELS_BASE = APP_DIR / "models"
MODELS_BASE.mkdir(exist_ok=True)

TEXT_MODEL_DIR = MODELS_BASE / "Final_Text_Disease_Model"
IMAGE_MODEL_FILE = MODELS_BASE / "plant_disease_model_fixed.keras"
REMEDY_CSV_PATH = MODELS_BASE / "plant_disease_dataset_clean_lemmatized03.csv"

# ---- your Google Drive IDs (from the links you sent) ----
GDRIVE_IMAGE_ID = "1oFVBpl-Q81kryI6h-TC94Wt7rcMd8AXn"
GDRIVE_TEXT_FOLDER_ID = "1L_yTMpvW5xFSKUFHQN-_mz5t2K9fjxnG"
GDRIVE_CSV_ID = "1hVEoCo-EecTqFdVWTP7c-nqSGl1iV3e6"


# ---------------------------------------------------------
# HELPERS TO DOWNLOAD FROM GOOGLE DRIVE (RUN ONCE)
# ---------------------------------------------------------
@st.cache_resource
def ensure_image_model_path() -> str:
    """Download image analysis component from Drive if missing."""
    if not IMAGE_MODEL_FILE.exists() or IMAGE_MODEL_FILE.stat().st_size == 0:
        st.write("📥 Downloading image analysis component…")
        gdown.download(
            id=GDRIVE_IMAGE_ID,
            output=str(IMAGE_MODEL_FILE),
            quiet=False,
        )
    return str(IMAGE_MODEL_FILE)


@st.cache_resource
def ensure_text_model_dir() -> str:
    """Download text analysis component from Drive if missing."""
    config_path = TEXT_MODEL_DIR / "config.json"
    if not config_path.exists():
        st.write("📥 Downloading text analysis component…")
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
    """Download treatment knowledge base from Drive if missing."""
    if not REMEDY_CSV_PATH.exists() or REMEDY_CSV_PATH.stat().st_size == 0:
        st.write("📥 Downloading treatment knowledge base…")
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
    model = TFAutoModelForSequenceClassification.from_pretrained(text_model_path)
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
    """
    csv_path = ensure_remedy_csv_path()
    df = pd.read_csv(csv_path)

    if "Disease" not in df.columns:
        return None

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

    records = []
    for disease_key, sub in df.groupby("disease_key"):
        if not sub["merged_label"].isna().all():
            merged_group = sub["merged_label"].value_counts().index[0]
        else:
            merged_group = "other"

        display_name = sub["Disease_raw"].value_counts().index[0]

        text_pieces = []
        if "Symptoms" in sub.columns:
            text_pieces = sub["Symptoms"].dropna().astype(str).unique().tolist()
        elif "clean_text" in sub.columns:
            text_pieces = sub["clean_text"].dropna().astype(str).unique().tolist()

        if not text_pieces:
            text_pieces = [display_name]

        rep_text = " ".join(text_pieces[:3])

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
    refiner = load_disease_refiner()
    if refiner is None:
        return None, None, None, None

    model = refiner["model"]
    index_df = refiner["index_df"]
    embs = refiner["embs"]

    coarse_key = str(coarse_merged).strip().lower()
    candidate_indices = index_df.index[index_df["merged_label"] == coarse_key].tolist()

    if not candidate_indices:
        norm_key = normalize_disease(coarse_key)
        candidate_indices = index_df.index[
            index_df["merged_label"].apply(normalize_disease) == norm_key
        ].tolist()

    if not candidate_indices:
        return None, None, None, None

    query_emb = model.encode(
        symptoms_text,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    candidate_embs = embs[candidate_indices]

    sims = util.cos_sim(query_emb, candidate_embs)[0]
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
# LOAD CORE COMPONENTS ONCE
# ---------------------------------------------------------
with st.spinner("Preparing assistant…"):
    tokenizer, nlp_model, label_encoder = load_nlp_model()
    cnn_model = load_image_model()
    img_idx_to_class = load_image_label_encoder()
    remedy_by_merged, remedy_by_norm = load_remedy_maps()
    display_name_by_merged = load_display_map()
    # disease_refiner loads lazily when needed


# ---------------------------------------------------------
# BACKEND PREDICT HELPERS
# ---------------------------------------------------------
def split_plant_and_disease(class_name):
    if not isinstance(class_name, str):
        class_name = str(class_name)
    c = class_name.replace("___", "_").replace("__", "_")
    parts = c.split("_")
    plant = parts[0].title() if parts else "Unknown"
    disease = " ".join(parts[1:]).replace("_", " ").title() if len(parts) > 1 else "Unknown"
    return plant, disease


def predict_text(symptoms: str):
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

    base_display = display_name_by_merged.get(
        merged_key, merged_key.replace("_", " ").title()
    )

    refined_name, disease_key, disease_remedy, sim = refine_disease_with_embeddings(
        symptoms, merged_key
    )

    if refined_name is not None:
        final_display = refined_name
        final_label_key = disease_key
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


def predict_image(image_file):
    img = Image.open(image_file).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)

    preds = cnn_model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    class_name = img_idx_to_class.get(idx, str(idx))
    plant, disease_full = split_plant_and_disease(class_name)

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
# SESSION STATE
# ---------------------------------------------------------
if "image_result" not in st.session_state:
    st.session_state["image_result"] = None
if "text_result" not in st.session_state:
    st.session_state["text_result"] = None
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None
if "symptom_text" not in st.session_state:
    st.session_state["symptom_text"] = ""


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='section-label'>Workspace</div>", unsafe_allow_html=True)
    st.write("Switch between **Image** and **Text** to analyze plant health.")
    st.divider()
    if st.button("Reset session"):
        st.session_state["image_result"] = None
        st.session_state["text_result"] = None
        st.session_state["uploaded_image"] = None
        st.session_state["symptom_text"] = ""
        st.success("Session cleared.")
    st.markdown("---")
    st.caption("Tip: use clear, close-up leaf photos and descriptive symptoms for best results.")


# ---------------------------------------------------------
# MAIN LAYOUT — TABS
# ---------------------------------------------------------
tab_img, tab_text = st.tabs(["📸 Leaf Image", "📝 Symptom Text"])

# ======================= IMAGE TAB =======================
with tab_img:
    st.markdown("<div class='section-label'>Image Input</div>", unsafe_allow_html=True)
    st.subheader("Analyze a leaf photo")

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        uploaded = st.file_uploader(
            "Upload a leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            st.session_state["uploaded_image"] = uploaded

        if st.session_state["uploaded_image"] is not None:
            st.image(
                st.session_state["uploaded_image"],
                caption="Preview",
                use_container_width=True,
            )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Run analysis", type="primary"):
                if st.session_state["uploaded_image"] is None:
                    st.warning("Please upload a leaf image first.")
                else:
                    with st.spinner("Analyzing leaf image…"):
                        prog = st.progress(0)
                        for i in range(0, 101, 25):
                            prog.progress(i)
                        res = predict_image(st.session_state["uploaded_image"])
                    st.session_state["image_result"] = res
                    st.success("Analysis complete.")

        with col_btn2:
            if st.button("Clear image"):
                st.session_state["uploaded_image"] = None
                st.session_state["image_result"] = None

    with c2:
        st.markdown("<div class='card'><h4>Result</h4>", unsafe_allow_html=True)

        res = st.session_state["image_result"]
        if res is None:
            st.markdown(
                "<span class='muted'>Upload a leaf image and run the analysis to see results here.</span>",
                unsafe_allow_html=True,
            )
        else:
            is_healthy = "healthy" in res["disease_display"].lower() or res["merged_label"] == "healthy"
            tag_class = "tag-healthy" if is_healthy else "tag-disease"
            tag_text = "Likely healthy" if is_healthy else "Likely diseased"

            st.markdown(
                f"<span class='tag {tag_class}'>{tag_text}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Plant:** {res['plant']}")
            st.markdown(f"**Condition detected:** {res['disease_display']}")
            st.markdown(f"**Confidence:** {res['confidence']*100:.2f}%")

            st.markdown("**Recommended treatment / next steps:**")
            st.write(res["remedy"])

        st.markdown("</div>", unsafe_allow_html=True)

# ======================= TEXT TAB =======================
with tab_text:
    st.markdown("<div class='section-label'>Text Input</div>", unsafe_allow_html=True)
    st.subheader("Describe what you see on the plant")

    st.markdown(
        "<span class='muted'>Mention leaf colour changes, spots, patterns, how fast it spreads, and where on the plant it appears.</span>",
        unsafe_allow_html=True,
    )

    c3, c4 = st.columns([1.1, 0.9])

    with c3:
        st.session_state["symptom_text"] = st.text_area(
            "Describe symptoms here…",
            height=160,
            label_visibility="collapsed",
            value=st.session_state["symptom_text"],
        )

        btn_analyze, btn_clear = st.columns(2)
        with btn_analyze:
            if st.button("Analyze description", type="primary"):
                if not st.session_state["symptom_text"].strip():
                    st.warning("Please describe the symptoms first.")
                else:
                    with st.spinner("Analyzing description…"):
                        prog2 = st.progress(0)
                        for i in range(0, 101, 25):
                            prog2.progress(i)
                        res_txt = predict_text(st.session_state["symptom_text"])
                    st.session_state["text_result"] = res_txt
                    st.success("Analysis complete.")

        with btn_clear:
            if st.button("Clear text"):
                st.session_state["symptom_text"] = ""
                st.session_state["text_result"] = None

    with c4:
        st.markdown("<div class='card'><h4>Result</h4>", unsafe_allow_html=True)

        res = st.session_state["text_result"]
        if res is None:
            st.markdown(
                "<span class='muted'>Enter symptoms and run the analysis to see a detailed interpretation here.</span>",
                unsafe_allow_html=True,
            )
        else:
            is_healthy = "healthy" in res["display_name"].lower() or res["coarse_label"] == "healthy"
            tag_class = "tag-healthy" if is_healthy else "tag-disease"
            tag_text = "Likely healthy" if is_healthy else "Likely diseased"

            st.markdown(
                f"<span class='tag {tag_class}'>{tag_text}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Condition detected:** {res['display_name']}")
            st.markdown(f"**Confidence:** {res['confidence']*100:.2f}%")

            # NOTE: we intentionally do NOT show the "Refined using symptom similarity..." sentence

            st.markdown("**Recommended treatment / next steps:**")
            st.write(res["remedy"])

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
f1, f2 = st.columns([3, 1])
with f1:
    st.caption(
        "For best results, avoid very blurry photos and include clear descriptions of all visible symptoms."
    )
with f2:
    st.caption("PlantDocBot · Demo interface")
