# rv10.py — Final Professional Plant Health Assistant (Cloud Ready)

import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import gdown

# ---------------------------------------------------------
# PAGE CONFIG (PROFESSIONAL LOOK)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Plant Health Assistant",
    page_icon="🌿",
    layout="wide",
)

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
}
.sub-title {
    font-size: 18px;
    color: #9ca3af;
}
.card {
    background: #0f172a;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
}
.label {
    color: #93c5fd;
    font-weight: 600;
}
.result {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌿 Plant Health Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered plant disease detection & treatment recommendation</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()

LABEL_ENCODER_FILE = APP_DIR / "label_encoder_merged.joblib"
IMAGE_LABEL_ENCODER_FILE = APP_DIR / "image_label_encoder.joblib"
DISEASE_TO_MERGED_CSV = APP_DIR / "disease_to_merged.csv"

MODELS_BASE = APP_DIR / "models"
MODELS_BASE.mkdir(exist_ok=True)

TEXT_MODEL_DIR = MODELS_BASE / "Final_Text_Disease_Model"
IMAGE_MODEL_FILE = MODELS_BASE / "plant_disease_model_cloud.h5"
REMEDY_CSV_PATH = MODELS_BASE / "plant_disease_dataset_clean_lemmatized03.csv"

# ✅ YOUR FINAL CLOUD MODEL IDS
GDRIVE_IMAGE_ID = "12aRYV9_laCwvonv20mJneshu5fBedZZL"
GDRIVE_TEXT_FOLDER_ID = "1L_yTMpvW5xFSKUFHQN-_mz5t2K9fjxnG"
GDRIVE_CSV_ID = "1hVEoCo-EecTqFdVWTP7c-nqSGl1iV3e6"

# ---------------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------------
@st.cache_resource
def ensure_image_model_path():
    if not IMAGE_MODEL_FILE.exists():
        with st.spinner("Downloading system files..."):
            gdown.download(id=GDRIVE_IMAGE_ID, output=str(IMAGE_MODEL_FILE), quiet=False)
    return str(IMAGE_MODEL_FILE)

@st.cache_resource
def ensure_text_model_dir():
    if not (TEXT_MODEL_DIR / "config.json").exists():
        with st.spinner("Downloading system files..."):
            TEXT_MODEL_DIR.mkdir(exist_ok=True)
            gdown.download_folder(
                id=GDRIVE_TEXT_FOLDER_ID,
                output=str(TEXT_MODEL_DIR),
                quiet=False,
                use_cookies=False,
            )
    return str(TEXT_MODEL_DIR)

@st.cache_resource
def ensure_remedy_csv_path():
    if not REMEDY_CSV_PATH.exists():
        with st.spinner("Downloading system files..."):
            gdown.download(id=GDRIVE_CSV_ID, output=str(REMEDY_CSV_PATH), quiet=False)
    return str(REMEDY_CSV_PATH)

# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------
def normalize_disease(d):
    d = str(d).lower()
    if "healthy" in d: return "healthy"
    if "bacterial" in d or "spot" in d: return "bacterial"
    if "blight" in d: return "blight"
    if "mildew" in d or "mold" in d: return "mildew"
    if "virus" in d or "mosaic" in d: return "viral"
    if "rot" in d or "canker" in d: return "rot_mold"
    if "scab" in d: return "scab"
    if "curl" in d: return "curl"
    return "other"

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_nlp_model():
    ensure_text_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
    model = TFAutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    return tokenizer, model, label_encoder

@st.cache_resource
def load_image_model():
    ensure_image_model_path()
    return tf.keras.models.load_model(IMAGE_MODEL_FILE, compile=False)

@st.cache_resource
def load_image_label_encoder():
    enc = joblib.load(IMAGE_LABEL_ENCODER_FILE)
    return {i: cls for i, cls in enumerate(enc.classes_)}

@st.cache_resource
def load_remedy_map():
    ensure_remedy_csv_path()
    df = pd.read_csv(REMEDY_CSV_PATH)
    df["merged_norm"] = df["merged_label"].astype(str).apply(normalize_disease)

    remedy_map = (
        df.groupby("merged_norm")["Remedy"]
        .agg(lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) else "No remedy available.")
        .to_dict()
    )
    remedy_map.setdefault("healthy", "No treatment needed.")
    remedy_map.setdefault("other", "No remedy available.")
    return remedy_map

# ---------------------------------------------------------
# LOAD ALL SYSTEMS
# ---------------------------------------------------------
with st.spinner("Initializing system..."):
    tokenizer, nlp_model, label_encoder = load_nlp_model()
    cnn_model = load_image_model()
    img_idx_to_class = load_image_label_encoder()
    treatment_map = load_remedy_map()

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def split_plant_and_disease(class_name):
    parts = str(class_name).replace("___","_").split("_")
    plant = parts[0].title()
    disease = " ".join(parts[1:]).replace("_"," ").title()
    return plant, disease

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------
def predict_text(symptoms):
    enc = tokenizer(symptoms, truncation=True, padding=True, max_length=256, return_tensors="tf")
    outputs = nlp_model(enc)
    pred = int(tf.argmax(outputs.logits, axis=1).numpy()[0])
    label = label_encoder.inverse_transform([pred])[0]
    key = normalize_disease(label)
    remedy = treatment_map.get(key, "No remedy available.")
    confidence = float(tf.nn.softmax(outputs.logits)[0][pred])
    return label.title(), confidence, remedy

def predict_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, 0)
    preds = cnn_model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    class_name = img_idx_to_class[idx]
    plant, disease = split_plant_and_disease(class_name)
    key = normalize_disease(disease)
    remedy = treatment_map.get(key, "No remedy available.")
    return plant, disease, confidence, remedy

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "image" not in st.session_state:
    st.session_state.image = None
if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📸 Image Analysis", "📝 Symptom Analysis"])

# ------------------ IMAGE TAB ------------------
with tab1:
    col1, col2 = st.columns([1,1])

    with col1:
        img_file = st.file_uploader("Upload plant leaf image", type=["jpg","jpeg","png"])
        if img_file:
            st.session_state.image = img_file
            st.image(img_file, use_column_width=True)

        if st.button("Analyze Image"):
            if st.session_state.image:
                with st.spinner("Analyzing image..."):
                    plant, disease, conf, remedy = predict_image(st.session_state.image)
                    st.session_state.result = (plant, disease, conf, remedy)
            else:
                st.warning("Please upload an image first.")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>Result</h4>", unsafe_allow_html=True)

        if st.session_state.result:
            plant, disease, conf, remedy = st.session_state.result
            st.markdown(f"<div class='result'><span class='label'>Plant:</span> {plant}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result'><span class='label'>Detected Issue:</span> {disease}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result'><span class='label'>Confidence:</span> {conf*100:.2f}%</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<b>Recommended Treatment</b>")
            st.write(remedy)
        else:
            st.info("Results will appear here.")

        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ TEXT TAB ------------------
with tab2:
    text_input = st.text_area("Describe symptoms observed on the plant")

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Analyze Symptoms"):
            if text_input.strip():
                with st.spinner("Analyzing symptoms..."):
                    disease, conf, remedy = predict_text(text_input)
                    st.session_state.result = (None, disease, conf, remedy)
            else:
                st.warning("Please enter symptoms.")

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>Result</h4>", unsafe_allow_html=True)
        if st.session_state.result:
            _, disease, conf, remedy = st.session_state.result
            st.markdown(f"<div class='result'><span class='label'>Detected Issue:</span> {disease}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result'><span class='label'>Confidence:</span> {conf*100:.2f}%</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<b>Recommended Treatment</b>")
            st.write(remedy)
        else:
            st.info("Results will appear here.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("Designed for professional agricultural decision support • Plant Health Assistant")
