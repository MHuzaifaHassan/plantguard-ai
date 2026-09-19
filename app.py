"""
PlantGuard AI — AI-Powered Tomato Leaf Disease Detection
Streamlit dashboard (UI redesign). The trained MobileNetV2 model, its
preprocessing and the class names are used exactly as before.
"""

import io
import html
import logging
import os
import re
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    TF_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    tf = None
    TF_IMPORT_ERROR = exc

logger = logging.getLogger("plantguard")

import inspect
# Newer Streamlit sizes buttons/images to their content by default; ask for full width in a version-safe way.
BTN_STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
               else {"use_container_width": True})
IMG_STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.image).parameters
               else {"use_container_width": True})

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard AI — Tomato Leaf Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Constants (existing ML pipeline settings — unchanged)
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.txt"
IMG_SIZE = (224, 224)

# ─────────────────────────────────────────────────────────────
# Educational content for the 10 classes
# ─────────────────────────────────────────────────────────────
DISEASES = {
    "bacterial_spot": {
        "name": "Bacterial Spot",
        "cause": "Bacterial",
        "category": "bacterial",
        "info": "Bacterial spot is caused by Xanthomonas bacteria. It typically appears as small, "
                "dark, water-soaked spots on leaves that may turn brown and sometimes show a yellow "
                "halo. Warm, wet conditions favor its spread.",
    },
    "early_blight": {
        "name": "Early Blight",
        "cause": "Fungal",
        "category": "fungal",
        "info": "Early blight is a fungal disease caused by Alternaria solani. It commonly produces "
                "brown spots with concentric rings, often on older leaves first, and may cause the "
                "surrounding tissue to yellow.",
    },
    "healthy": {
        "name": "Healthy",
        "cause": "No disease pattern detected",
        "category": "healthy",
        "info": "A healthy tomato leaf usually shows even green coloration without distinct spots, "
                "lesions, curling or discoloration. The model did not find strong signs of the "
                "disease patterns it was trained to recognize.",
    },
    "late_blight": {
        "name": "Late Blight",
        "cause": "Oomycete pathogen",
        "category": "fungal",
        "info": "Late blight is a tomato disease caused by the pathogen Phytophthora infestans. It "
                "can cause dark lesions and rapid leaf damage under favorable conditions, "
                "especially in cool, moist weather.",
    },
    "leaf_mold": {
        "name": "Leaf Mold",
        "cause": "Fungal",
        "category": "fungal",
        "info": "Leaf mold is a fungal disease caused by Passalora fulva. It often appears as pale "
                "green or yellowish patches on the upper leaf surface, with olive-green to brownish "
                "fuzzy growth underneath. High humidity favors it.",
    },
    "septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "cause": "Fungal",
        "category": "fungal",
        "info": "Septoria leaf spot is a fungal disease caused by Septoria lycopersici. It typically "
                "forms many small circular spots with dark borders and lighter centers, usually "
                "starting on the lower leaves.",
    },
    "spider_mites": {
        "name": "Spider Mites (Two-spotted Spider Mite)",
        "cause": "Pest",
        "category": "pest",
        "info": "Two-spotted spider mites are tiny pests that feed on leaf tissue. Affected leaves "
                "can show fine stippling, yellowing or bronzing, and sometimes delicate webbing. "
                "Hot, dry conditions favor them.",
    },
    "target_spot": {
        "name": "Target Spot",
        "cause": "Fungal",
        "category": "fungal",
        "info": "Target spot is a fungal disease caused by Corynespora cassiicola. It can produce "
                "brown lesions with concentric ring patterns on leaves and may also affect stems "
                "and fruit in warm, humid conditions.",
    },
    "mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "cause": "Viral",
        "category": "viral",
        "info": "Tomato mosaic virus is a viral disease that can cause mottled light and dark green "
                "patterns on leaves, along with distortion and stunted growth. It spreads easily "
                "through contact, tools and contaminated plant material.",
    },
    "yellow_leaf_curl": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "cause": "Viral (spread by whiteflies)",
        "category": "viral",
        "info": "Tomato yellow leaf curl virus is a viral disease transmitted by whiteflies. Affected "
                "plants can show upward-curling, yellowing leaves and stunted growth.",
    },
}

# Ordered keyword rules used to map whatever is written in class_names.txt
# (e.g. "Tomato___Late_blight" or "Late Blight") onto the content above.
MATCH_RULES = [
    ("healthy", ["healthy"]),
    ("bacterial_spot", ["bacterialspot"]),
    ("early_blight", ["earlyblight"]),
    ("late_blight", ["lateblight"]),
    ("leaf_mold", ["leafmold", "leafmould"]),
    ("septoria_leaf_spot", ["septoria"]),
    ("spider_mites", ["spidermite", "twospotted"]),
    ("target_spot", ["targetspot"]),
    ("mosaic_virus", ["mosaic"]),
    ("yellow_leaf_curl", ["yellowleafcurl", "curlvirus"]),
]

NEXT_STEPS = {
    "fungal": [
        "Inspect nearby leaves and neighboring plants for similar symptoms.",
        "Remove heavily affected plant material according to local agricultural guidance.",
        "Avoid unnecessary leaf wetness: water at the base of the plant and allow good airflow.",
        "Monitor the plant regularly for changes.",
        "Consult a local agriculture expert for confirmed diagnosis and treatment.",
    ],
    "bacterial": [
        "Inspect nearby leaves and neighboring plants for similar symptoms.",
        "Avoid handling plants while they are wet, and clean tools between plants.",
        "Avoid unnecessary leaf wetness: water at the base of the plant.",
        "Monitor the plant regularly for changes.",
        "Consult a local agriculture expert for confirmed diagnosis and treatment.",
    ],
    "viral": [
        "Inspect nearby plants for similar symptoms and for insect pests such as whiteflies.",
        "Where possible, keep suspect plants separate and clean hands and tools after handling them.",
        "Remove heavily affected plant material according to local agricultural guidance.",
        "Monitor the plant regularly for changes.",
        "Consult a local agriculture expert for confirmed diagnosis and treatment.",
    ],
    "pest": [
        "Check the undersides of leaves for tiny mites, stippling or fine webbing.",
        "Inspect nearby plants, since pests can move between them.",
        "Remove heavily affected plant material according to local agricultural guidance.",
        "Monitor the plant regularly for changes.",
        "Consult a local agriculture expert for confirmed diagnosis and treatment.",
    ],
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
ICONS = {
    "upload": '<path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M20 16v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 4 13c0-6 6-9 16-9 0 10-3 16-9 16z"/><path d="M4 21c3-6 6-9 11-12"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4 6h.01M4 12h.01M4 18h.01"/>',
    "bars": '<path d="M5 20V11M12 20V4M19 20v-6"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h6v6H9z"/>',
}


def icon(name: str, size: int = 20) -> str:
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


def render(markup: str) -> None:
    """Render HTML through st.markdown. Indentation and blank lines are removed
    so Markdown never mistakes the HTML for a code block."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def resolve_class(raw_name: str):
    norm = re.sub(r"[^a-z0-9]", "", raw_name.lower())
    for key, keywords in MATCH_RULES:
        if any(k in norm for k in keywords):
            return key
    return None


def pretty_name(raw_name: str) -> str:
    key = resolve_class(raw_name)
    if key:
        return DISEASES[key]["name"]
    cleaned = re.sub(r"^tomato[_\s-]*", "", raw_name, flags=re.I)
    return re.sub(r"[_\s]+", " ", cleaned).strip().title() or raw_name


def confidence_level(pct: float):
    """>=80 High · 60–79.99 Moderate · <60 Low"""
    if pct >= 80:
        return "High Confidence", "high"
    if pct >= 60:
        return "Moderate Confidence", "moderate"
    return "Low Confidence", "low"


SAMPLE_DIR = BASE_DIR / "test_images"


def get_samples(limit: int = 4) -> list:
    """Real images from the project's test_images/ folder (empty list if none)."""
    try:
        files = sorted(
            p for p in SAMPLE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        return files[:limit]
    except Exception:
        return []


def build_report(file_name: str, probs: np.ndarray, class_names: list) -> str:
    from datetime import datetime
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    pct = float(probs[top] * 100)
    label, _ = confidence_level(pct)
    lines = [
        "PlantGuard AI — Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Image: {file_name}",
        "",
        f"Prediction: {pretty_name(class_names[top])}",
        f"Confidence: {pct:.2f}% ({label})",
        "",
        "Top 3 predictions:",
    ]
    for rank, idx in enumerate(order[:3], start=1):
        lines.append(f"  {rank}. {pretty_name(class_names[int(idx)])} — {probs[int(idx)] * 100:.2f}%")
    lines += [
        "",
        "Model: MobileNetV2 (transfer learning), input 224 x 224",
        "Note: This is an AI model prediction, not a confirmed diagnosis. "
        "Please verify with appropriate agricultural expertise.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Model loading (cached) and prediction (existing logic)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    """Returns (model, class_names, error_message)."""
    if tf is None:
        logger.error("TensorFlow import failed: %s", TF_IMPORT_ERROR)
        return None, [], "TensorFlow could not be loaded. Please check your installation."
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
    except Exception:
        logger.exception("Model loading failed")
        return None, [], (
            "The AI model could not be loaded. Make sure "
            "<code>models/plant_disease_model.keras</code> exists and try again."
        )
    try:
        names = [
            line.strip()
            for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        logger.exception("Class names loading failed")
        return None, [], (
            "The class list could not be read. Make sure "
            "<code>models/class_names.txt</code> exists."
        )
    try:
        if int(model.output_shape[-1]) != len(names):
            logger.error("Model outputs %s classes but class_names.txt has %s",
                         model.output_shape[-1], len(names))
            return None, [], (
                "The model and <code>class_names.txt</code> do not match "
                "(different number of classes)."
            )
    except Exception:
        pass
    return model, names, None


def run_prediction(model, image: Image.Image) -> np.ndarray:
    """Same pipeline as before: 224×224 resize → MobileNetV2 preprocess_input →
    model.predict. Returns the class probability vector."""
    resized = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(resized, dtype="float32")
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    probs = np.asarray(model.predict(arr, verbose=0))[0].astype("float64")
    # Safety net: only applied if the model returned raw logits instead of probabilities.
    if probs.min() < 0 or abs(probs.sum() - 1.0) > 1e-3:
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
    return probs


# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Figtree:wght@400;500;600;700&display=swap');

:root{
  --bg:#F4F6F3; --card:#FFFFFF; --ink:#14291E; --muted:#5B6B61; --line:#E0E6DF;
  --forest:#1D5A38; --leaf:#3F8F5B; --mint:#E6F1E8; --sage:#A9C4B0;
  --amber:#B7791F; --amber-bg:#FBF1DD; --rose:#B4483C; --rose-bg:#F8E4E1;
  --radius:20px;
  --shadow:0 1px 2px rgba(20,41,30,.05),0 8px 24px rgba(20,41,30,.06);
}

html, body, .stApp{ background:var(--bg); color:var(--ink); color-scheme:light; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp small,
[data-testid="stMarkdownContainer"], [data-testid="stFileUploaderDropzone"]{ font-family:'Figtree',system-ui,-apple-system,'Segoe UI',sans-serif; }
/* Keep Streamlit's icon font intact (otherwise icon names like "upload" show up as text) */
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded,
span[class*="material"], [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"]{
  font-family:'Material Symbols Rounded','Material Icons' !important; }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
.block-container{ max-width:1180px; padding:1.25rem 1.5rem 0 1.5rem; }
h1,h2,h3{ font-family:'Fraunces',Georgia,serif; color:var(--ink); }

/* ── Navbar ── */
.pg-nav{ display:flex; align-items:center; justify-content:space-between; gap:1.5rem;
  background:var(--card); border:1px solid var(--line); border-radius:999px;
  padding:.7rem 1.1rem .7rem 1.4rem; box-shadow:var(--shadow); }
.pg-brand{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:1.25rem; color:var(--forest); white-space:nowrap; }
.pg-links{ display:flex; gap:.35rem; margin-left:auto; }
.pg-links a{ color:var(--muted) !important; text-decoration:none !important; font-weight:500; font-size:.95rem;
  padding:.45rem .9rem; border-radius:999px; transition:background .15s,color .15s; }
.pg-links a:hover{ background:var(--mint); color:var(--forest) !important; }
.pg-status{ display:flex; align-items:center; gap:.5rem; font-size:.85rem; font-weight:600; white-space:nowrap;
  padding:.4rem .8rem; border-radius:999px; background:var(--mint); color:var(--forest); }
.pg-status .dot{ width:.6rem; height:.6rem; border-radius:50%; background:#2FB463; box-shadow:0 0 0 3px rgba(47,180,99,.2); }
.pg-status.offline{ background:var(--rose-bg); color:var(--rose); }
.pg-status.offline .dot{ background:var(--rose); box-shadow:0 0 0 3px rgba(180,72,60,.18); }

/* ── Hero ── */
.pg-anchor{ position:relative; top:-1rem; height:0; }
.pg-hero{ padding:3.2rem 0 1rem 0; }
.pg-badge{ display:inline-block; background:var(--mint); color:var(--forest); font-weight:600; font-size:.85rem;
  padding:.4rem .9rem; border-radius:999px; border:1px solid #CFE3D3; }
.pg-hero h1{ font-size:clamp(2.2rem,4.4vw,3.5rem); line-height:1.08; font-weight:600; letter-spacing:-.02em;
  margin:1.1rem 0 1rem 0; padding:0; max-width:14em; }
.pg-hero p.lead{ font-size:1.12rem; line-height:1.6; color:var(--muted); max-width:30em; margin:0 0 1.6rem 0; }
.pg-facts{ display:flex; flex-wrap:wrap; gap:.6rem; }
.pg-fact{ display:flex; align-items:center; gap:.5rem; font-size:.9rem; font-weight:500; color:var(--ink);
  background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.55rem .8rem; }
.pg-fact svg{ color:var(--leaf); }

/* ── Cards ── */
.pg-card, .st-key-upload_card{ background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
  box-shadow:var(--shadow); padding:1.6rem; }
.st-key-upload_card{ margin-top:1.4rem; }
.pg-card-title{ display:flex; align-items:center; gap:.6rem; font-family:'Fraunces',Georgia,serif; font-size:1.25rem;
  font-weight:600; color:var(--ink); margin-bottom:.35rem; }
.pg-card-title .ico{ display:grid; place-items:center; width:2.2rem; height:2.2rem; border-radius:12px; background:var(--mint); color:var(--forest); }
.pg-muted{ color:var(--muted); font-size:.95rem; line-height:1.55; }
.pg-card-title + .pg-muted{ margin-bottom:1rem; }

/* Uploader */
[data-testid="stFileUploaderDropzone"]{ background:var(--mint); border:1.5px dashed var(--sage); border-radius:16px; padding:1.4rem; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--leaf); }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span{ color:var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button{ border-radius:10px; border:1px solid var(--sage); background:#fff; color:var(--forest); font-weight:600; }
[data-testid="stFileUploaderFile"]{ display:none; }

/* Image preview */
[data-testid="stImage"] img{ width:100%; max-height:320px; object-fit:cover; border-radius:16px; border:1px solid var(--line); }
.pg-file{ display:flex; align-items:center; justify-content:space-between; gap:.8rem; flex-wrap:wrap; margin:.9rem 0 1rem 0; }
.pg-file b{ display:block; font-weight:600; word-break:break-all; }
.pg-file small{ color:var(--muted); }
.pg-chip{ font-size:.8rem; font-weight:600; padding:.3rem .7rem; border-radius:999px; white-space:nowrap; }
.pg-chip.ready{ background:var(--mint); color:var(--forest); }
.pg-chip.done{ background:var(--forest); color:#fff; }

/* Buttons */
.stButton{ width:100%; }
.stButton > button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"]{
  width:100%; min-height:3rem; border-radius:12px; font-weight:600; font-size:1rem; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--forest); border:1px solid var(--forest); color:#fff; box-shadow:0 6px 16px rgba(29,90,56,.25); }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{ background:#164A2D; border-color:#164A2D; color:#fff; }
.stButton > button:focus-visible{ outline:3px solid rgba(63,143,91,.45); outline-offset:2px; }
.stButton > button:disabled{ background:#C9D3CB; border-color:#C9D3CB; color:#fff; box-shadow:none; }

/* ── Results ── */
.pg-section-title{ font-family:'Fraunces',Georgia,serif; font-size:1.7rem; font-weight:600; margin:0 0 .3rem 0; letter-spacing:-.01em; }
.pg-section-sub{ color:var(--muted); margin:0 0 1.4rem 0; }
.pg-result-name{ font-family:'Fraunces',Georgia,serif; font-size:clamp(1.9rem,3.2vw,2.6rem); font-weight:600; line-height:1.15; margin:.6rem 0 1.2rem 0; color:var(--forest); }
.pg-conf-row{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:.7rem; }
.pg-conf-label{ color:var(--muted); font-size:.9rem; font-weight:500; }
.pg-conf-num{ font-size:2.4rem; font-weight:700; line-height:1.05; letter-spacing:-.02em; }
.pg-pill{ font-size:.85rem; font-weight:600; padding:.4rem .85rem; border-radius:999px; }
.pg-pill.high{ background:var(--mint); color:var(--forest); }
.pg-pill.moderate{ background:var(--amber-bg); color:var(--amber); }
.pg-pill.low{ background:var(--rose-bg); color:var(--rose); }
.pg-bar{ height:12px; border-radius:999px; background:#E9EEE8; overflow:hidden; }
.pg-bar > span{ display:block; height:100%; border-radius:999px; background:var(--leaf); }
.pg-bar.moderate > span{ background:var(--amber); }
.pg-bar.low > span{ background:var(--rose); }
.pg-note{ margin:1rem 0 0 0; color:var(--muted); font-size:.92rem; line-height:1.55; }
.pg-disclaimer{ margin-top:1.1rem; padding:.8rem 1rem; border-radius:12px; background:#F1F4F0; color:var(--muted); font-size:.88rem; line-height:1.5; }

.pg-pred{ padding:.85rem 0; border-top:1px solid var(--line); }
.pg-pred:first-of-type{ border-top:none; padding-top:.3rem; }
.pg-pred-head{ display:flex; justify-content:space-between; gap:1rem; margin-bottom:.5rem; font-weight:500; }
.pg-pred-head .rank{ color:var(--muted); margin-right:.6rem; font-weight:600; }
.pg-pred-head .pct{ font-weight:700; font-variant-numeric:tabular-nums; }
.pg-pred .pg-bar{ height:8px; }
.pg-pred .pg-bar > span{ background:var(--sage); }
.pg-pred.top .pg-bar > span{ background:var(--forest); }

.pg-tag{ display:inline-block; font-size:.8rem; font-weight:600; color:var(--forest); background:var(--mint); padding:.3rem .7rem; border-radius:999px; margin:.2rem 0 .9rem 0; }
.pg-steps-list{ margin:.4rem 0 0 0; padding:0; list-style:none; }
.pg-steps-list li{ position:relative; padding:.55rem 0 .55rem 1.9rem; line-height:1.5; border-top:1px solid var(--line); }
.pg-steps-list li:first-child{ border-top:none; }
.pg-steps-list li::before{ content:""; position:absolute; left:.2rem; top:.95rem; width:.7rem; height:.7rem; border-radius:50%; background:var(--leaf); opacity:.85; }
.pg-healthy{ font-weight:600; font-size:1.05rem; color:var(--forest); margin:.4rem 0 .5rem 0; }

/* ── Model info ── */
.pg-model-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:1.2rem; }
.pg-model-item{ border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; background:#FAFBF9; }
.pg-model-item span{ display:block; color:var(--muted); font-size:.85rem; margin-bottom:.25rem; }
.pg-model-item b{ font-weight:600; font-size:1.02rem; }

/* ── How it works ── */
.pg-flow{ display:grid; grid-template-columns:1fr auto 1fr auto 1fr; gap:1rem; align-items:stretch; }
.pg-step{ background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:1.5rem; box-shadow:var(--shadow); }
.pg-step .num{ font-family:'Fraunces',Georgia,serif; font-size:1.05rem; font-weight:600; color:var(--leaf); }
.pg-step .ico{ display:grid; place-items:center; width:2.8rem; height:2.8rem; border-radius:14px; background:var(--mint); color:var(--forest); margin:.7rem 0 1rem 0; }
.pg-step h3{ font-size:1.2rem; margin:0 0 .35rem 0; padding:0; }
.pg-step p{ color:var(--muted); margin:0; line-height:1.55; font-size:.95rem; }
.pg-arrow{ align-self:center; color:var(--sage); font-size:1.6rem; }

/* ── Footer ── */
.pg-footer{ margin-top:4rem; padding:2.2rem 0 2.4rem 0; border-top:1px solid var(--line); text-align:center; }
.pg-footer .brand{ font-family:'Fraunces',Georgia,serif; font-size:1.3rem; font-weight:600; color:var(--forest); }
.pg-footer p{ margin:.3rem 0; color:var(--muted); font-size:.93rem; }
.pg-footer .edu{ font-size:.85rem; max-width:40em; margin:.9rem auto 0 auto; }

.pg-spacer{ height:2.4rem; }

.stElementContainer:has(> .stButton), .stElementContainer:has(> .stDownloadButton){ width:100% !important; }
/* Secondary + download buttons */
.stButton > button[kind="secondary"], [data-testid="stBaseButton-secondary"], .stDownloadButton > button{
  background:#fff; border:1px solid var(--sage); color:var(--forest); box-shadow:none; }
.stButton > button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover, .stDownloadButton > button:hover{
  background:var(--mint); border-color:var(--leaf); color:var(--forest); }
.stDownloadButton{ width:100%; }
.stDownloadButton > button{ width:100%; min-height:3rem; border-radius:12px; font-weight:600; }

/* Dropzone instructions */
[data-testid="stFileUploaderDropzoneInstructions"]{ display:flex !important; flex-direction:column; }
[data-testid="stFileUploaderDropzoneInstructions"] span{ display:block !important; visibility:visible !important; }

/* Sample leaves */
.pg-sample-title{ font-weight:600; margin:1.1rem 0 .1rem 0; }
.st-key-samples [data-testid="stImage"] img{ height:92px; max-height:92px; border-radius:12px; }
.st-key-samples .stButton > button{ min-height:2.3rem; font-size:.9rem; margin-top:.3rem; }

/* Expander */
[data-testid="stExpander"]{ border:1px solid var(--line) !important; border-radius:16px !important; background:var(--card); box-shadow:var(--shadow); }
[data-testid="stExpander"] summary{ font-weight:600; color:var(--ink); }

/* Supported conditions + tips */
.pg-chips{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; }
.pg-cond{ font-size:.9rem; font-weight:500; padding:.45rem .8rem; border-radius:999px; background:#FAFBF9; border:1px solid var(--line); }
.pg-cond.healthy{ background:var(--mint); border-color:#CFE3D3; color:var(--forest); }

/* ── Responsive ── */
@media (max-width:900px){
  .pg-links{ display:none; }
  .pg-nav{ border-radius:20px; }
  .pg-model-grid{ grid-template-columns:repeat(2,1fr); }
}
@media (max-width:700px){
  .block-container{ padding:1rem 1rem 0 1rem; }
  .pg-hero{ padding-top:2rem; }
  .pg-flow{ grid-template-columns:1fr; }
  .pg-arrow{ transform:rotate(90deg); justify-self:center; }
  .pg-model-grid{ grid-template-columns:1fr; }
  .pg-card, .st-key-upload_card{ padding:1.2rem; }
}
@media (prefers-reduced-motion:reduce){ *{ transition:none !important; scroll-behavior:auto !important; } }
</style>
"""


# ─────────────────────────────────────────────────────────────
# UI sections
# ─────────────────────────────────────────────────────────────
def render_navbar(model_online: bool) -> None:
    status_cls = "" if model_online else "offline"
    status_txt = "AI Model Online" if model_online else "AI Model Offline"
    render(f"""
    <div class="pg-nav">
        <div class="pg-brand">🌱 PlantGuard AI</div>
        <nav class="pg-links">
            <a href="#dashboard" target="_self">Dashboard</a>
            <a href="#how-it-works" target="_self">How It Works</a>
            <a href="#about" target="_self">About</a>
        </nav>
        <div class="pg-status {status_cls}"><span class="dot"></span>{status_txt}</div>
    </div>
    <div id="dashboard" class="pg-anchor"></div>
    """)


def render_hero_text() -> None:
    render(f"""
    <div class="pg-hero">
        <span class="pg-badge">Powered by MobileNetV2 • Transfer Learning</span>
        <h1>Detect Tomato Leaf Diseases with AI</h1>
        <p class="lead">Upload a tomato leaf image and let our AI model analyze it for potential
        disease patterns.</p>
        <div class="pg-facts">
            <div class="pg-fact">{icon('leaf', 18)}10 leaf conditions</div>
            <div class="pg-fact">{icon('chip', 18)}224 × 224 input</div>
            <div class="pg-fact">{icon('bars', 18)}Top-3 probabilities</div>
        </div>
    </div>
    """)


def render_result(result: dict, class_names: list) -> None:
    probs = np.array(result["probs"])
    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    raw_name = class_names[top_idx]
    key = resolve_class(raw_name)
    display = pretty_name(raw_name)
    pct = float(probs[top_idx] * 100)
    label, level = confidence_level(pct)

    render('<div id="results" class="pg-anchor"></div>')
    render('<div class="pg-section-title">Analysis results</div>'
           '<div class="pg-section-sub">Predictions come from the trained MobileNetV2 model.</div>')

    # Row 1: diagnosis + top predictions
    col_a, col_b = st.columns([1.15, 1], gap="large")
    with col_a:
        if level == "low":
            note = ("The model is not very sure about this image. Try a clearer, well-lit photo "
                    "of a single leaf.")
        elif level == "moderate":
            note = ("The model finds this the most likely match, but other conditions are "
                    "possible. Compare with the other predictions.")
        else:
            note = "The model is fairly confident about this prediction."
        render(f"""
        <div class="pg-card">
            <div class="pg-card-title"><span class="ico">{icon('leaf')}</span>AI Diagnosis</div>
            <div class="pg-result-name">{html.escape(display)}</div>
            <div class="pg-conf-row">
                <div>
                    <div class="pg-conf-label">Confidence</div>
                    <div class="pg-conf-num">{pct:.2f}%</div>
                </div>
                <span class="pg-pill {level}">{label}</span>
            </div>
            <div class="pg-bar {level}"><span style="width:{min(pct, 100):.2f}%"></span></div>
            <p class="pg-note">{note}</p>
            <div class="pg-disclaimer">This is an AI model prediction, not a confirmed diagnosis.</div>
        </div>
        """)
    with col_b:
        rows = ""
        for rank, idx in enumerate(order[:3], start=1):
            p = float(probs[idx] * 100)
            top_cls = " top" if rank == 1 else ""
            rows += f"""
            <div class="pg-pred{top_cls}">
                <div class="pg-pred-head">
                    <span><span class="rank">{rank}</span>{html.escape(pretty_name(class_names[int(idx)]))}</span>
                    <span class="pct">{p:.2f}%</span>
                </div>
                <div class="pg-bar"><span style="width:{min(p, 100):.2f}%"></span></div>
            </div>"""
        render(f"""
        <div class="pg-card">
            <div class="pg-card-title"><span class="ico">{icon('bars')}</span>Top Predictions</div>
            <div class="pg-muted">The three most likely conditions according to the model.</div>
            {rows}
        </div>
        """)

    render('<div class="pg-spacer" style="height:1.2rem"></div>')

    # Row 2: disease info + next steps
    info = DISEASES.get(key) if key else None
    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        if info:
            body = html.escape(info["info"])
            tag = html.escape(info["cause"])
            render(f"""
            <div class="pg-card">
                <div class="pg-card-title"><span class="ico">{icon('info')}</span>About {html.escape(info['name'])}</div>
                <span class="pg-tag">{tag}</span>
                <div class="pg-muted" style="font-size:1rem">{body}</div>
            </div>
            """)
        else:
            render(f"""
            <div class="pg-card">
                <div class="pg-card-title"><span class="ico">{icon('info')}</span>About {html.escape(display)}</div>
                <div class="pg-muted">No additional information is available for this class.</div>
            </div>
            """)
    with col_d:
        if info and info["category"] == "healthy":
            inner = """
            <div class="pg-healthy">Your leaf appears healthy according to the AI model.</div>
            <div class="pg-muted" style="font-size:1rem">Continue regular monitoring and good
            plant-care practices.</div>
            """
        else:
            steps = NEXT_STEPS.get(info["category"] if info else "fungal", NEXT_STEPS["fungal"])
            items = "".join(f"<li>{html.escape(s)}</li>" for s in steps)
            inner = f'<ul class="pg-steps-list">{items}</ul>'
        render(f"""
        <div class="pg-card">
            <div class="pg-card-title"><span class="ico">{icon('list')}</span>Recommended Next Steps</div>
            {inner}
            <div class="pg-disclaimer">General educational guidance only. The AI prediction is not a
            guaranteed diagnosis.</div>
        </div>
        """)

    render('<div class="pg-spacer" style="height:1.2rem"></div>')

    # Row 3: all probabilities + download
    col_e, col_f = st.columns([1.6, 1], gap="large")
    with col_e:
        with st.expander("Show all class probabilities"):
            all_rows = ""
            for rank, idx in enumerate(order, start=1):
                p = float(probs[int(idx)] * 100)
                top_cls = " top" if rank == 1 else ""
                all_rows += f"""
                <div class="pg-pred{top_cls}">
                    <div class="pg-pred-head">
                        <span><span class="rank">{rank}</span>{html.escape(pretty_name(class_names[int(idx)]))}</span>
                        <span class="pct">{p:.2f}%</span>
                    </div>
                    <div class="pg-bar"><span style="width:{min(p, 100):.2f}%"></span></div>
                </div>"""
            render(all_rows)
    with col_f:
        st.download_button(
            "⬇ Download result (.txt)",
            data=build_report(result.get("name", "uploaded image"), probs, class_names),
            file_name="plantguard_result.txt",
            mime="text/plain",
            **BTN_STRETCH,
        )

    render('<div class="pg-spacer"></div>')


def render_conditions_and_tips() -> None:
    chips = "".join(
        f'<span class="pg-cond{" healthy" if d["category"] == "healthy" else ""}">{html.escape(d["name"])}</span>'
        for d in DISEASES.values()
    )
    tips = [
        "Use a single leaf and keep it in focus.",
        "Photograph in good natural light. Avoid harsh shadows and glare.",
        "Use a plain background when possible.",
        "Let the leaf fill most of the frame.",
        "Avoid filters or heavy editing.",
    ]
    tip_items = "".join(f"<li>{html.escape(t)}</li>" for t in tips)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render(f"""
        <div class="pg-card">
            <div class="pg-card-title"><span class="ico">{icon('leaf')}</span>Supported conditions</div>
            <div class="pg-muted">The model is trained to recognize these 10 tomato leaf conditions.</div>
            <div class="pg-chips">{chips}</div>
        </div>
        """)
    with c2:
        render(f"""
        <div class="pg-card">
            <div class="pg-card-title"><span class="ico">{icon('check')}</span>Tips for a good photo</div>
            <ul class="pg-steps-list">{tip_items}</ul>
        </div>
        """)
    render('<div class="pg-spacer"></div>')


def render_model_info() -> None:
    render(f"""
    <div id="about" class="pg-anchor"></div>
    <div class="pg-card">
        <div class="pg-card-title"><span class="ico">{icon('cpu')}</span>Model information</div>
        <div class="pg-muted">Technical details of the model behind PlantGuard AI.</div>
        <div class="pg-model-grid">
            <div class="pg-model-item"><span>AI Model</span><b>MobileNetV2</b></div>
            <div class="pg-model-item"><span>Technique</span><b>Transfer Learning</b></div>
            <div class="pg-model-item"><span>Input Size</span><b>224 × 224</b></div>
            <div class="pg-model-item"><span>Classes</span><b>10 Tomato Leaf Conditions</b></div>
            <div class="pg-model-item"><span>Base Model</span><b>ImageNet pretrained</b></div>
            <div class="pg-model-item"><span>Base Layers</span><b>Frozen during initial training</b></div>
        </div>
    </div>
    <div class="pg-spacer"></div>
    """)


def render_how_it_works() -> None:
    render(f"""
    <div id="how-it-works" class="pg-anchor"></div>
    <div class="pg-section-title">How It Works</div>
    <div class="pg-section-sub">From photo to prediction in three steps.</div>
    <div class="pg-flow">
        <div class="pg-step">
            <div class="num">01</div>
            <div class="ico">{icon('upload')}</div>
            <h3>Upload</h3>
            <p>Upload a tomato leaf image.</p>
        </div>
        <div class="pg-arrow">→</div>
        <div class="pg-step">
            <div class="num">02</div>
            <div class="ico">{icon('chip')}</div>
            <h3>AI Analysis</h3>
            <p>MobileNetV2 processes the image.</p>
        </div>
        <div class="pg-arrow">→</div>
        <div class="pg-step">
            <div class="num">03</div>
            <div class="ico">{icon('check')}</div>
            <h3>Prediction</h3>
            <p>The system returns the predicted condition and confidence.</p>
        </div>
    </div>
    """)


def render_footer() -> None:
    render("""
    <div class="pg-footer">
        <div class="brand">PlantGuard AI</div>
        <p>AI-powered tomato leaf disease detection.</p>
        <p>Built with Python • TensorFlow • MobileNetV2 • Streamlit</p>
        <p class="edu">Educational project — AI predictions should be verified with appropriate
        agricultural expertise.</p>
    </div>
    """)


def scroll_to_results() -> None:
    """Best-effort smooth scroll to the results after an analysis."""
    try:
        import streamlit.components.v1 as components
        components.html(
            "<script>const el=window.parent.document.getElementById('results');"
            "if(el){el.scrollIntoView({behavior:'smooth',block:'start'});}</script>",
            height=0,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
def main() -> None:
    render(CSS)

    model, class_names, load_error = load_assets()
    model_ready = model is not None

    render_navbar(model_ready)

    # ── Hero + upload ──
    hero_left, hero_right = st.columns([1.05, 0.95], gap="large")

    with hero_left:
        render_hero_text()

    image = None
    file_key = None
    file_name = None
    file_data = None
    uploaded = None

    with hero_right:
        with st.container(border=True, key="upload_card"):
            render(f"""
            <div class="pg-card-title"><span class="ico">{icon('upload')}</span>Analyze Your Tomato Leaf</div>
            <div class="pg-muted">Upload a clear image of a tomato leaf to receive an AI-powered prediction.</div>
            """)

            uploaded = st.file_uploader(
                "Upload a tomato leaf image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            # Image source: an uploaded file wins, otherwise a chosen sample leaf
            sample_path = st.session_state.get("sample_path")
            if uploaded is not None:
                file_name, file_data = uploaded.name, uploaded.getvalue()
            elif sample_path and Path(sample_path).exists():
                try:
                    file_name, file_data = Path(sample_path).name, Path(sample_path).read_bytes()
                except Exception:
                    logger.exception("Could not read sample image")
                    st.session_state.pop("sample_path", None)

            if file_data is not None:
                try:
                    img = Image.open(io.BytesIO(file_data))
                    img.load()
                    image = img.convert("RGB")
                    file_key = f"{file_name}-{len(file_data)}"
                except Exception:
                    logger.exception("Could not read image")
                    st.error("This file couldn't be read as an image. "
                             "Please upload a valid JPG, JPEG or PNG file.")

            if image is not None:
                done = (st.session_state.get("result") or {}).get("key") == file_key
                chip = ('<span class="pg-chip done">Analysis complete</span>' if done
                        else '<span class="pg-chip ready">Ready to analyze</span>')
                st.image(image, **IMG_STRETCH)
                source = "" if uploaded is not None else " • sample leaf"
                render(f"""
                <div class="pg-file">
                    <div><b>{html.escape(file_name)}</b>
                    <small>{image.width} × {image.height} px • {len(file_data) / 1024:.0f} KB{source}</small></div>
                    {chip}
                </div>
                """)
            else:
                render('<div class="pg-note" style="margin-top:.9rem">No image yet. '
                       'Upload a JPG, JPEG or PNG to get started.</div>')

            # Sample leaves (from test_images/), shown until an image is chosen
            samples = get_samples(4)
            if samples and image is None:
                render('<div class="pg-sample-title">No leaf photo handy? Try a sample</div>'
                       '<div class="pg-muted" style="margin-bottom:.6rem">Real images from the project\'s test set.</div>')
                with st.container(key="samples"):
                    cols = st.columns(len(samples))
                    for i, (col, path) in enumerate(zip(cols, samples), start=1):
                        with col:
                            st.image(str(path), **IMG_STRETCH)
                            if st.button(f"Sample {i}", key=f"sample_{i}", **BTN_STRETCH):
                                st.session_state["sample_path"] = str(path)
                                st.rerun()
            elif uploaded is None and image is not None and st.session_state.get("sample_path"):
                if st.button("Remove sample", key="remove_sample", **BTN_STRETCH):
                    st.session_state.pop("sample_path", None)
                    st.rerun()

            if load_error:
                render(
                    f'<div class="pg-disclaimer" style="background:var(--rose-bg);color:var(--rose)">'
                    f'{load_error}</div>'
                )

            if image is not None:
                analyze = st.button("🔍 Analyze Leaf", type="primary", disabled=not model_ready, **BTN_STRETCH)
                if analyze:
                    try:
                        with st.spinner("Analyzing leaf…"):
                            probs = run_prediction(model, image)
                        st.session_state["result"] = {
                            "key": file_key, "name": file_name, "probs": probs.tolist(),
                        }
                        st.session_state["just_analyzed"] = True
                        st.rerun()
                    except Exception:
                        logger.exception("Prediction failed")
                        st.error("Something went wrong while analyzing this image. "
                                 "Please try again with a different photo.")

    # ── Results (only for the currently uploaded image) ──
    result = st.session_state.get("result")
    if image is not None and result and result.get("key") == file_key and model_ready:
        render('<div class="pg-spacer" style="height:1rem"></div>')
        try:
            render_result(result, class_names)
        except Exception:
            logger.exception("Rendering results failed")
            st.error("The result could not be displayed. Please run the analysis again.")
        if st.session_state.pop("just_analyzed", False):
            scroll_to_results()
    else:
        render('<div class="pg-spacer"></div>')

    render_conditions_and_tips()
    render_how_it_works()
    render('<div class="pg-spacer"></div>')
    render_model_info()
    render_footer()


main()