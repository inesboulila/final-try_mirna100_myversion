# ================================================================
#  MIRNA PREDICTION APP — FINAL VERSION
#  Gradient Boosting + Calibration + SHAP
#  Model: model_final_GB.pkl
#  Run: streamlit run app_final.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
import shap

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="miRNA Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main {
        background-color: #0a0e1a;
        color: #e8edf5;
    }

    .stApp {
        background-color: #0a0e1a;
    }

    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #00d4aa !important;
        letter-spacing: -0.5px;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f1628 0%, #151d35 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(0, 212, 170, 0.05);
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #6b7fa3;
        margin-top: 4px;
    }

    .up-color   { color: #00d4aa; }
    .down-color { color: #ff4b6e; }
    .neutral    { color: #a0aec0; }

    .prediction-box-up {
        background: linear-gradient(135deg, #003d30 0%, #004d3c 100%);
        border: 2px solid #00d4aa;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 212, 170, 0.15);
    }

    .prediction-box-down {
        background: linear-gradient(135deg, #3d001a 0%, #4d0022 100%);
        border: 2px solid #ff4b6e;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 75, 110, 0.15);
    }

    .tag {
        display: inline-block;
        background: #1e3a5f;
        color: #7eb8f7;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px;
        letter-spacing: 0.5px;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #6b7fa3;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
        margin: 20px 0 12px 0;
    }

    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #a0aec0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #0097ff);
        color: #0a0e1a;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        width: 100%;
        letter-spacing: 1px;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.3);
    }

    .sidebar .sidebar-content {
        background-color: #0d1220;
    }

    div[data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e3a5f;
    }

    .stSlider > div > div {
        color: #00d4aa;
    }

    .warning-box {
        background: #1a1200;
        border: 1px solid #f0a500;
        border-radius: 8px;
        padding: 12px 16px;
        color: #f0a500;
        font-size: 0.85rem;
    }

    .info-box {
        background: #001a2e;
        border: 1px solid #0097ff;
        border-radius: 8px;
        padding: 12px 16px;
        color: #7eb8f7;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  LOAD MODEL AND DATA
# ════════════════════════════════════════════════════════════════
MODEL_PATH = "model_final_GB.pkl"
DATA_PATH  = "microrna_top100_balanced_final.xlsx"

@st.cache_resource
def load_model_and_data():
    """Load model, fit calibration, prepare SHAP explainer."""

    with open(MODEL_PATH, 'rb') as f:
        saved = pickle.load(f)

    pipeline   = saved['pipeline']
    features   = saved['features']
    target_cols = saved['target_cols']
    smoothing  = saved['smoothing']

    # Load training data for calibration and SHAP
    df = pd.read_excel(DATA_PATH)

    # Apply same cleaning
    text_cols = ['microrna', 'microrna_group_simplified',
                 'parasite', 'organism', 'infection', 'cell type']
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df['parasite']  = df['parasite'].str.replace(r'\s+', '', regex=True)
    df['cell type'] = df['cell type'].str.replace(
        r'bmdm.*', 'bmdm', regex=True).str.replace(
        r'blood serum.*', 'other', regex=True)
    df = df.drop(columns=['infection'])

    def strip_prefix(name):
        for prefix in ['hsa-', 'mmu-', 'cfa-']:
            if str(name).startswith(prefix):
                return str(name)[len(prefix):]
        return str(name)

    df['microrna']                  = df['microrna'].apply(strip_prefix)
    df['microrna_group_simplified'] = df['microrna_group_simplified'].apply(strip_prefix)
    df['organism_label']            = df['organism']
    df['organism']                  = df['organism'].map({'human': 1, 'mouse': 0})
    df['super_scenario']            = df['parasite'] + "_" + df['cell type'] + "_" + df['organism_label']
    df['mirna_organism']            = df['microrna'] + "_" + df['organism_label']
    df['parasite_solo']             = df['parasite']
    df['time_phase']                = df['time'].apply(
        lambda x: 'early' if x <= 8 else ('mid' if x <= 12 else 'late'))

    X = df[features]
    y = df['is_upregulated']

    # Fit calibration wrapper using cross-validation
    # This makes the probability scores scientifically trustworthy
    calibrated = CalibratedClassifierCV(
        pipeline, method='isotonic', cv=5
    )
    calibrated.fit(X, y)

    # Fit plain pipeline on full data for SHAP
    pipeline.fit(X, y)
    X_transformed = pipeline.named_steps['pre'].transform(X)
    gb_model      = pipeline.named_steps['model']
    explainer     = shap.TreeExplainer(gb_model)

    known_mirnas = set(df['microrna'].unique())

    return {
        'calibrated' : calibrated,
        'pipeline'   : pipeline,
        'explainer'  : explainer,
        'X_transformed': X_transformed,
        'features'   : features,
        'target_cols': target_cols,
        'smoothing'  : smoothing,
        'known_mirnas': known_mirnas,
        'df'         : df,
    }

# ════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def strip_prefix(name):
    for prefix in ['hsa-', 'mmu-', 'cfa-']:
        if str(name).startswith(prefix):
            return str(name)[len(prefix):]
    return str(name)

def build_input_row(mirna, mirna_group, parasite, organism_label,
                    cell_type, time_val):
    """Build the feature row exactly as the model expects."""
    mirna_clean  = strip_prefix(mirna.strip().lower())
    group_clean  = strip_prefix(mirna_group.strip().lower())
    para_clean   = parasite.strip().lower().replace(' ', '')
    org_clean    = organism_label.strip().lower()
    cell_clean   = cell_type.strip().lower()
    cell_clean   = 'bmdm' if 'bmdm' in cell_clean else cell_clean

    super_scenario = para_clean + "_" + cell_clean + "_" + org_clean
    mirna_organism = mirna_clean + "_" + org_clean
    parasite_solo  = para_clean
    time_phase     = 'early' if time_val <= 8 else ('mid' if time_val <= 12 else 'late')
    organism_num   = 1 if org_clean == 'human' else 0

    return pd.DataFrame([{
        'microrna'                 : mirna_clean,
        'microrna_group_simplified': group_clean,
        'super_scenario'           : super_scenario,
        'mirna_organism'           : mirna_organism,
        'parasite_solo'            : parasite_solo,
        'time_phase'               : time_phase,
        'organism'                 : organism_num,
        'time'                     : time_val,
    }])


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧬 miRNA Predictor")
    st.markdown("---")

    st.markdown("""
    <div class='section-header'>Model Info</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-card'>
        <div class='metric-value up-color'>81.46%</div>
        <div class='metric-label'>Accuracy</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value up-color'>0.9047</div>
        <div class='metric-label'>AUC Score</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header'>Validation</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    ✓ Repeated 5-Fold CV (25 folds)<br>
    ✓ Leave-One-miRNA-Out tested<br>
    ✓ New miRNA gap: 0.004 AUC<br>
    ✓ Calibrated confidence scores<br>
    ✓ No data leakage
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header'>Algorithm</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='tag'>Gradient Boosting</div>
    <div class='tag'>Target Encoding</div>
    <div class='tag'>Isotonic Calibration</div>
    <div class='tag'>SHAP Explainability</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header'>Best Parameters</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-card'>
    <span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#a0aec0'>
    n_estimators: 100<br>
    max_depth: 5<br>
    learning_rate: 0.15<br>
    min_samples_leaf: 12<br>
    subsample: 0.8<br>
    smoothing: 30
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header'>About</div>
    """, unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#6b7fa3;font-size:0.8rem'>"
        "Predicts miRNA up/downregulation in immune cells "
        "infected with Leishmania parasites.<br><br>"
        "Mir-Acle Biotech — PFE Project 2026"
        "</span>",
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ════════════════════════════════════════════════════════════════

st.markdown("# miRNA Regulation Predictor")
st.markdown(
    "<span style='color:#6b7fa3;font-size:0.95rem'>"
    "Predicts whether a miRNA will be upregulated or downregulated "
    "in immune cells infected with Leishmania — powered by Gradient Boosting "
    "with calibrated confidence scores and SHAP explainability."
    "</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# Load model
with st.spinner("Loading model and calibration layer..."):
    try:
        model_data = load_model_and_data()
        st.success("Model loaded successfully", icon="✓")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ── Input form ─────────────────────────────────────────────────
st.markdown("""
<div class='section-header'>Input — Experimental Conditions</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    mirna_input = st.text_input(
        "miRNA Name",
        value="miR-146a",
        help="Enter miRNA name with or without species prefix (hsa-, mmu-)"
    )
    mirna_group = st.text_input(
        "miRNA Group / Family",
        value="miR-146a",
        help="Simplified group name e.g. miR-146a, let-7a"
    )

with col2:
    parasite = st.selectbox(
        "Parasite Species",
        options=[
            "L.major", "L.donovani", "L.amazonensis", "L.infantum"
        ]
    )
    organism = st.selectbox(
        "Host Organism",
        options=["Human", "Mouse"]
    )

with col3:
    cell_type = st.selectbox(
        "Cell Type",
        options=["PBMC", "THP-1", "BMDM", "RAW 264.7", "HMDM", "Other"]
    )
    time_point = st.selectbox(
        "Time Point (hours post infection)",
        options=[3, 6, 8, 12, 24, 30, 48],
        index=4
    )

st.markdown("")
predict_btn = st.button("🔬  PREDICT REGULATION", use_container_width=True)

# ════════════════════════════════════════════════════════════════
#  PREDICTION
# ════════════════════════════════════════════════════════════════

if predict_btn:

    st.markdown("---")
    st.markdown("""
    <div class='section-header'>Prediction Result</div>
    """, unsafe_allow_html=True)

    # Build input
    input_row = build_input_row(
        mirna_input, mirna_group, parasite,
        organism, cell_type, time_point
    )

    # Check if miRNA is known or new
    mirna_clean   = strip_prefix(mirna_input.strip().lower())
    is_known      = mirna_clean in model_data['known_mirnas']

    # Calibrated prediction
    try:
        proba_cal = model_data['calibrated'].predict_proba(input_row)[0]
        pred_cal  = 1 if proba_cal[1] >= 0.5 else 0
        conf_up   = proba_cal[1]
        conf_down = proba_cal[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # ── Main result display ────────────────────────────────────
    col_pred, col_conf, col_info = st.columns([2, 1, 1])

    with col_pred:
        if pred_cal == 1:
            st.markdown(f"""
            <div class='prediction-box-up'>
                <div style='font-size:3rem'>▲</div>
                <div style='font-family:IBM Plex Mono;font-size:1.8rem;
                            font-weight:600;color:#00d4aa;margin:8px 0'>
                    UPREGULATED
                </div>
                <div style='color:#6b7fa3;font-size:0.85rem'>
                    {mirna_input.upper()} is predicted to increase
                    expression upon infection
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='prediction-box-down'>
                <div style='font-size:3rem'>▼</div>
                <div style='font-family:IBM Plex Mono;font-size:1.8rem;
                            font-weight:600;color:#ff4b6e;margin:8px 0'>
                    DOWNREGULATED
                </div>
                <div style='color:#6b7fa3;font-size:0.85rem'>
                    {mirna_input.upper()} is predicted to decrease
                    expression upon infection
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_conf:
        st.markdown(f"""
        <div class='metric-card' style='text-align:center'>
            <div class='metric-value up-color'>{conf_up*100:.1f}%</div>
            <div class='metric-label'>Confidence UP</div>
        </div>
        <div class='metric-card' style='text-align:center'>
            <div class='metric-value down-color'>{conf_down*100:.1f}%</div>
            <div class='metric-label'>Confidence DOWN</div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        known_label = "Known miRNA" if is_known else "New miRNA"
        known_color = "#00d4aa" if is_known else "#f0a500"
        known_note  = ("Seen during training" if is_known
                       else "Never seen during training —<br>prediction based on biological context")

        time_ph = ('Early (≤8h)' if time_point <= 8 else
                   'Mid (≤12h)'  if time_point <= 12 else 'Late (>12h)')

        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-family:IBM Plex Mono;font-size:0.9rem;
                        color:{known_color};font-weight:600'>
                {known_label}
            </div>
            <div style='color:#6b7fa3;font-size:0.75rem;margin-top:6px'>
                {known_note}
            </div>
        </div>
        <div class='metric-card'>
            <div style='font-family:IBM Plex Mono;font-size:0.8rem;
                        color:#7eb8f7'>
                Time Phase: {time_ph}
            </div>
            <div style='font-family:IBM Plex Mono;font-size:0.8rem;
                        color:#7eb8f7;margin-top:4px'>
                Calibrated: ✓
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Warning for low confidence ─────────────────────────────
    if max(conf_up, conf_down) < 0.65:
        st.markdown("""
        <div class='warning-box'>
        ⚠ Low confidence prediction. The model has limited information
        about this specific miRNA in this biological context.
        Treat this result with caution and consider experimental validation.
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP explanation ───────────────────────────────────────
    st.markdown("""
    <div class='section-header' style='margin-top:24px'>
    SHAP — Why Did the Model Make This Prediction?
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<span style='color:#6b7fa3;font-size:0.85rem'>"
        "Each bar shows how much a feature pushed the prediction "
        "toward Upregulated (green) or Downregulated (red)."
        "</span>",
        unsafe_allow_html=True
    )

    try:
        input_transformed = model_data['pipeline'].named_steps['pre'].transform(input_row)
        shap_vals         = model_data['explainer'].shap_values(input_transformed)
        shap_up_vals      = (shap_vals[1][0] if isinstance(shap_vals, list)
                             else shap_vals[0])

        features    = model_data['features']
        shap_pairs  = list(zip(features, shap_up_vals))
        shap_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0f1628')
        ax.set_facecolor('#0f1628')

        names  = [p[0] for p in shap_sorted]
        values = [p[1] for p in shap_sorted]
        colors = ['#00d4aa' if v > 0 else '#ff4b6e' for v in values]

        bars = ax.barh(names, values, color=colors,
                       edgecolor='none', height=0.55)

        ax.axvline(0, color='#2a3a5f', linewidth=1.5)
        ax.set_xlabel('SHAP value (impact on prediction)',
                      color='#6b7fa3', fontsize=10)
        ax.set_title(
            f'Feature Contributions for {mirna_input} prediction',
            color='#e8edf5', fontsize=11, fontweight='bold',
            fontfamily='monospace'
        )
        ax.tick_params(colors='#a0aec0', labelsize=9)
        ax.spines['bottom'].set_color('#1e3a5f')
        ax.spines['left'].set_color('#1e3a5f')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

        up_patch   = mpatches.Patch(color='#00d4aa', label='Pushes → Upregulated')
        down_patch = mpatches.Patch(color='#ff4b6e', label='Pushes → Downregulated')
        ax.legend(handles=[up_patch, down_patch],
                  facecolor='#0f1628', edgecolor='#1e3a5f',
                  labelcolor='#a0aec0', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    # ── Input summary ──────────────────────────────────────────
    st.markdown("""
    <div class='section-header' style='margin-top:24px'>Input Summary</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='tag'>miRNA: {mirna_clean}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='tag'>Parasite: {parasite}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='tag'>Organism: {organism}</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='tag'>Cell: {cell_type} | {time_point}h</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  BATCH PREDICTION TAB
# ════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class='section-header'>Batch Prediction — Test Multiple miRNAs at Once</div>
""", unsafe_allow_html=True)

st.markdown(
    "<span style='color:#6b7fa3;font-size:0.85rem'>"
    "Upload a CSV file with columns: microrna, microrna_group_simplified, "
    "parasite, organism, cell type, time"
    "</span>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader("Upload CSV", type=['csv'])

if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
        required = ['microrna', 'microrna_group_simplified',
                    'parasite', 'organism', 'cell type', 'time']
        missing  = [c for c in required if c not in batch_df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            results = []
            for _, row in batch_df.iterrows():
                input_row = build_input_row(
                    str(row['microrna']),
                    str(row['microrna_group_simplified']),
                    str(row['parasite']),
                    str(row['organism']),
                    str(row['cell type']),
                    int(row['time'])
                )
                proba = model_data['calibrated'].predict_proba(input_row)[0]
                pred  = 'Upregulated' if proba[1] >= 0.5 else 'Downregulated'
                results.append({
                    'microrna'     : row['microrna'],
                    'parasite'     : row['parasite'],
                    'organism'     : row['organism'],
                    'cell type'    : row['cell type'],
                    'time'         : row['time'],
                    'prediction'   : pred,
                    'confidence_up': f"{proba[1]*100:.1f}%",
                    'confidence_down': f"{proba[0]*100:.1f}%",
                    'known_mirna'  : str(row['microrna']).lower() in model_data['known_mirnas'],
                })

            result_df = pd.DataFrame(results)

            # Color code the prediction column
            def color_pred(val):
                if val == 'Upregulated':
                    return 'color: #00d4aa; font-weight: bold'
                return 'color: #ff4b6e; font-weight: bold'

            st.dataframe(
                result_df.style.applymap(color_pred, subset=['prediction']),
                use_container_width=True
            )

            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "⬇ Download Results CSV",
                csv_out,
                "mirna_predictions.csv",
                "text/csv"
            )

    except Exception as e:

        st.error(f"Batch prediction error: {e}")


