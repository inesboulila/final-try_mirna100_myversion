# ================================================================
#  miRNA PREDICTION APP — v5 True Final
#  Model : model_final_v5.pkl
#  Run   : streamlit run app_v5.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="miRNA Predictor — v5",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "model_final_v5.pkl"


# ── Helper functions ───────────────────────────────────────────
def strip_prefix(name: str) -> str:
    for prefix in ['hsa-', 'mmu-', 'cfa-']:
        if str(name).startswith(prefix):
            return str(name)[len(prefix):]
    return str(name)


def strip_arm(name: str) -> str:
    name = str(name)
    if name.endswith('-3p') or name.endswith('-5p'):
        return name[:-3]
    return name


def get_time_bucket(hours: int) -> str:
    if hours <= 6:
        return 'very_early'
    if hours <= 12:
        return 'transition'
    return 'late'


def build_input_row(mirna, mirna_group, parasite, organism_label,
                    cell_type, time_val):
    mirna_c = strip_prefix(mirna.strip().lower())
    group_c = strip_prefix(mirna_group.strip().lower())
    para_c  = parasite.strip().lower().replace(' ', '')
    org_c   = organism_label.strip().lower()
    cell_c  = cell_type.strip().lower()
    cell_c  = 'bmdm' if 'bmdm' in cell_c else cell_c
    org_num = 1 if org_c == 'human' else 0
    t_bkt   = get_time_bucket(time_val)

    return pd.DataFrame([{
        'mirna_organism'           : f"{mirna_c}_{org_c}",
        'microrna_group_simplified': group_c,
        'family_organism'          : f"{group_c}_{org_c}",
        'family_parasite'          : f"{group_c}_{para_c}",
        'family_cell'              : f"{group_c}_{cell_c}",
        'cell_parasite_org'        : f"{cell_c}_{para_c}_{org_c}",
        'time_bucket'              : t_bkt,
        'organism'                 : org_num,
        'time'                     : float(time_val),
    }])


# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

try:
    saved = load_model()
except FileNotFoundError:
    st.error(
        f"**Missing file:** `{MODEL_PATH}` not found. "
        "Place it in the same directory and restart."
    )
    st.stop()

pipeline     = saved['pipeline']
known_mirnas = saved['known_mirnas']
final_auc    = saved.get('final_auc',     None)
final_acc    = saved.get('final_accuracy', None)
lomo_auc     = saved.get('lomo_auc',      None)


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🧬 miRNA Regulation Predictor")
st.markdown(
    "Predicts whether a miRNA will be **upregulated** or **downregulated** "
    "in immune cells infected with *Leishmania* — "
    "Gradient Boosting v5 with family-level interaction features."
)
st.divider()

# ── Model performance strip ────────────────────────────────────
if final_auc:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CV AUC",      f"{final_auc:.4f}")
    m2.metric("CV Accuracy", f"{final_acc*100:.2f}%" if final_acc else "—")
    m3.metric("LOMO AUC",    f"{lomo_auc:.4f}" if lomo_auc else "—")
    m4.metric("Training rows", str(saved.get('training_rows', '—')))
    st.divider()


# ══════════════════════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════════════════════
st.subheader("Experimental conditions")

col1, col2, col3 = st.columns(3)

with col1:
    mirna_input = st.text_input(
        "miRNA name",
        value="mir-146a-3p",
        help="Enter miRNA name with or without species prefix (hsa-, mmu-, cfa-) "
             "and with or without arm suffix (-3p / -5p). Both are stripped automatically."
    )
    mirna_group = st.text_input(
        "miRNA group / family",
        value="mir-146a",
        help="Simplified family name e.g. mir-146a, let-7a"
    )

with col2:
    parasite = st.selectbox(
        "Parasite species",
        ["L.major", "L.donovani", "L.amazonensis", "L.infantum"]
    )
    organism = st.selectbox(
        "Host organism",
        ["Human", "Mouse"]
    )

with col3:
    cell_type = st.selectbox(
        "Cell type",
        ["THP-1", "PBMC", "BMDM", "RAW 264.7", "HMDM", "Other"]
    )
    time_point = st.selectbox(
        "Time point (hours post-infection)",
        [3, 6, 8, 12, 24, 30, 48],
        index=3,
        help="≤6h = very early  |  ≤12h = transition  |  >12h = late"
    )

predict_btn = st.button("🔬 Predict regulation", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════
if predict_btn:

    input_row   = build_input_row(mirna_input, mirna_group, parasite,
                                  organism, cell_type, time_point)
    mirna_clean = strip_prefix(mirna_input.strip().lower())
    is_known    = mirna_clean in known_mirnas

    try:
        proba     = pipeline.predict_proba(input_row)[0]
        pred      = 1 if proba[1] >= 0.5 else 0
        prob_up   = float(proba[1])
        prob_down = float(proba[0])
    except Exception as e:
        st.error(f"**Prediction error:** {e}")
        st.stop()

    st.divider()
    st.subheader("Prediction")

    # ── 3-column result block ──────────────────────────────────
    res_col, conf_col, info_col = st.columns([2, 1, 1])

    with res_col:
        if pred == 1:
            st.success(
                f"## ⬆ Upregulated\n"
                f"{mirna_input.upper()} is predicted to increase "
                "expression upon infection."
            )
        else:
            st.error(
                f"## ⬇ Downregulated\n"
                f"{mirna_input.upper()} is predicted to decrease "
                "expression upon infection."
            )

    with conf_col:
        st.metric("Confidence up",   f"{prob_up   * 100:.1f}%")
        st.metric("Confidence down", f"{prob_down * 100:.1f}%")

    with info_col:
        known_label = "Known miRNA" if is_known else "New miRNA"
        known_note  = ("Seen during training" if is_known
                       else "Not seen during training — prediction based on family context")
        time_label  = get_time_bucket(time_point).replace('_', ' ').title()
        st.info(f"**{known_label}**  \n{known_note}")
        st.info(
            f"Time bucket: **{time_label}**  \n"
            f"Organism: **{organism.lower()}** ({1 if organism == 'Human' else 0})"
        )

    # ── Progress bar ───────────────────────────────────────────
    st.progress(
        prob_up,
        text=f"↑ {prob_up*100:.1f}%  |  ↓ {prob_down*100:.1f}%"
    )

    # ── Low confidence warning ─────────────────────────────────
    if max(prob_up, prob_down) < 0.65:
        st.warning(
            "⚠ Low confidence prediction. The model has limited information "
            "about this miRNA in this biological context. "
            "Consider experimental validation."
        )

    # ── Processing log ─────────────────────────────────────────
    mirna_clean_display = strip_prefix(mirna_input.strip().lower())
    st.info(
        f"Model processing log: name interpreted as **{mirna_clean_display}** "
        f"(prefix stripped)  |  family: **{strip_prefix(mirna_group.strip().lower())}**  |  "
        f"time bucket: **{get_time_bucket(time_point)}**"
    )

    # ── Input summary expander ─────────────────────────────────
    with st.expander("Input summary"):
        st.dataframe(input_row, use_container_width=True, hide_index=True)

    # ── Interaction features expander ──────────────────────────
    with st.expander("Derived interaction features"):
        st.caption(
            "These are the engineered columns fed into the model, "
            "built from your inputs."
        )
        display_cols = {
            'mirna_organism':            input_row['mirna_organism'].iloc[0],
            'family_organism':           input_row['family_organism'].iloc[0],
            'family_parasite':           input_row['family_parasite'].iloc[0],
            'family_cell':               input_row['family_cell'].iloc[0],
            'cell_parasite_org':         input_row['cell_parasite_org'].iloc[0],
            'time_bucket':               input_row['time_bucket'].iloc[0],
        }
        st.dataframe(
            pd.DataFrame([display_cols]),
            use_container_width=True,
            hide_index=True
        )