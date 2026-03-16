import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Prediction App",
    page_icon="🧬",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("miracle_ai_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("encoders.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("**Missing file:** `miracle_ai_model.pkl` not found. Place it in the same directory and restart.")
    st.stop()

try:
    encoders     = load_encoders()
    use_encoders = True
except FileNotFoundError:
    st.warning("**`encoders.pkl` not found.** Assuming all features are numeric.")
    use_encoders = False

# ── Feature columns ───────────────────────────────────────────
TEXT_COLUMNS = [
    'microrna',
    'microrna_group_simplified',
    'parasite',
    'organism',
    'infection',
    'cell type',
]

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🧬 AI Prediction App")
st.markdown("Enter the features below to predict **is_upregulated**.")
st.divider()

# ══════════════════════════════════════════════════════════════
# INPUT FIELDS  (2-column grid, alternating)
# ══════════════════════════════════════════════════════════════
st.subheader("Experimental features")

col1, col2 = st.columns(2)
inputs      = {}
grid_cols   = [col1, col2, col1, col2, col1, col2]

for field_col, col_name in zip(grid_cols, TEXT_COLUMNS):
    with field_col:
        inputs[col_name] = st.text_input(
            col_name,
            placeholder=f"Enter {col_name}",
        )

with col1:
    inputs['time'] = st.number_input(
        "time",
        min_value=0,
        step=1,
        value=0,
        help="Hours post-infection.",
    )

st.divider()

# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════
if st.button("Predict", type="primary"):

    data        = []
    unseen_cols = []

    # ── Encode categoricals ───────────────────────────────────
    for col_name in TEXT_COLUMNS:
        val = inputs[col_name]
        if use_encoders:
            try:
                encoded = encoders[col_name].transform([val])[0]
            except Exception:
                encoded = 0
                unseen_cols.append(col_name)
            data.append(encoded)
        else:
            data.append(val)

    # ── Append numeric ────────────────────────────────────────
    data.append(inputs['time'])
    X = np.array([data])

    # ── Run model ─────────────────────────────────────────────
    prediction = model.predict(X)
    pred_label = int(prediction[0])
    proba      = model.predict_proba(X)[0]
    prob_up    = proba[1]
    prob_down  = proba[0]
    confidence = proba[pred_label]

    # ── Result label ──────────────────────────────────────────
    st.subheader("Prediction")

    if pred_label == 1:
        st.success("## ⬆ Upregulated")
    else:
        st.error("## ⬇ Downregulated")

    # ── Probability breakdown ─────────────────────────────────
    st.markdown("**Probability breakdown:**")
    p_col1, p_col2 = st.columns(2)
    p_col1.metric("Upregulated",   f"{prob_up   * 100:.1f}%")
    p_col2.metric("Downregulated", f"{prob_down * 100:.1f}%")

    st.progress(
        float(prob_up),
        text=f"↑ {prob_up*100:.1f}%  |  ↓ {prob_down*100:.1f}%"
    )

    st.info(f"Confidence: **{confidence * 100:.1f}%**")

    # ── Unseen value warning ──────────────────────────────────
    if unseen_cols:
        st.warning(
            "⚠ The following columns had values never seen in training: "
            + ", ".join(f"`{c}`" for c in unseen_cols)
        )

    # ── Input summary expander ────────────────────────────────
    with st.expander("Input summary"):
        summary_data = {col_name: [inputs[col_name]] for col_name in TEXT_COLUMNS}
        summary_data['time'] = [inputs['time']]
        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True,
        )
