import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="miRNA Predictor", page_icon="🧬")
st.title("🧬 miRNA Expression Predictor")
st.markdown("Load your saved model and predict whether a miRNA is **UP** or **DOWN** regulated.")

# ── Helper functions (copied from your original code) ──────────────────────────
def strip_prefix(name):
    for prefix in ['hsa-', 'mmu-', 'cfa-']:
        if str(name).startswith(prefix):
            return str(name)[len(prefix):]
    return str(name)

def build_row(mirna, group, parasite, organism_label, cell, time_val):
    mirna_c = strip_prefix(mirna.strip().lower())
    group_c = strip_prefix(group.strip().lower())
    para_c  = parasite.strip().lower().replace(' ', '')
    org_c   = organism_label.strip().lower()
    cell_c  = cell.strip().lower()
    cell_c  = 'bmdm' if 'bmdm' in cell_c else cell_c
    org_num = 1 if org_c == 'human' else 0
    t_bkt   = 'very_early' if time_val <= 6 else ('transition' if time_val <= 12 else 'late')
    return pd.DataFrame([{
        'mirna_organism'           : mirna_c + "_" + org_c,
        'microrna_group_simplified': group_c,
        'family_organism'          : group_c + "_" + org_c,
        'family_parasite'          : group_c + "_" + para_c,
        'family_cell'              : group_c + "_" + cell_c,
        'cell_parasite_org'        : cell_c + "_" + para_c + "_" + org_c,
        'time_bucket'              : t_bkt,
        'organism'                 : org_num,
        'time'                     : time_val,
    }])

# ── Load model ─────────────────────────────────────────────────────────────────
st.sidebar.header("1️⃣ Load Model")
uploaded = st.sidebar.file_uploader(r"C:\Users\MSI\Desktop\PFE\PHASE 2Upload model_final_v5.pkl", type="pkl")

model_data = None
if uploaded:
    model_data = pickle.load(uploaded)
    st.sidebar.success(f"✅ Model loaded  (v{model_data.get('version','?')})")
    st.sidebar.write(f"Training rows : {model_data.get('training_rows','?')}")
    st.sidebar.write(f"CV AUC        : {model_data.get('final_auc', '?'):.4f}")
    st.sidebar.write(f"LOMO AUC      : {model_data.get('lomo_auc', '?'):.4f}")

# ── Input form ─────────────────────────────────────────────────────────────────
st.header("2️⃣ Enter Inputs")

col1, col2 = st.columns(2)

with col1:
    mirna   = st.text_input("miRNA name",          value="mir-146a-3p",   help="e.g. mir-146a-3p or hsa-mir-146a-3p")
    group   = st.text_input("miRNA family / group", value="mir-146a",      help="e.g. mir-146a")
    parasite= st.text_input("Parasite",             value="L.major",       help="e.g. L.major, L.amazonensis")

with col2:
    organism= st.selectbox("Organism", ["Human", "Mouse"])
    cell    = st.text_input("Cell type", value="THP-1", help="e.g. THP-1, BMDM, macrophage")
    time_val= st.number_input("Time point (hours)", min_value=0, max_value=200, value=24, step=1)

# ── Predict ────────────────────────────────────────────────────────────────────
st.header("3️⃣ Predict")

if st.button("🔮 Predict", use_container_width=True):
    if model_data is None:
        st.error("Please upload the model file first (sidebar).")
    else:
        pipeline = model_data['pipeline']
        row      = build_row(mirna, group, parasite, organism, cell, time_val)
        proba    = pipeline.predict_proba(row)[0]
        pred     = 'UP ⬆️' if proba[1] >= 0.5 else 'DOWN ⬇️'
        conf     = max(proba)

        color = "green" if proba[1] >= 0.5 else "red"
        st.markdown(f"### Result: :{color}[**{pred}**]")

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction",    pred)
        c2.metric("Confidence",    f"{conf*100:.1f}%")
        c3.metric("P(UP) / P(DOWN)", f"{proba[1]*100:.1f}% / {proba[0]*100:.1f}%")

        known = model_data.get('known_mirnas', set())
        clean_mirna = strip_prefix(mirna.strip().lower())
        if clean_mirna not in known:
            st.info(f"ℹ️ **{mirna}** is not in the training set — prediction is based on family & context features.")

        with st.expander("See input details sent to model"):
            st.dataframe(row)

# ── Batch prediction ───────────────────────────────────────────────────────────
st.divider()
st.header("4️⃣ Batch Prediction (optional)")
st.markdown("Upload a CSV with columns: `mirna, group, parasite, organism, cell_type, time`")

batch_file = st.file_uploader("Upload CSV", type="csv")
if batch_file and model_data:
    bdf = pd.read_csv(batch_file)
    st.write("Preview:", bdf.head())
    if st.button("Run batch prediction"):
        pipeline = model_data['pipeline']
        results  = []
        for _, r in bdf.iterrows():
            row   = build_row(r['mirna'], r['group'], r['parasite'],
                              r['organism'], r['cell_type'], int(r['time']))
            proba = pipeline.predict_proba(row)[0]
            results.append({
                'mirna'     : r['mirna'],
                'time'      : r['time'],
                'prediction': 'UP' if proba[1] >= 0.5 else 'DOWN',
                'confidence': f"{max(proba)*100:.1f}%",
                'P(UP)'     : f"{proba[1]*100:.1f}%",
            })
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
        csv = res_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download results", csv, "predictions.csv", "text/csv")
