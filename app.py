from pathlib import Path
import sys, joblib, pandas as pd, numpy as np, streamlit as st
import plotly.express as px
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.neuromargin.audio import inspect_audio

st.set_page_config(page_title="NeuroMargin | Voice Research",page_icon="🧠",layout="wide")
st.title("🧠 NeuroMargin")
st.caption("Voice-based Parkinson's research prototype | WAV analysis, SVM and mathematical visualizations")
model_path=ROOT/"models/voice_model.joblib"
if not model_path.exists():
    st.error("The model has not been trained yet. Close the dashboard and run START_NEUROMARGIN.bat again.")
    st.stop()
bundle=joblib.load(model_path)
model=bundle["model"]; cols=bundle["feature_columns"]
pages=st.tabs(["Voice analysis","Results & metrics","Acoustic features","Methodology & limitations"])
with pages[0]:
    st.subheader("Upload a WAV recording")
    st.write("Upload a clear voice recording. The model accepts mono or stereo WAV and resamples it to 16 kHz mono.")
    audio=st.file_uploader("Choose WAV file",type=["wav"])
    if audio:
        try:
            y,meta,feat=inspect_audio(audio)
            row=pd.DataFrame([feat]).reindex(columns=cols,fill_value=0)
            score=float(model.decision_function(row)[0])
            probs=model.predict_proba(row)[0]
            label=int(model.predict(row)[0])
            st.audio(audio)
            a,b,c=st.columns(3); a.metric("Duration",f"{meta['duration_seconds']} s"); b.metric("Original sample rate",f"{meta['original_sample_rate']} Hz"); c.metric("Model sample rate","16,000 Hz")
            st.divider()
            st.subheader("Experimental model output")
            if label==1: st.error("PD-associated voice pattern")
            else: st.success("Healthy-control-associated voice pattern")
            st.metric("Model score",f"{score:.4f}")
            st.metric("Model probability output (not clinically calibrated)",f"{probs[label]*100:.1f}%")
            st.warning("Research demonstration only. This is not a medical diagnosis or a validated screening test. The training corpus is small (37 participants), and a voice sample can be affected by language, age, sex, microphone, room acoustics, task, and recording quality. An arbitrary WAV may be outside the training distribution.")
            with st.expander("Inspect extracted acoustic features"):
                st.dataframe(pd.DataFrame([feat]).T.rename(columns={0:"value"}),use_container_width=True)
        except Exception as e: st.error(f"Could not analyze this WAV: {e}")
with pages[1]:
    st.subheader("Participant-independent evaluation")
    m=ROOT/"results/tables/model_comparison.csv"
    if m.exists():
        df=pd.read_csv(m); st.dataframe(df,use_container_width=True)
        st.caption("Out-of-fold metrics use participant-grouped splits so recordings from one person do not appear in both training and test folds.")
    else: st.info("Run training first.")
    for fn,cap in [("model_comparison.png","Model comparison"),("confusion_matrix.png","Out-of-fold confusion matrix"),("roc_curve.png","Out-of-fold ROC curve")]:
        p=ROOT/"results/figures"/fn
        if p.exists(): st.image(str(p),caption=cap,use_container_width=True)
with pages[2]:
    st.subheader("Acoustic and mathematical visualizations")
    for fn,cap in [("class_distribution.png","Dataset class distribution"),("pca_features.png","PCA projection of acoustic feature vectors"),("feature_importance.png","Random Forest feature importance"),("cg_convergence.png","Conjugate Gradient residual convergence")]:
        p=ROOT/"results/figures"/fn
        if p.exists(): st.image(str(p),caption=cap,use_container_width=True)
    p=ROOT/"results/tables/dataset_summary.csv"
    if p.exists(): st.dataframe(pd.read_csv(p),use_container_width=True)
with pages[3]:
    st.subheader("Dataset and method")
    st.markdown("""
**Training corpus:** Mobile Device Voice Recordings at King's College London (MDVR-KCL), Zenodo record 2867216. The corpus contains English read-text and spontaneous-dialogue WAV recordings from 21 healthy controls and 16 participants with Parkinson's disease (37 participants total).

**Pipeline:** audio decoding → mono conversion → 16 kHz resampling → fixed-size MFCC/delta/spectral/pitch feature extraction → standardized RBF-SVM.

**Validation:** StratifiedGroupKFold participant-disjoint out-of-fold evaluation. Model comparisons include RBF-SVM, logistic regression, Random Forest and Extra Trees.

**Mathematical components:** PCA visualization and a Conjugate Gradient numerical optimization demonstration are included. The SVM uses a regularized kernel decision boundary.
""")
    st.warning("This is an educational research prototype, not a diagnostic device. Validation on 37 participants does not establish generalization to all populations, languages, recording devices, or speech tasks. Use clinical assessment for health decisions.")
    st.markdown("Dataset citation: Jaeger, H., Trivedi, D., & Stadtschnitzer, M. (2019). MDVR-KCL. Zenodo. https://doi.org/10.5281/zenodo.2867216 (CC BY 4.0).")
