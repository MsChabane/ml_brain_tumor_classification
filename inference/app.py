import streamlit as st
import json
from PIL import Image
import os


CONFIG = {
    "CNN": {
        "model_dir": "./reports/cnn",
        "data_source": "Kaggle – Brain Tumor MRI Dataset (Image-based)"
    },
    "ML": {
        "model_dir": "./reports/ml",
        "data_source": "Built dataset from the previous project "
    }
}



st.title("🧠 Brain Tumor Classification – Evaluation Dashboard")


tab_cnn, tab_ml = st.tabs(["📷 CNN Model", "📄 Machine Learning Model"])



def show_results(model_name):
    model_dir = CONFIG[model_name]["model_dir"]
    data_source = CONFIG[model_name]["data_source"]

    metrics_path = os.path.join(model_dir, "result.json")
    roc_path = os.path.join(model_dir, "roc_curve.png")
    cm_path = os.path.join(model_dir, "confusion_matrix.png")


    st.subheader("📌 Data Information")
    st.info(data_source)


    st.subheader("📊 Model Metrics")

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", round(metrics.get("accuracy", 0), 4))
            col2.metric("F1 Score", round(metrics.get("f1_score", 0), 4))
            col3.metric("Recall", round(metrics.get("recall", 0), 4))
            col4.metric("Precision", round(metrics.get("precision", 0), 4))

        except Exception as e:
            st.error(f"❌ Error reading metrics: {e}")

    else:
        st.error(f"❌ Metrics file not found: {metrics_path}")


    st.subheader("📈 ROC Curve")

    if os.path.exists(roc_path):
        try:
            roc_img = Image.open(roc_path)
            st.image(roc_img, caption=f"{model_name} ROC Curve", use_column_width=True)
        except:
            st.error("❌ Cannot load ROC image!")
    else:
        st.error(f"❌ ROC image not found at: {roc_path}")

    
    st.subheader("🔳 Confusion Matrix")

    if os.path.exists(cm_path):
        try:
            cm_img = Image.open(cm_path)
            st.image(cm_img, caption=f"{model_name} Confusion Matrix", use_column_width=True)
        except:
            st.error("❌ Cannot load confusion matrix image!")
    else:
        st.error(f"❌ Confusion Matrix not found at: {cm_path}")



with tab_cnn:
    show_results("CNN")

with tab_ml:
    show_results("ML")
