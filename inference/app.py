import streamlit as st
import json
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type:  ignore
import joblib
import pandas as pd

@st.cache_resource
def load_cnn_model():
    model_path = "./models/cnn/model.keras"  
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None

@st.cache_resource
def load_ml_models():
    ml_clf = joblib.load("./models/ml/model.pkl" ) 
    gender_enc = joblib.load("./models/ml/gender_encoder.pkl" ) 
    seizures_enc = joblib.load("./models/ml/seizures_encoder.pkl" ) 
    return ml_clf,gender_enc,seizures_enc
    

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


tab_cnn, tab_ml,tab_cnn_predict,tab_ml_predict = st.tabs(["📷 CNN Model", "📄 Machine Learning Model",'🖼️ Predict With CNN','🖥 Predict With ML'])



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
            st.image(roc_img, caption=f"{model_name} ROC Curve",  width='stretch')
        except:
            st.error("❌ Cannot load ROC image!")
    else:
        st.error(f"❌ ROC image not found at: {roc_path}")

    
    st.subheader("🔳 Confusion Matrix")

    if os.path.exists(cm_path):
        try:
            cm_img = Image.open(cm_path)
            st.image(cm_img, caption=f"{model_name} Confusion Matrix",  width='stretch')
        except:
            st.error("❌ Cannot load confusion matrix image!")
    else:
        st.error(f"❌ Confusion Matrix not found at: {cm_path}")

def preprocess_image(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def ml_predict(features):
    features['gender']=gender_enc.transform(features['gender'])
    features['seizures']=seizures_enc.transform(features['seizures'])
    preds=ml_clf.predict(features)
    proba=ml_clf.predict_proba(features)[0]
    return preds,proba
    
    
    


with tab_cnn_predict:
    st.header("🖼️ Predict Brain Tumor Using CNN Model")

    cnn_model = load_cnn_model()

    if cnn_model is None:
        st.error("❌ CNN model not found! Train the model first.")
    else:
        uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            
            img=tf.keras.preprocessing.image.load_img( uploaded_file, target_size=(224, 224) )
            
            st.image(img, caption="Uploaded Image")

            
            processed = preprocess_image(img)

            with st.spinner("In progress ..."):
                pred = cnn_model.predict(processed)[0][0]

                st.subheader("🔍 Prediction Result")
                if pred >= 0.5:
                    st.error(f"🧠 Tumor Detected (Probability: {pred:.4f})")
                else:
                    st.success(f"✔️ No Tumor Detected (Probability: {1-pred:.4f})")

with tab_ml_predict:
    st.header("🖼️ Predict Brain Tumor Using Machine Learning Model")
    ml_clf,gender_enc,seizures_enc = load_ml_models()
    if not ml_clf or not gender_enc or not seizures_enc:
        st.error("❌ Machine Learning models are not found!...")
        
    st.title("🧠 Brain Tumor Prediction Form")

    with st.form(key="tumor_form"):
        
        col_1,col_2= st.columns(2)
        with col_1:
            age = st.number_input("Age", min_value=0, max_value=130, value=30)
            gender = st.selectbox("Gender", options=["M", "F"],)
            antecedents = st.number_input("Antecedents", min_value=0, max_value=3,step=1, value=0)
            headaches = st.number_input("Headaches", min_value=0, max_value=2, value=0,step=1)
            seizures = st.selectbox("Seizures (e.g., C, M, etc.)",options=['M',"TC","S","C"] )
            fatigue = st.number_input("Fatigue", min_value=0, max_value=2, value=0,step=1)
            drowsiness = st.number_input("Drowsiness", min_value=0, max_value=2, value=0,step=1)
            sleep_mb = st.number_input("Sleep problems", min_value=0, max_value=2, value=0,step=1)
        with col_2:
            memory_mb = st.number_input("Memory problems", min_value=0, max_value=2, value=0,step=1)
            pressure = st.number_input("Pressure", min_value=0, max_value=3, value=0)
            balance_loss = st.number_input("Balance loss", min_value=0, max_value=1, value=0,step=1)
            judgment_degradation = st.number_input("Judgment degradation", min_value=0, max_value=3, value=0,step=1)
            sense_degradation = st.number_input("Sense degradation", min_value=0, max_value=3, value=0,step=1)
            lactation = st.number_input("Lactation", min_value=0, max_value=2, value=0,step=1)
            swallowing = st.number_input("Swallowing", min_value=0, max_value=3, value=0,step=1)
            muscle = st.number_input("Muscle issues", min_value=0, max_value=3, value=0,step=1)

        

        submit = st.form_submit_button("Submit")

        if submit:
            input_data = {
                "age": age,
                "gender": gender,
                "antecedents": antecedents,
                "headaches": headaches,
                "seizures": seizures,
                "fatigue": fatigue,
                "drowsiness": drowsiness,
                "sleep_mb": sleep_mb,
                "memory_mb": memory_mb,
                "pressure": pressure,
                "balance_loss": balance_loss,
                "judgment_degradation": judgment_degradation,
                "sense_degradation": sense_degradation,
                "lactation": lactation,
                "swallowing": swallowing,
                "muscle": muscle,
                
            }
            features=pd.DataFrame([input_data])
            with st.spinner() :
                pred,prob=ml_predict(features)
                
                st.subheader("🔍 Prediction Result")
                if pred == 1:
                    st.error(f"🧠 Tumor Detected (Probability: {prob[1]:.4f})")
                else:
                    st.success(f"✔️ No Tumor Detected (Probability: {prob[0]:.4f})")
                

with tab_cnn:
    show_results("CNN")

with tab_ml:
    show_results("ML")
