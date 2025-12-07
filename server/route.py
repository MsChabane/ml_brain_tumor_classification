import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from fastapi import APIRouter,UploadFile,File,Depends
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #type: ignore 
from .schemas import Predictor,input_ml
import uuid
import io
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "models/cnn/model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
MODEL_ML_PATH = "models/ml/model.pkl"
ENCODE_GENDER_PATH ="models/ml/gender_encoder.pkl"
ENCODE_SEIZ_PATH ="models/ml/seizures_encoder.pkl"
model_ml = joblib.load(MODEL_ML_PATH)
encode_gender=joblib.load(ENCODE_GENDER_PATH)
encode_seiz=joblib.load(ENCODE_SEIZ_PATH)


router =APIRouter()

@router.post("/predict/cnn", response_model=Predictor)
async def predict_cnn(image: UploadFile = File(...)):
    contents = await image.read()
    img = tf.keras.preprocessing.image.load_img(
        io.BytesIO(contents),
        target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)[0][0]
    prob_yes = round(float(preds),4)
    prob_no =round(float(1-prob_yes),4)
    prediction_label = "yes" if prob_yes >= 0.5 else "no"
    accuracy = prob_yes if prediction_label == "yes" else prob_no
    cnn_hyperparams = {
    "img_size": (224, 224),
    "learning_rate": 0.001,
    "batch_size": 16,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "epochs": 50,
    "architecture": "MobileNetV2"   
}
    predict_cnn=Predictor(
        model_name='Cnn',
        params=cnn_hyperparams,
        prediction=prediction_label,
        accuracy=accuracy
    )
    
    return predict_cnn




@router.post("/predict/ml", response_model=Predictor)
async def predict_ml(input_data:input_ml):
    input={
        'age':input_data.age, 
        'gender': input_data.gender,
        'antecedents':input_data.antecedents,
        'headaches':input_data.headaches,
        'seizures': input_data.seizures,
        'fatigue':input_data.fatigue,
        'drowsiness':input_data.drowsiness,	
        'sleep_mb':	input_data.sleep_pb,
        'memory_mb': input_data.memory_pb,
        'pressure': input_data.pressure,
        'balance_loss': input_data.balance_loss,
        'judgment_degradation': input_data.judgment_degradation,
        'sense_degradation': input_data.sense_degradation,
        'lactation': input_data.lactation,
        'swallowing': input_data.swallowing,
        'muscle': input_data.muscle,
    }
    df = pd.DataFrame([input])
    
    df['gender']= encode_gender.transform(df['gender'])
    df['seizures']=encode_seiz.transform(df['seizures'])

    preds = model_ml.predict_proba(df)[0]

    proba_no = round(float(preds[0]),4)
    proba_yes = round(float(preds[1]),4)
    prediction_label,accuracy = ("yes",proba_yes) if proba_yes >= proba_no else ("no",proba_no)
    
    ml_hyperparams = {
        "penalty":"l2",
        "C":0.5,
        "solver":"lbfgs",
        "max_iter":1000,
        "random_state":42   
    }
    predict_ml=Predictor(
        
        model_name='LogisticRegression',
        params=ml_hyperparams,
        prediction=prediction_label,
        accuracy=accuracy 
    )
    return predict_ml


