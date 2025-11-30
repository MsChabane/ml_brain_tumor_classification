import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from fastapi import APIRouter,UploadFile,File
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from .schemas import PredictionResponse
import io
import numpy as np

MODEL_PATH = "models/cnn/model.keras"
model = tf.keras.models.load_model(MODEL_PATH)



router =APIRouter()

@router.post("/predict/cnn", response_model=PredictionResponse)
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
    return PredictionResponse( prediction=prediction_label,probability_class_yes=prob_yes,probability_class_no=prob_no )



@router.post("/predict/ml")
def predict_ml():
    return {"status":'not implemented yet'}


