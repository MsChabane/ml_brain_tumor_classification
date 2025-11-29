# Brain Tumor Classification Project

## Project Overview

This project implements a **brain tumor binary classification system** using two approaches:

1. **Convolutional Neural Network (CNN)** – classifies brain tumor images directly.
2. **Machine Learning (ML)** – classifies brain tumors based on **patient features**, such as age, gender, and symptoms (e.g., balance loss, sleep problems, headache).

The project is modular and organized into **pipelines, models, notebooks, and a FastAPI server** for serving predictions.

---

## Features

### CNN Approach:

- **Input:** Brain MRI images
- **Architecture:** Custom CNN (or pre-trained model can be integrated)
- **Output:** Binary classification (Tumor / No Tumor)
- **Workflow:** `preprocess → train → evaluate → save_model`

### ML Approach:

- **Input:** Patient features, such as:
  - Age
  - Gender
  - Symptoms: balance loss, sleep problems, headache...etc.
- **Algorithms:** Scikit-learn classifiers (e.g., Random Forest, SVM)
- **Output:** Binary classification (Tumor / No Tumor)
- **Workflow:** `preprocess → train → evaluate → save_model`

---

## Getting Started

### Requirements

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- FastAPI
- joblib, numpy, pandas, matplotlib, etc.

```bash
pip install -r requirements.txt
```

## Running Pipelines

- CNN Pipeline

```bach
python pipelines/run_cnn_pipeline.py
```

- ML Pipeline

```bash
python pipelines/run_ml_pipeline.py
```

### Both pipelines will:

1.  Preprocess data

2.  Train the model

3.  Evaluate performance

4.  Save the model to the models/ folder

## Starting the FastAPI Server

```bash
uvicorn server.main:app --reload
```

## Running Streamlit Inference App

```bash
streamlit run inference/app.py
```
