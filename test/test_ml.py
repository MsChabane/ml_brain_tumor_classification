from pipelines.ml.load_data import load_data
from pipelines.ml.clean import clean_data
from pipelines.ml.prepare_data import prepare_data
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd 
from pipelines.ml.train_model import train

def test_load_data():
    df = load_data('./data/data.csv')
    
    assert type(df) ==pd.DataFrame
    assert df.shape[0]==20000

def test_clean_data():
    df = load_data('./data/data.csv')
    df = clean_data(df)

    assert type(df) ==pd.DataFrame


def test_prepare_data():
    df = load_data('./data/data.csv')
    df = clean_data(df)
    prepared_data=prepare_data(df,'Tumor')

    assert type(prepared_data)==tuple
    assert len(prepared_data)==4
    assert type(prepared_data[0]) ==pd.DataFrame
    assert type(prepared_data[1]) ==pd.DataFrame
    assert type(prepared_data[2]) ==pd.Series
    assert type(prepared_data[3]) ==pd.Series

    assert "Tumor" not in  prepared_data[0].columns
    assert "Tumor" not in  prepared_data[1].columns



def test_train():
    
    X_train = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [1.0, 1.1],
        [1.1, 1.0]
    ])
    y_train = np.array([0, 0, 1, 1])

    model = train(X_train, y_train)

    
    assert isinstance(model, LogisticRegression)

    
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)
