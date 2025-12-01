import pandas as pd 
from sklearn.linear_model import LogisticRegression 



def train (X_train,y_train):
    model=LogisticRegression(
    penalty="l2",
    C=0.5,
    solver="lbfgs",
    max_iter=1000,
    random_state=42)
    model.fit(X_train,y_train)
    return model






