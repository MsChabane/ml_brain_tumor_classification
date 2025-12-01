import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from .load_data import load_data
from .clean import clean_data
import joblib



def prepare_data (df:pd.DataFrame,target:str):
    X,y=df.drop(target,axis=1),df[target]
    gender_enc =LabelEncoder().fit(X['gender'])
    X['gender']=gender_enc.transform(X['gender'])
    joblib.dump(gender_enc,'./models/ml/gender_encoder.pkl')
    seizures_enc =LabelEncoder().fit(X['seizures'])
    X['seizures']=seizures_enc.transform(X['seizures'])
    joblib.dump(seizures_enc,'./models/ml/seizures_encoder.pkl')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return   X_train,X_test,y_train,y_test

if __name__=='__main__':
    df=load_data('./data/data.csv')
    print(df.shape)
    
    df=clean_data(df)
    print(df.shape)

    prep=prepare_data(df,'Tumor')
    print (len(prep))



