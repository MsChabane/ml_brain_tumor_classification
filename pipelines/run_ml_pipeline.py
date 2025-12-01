from ml.load_data import load_data
from ml.clean import clean_data
from ml.prepare_data import prepare_data
from ml.train_model import train
from ml.evaluate import evaluate
import joblib



def run_pipeline():
    print("pipeline is running...")
    df=load_data('./data/data.csv')
    df=clean_data(df)
    X_train,X_test,y_train,y_test=prepare_data(df,'Tumor')
    model=train(X_train,y_train)
    evaluate(model,X_test,y_test)
    joblib.dump(model,"./models/ml/model.pkl")



if __name__ =='__main__':
    run_pipeline()