from cnn.evaluate import evaluate
from cnn.load_data import load_data_to
from cnn.prepare import prepare
from cnn.train import train
import os

def run_pipeline():
    print("pipeline is running...")
    print(">> loading the data... ")
    path='./data'
    os.makedirs(path,exist_ok=True)
    load_data_to(path)
    print('>> prepare the data ...')
    train_path=os.path.join(path,'Brain_Tumor_Datasets','train')
    test_path=os.path.join(path,'Brain_Tumor_Datasets','test')
    train_data,val_data,test_data= prepare(train_path,test_path)
    print('>> train the model ...')
    model = train(train_data,val_data,checkpoint_path='./models/cnn/model.keras')
    
    print('>> evalate the model ...')
    evaluate(model,test_data)
    print("pipeline is completed ")
    



if __name__ =='__main__':
    run_pipeline()