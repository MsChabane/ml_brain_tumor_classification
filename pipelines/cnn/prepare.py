import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_df(path):
  labels=[]
  images_path=[]
  for i in os.listdir(path):

    for j in os.listdir(os.path.join(path,i)):
      images_path.append(os.path.join(path,i,j))
      labels.append(i)

  return images_path,labels

def prepare(train_path,test_path)->tuple:
    images,labels=get_df(train_path)
    train_df=pd.DataFrame({'path':images,'label':labels})
    images,labels=get_df(test_path)
    test_df=pd.DataFrame({'path':images,'label':labels})
    train_df,val_df=train_test_split(train_df,test_size=0.2,random_state=42)
    train = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='label',
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )

    val = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='label',
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    test = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='label',
        target_size=(224,224),
        batch_size=32,
        shuffle=False,
        class_mode='binary'
    )
    return train,val,test


