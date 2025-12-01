import pandas as pd
from .load_data import load_data



def clean_data(df:pd.DataFrame):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

if __name__=='__main__':
    df=load_data('./data/data.csv')
    print(df.shape)
    
    df=clean_data(df)
    print(df.shape)
    