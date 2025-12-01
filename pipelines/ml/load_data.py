import pandas as pd 


def load_data (path):
    return pd.read_csv(path)

if __name__=='__main__':
    df=load_data('./data/data.csv')
    print(df.shape)


