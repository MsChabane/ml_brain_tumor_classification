import kagglehub
import shutil
import os

def load_data_to(distination):
    
    path = kagglehub.dataset_download("luluw8071/brain-tumor-mri-datasets")
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(distination, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print(f'data is stored in {distination}')




if __name__ =='__main__':
    path='./data'
    os.makedirs(path,exist_ok=True)
    load_data_to(path)