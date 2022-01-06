import os

import torch
from torch.utils.data import Dataset

import pandas as pd

from utils import read_and_process_file

class SignatureDataset(Dataset):
    def __init__(self,train,df=None):
        self.train = train
        
        if isinstance(df,pd.DataFrame):
            self.df = df
        else:
            self.df = pd.read_csv(os.path.join('.',f'sign_data/{"train" if train else "test"}_data.csv'),names=['x1','x2','label'])

    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        base_path = f'sign_data/{"train" if self.train else "test" }'

        x1 = torch.from_numpy(read_and_process_file(os.path.join(base_path,row['x1']))).unsqueeze(0).float()
        x2 = torch.from_numpy(read_and_process_file(os.path.join(base_path,row['x2']))).unsqueeze(0).float()

        label = row['label']

        return (x1,x2,label)


    def __len__(self):
        return len(self.df)