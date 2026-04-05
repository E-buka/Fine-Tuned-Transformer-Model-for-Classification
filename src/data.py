import pandas as pd
import torch
import re
import numpy as np
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, random_split
from config import LABEL_COL, MAX_SEQ_LEN


class MakeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts 
        self.labels = labels 
        self.tokenizer = tokenizer    

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        texts = self.texts.iloc[idx]
        ids = self.tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return {"input_ids": ids["input_ids"],
                "attention_mask": ids["attention_mask"],
                "labels":label,
        }
    

def split_data(full_data):
    train_size = int(0.7 * len(full_data))
    hold_size = len(full_data) - train_size 
    val_size = int(hold_size/2)
    test_size = hold_size - val_size
    
    train_df, hold_df = random_split(full_data, [train_size, hold_size])
    test_df, val_df = random_split(hold_df, [test_size, val_size])
    return train_df, test_df, val_df 

def encode_text(text, vocab, max_seq_length):    
    tokens = text.lower().split()
    ids = [vocab.get(id, vocab["<unk>"]) for id in tokens]
    ids = ids[:max_seq_length]
    if len(ids) < max_seq_length:
        ids += [vocab["<pad>"]] * (max_seq_length - len(ids))
    return torch.tensor(ids, dtype=torch.long)
        

def text_cleaner(text):
    text  = re.sub(r"http\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_csv_data(path: str, feature_col, label_col, encoding):
    df = pd.read_csv(path, usecols=[feature_col, label_col], encoding=encoding)
    
    df[feature_col] = df[feature_col].apply(text_cleaner)
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    
    df[label_col] = df[label_col].mask(df[label_col]==4, 1)
    return df
    

def stratified_split(full_data):
    train_size = int(0.7 * len(full_data))
    hold_size = len(full_data) - train_size 
    val_size = int(hold_size/2)
    test_size = hold_size - val_size
    
    train_idx, hold_idx = train_test_split(np.arange(len(full_data)),
                                           test_size=hold_size,
                                           shuffle=True,
                                           stratify=full_data[LABEL_COL])
    
    train_df = full_data.iloc[train_idx].reset_index(drop=True)
    hold_data = full_data.iloc[hold_idx].reset_index(drop=True)
    
    val_idx, test_idx = train_test_split(np.arange(len(hold_data)),
                                         test_size=test_size,
                                         shuffle=True,
                                         stratify=hold_data[LABEL_COL])
    
    val_df = hold_data.iloc[val_idx].reset_index(drop=True)
    test_df = hold_data.iloc[test_idx].reset_index(drop=True)
    
    return train_df, test_df, val_df 

