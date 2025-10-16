import pandas as pd
from datasets import load_dataset, load_from_disk
import os


def load_data(path):
    df = load_from_disk(path)
    return df

    
def split_data(df_path):
    train_data = df_path['train']
    test_data = df_path['test']
    valid_data = df_path['validation']
    return train_data, test_data, valid_data


def save_data(data_path, train_data, test_data, valid_data):
    os.makedirs(data_path, exist_ok=True)
    train_data.save_to_disk(os.path.join(data_path, 'train'))
    test_data.save_to_disk(os.path.join(data_path, 'test'))
    valid_data.save_to_disk(os.path.join(data_path, 'validation'))
    
    
def main():
    data_path = os.path.join("data", "raw")
    df = load_data('data/external/samsum_dataset')
    train_data, test_data, valid_data = split_data(df)
    save_data(data_path, train_data, test_data, valid_data)
    
    
if __name__ == '__main__':
    main()

