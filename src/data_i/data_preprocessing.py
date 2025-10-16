from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import os

def load_data(train_data_path, test_data_path, valid_data_path):
    train_data = load_from_disk(train_data_path)
    test_data = load_from_disk(test_data_path)
    valid_data = load_from_disk(valid_data_path)
    
    return train_data, test_data, valid_data

def load_tokenizer(model_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)
    return tokenizer

def preprocess_batch(batch, tokenizer):
    
    inputs = tokenizer(
        batch['dialogue'],
        max_length=1024,
        truncation=True,
        padding='max_length'
    )
    
    targets = tokenizer(
        batch['summary'],
        max_length=128,
        truncation=True,
        padding='max_length'
    )
    
        

    labels = targets["input_ids"]
    # replace padding token id's in labels by -100 so they are ignored by the loss
    labels = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels
    ]

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }

def tokenize_data(train_data, test_data, valid_data, tokenizer, batch_size=1000):
    train_data = train_data.map(lambda batch: preprocess_batch(batch, tokenizer), batched=True, batch_size=batch_size)
    test_data = test_data.map(lambda batch: preprocess_batch(batch, tokenizer), batched=True, batch_size=batch_size)
    valid_data = valid_data.map(lambda batch: preprocess_batch(batch, tokenizer), batched=True, batch_size=batch_size)
    
    return train_data, test_data, valid_data


def save_data(data_path, train_data, test_data, valid_data):
    os.makedirs(data_path, exist_ok=True)
    train_data.save_to_disk(os.path.join(data_path, 'train'))
    test_data.save_to_disk(os.path.join(data_path, 'test'))
    valid_data.save_to_disk(os.path.join(data_path, 'validation'))


def main():
    train_data_path = os.path.join("data", "raw", "train")
    test_data_path = os.path.join("data", "raw", "test")
    valid_data_path = os.path.join("data", "raw", "validation")
    
    train_data, test_data, valid_data = load_data(train_data_path, test_data_path, valid_data_path)
    tokenizer = load_tokenizer("t5-small")
    # tokenizer = AutoTokenizer.from_pretrained("t5-small")

    train_data, test_data, valid_data = tokenize_data(train_data, test_data, valid_data, tokenizer)
    
    data_path = os.path.join("data", "tokenized")
    save_data(data_path, train_data, test_data, valid_data)
    
if __name__ == '__main__':
    main()

