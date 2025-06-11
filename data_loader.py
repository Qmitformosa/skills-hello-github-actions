from transformers import DistilBertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

def tokenize_data(example, tokenizer):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

def get_imdb_dataloaders(batch_size=8):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = load_dataset('imdb')

    tokenized_train = dataset['train'].map(lambda x: tokenize_data(x, tokenizer), batched=True)
    tokenized_test = dataset['test'].map(lambda x: tokenize_data(x, tokenizer), batched=True)

    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size)

    return train_loader, test_loader