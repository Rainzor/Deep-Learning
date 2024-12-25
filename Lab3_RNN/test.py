import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from transformers import BertTokenizer
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

from models.utils import best_config
from models.model import RNNClassifier

@dataclass
class YelpData:
    text: str
    star: int

class YelpDataset(Dataset):
    def __init__(self, data_dir, tokenizer, train=True, max_length=512, reload_=False):
        """
        Dataset constructor
        :param data_dir: Directory of the data files
        :param train: Whether to load training data
        :param tokenizer_name: Name of the tokenizer to use
        :param max_length: Maximum length for padding and truncation
        """
        self.is_train = train
        self.data_path = os.path.join(data_dir, 'train.json') if train else os.path.join(data_dir, 'test.json')
        self.raw_data = self._read_json(self.data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = None
        self.save_path = os.path.join(data_dir, 'train.pt') if train else os.path.join(data_dir, 'test.pt')

        if not os.path.exists(self.save_path) or reload_:
            if not reload_:
                print("Preprocessed data not found, it is first time to preprocess the data")
            else:
                print("Force to reload the data")
            self.data = self._preprocess(self.raw_data)
            torch.save(self.data, self.save_path)
            print(f"Preprocessed data saved to {self.save_path}")
        
        self.data = torch.load(self.save_path, weights_only=False)
        
    
    def _read_json(self, file_path):
        """
        Load training/test data from the specified directory
        :param data_dir: Directory containing the data files
        :param train: Whether to load the training data
        :return: List of data instances
        """
        data = []
        if self.is_train:
            start = 1000
        else:
            start = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line_num < start: # skip the first 1000 lines for training data
                        continue
                    rdata = json.loads(line)
                    text = rdata.get('text', None)
                    star = rdata.get('stars', None)
                    
                    if text is not None and star is not None:
                        data.append(YelpData(text=text, star=star))
                    else:
                        print(f"{line_num} data is invalid")
                except json.JSONDecodeError as e:
                    print(f"Fails to decode line {line_num}")
        
        return data
    
    def _preprocess(self, raw_data):
        tokenizer_list = []
        mask_list = []
        label_list = []
        with tqdm(total=len(raw_data), desc="Preprocessing data") as pbar:
            for review in raw_data:
                text = review.text
                label = review.star - 1 # Convert 1-5 to 0-4
                
                # Tokenize, pad and truncate the text
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,  # Add [CLS] and [SEP]
                    truncation=True,  # Truncate text if it exceeds max_length
                    padding='max_length',  # Pad text to max_length
                    max_length=self.max_length,
                    return_attention_mask=True,
                    return_tensors='pt'  # Return PyTorch tensors
                )
                
                tokenizer_list.append(encoding['input_ids'].squeeze(0))
                mask_list.append(encoding['attention_mask'].squeeze(0))
                label_list.append(torch.tensor(label, dtype=torch.long))
                pbar.update(1)
        
        return {
            'input_ids': torch.stack(tokenizer_list),
            'attention_mask': torch.stack(mask_list),
            'label': torch.stack(label_list)
        }


    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        if self.data is None:
            text = self.raw_data[idx].text
            label = self.raw_data[idx].star-1
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }  
        else:
            input_ids = self.data['input_ids'][idx]
            attention_mask = self.data['attention_mask'][idx]
            label = self.data['label'][idx]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label':  label
            }


def main():
    # Load the best configuration
    data_dir = 'data'
    output_dir = 'output'
    tokenizer_dir = 'tokenizer'

    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    test_dataset = YelpDataset(data_dir, tokenizer, train=False, max_length=512, reload_=False)
    test_loader = DataLoader(test_dataset, batch_size=128 ,shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_config.vocab_size = tokenizer.vocab_size
    model = RNNClassifier(best_config).to(device)

    model_state_dict = torch.load(os.path.join(output_dir, 'model.pth'), weights_only=True)
    model.load_state_dict(model_state_dict)

    # Evaluate the model
    def accuracy(preds, y):
        """
        Returns accuracy per batch
        :param preds: Predictions from the model
        :param y: Correct labels
        :return: Accuracy per batch
        """
        preds = torch.argmax(preds, dim=1)
        correct = (preds == y).float()
        acc = correct.sum() / len(correct)
        return acc
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    with torch.no_grad():
        model.eval()
        test_loss = 0
        test_acc = 0
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output, label)
            acc = accuracy(output, label)
            test_loss += loss.item()
            test_acc += acc.item()
        
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%')

    # Draw the output
    
    def get_sv(filename):
        train_file = filename + '_train.csv'
        val_file = filename + '_val.csv'
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        return df_train, df_val
    
    df_train, df_val = get_sv('output/best')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(df_train['Step'], df_train['Value'], label='Train Loss')
    
    ax[1].plot(df_val['Step'], df_val['Value'], label='Val Accuracy')

    ax[0].set_title('Train Loss')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].set_title('Val Accuracy')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()

    
if __name__ == '__main__':
    main()
