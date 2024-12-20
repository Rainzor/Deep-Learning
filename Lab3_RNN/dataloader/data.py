import os
import json
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


# Define the data structure
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
        
        self.data = torch.load(self.save_path)
    
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

def collate_fn(batch):
    
    lengths = [torch.sum(item['attention_mask']) for item in batch]
    sorted_lengths, sorted_idx = torch.sort(torch.tensor(lengths), descending=True)
    batch_sorted = [batch[i] for i in sorted_idx]

    input_ids = torch.stack([item['input_ids'] for item in batch_sorted])
    attention_mask = torch.stack([item['attention_mask'] for item in batch_sorted])
    label = torch.stack([item['label'] for item in batch_sorted])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': label
    }