import os
import json
import jsonlines
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

class InputFeatures(object):
    
    def __init__(self, input_id, attention_mask, label):
        self.input_id = input_id
        self.attention_mask = attention_mask
        self.label = label

class NLIProcessor():

    '''
        Task type: Entity Typing
        File type: json
        Download url: https://github.com/thunlp/ERNIE
        A piece:
            "sent": "The British ... Google Maps ."
            "labels": ["person"]
            "start": 55
            "end": 64
    '''

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def load_dataset(self, mode):
        file_path = os.path.join(self.args.data_dir, mode+'.jsonl')
        raw_data = []
        with open(file_path, "r") as f:
            for item in jsonlines.Reader(f):
                raw_data.append(item)
        label_list = []
        for piece in raw_data:
            label = piece['label_text']
            if label not in label_list:
                label_list.append(label)
        return raw_data, sorted(label_list)
    
    def create_features(self, mode):
        CLS_TOKEN = self.tokenizer.cls_token
        SEP_TOKEN = self.tokenizer.sep_token
        PAD_TOKEN = self.tokenizer.pad_token
        features = []
        raw_data, _ = self.load_dataset(mode)
        label_map = {label : i for i, label in enumerate(self.args.label_list)}
        for piece in tqdm(raw_data, desc="Create Features"):
            text1 = piece['text1']
            text2 = piece['text2']
            text1 = self.tokenizer.tokenize(text1)
            text2 = self.tokenizer.tokenize(text2)
            inputs = [CLS_TOKEN] + text1 + [SEP_TOKEN] + text2
            if len(inputs) > self.args.max_seq_length:
                inputs = inputs[:self.args.max_seq_length]

            seq_len = len(inputs)
            padding_length = self.args.max_seq_length - seq_len
            padding = [PAD_TOKEN]*padding_length
            inputs += padding
            input_id = self.tokenizer.convert_tokens_to_ids(inputs)
            attention_mask = [1]*seq_len + [0]*padding_length
            label = label_map[piece['label_text']]

            assert len(input_id) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            features.append(
                InputFeatures(
                    input_id=input_id,
                    attention_mask=attention_mask,
                    label=label
                )
            )
        return features

    def create_dataloader(self, mode):
        features = self.create_features(mode)
        sampler = {'train': RandomSampler, 'dev': SequentialSampler, 'test': SequentialSampler}
        batch_size = {'train': self.args.train_batch_size, 'dev': self.args.eval_batch_size, 'test': self.args.eval_batch_size}
        input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, attention_mask, labels)
        data_sampler = sampler[mode](data)
        return DataLoader(data, sampler=data_sampler, batch_size=batch_size[mode])
