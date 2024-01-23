import os
import json
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

class InputFeatures(object):
    
    def __init__(self, input_id, attention_mask, label, ent_mask):
        self.input_id = input_id
        self.attention_mask = attention_mask
        self.label = label
        self.ent_mask = ent_mask

class TypingProcessor():

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
        file_path = os.path.join(self.args.data_dir, mode+'.json')
        with open(file_path, "r") as f:
            raw_data = json.load(f)
        label_list = []
        for piece in raw_data:
            for label in piece['labels']:
                if label not in label_list:
                    label_list.append(label)
        return raw_data, sorted(label_list)
    
    def create_features(self, mode):
        CLS_TOKEN = self.tokenizer.cls_token
        SEP_TOKEN = self.tokenizer.sep_token
        PAD_TOKEN = self.tokenizer.pad_token
        ADDITIONAL_SPECIAL_TOKENS = ['<e1>', '</e1>']
        self.tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        features = []
        raw_data, _ = self.load_dataset(mode)
        label_map = {label : i for i, label in enumerate(self.args.label_list)}
        for piece in tqdm(raw_data, desc="Create Features"):
            inputs = piece['sent']
            s, e = piece["start"], piece["end"]
            inputs = inputs[:s] + ' <e1> ' + inputs[s:e] + ' </e1> ' + inputs[e:]
            inputs = self.tokenizer.tokenize(inputs)
            inputs = self.clip_inputs(inputs)
            inputs = [CLS_TOKEN] + inputs + [SEP_TOKEN]
            s, e = inputs.index('<e1>'), inputs.index('</e1>')
            inputs[s] = inputs[e] = '#'

            seq_len = len(inputs)
            padding_length = self.args.max_seq_length - seq_len
            padding = [PAD_TOKEN]*padding_length
            inputs += padding
            input_id = self.tokenizer.convert_tokens_to_ids(inputs)
            attention_mask = [1]*seq_len + [0]*padding_length
            ent_mask = [0] * self.args.max_seq_length
            ent_mask[s] = 1
            label = [0]*len(label_map)
            for l in piece['labels']:
                l_= label_map[l]
                label[l_] = 1

            assert len(input_id) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(ent_mask) == self.args.max_seq_length
            features.append(
                InputFeatures(
                    input_id=input_id,
                    attention_mask=attention_mask,
                    label=label,
                    ent_mask=ent_mask
                )
            )
        return features

    def clip_inputs(self, inputs):
        upboard = self.args.max_seq_length-2 # reserve pos for cls & sep
        if len(inputs) <= upboard:
            return inputs
        p0 = inputs.index('<e1>')
        p1 = inputs.index('</e1>')
        assert p0 < p1
        assert p1-p0+1 <= upboard
        if p1 < upboard:
            return inputs[:upboard]
        elif len(inputs) - p0 <= upboard:
            return inputs[len(inputs)-upboard:]
        else:
            return inputs[p1-upboard:p1]

    def create_dataloader(self, mode):
        features = self.create_features(mode)
        sampler = {'train': RandomSampler, 'dev': SequentialSampler, 'test': SequentialSampler}
        batch_size = {'train': self.args.train_batch_size, 'dev': self.args.eval_batch_size, 'test': self.args.eval_batch_size}
        input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        labels = torch.tensor([f.label for f in features], dtype=torch.float)
        ent_mask = torch.tensor([f.ent_mask for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, attention_mask, labels, ent_mask)
        data_sampler = sampler[mode](data)
        return DataLoader(data, sampler=data_sampler, batch_size=batch_size[mode])
