import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

class InputFeatures(object):

    def __init__(self, input_id, attention_mask, label, e1_mask, e2_mask):

        self.input_id = input_id
        self.attention_mask = attention_mask
        self.label = label
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

class REProcessor():

    '''
        Task type: Relation Classification
        File type: json
        Download url: https://github.com/thunlp/ERNIE
        A piece:
            "label": "org:founded", 
            "text": "Zagat Survey ... the decision .", 
            "ents": [["Zagat", 1, 5, 0.5], ["1979", 82, 86, 0.5]], 
            "ann": [["Q140258", 0, 12, 0.57093775], ["Q7804542", 60, 78, 0.532475]]
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
            if piece['label'] not in label_list:
                label_list.append(piece['label'])
        # ----------fix a bug here-----------
        data_size = len(raw_data)
        for i in range(data_size):
            for j in range(2):
                if raw_data[i]['ents'][j][1] == 1:
                    raw_data[i]['ents'][j][1] = 0
        # ----------fix a bug here-----------
        return raw_data, sorted(label_list)

    def create_features(self, mode):
        CLS_TOKEN = self.tokenizer.cls_token
        SEP_TOKEN = self.tokenizer.sep_token
        PAD_TOKEN = self.tokenizer.pad_token
        ADDITIONAL_SPECIAL_TOKENS = ['<e1>', '</e1>', '<e2>', '</e2>']
        self.tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        features = []
        raw_data, _ = self.load_dataset(mode)
        label_map = {label : i for i, label in enumerate(self.args.label_list)}
        for piece in tqdm(raw_data, desc="Create Features"):
            inputs = piece['text']
            ents = piece['ents']
            etokens = [['<e1>', '</e1>'], ['<e2>', '</e2>']]
            etokens_ = [[' <e1> ', ' </e1> '], [' <e2> ', ' </e2> ']]
            bias = 0 if ents[0][1] < ents[1][1] else 1
            inputs = inputs[:ents[bias][1]] + etokens_[bias][0] + \
                    inputs[ents[bias][1]:ents[bias][2]] + etokens_[bias][1] + \
                        inputs[ents[bias][2]:ents[1-bias][1]] + etokens_[1-bias][0] + \
                            inputs[ents[1-bias][1]:ents[1-bias][2]] + etokens_[1-bias][1] + \
                                inputs[ents[1-bias][2]:]
            inputs = self.tokenizer.tokenize(inputs)
            inputs = self.clip_inputs(inputs, etokens, bias)
            inputs = [CLS_TOKEN] + inputs + [SEP_TOKEN]
            h0, h1, t0, t1 = inputs.index('<e1>'), inputs.index('</e1>'), inputs.index('<e2>'), inputs.index('</e2>')
            inputs[h0] = inputs[h1] = '@'
            inputs[t0] = inputs[t1] = '#'

            seq_len = len(inputs)
            padding_length = self.args.max_seq_length - seq_len
            padding = [PAD_TOKEN]*padding_length
            inputs += padding
            input_id = self.tokenizer.convert_tokens_to_ids(inputs)
            attention_mask = [1]*seq_len + [0]*padding_length
            e1_mask = [0] * self.args.max_seq_length
            e2_mask = [0] * self.args.max_seq_length
            # e1_mask[h0:h1+1] = [1] * (h1-h0+1)
            # e2_mask[t0:t1+1] = [1] * (t1-t0+1)
            e1_mask[h0] = 1
            e2_mask[t0] = 1
            
            label = label_map[piece['label']]
                   
            assert len(input_id) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(e1_mask) == self.args.max_seq_length
            assert len(e2_mask) == self.args.max_seq_length

            features.append(
                InputFeatures(input_id=input_id,
                    attention_mask=attention_mask,
                    label=label,
                    e1_mask=e1_mask,
                    e2_mask=e2_mask)
                )
        return features

    def clip_inputs(self, inputs, etokens, bias):
        upboard = self.args.max_seq_length-2 # reserve pos for cls & sep
        if len(inputs) <= upboard:
            return inputs
        p0 = inputs.index(etokens[bias][0])
        p1 = inputs.index(etokens[bias][1])
        p2 = inputs.index(etokens[1-bias][0])
        p3 = inputs.index(etokens[1-bias][1])
        assert p0 < p1 < p2 < p3
        assert (p1-p0)+(p3-p2)+2 <= upboard
        if p3-p0+1 <= upboard:
            if p3 < upboard:
                return inputs[:upboard]
            elif len(inputs) - p0 <= upboard:
                return inputs[len(inputs)-upboard:]
            else:
                extra = upboard - (p3-p0+1)
                return inputs[p0-extra:p3+1]
        else:
            extra = (p3-p0+1) - upboard
            return inputs[p0:p1+1] + inputs[p1+1+extra:p3+1]

    def create_dataloader(self, mode):
        features = self.create_features(mode)
        sampler = {'train': RandomSampler, 'dev': SequentialSampler, 'test': SequentialSampler}
        batch_size = {'train': self.args.train_batch_size, 'dev': self.args.eval_batch_size, 'test': self.args.eval_batch_size}
        input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        labels = torch.tensor([f.label for f in features], dtype=torch.long)
        e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)
        e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, attention_mask, labels, e1_mask, e2_mask)
        data_sampler = sampler[mode](data)
        return DataLoader(data, sampler=data_sampler, batch_size=batch_size[mode])
