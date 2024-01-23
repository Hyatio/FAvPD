import os
import jsonlines
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

class InputFeatures(object):

    def __init__(self, input_id, attention_mask, token_type_id, label):

        self.input_id = input_id
        self.attention_mask = attention_mask
        self.token_type_id = token_type_id
        self.label = label
    
class SWAGProcessor():

    '''
        Task type: SWAG QA(4 option)
        File type: jsonlines
        Download url: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
        A piece:
            "question": "A junior ... to take?",
            "answer": "Tell the attending that he cannot fail to disclose this mistake", 
            "options": {
                            "A": "Disclose the error to the patient and put it in the operative report",
                            "B": "Tell the attending that he cannot fail to disclose this mistake",
                            "C": "Report the physician to the ethics committee",
                            "D": "Refuse to dictate the operative report"
                        },
            "meta_info": "step1",
            "answer_idx": "B",
            "metamap_phrases": ["junior orthopaedic surgery resident", ..., "resident to take"]
    '''

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def load_dataset(self, mode):
        file_path = os.path.join(self.args.data_dir, mode+'.jsonl')
        with jsonlines.open(file_path, 'r') as reader:
            raw_data = list(reader)
        label_list = []
        for piece in raw_data:
            if piece['answer_idx'] not in label_list:
                label_list.append(piece['answer_idx'])
        return raw_data, sorted(label_list)
    
    def create_features(self, mode):
        CLS_TOKEN = self.tokenizer.cls_token
        SEP_TOKEN = self.tokenizer.sep_token
        PAD_TOKEN = self.tokenizer.pad_token
        features = []
        raw_data, _ = self.load_dataset(mode)
        label_map = {label : i for i, label in enumerate(self.args.label_list)}
        for piece in tqdm(raw_data, desc="Create Features"):
            line = ''
            for op in ['A', 'B', 'C', 'D']:
                line += '(' + op + ') ' + piece['options'][op] + ' '
            line = line.replace('\n', '').replace('\"', ' ')
            with open('options.txt', 'a') as f:
                f.write(line+'\n')
            inputs = []
            inputs_0 = piece['question']
            inputs_0 = self.tokenizer.tokenize(inputs_0)
            # assert 4-options A, B, C, D
            input_id, attention_mask, token_type_id = [], [], []
            assert sorted(piece['options'].keys()) == self.args.label_list
            for option in self.args.label_list:
                inputs_1 = piece['options'][option]
                inputs_1 = self.tokenizer.tokenize(inputs_1)
                Q, A = self.clip_inputs(inputs_0, inputs_1)
                QA = [CLS_TOKEN] + Q + [SEP_TOKEN] + A + [SEP_TOKEN]
                seq_len = len(QA)
                padding_length = self.args.max_seq_length - seq_len
                QA += [PAD_TOKEN]*padding_length
                input_id_ = self.tokenizer.convert_tokens_to_ids(QA)
                attention_mask_ = [1]*seq_len + [0]*padding_length
                token_type_id_ = [0]*(len(Q)+2) + [1]*(len(A)+1) + [0]*padding_length
                assert len(input_id_) == self.args.max_seq_length
                assert len(attention_mask_) == self.args.max_seq_length
                assert len(token_type_id_) == self.args.max_seq_length
                input_id.append(input_id_)
                attention_mask.append(attention_mask_)
                token_type_id.append(token_type_id_)
            
            label = label_map[piece['answer_idx']]
            assert len(input_id) == len(self.args.label_list)
            assert len(attention_mask) == len(self.args.label_list)
            assert len(token_type_id) == len(self.args.label_list)
            features.append(
                InputFeatures(input_id=input_id,
                    attention_mask=attention_mask,
                    token_type_id=token_type_id,
                    label=label)
                )
        return features
    
    def clip_inputs(self, inputs_0, inputs_1):
        # clip inputs_0, keep inputs_1
        # reserve pos for [CLS]*1 and [SEP]*2
        upboard = self.args.max_seq_length - len(inputs_1) - 3
        if len(inputs_0) <= upboard:
            return inputs_0, inputs_1
        else:
            return inputs_0[:upboard], inputs_1

    def create_dataloader(self, mode):
        features = self.create_features(mode)
        sampler = {'train': RandomSampler, 'dev': SequentialSampler, 'test': SequentialSampler}
        batch_size = {'train': self.args.train_batch_size, 'dev': self.args.eval_batch_size, 'test': self.args.eval_batch_size}
        input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
        attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_id for f in features], dtype=torch.long)
        labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
        data_sampler = sampler[mode](data)
        return DataLoader(data, sampler=data_sampler, batch_size=batch_size[mode])