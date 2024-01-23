import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
PAD_LABEL_ID = -100

class InputFeatures(object):
    
    def __init__(self, input_id, attention_mask, label):
        self.input_id = input_id
        self.attention_mask = attention_mask
        self.label = label

class NERProcessor():

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
    
    def create_features(self, mode):
        CLS_TOKEN = self.tokenizer.cls_token
        SEP_TOKEN = self.tokenizer.sep_token
        PAD_TOKEN = self.tokenizer.pad_token
        features = []
        raw_data, _ = self.load_dataset(mode)
        label_map = {label : i for i, label in enumerate(self.args.label_list)}
        for piece in tqdm(raw_data, desc="Create Features"):
            tokens = []
            labels = []
            for token, label in zip(piece[0], piece[1]):
                token = self.tokenizer.tokenize(token)
                tokens.extend(token)
                label_id = label_map[label]
                labels.extend([label_id]+[PAD_LABEL_ID]*(len(token)-1))
            if len(tokens) > self.args.max_seq_length - 2:
                tokens = tokens[:self.args.max_seq_length-2]
                labels = labels[:self.args.max_seq_length-2]
            inputs = [CLS_TOKEN] + tokens + [SEP_TOKEN]
            labels = [PAD_LABEL_ID] + labels + [PAD_LABEL_ID]

            seq_len = len(inputs)
            padding_length = self.args.max_seq_length - seq_len
            padding = [PAD_TOKEN] * padding_length
            padding_labels = [PAD_LABEL_ID] * padding_length
            inputs += padding
            input_id = self.tokenizer.convert_tokens_to_ids(inputs)
            attention_mask = [1] * seq_len + [0] * padding_length
            labels += padding_labels

            assert len(input_id) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(labels) == self.args.max_seq_length
            features.append(InputFeatures(
                input_id=input_id,
                attention_mask=attention_mask,
                label=labels
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

class FewNERDProcessor(NERProcessor):

    ''''
        Task type: NER
        Task name: Few-NERD
        File type: txt
        Download path: https://ningding97.github.io/fewnerd/
        A piece:
            word        label
            --------------------------
            The	        O
            final	    O
            season	    O
            of	        O
            minor	    O
            league	    O
            play	    O
            Elkin	    location-park
            Memorial	location-park
            Park	    location-park
            saw	        O
            season      O
            attendance	O
            of          O
            16,322	    O
    '''

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def load_dataset(self, mode):
        '''
            A piece in raw_data:
            [
                ['The', 'final', 'season', ..., 'Park', 'saw'], #words
                ['O', 'O', 'O', ..., 'location-park', 'O'], #labels
            ]
        '''
        file_path = os.path.join(self.args.data_dir, mode+'.txt')
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # ----------fix some bugs here-----------
        lines[-1] += '\n'
        lines.append('\n')
        if mode == 'train': del lines[204314]
        # ----------fix some bugs here-----------
        words, labels, label_list = [], [], []
        raw_data = []
        for line in lines:
            assert line[-1] == '\n'
            if line == '\n':
                assert len(words) > 0
                assert len(words) == len(labels)
                raw_data.append([words, labels])
                words = []
                labels = []
            else:
                line = line[:-1]
                line = line.split('\t')
                assert len(line) == 2
                words.append(line[0])
                labels.append(line[1])
                if line[1] not in label_list:
                    label_list.append(line[1])
        return raw_data, sorted(label_list)


class BC5CDRProcessor(NERProcessor):

    '''
            Task type: NER
            Task name: BC5CDR
            File type: tsv
            Download path: https://github.com/ncbi-nlp/BLUE_Benchmark/releases/tag/0.1
            A piece:
                word                unknown     start   label
                -----------------------------------------------
                Naloxone            227508	    0	    O
                reverses            -	        9	    O
                the                 -	        18	    O
                antihypertensive	-	        22	    O
                effect	            -	        39	    O
                of	                -	        46	    O
                clonidine	        -	        49	    O
                .	                -	        58	    O
        '''

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
    
    def load_dataset(self, mode):
        '''
            A piece in raw_data:
            [
                ['Lidocaine', '-', 'induced', 'cardiac', 'asystole', '.'], #words
                ['O', 'O', 'O', 'B', 'I' 'O'], #labels
            ]
        '''
        file_path = os.path.join(self.args.data_dir, mode+'.tsv')
        lines = pd.read_csv(file_path, sep='\t', header=None, names=['word','unknown', 'start', 'label'])
        words, labels = [], []
        raw_data = []
        for word, unknown, label in zip(lines['word'], lines['unknown'], lines['label']):
            # ----------fix a bug here-----------
            if type(word) != str: word = 'null'
            # ----------fix a bug here-----------
            if unknown == '-':
                words.append(word)
                labels.append(label)
            else:
                raw_data.append([words, labels])
                words = [word]
                labels= [label]
        del raw_data[0]
        raw_data.append([words, labels])
        label_list = ['B', 'I', 'O']
        return raw_data, label_list
