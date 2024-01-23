import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class InputFeatures():
    
    def __init__(self, input_id, attention_mask, pos_id, vm, label_tip, label_label):
        self.input_id = input_id
        self.attention_mask = attention_mask
        self.pos_id = pos_id
        self.label_tip = label_tip
        self.vm = vm
        self.label_label = label_label
        '''
            label_tip index
            0: special tokens (e.g. [CLS], [SEP], [PAD], ... )
            -1: text (except for entity)
            -2: knowledge prompt (except for conjunction)
            -3: entity
            -4: conjunction (e.g. 'is a')
            +: label token id
        '''

class AdaptingProcessor():
    
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.multi_label = \
            args.prompt_type == 'MLPrompt' \
                or args.prompt_type == 'KMLPrompt'

    def load_sentences(self, data_dir):
        with open(data_dir, 'r') as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        return lines

    def mask_tokens(self, input_ids, label_tips):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        """ text: 15% Predict"""
        """ label: 100% Predict """
        """ knowledge: 50% Predict"""

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        label_ids = input_ids.clone()
        batch_size, seq_length = input_ids.shape
        label_tips.masked_fill_(input_ids == self.tokenizer.unk_token_id, value=0)
        
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(label_ids.shape, self.args.mlm_probability)
        # Apply default probability to the text part
        # And modify the probability for knowledge prompt part
        for i in range(batch_size):
            for j in range(seq_length):
                if int(label_tips[i][j]) == -2:
                    probability_matrix[i][j] = self.args.k_mlm_probability
                elif int(label_tips[i][j]) <= -3:
                    probability_matrix[i][j] = 0.0
        
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in label_ids.tolist()
        ]
        assert (label_tips >= 0).sum() == torch.tensor(special_tokens_mask).sum()
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = label_ids.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        label_ids[~masked_indices] = CrossEntropyLoss().ignore_index  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(label_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(label_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer)-4, label_ids.shape, dtype=torch.long)
        # -X(X is equal or larger than nums of special tokens, here we get 4): del the special tokens which is out of 'vocab.txt'
        input_ids[indices_random] = random_words[indices_random]

        t_label_ids = label_ids.masked_fill(label_tips!=-1, value=-100)
        k_label_ids = label_ids.masked_fill(label_tips!=-2, value=-100)
        l_label_ids = torch.zeros(batch_size, seq_length) if self.multi_label \
                else label_tips.masked_fill(label_tips<=0, value=-100)
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, t_label_ids, k_label_ids, l_label_ids
    
    def create_features(self):
        CLS_TOKEN = self.tokenizer.cls_token
        SEP_TOKEN = self.tokenizer.sep_token
        PAD_TOKEN = self.tokenizer.pad_token
        MASK_TOKEN = self.tokenizer.mask_token
        features = []
        data_dir = os.path.join(self.args.data_dir, self.args.prompt_type+'.txt')
        lines = self.load_sentences(data_dir)
        for line in tqdm(lines, desc="Create Features"):
            suffix = CLS_TOKEN + ' ' + line
            sent_parts = []
            istrunk = []
            seq_len = 0
            while '||' in suffix:
                assert suffix.count("||") % 3 == 0
                prefix, ent, prompt, suffix = suffix.split('||', 3)
                prefix = self.tokenizer.tokenize(prefix)
                if ent[0] == '#' and ent[-1] == '#':
                    # LPrompt: Label Prompt
                    ent = ent[1:-1]
                    ent = self.tokenizer.tokenize(ent)
                    if self.multi_label:
                        prompt_ = []
                        prompt_length = 0
                        for lprompt in prompt.split('$'):
                            lprompt = self.tokenizer.tokenize(lprompt)
                            prompt_.append(lprompt)
                            if len(lprompt) > prompt_length:
                                prompt_length = len(lprompt)
                        prompt_type = 'MLP'
                    else:
                        prompt_type = 'LP'
                        prompt_ = self.tokenizer.tokenize(prompt)
                        prompt_length = len(prompt_)

                else:
                    # KPrompt: Knowledge Prompt
                    ent = self.tokenizer.tokenize(ent)
                    prompt_type = 'KP'
                    prompt_ = self.tokenizer.tokenize(prompt)
                    prompt_length = len(prompt_)
                sent_parts.append((prefix, ent, prompt_, prompt_type, prompt_length))
                istrunk.extend([1]*len(prefix+ent)+[0]*prompt_length)
                seq_len += len(prefix+ent) + prompt_length
            suffix  = self.tokenizer.tokenize(suffix)
            istrunk.extend([1]*len(suffix))
            seq_len += len(suffix)

            input_id = []
            label_tip = []
            label_label = [] #(start_pos_0, label_token_length_0, label_token_id_0_0, label_token_id_0_1...)
            pos_id = []
            last_pos = 0
            for spart in sent_parts:
                prompt_type = spart[3]
                prompt_length = spart[4]
                label_tip.extend([-1]*len(spart[0]))
                label_tip.extend([-3]*len(spart[1]))
                label_tip.extend([-4]*2)
                # In our settings: we assert
                # 1# prompts begin with linking verbs (e.g. 'is a')
                # 2# linking verbs occupy 2 verbs in BertTokenizer/RobertaTokenizer
                if prompt_type  == 'LP':
                    input_id.extend(spart[0]+spart[1]+spart[2][:2]+[MASK_TOKEN]*(prompt_length-2))
                    label_tip.extend(self.tokenizer.convert_tokens_to_ids(spart[2][2:]))
                elif prompt_type == 'KP':
                    input_id.extend(spart[0]+spart[1]+spart[2])
                    label_tip.extend([-2]*(prompt_length-2))
                elif prompt_type == 'MLP':
                    input_id.extend(spart[0]+spart[1]+spart[2][0][:2]+[MASK_TOKEN]*(prompt_length-2))
                    label_tip.extend([0]*(prompt_length-2))
                    for lp in spart[2]:
                        start_pos = len(input_id)-prompt_length+2
                        label_token_length = len(lp)-2
                        if start_pos+label_token_length >= self.args.max_seq_length:
                            continue
                        label_label.append(start_pos)
                        label_label.append(label_token_length)
                        label_label.extend(self.tokenizer.convert_tokens_to_ids(lp[2:]))
                else:
                    raise ValueError("prompt type error")
                pos_id.extend(list(range(last_pos, last_pos+len(spart[0]+spart[1])+prompt_length)))
                last_pos += len(spart[0]+spart[1])
            input_id.extend(suffix)
            label_tip.extend([-1]*len(suffix))
            pos_id.extend(list(range(last_pos, last_pos+len(suffix))))
            import pdb
            if not len(input_id) == len(label_tip) == len(pos_id) == len(istrunk) == seq_len:
                pdb.set_trace()

            seeings, vm = [], []
            for spart in sent_parts:
                #   index   (prefix, entity, prompt)
                #   0       (1,      1,      0     )
                #   1       (1,      1,      1     )
                #   2       (0,      1,      0     )
                #   3       (0,      0,      0     )
                l0, l1, l2 = len(spart[0]), len(spart[1]), spart[4]
                seeing = []
                seeing.append([1]*(l0+l1)+[0]*l2)
                seeing.append([1]*(l0+l1+l2))
                seeing.append([0]*l0+[1]*l1+[0]*l2)
                seeing.append([0]*(l0+l1+l2))
                seeings.append(seeing)
            seeing_suffix = [[0]*len(suffix), [1]*len(suffix)]
            for i, spart in enumerate(sent_parts):
                seeing_ = [[],[],[]]
                for j, seeing in enumerate(seeings):
                    seeing_[0].extend(seeing[0])
                    if j == i:
                        seeing_[1].extend(seeing[1])
                        seeing_[2].extend(seeing[2])
                    else:
                        seeing_[1].extend(seeing[0])
                        seeing_[2].extend(seeing[3])
                seeing_[0].extend(seeing_suffix[1])
                seeing_[1].extend(seeing_suffix[1])
                seeing_[2].extend(seeing_suffix[0])
                l0, l1, l2 = len(spart[0]), len(spart[1]), spart[4]
                for i_, j in enumerate([l0, l1, l2]):
                    vm.extend([seeing_[i_] for k in range(j)])
            if len(sent_parts) == 0: # plain case
                vm.extend(seeing_suffix[1] for k in suffix)
            else:
                vm.extend(seeing_[0] for k in suffix)
            tvm = torch.tensor(vm)
            assert tvm.shape == (seq_len, seq_len)
            
            assert len(label_label) <= self.args.max_seq_length
            if seq_len > self.args.max_seq_length-1:
                input_id = input_id[:self.args.max_seq_length-1] + [SEP_TOKEN]
                input_id = self.tokenizer.convert_tokens_to_ids(input_id)
                attention_mask = [1] * self.args.max_seq_length
                pos_id = pos_id[:self.args.max_seq_length-1]
                p = self.args.max_seq_length-2
                while not istrunk[p]:
                    p -= 1
                pos_id += [pos_id[p]+1]
                label_tip[0] = 0 # CLS_TOKEN is a special token
                label_tip = label_tip[:self.args.max_seq_length-1] + [0] # SEP_TOKEN is a special token
                vm = vm[:self.args.max_seq_length-1]
                for i in range(self.args.max_seq_length-1):
                    if istrunk[i]:
                        vm[i] = vm[i][:self.args.max_seq_length-1] + [1]
                    else:
                        vm[i] = vm[i][:self.args.max_seq_length-1] + [0]
                vm.append(istrunk[:self.args.max_seq_length-1]+[1])
            else:
                input_id += [SEP_TOKEN]
                label_tip[0] = 0 # CLS_TOKEN is a special token
                label_tip += [0] # SEP_TOKEN is a special token
                seq_len = len(input_id)
                padding_length = self.args.max_seq_length-seq_len
                padding = [PAD_TOKEN]*padding_length
                zero_padding = [0]*padding_length # PAD_TOKEN is a special token
                pos_padding = [self.args.max_seq_length-1]*padding_length
                padding_all = [0]*self.args.max_seq_length
                input_id += padding
                input_id = self.tokenizer.convert_tokens_to_ids(input_id)
                attention_mask = [1]*seq_len + zero_padding
                label_tip += zero_padding
                p = seq_len-2
                while not istrunk[p]:
                    p -= 1
                pos_id += [pos_id[p]+1]
                pos_id += pos_padding
                for i in range(seq_len-1):
                    if istrunk[i]:
                        vm[i] = vm[i][:self.args.max_seq_length-1]+[1]+zero_padding
                    else:
                        vm[i] = vm[i][:self.args.max_seq_length-1]+[0]+zero_padding
                vm.append(istrunk+[1]+zero_padding)
                vm.extend([padding_all for i in range(padding_length)])
            
            assert len(input_id) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(pos_id) == self.args.max_seq_length
            assert len(label_tip) == self.args.max_seq_length
            tvm = torch.tensor(vm)
            assert tvm.shape == (self.args.max_seq_length, self.args.max_seq_length)
            label_label += [-1]*(self.args.max_seq_length-len(label_label))
            assert len(label_label) == self.args.max_seq_length

            features.append(
                InputFeatures(
                    input_id = input_id,
                    attention_mask = attention_mask,
                    pos_id = pos_id,
                    label_tip = label_tip,
                    vm = vm,
                    label_label = label_label
                )
            )
        return features

    def create_dataloader(self):
        features = self.create_features()
        input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
        attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        pos_ids = torch.tensor([f.pos_id for f in features], dtype=torch.long)
        label_tips = torch.tensor([f.label_tip for f in features], dtype=torch.long)
        vms = torch.tensor([f.vm for f in features], dtype=torch.uint8)
        label_labels = torch.tensor([f.label_label for f in features], dtype=torch.long)
        input_ids_ = input_ids.clone()
        static_input_ids, static_t_label_ids, static_k_label_ids, \
            static_l_label_ids = self.mask_tokens(input_ids_, label_tips)
        data = TensorDataset(input_ids, attention_masks, pos_ids, label_tips, vms, label_labels, \
            static_input_ids, static_t_label_ids, static_k_label_ids, static_l_label_ids)
        data_sampler = RandomSampler(data)
        return DataLoader(data, sampler=data_sampler, batch_size=self.args.train_batch_size)