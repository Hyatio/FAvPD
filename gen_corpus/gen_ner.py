import os
import sys
sys.path.append('.')
import json
import datetime
import pandas as pd
from tqdm import tqdm
from process.utils.corpus import Gen_Corpus

class Gen_NER(Gen_Corpus):

    def __init__(self, mode):
        self.mode = mode
        self.cache_dir = os.path.join(self.dir_path, 'cache')
        self.output_path = os.path.join(self.cache_dir, str(datetime.datetime.now()))
        self.raw_data = self.load_raw_data()

    def get_label_list(self):
        '''
            [
                [],
                ['person-artist/author', 'person-actor', 'person-actor'],
                ['art-writtenart', 'person-artist/author', 'art-writtenart', 'person-director', 'person-other', 'person-other'],
                ['organization-other'],
                ...
            ]
        '''
        label_list = []
        for piece in self.raw_data:
            prev_label = 'NA'
            labels = []
            for label in piece[1]:
                if label != 'O' and label != prev_label:
                    labels.append(label)
                prev_label = label
            label_list.append(labels)
        return label_list

    def gen_plain(self):
        plain = []
        for piece in self.raw_data:
            plain.append(piece['text'])
        return plain

    def get_wiki_prompts(self):
        wiki_file = 'wiki-'+self.task_name+'.txt'
        wiki_dir = os.path.join(self.cache_dir, wiki_file)
        return self.txt2list(wiki_dir)

    def gen_KPrompt(self):
        '''
            [
                "<prefix>||<entity>||<KPrompt>||<suffix>",
                ...
            ]
        '''
        wiki_dscbs = self.get_wiki_prompts()
        promptSents = []
        assert len(self.raw_data) == len(wiki_dscbs)
        for piece, wiki_dscb in zip(self.raw_data, wiki_dscbs):
            sent = piece['text']
            ents = piece['ents']
            dscbs = [] if wiki_dscb == '' else wiki_dscb.split('||')
            assert len(ents) == len(dscbs)
            promptSent = ''
            start = 0
            for ent, dscb in zip(ents, dscbs):
                if ent[3] > 0.3 and dscb != 'No description defined' \
                    and self.subword_detect(sent, ent[1], ent[2]):
                    if '(' in dscb and ')' in dscb:
                        dscb = self.rm_comment(dscb)
                    kprompt = self.add_conj(dscb)
                    promptSent += sent[start:ent[1]] + '||' + \
                        sent[ent[1]:ent[2]] + '||' + kprompt + '||'
                else:
                    promptSent += sent[start:ent[2]]
                start = ent[2]
            promptSent += sent[start:]
            promptSents.append(promptSent)
        return promptSents

class Gen_FewNERD(Gen_NER):

    '''
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
    # fewshot = 200

    def __init__(self, mode):
        self.task_name = 'FewNERD'
        self.dir_path = '../DATA/dataset/FewNERD'
        self.data_dir = os.path.join(self.dir_path, mode+'.txt') \
            if mode != 'fewshot' else os.path.join(self.dir_path, mode+'.json')
        super().__init__(mode)

    def load_raw_data(self):
        # ------ few shot ------
        if self.mode == 'fewshot':
            with open(self.data_dir, 'r') as f:
                data = f.read()
                raw_data = json.loads(data)
            return raw_data
        # ------ -------- ------
        with open(self.data_dir, 'r') as f:
            lines = f.readlines()
        # ----------fix some bugs here-----------
        lines[-1] += '\n'
        lines.append('\n')
        if self.mode == 'train': del lines[204314]
        # ----------fix some bugs here-----------
        words, labels = [], []
        raw_data = []
        for line in lines:
            assert line[-1] == '\n'
            if line == '\n':
                assert len(words) > 0
                assert len(words) == len(labels)
                raw_data.append([words, labels])
                '''
                    A piece in raw_data:
                        [
                            ['The', 'final', 'season', ..., 'Park', 'saw'], #words
                            ['O', 'O', 'O', ..., 'location-park', 'O'], #labels
                        ]
                '''
                words = []
                labels = []
            else:
                line = line[:-1]
                line = line.split('\t')
                assert len(line) == 2
                words.append(line[0])
                labels.append(line[1])
        return raw_data
    
    def get_label_set(self, isCoarseGrained=False):
        labelSet = []
        label_list = self.get_label_list()
        for labels in label_list:
            for label in labels:
                label = label.split('-')[0] if isCoarseGrained else label
                labelSet.append(label)
        return sorted(set(labelSet))

    def gen_fewshot(self, shots=10, isCoarseGrained=False):
        assert self.mode == 'train'
        label_list = self.get_label_list()
        num_labels = len(self.get_label_set(isCoarseGrained))
        flag = False
        label_count = {}
        fewshot_rawdata = []
        for piece, labels in tqdm(zip(self.raw_data, label_list), desc='few shotting'):
            if flag:
                break
            append_flag = False
            for label in labels:
                label = label.split('-')[0] if isCoarseGrained else label
                if label in label_count.keys():
                    if label_count[label] < shots:
                        append_flag = True
                    label_count[label] += 1
                else:
                    append_flag = True
                    label_count[label] = 1
            if append_flag:
                fewshot_rawdata.append(piece)
            else:
                for label in labels:
                    label = label.split('-')[0] if isCoarseGrained else label
                    label_count[label] -= 1
            if len(label_count.keys()) == num_labels:
                flag = True
                for k in label_count.keys():
                    if label_count[k] < shots:
                        flag = False
                        break
        print(len(fewshot_rawdata))
        print(label_count)
        return fewshot_rawdata

    def get_entity_list(self):
        '''
            [
                [],
                ['Hicks', 'Ellaline Terriss', 'Edmund Payne'],
                ['Time', 'George Axelrod', 'The Seven Year Itch', 'Richard Quine', 'Holden', 'Richard Benson'],
                ['IAEA'],
                ...
            ]
        '''
        ent_list = []
        for piece in self.raw_data:
            prev_label = 'O'
            ent = None
            ents = []
            for word, label in zip(piece[0], piece[1]):
                if label != 'O':
                    if label == prev_label:
                        ent += ' '+word
                    else:
                        if ent is not None:
                            ents.append(ent)
                        ent = word
                prev_label = label
            if ent is not None:
                ents.append(ent)
            ent_list.append(ents)
        return ent_list

    def get_label2dscb(self, granularity):
        '''
            {
                "O": "O",
                "art-broadcastprogram": "broadcast program",
                "art-film": "film",
                ...

            }
        '''
        # granularity: fine-grained(fg), coarse-grained(cg)
        label_set_dir = os.path.join(self.cache_dir, 'label-set.txt')
        fg_dir = os.path.join(self.cache_dir, 'label-dscb-fg.txt')
        cg_dir = os.path.join(self.cache_dir, 'label-dscb-cg.txt')
        label_set = self.txt2list(label_set_dir)
        fg = self.txt2list(fg_dir)
        cg = self.txt2list(cg_dir)
        fg_dict, cg_dict = {}, {}
        assert len(label_set) == len(fg) == len(cg)
        for l, f, c in zip(label_set, fg, cg):
            fg_dict[l] = f
            cg_dict[l] = c
        if granularity == 'fg':
            return fg_dict
        else:
            return cg_dict

    def gen_LPrompt(self, plain, entity_list, label_list):
        promptSents = []
        assert len(plain) == len(entity_list) == len(label_list)
        label2dscb = self.get_label2dscb('fg')
        for sent, ents, labels in zip(plain, entity_list, label_list):
            assert len(ents) == len(labels)
            if len(ents) == 0:
                promptSents.append(sent)
                continue
            prefix = ''
            suffix = sent
            for ent, label in zip(ents, labels):
                sep = suffix.split(ent, 1)
                prefix += sep[0] + '||' + ent + '||' + self.add_conj(label2dscb[label]) + '||'
                suffix = sep[1]
            promptSent = prefix + suffix
            promptSent = promptSent.replace(' ||', '||')
            promptSent = promptSent.replace('|| ', '||')
            promptSents.append(promptSent)
        return promptSents

class Gen_BC5CDR(Gen_Corpus):

    '''
        Task type: NER
        Task name: BC5CDR
        File type: tsv
        Download url: https://github.com/ncbi-nlp/BLUE_Benchmark/releases/tag/0.1
        A piece:
            word                flag        start   label
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

    def __init__(self, mode, sub):
        self.mode = mode
        self.dir_path = os.path.join('../DATA/dataset/BC5CDR', sub)
        self.data_dir = os.path.join(self.dir_path, mode+'.tsv')
        self.cache_dir = os.path.join(self.dir_path, 'cache')
        self.output_path = os.path.join(self.cache_dir, str(datetime.datetime.now())+'.txt')
        self.raw_data = self.load_raw_data()

    def load_raw_data(self):
        names = ['word', 'flag', 'start', 'label']
        return pd.read_csv(self.data_dir, sep='\t', header=None, names=names)

    def gen_plain(self):
        sent = None
        plain = []
        for word, flag in zip(self.raw_data['word'], self.raw_data['flag']):    
            # ----------fix a bug here-----------
            if type(word) != str: word = 'null'
            # ----------fix a bug here-----------
            if flag == '-':
                sent += ' '+word
            else:
                if sent is not None:
                    plain.append(sent)
                sent = word
        plain.append(sent)
        return plain
    
    def get_entity_set(self):
        '''
            [absence epilepsy, absence seizure, absence seizures, ... ]
        '''
        ent = None
        ent_all = []
        for word, label in zip(self.raw_data['word'], self.raw_data['label']):
            if label == 'B':
                if ent is not None:
                    ent = ent.lower()
                    if ent not in ent_all:
                        ent_all.append(ent)
                ent = word
            elif label == 'I':
                ent += ' '+word
        ent = ent.lower()
        if ent not in ent_all:
            ent_all.append(ent)
        ent_all = sorted(ent_all)
        return ent_all

    def get_entity_list(self):
        '''
            [
                [],
                ['hypertensive'],
                ['hypotensive'],
                [],
                ...
            ]
        '''
        ent = None
        ent_all = None
        ent_list = []
        for word, flag, label in zip(self.raw_data['word'], self.raw_data['flag'], self.raw_data['label']):
            if flag == '-':
                if label == 'B':
                    if ent is not None:
                        ent_all.append(ent)
                    ent = word
                elif label == 'I':
                    ent += ' '+word
            else:
                if ent is not None:
                    ent_all.append(ent)
                if label == 'B':
                    ent = word
                else:
                    ent = None
                if ent_all is not None:
                    ent_list.append(ent_all)
                ent_all = []
        ent_list.append(ent_all)
        return ent_list

    def gen_LPrompt(self, plain, entity_list):
        promptSents = []
        assert len(plain) == len(entity_list)
        for sent, entities in zip(plain, entity_list):
            if len(entities) == 0:
                promptSents.append(sent)
                continue
            prefix = ''
            suffix = sent
            for ent in entities:
                sep = suffix.split(ent, 1)
                prefix += sep[0] + '||' + ent + '||' + 'is a disease' + '||'
                suffix = sep[1]
            promptSent = prefix + suffix
            promptSents.append(promptSent)
        return promptSents

def main():
    gener = Gen_FewNERD('fewshot')
    gener.list2txt(gener.gen_KPrompt())

if __name__ == "__main__":
    main()