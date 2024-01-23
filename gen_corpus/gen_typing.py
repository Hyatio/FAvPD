import os
import sys
sys.path.append('.')
import json
import datetime
from tqdm import tqdm
from process.utils.corpus import Gen_Corpus, Wikidata

class Gen_Typing(Gen_Corpus):

    '''
        Task type: Entity Typing
        File type: json
        Download url: https://github.com/thunlp/ERNIE
        A piece:
            "sent": "The British ... Google Maps ."
            "labels": ["person"]
            "start": 55
            "end": 64
            "ents": [["Q849763", 4, 11, 0.120101936], ..., ["Q1662473", 12, 46, 0.6343167]]
    '''

    def __init__(self, mode):
        self.mode = mode
        self.data_dir = os.path.join(self.dir_path, mode+'.json')
        self.cache_dir = os.path.join(self.dir_path, 'cache')
        self.output_path = os.path.join(self.cache_dir, str(datetime.datetime.now()))
        self.raw_data = self.load_raw_data()
    
    def load_raw_data(self):
        with open(self.data_dir, 'r') as f:
            data = f.read()
            raw_data = json.loads(data)
        return raw_data

    def gen_plain(self):
        plain = []
        for piece in self.raw_data:
            plain.append(piece['sent'])
        return plain

    def get_entity_list(self):
        '''
            ["Web users", "he", "They", ..., ]
        '''
        entity_list = []
        for piece in self.raw_data:
            s, e = piece['start'], piece['end']
            entity = piece['sent'][s:e]
            entity_list.append(entity)
        return entity_list

    def get_wiki_prompts(self):
        wiki_file = 'wiki-'+self.task_name+'.txt'
        wiki_dir = os.path.join(self.cache_dir, wiki_file)
        return self.txt2list(wiki_dir)
    
    def get_wiki_titles(self):
        wiki_file = 'wikiTitle-'+self.task_name+'.txt'
        wiki_dir = os.path.join(self.cache_dir, wiki_file)
        return self.txt2list(wiki_dir)

    def get_label_set(self, isCoarseGrained=False):
        labelSet = []
        for piece in self.raw_data:
            for label in piece['labels']:
                label = '/'+label.split('/')[1] if isCoarseGrained else label
                labelSet.append(label)
        return sorted(set(labelSet))

    def get_label_statistics(self):
        label_dict = {}
        labelSet = self.get_label_set()
        for label in labelSet:
            label_dict[label] = 0
        for piece in self.raw_data:
            for label in piece['labels']:
                label_dict[label] += 1
        return label_dict

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
            sent = piece['sent']
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

class Gen_OpenEntity(Gen_Typing):

    def __init__(self, mode):
        self.task_name = 'OpenEntity'
        self.dir_path = '../DATA/dataset/OpenEntity'
        super().__init__(mode)

    def gen_LPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt>||<suffix>",
                ...
            ]
        '''
        promptSents = []
        for piece in self.raw_data:
            sent = piece['sent']
            labels = piece['labels']
            s, e = piece['start'], piece['end']
            if labels:
                lprompt = self.add_conj(labels[0])
                sent = sent[:s] + '||#' + sent[s:e] + '#||' + lprompt + '||' + sent[e:]
            promptSents.append(sent)
        return promptSents
        
    def gen_MLPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt_0>$<LPrompt_1>$...$<LPrompt_n>||<suffix>",
                ...
            ]
        '''
        promptSents = []
        for piece in self.raw_data:
            sent = piece['sent']
            labels = piece['labels']
            s, e = piece['start'], piece['end']
            lprompt = ''
            for label in labels:
                lprompt += self.add_conj(label) + '$'
            if lprompt:
                sent = sent[:s] + '||#' + sent[s:e] + '#||' + lprompt[:-1] + '||' + sent[e:]
            promptSents.append(sent)
        return promptSents
    
    def gen_KMLPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt>||<suffix>",
                "<prefix>||<entity>||<KPrompt>||<suffix>",
                ...
            ]
        '''
        wiki_dscbs = self.get_wiki_prompts()
        promptSents = []
        for piece, wiki_dscb in zip(self.raw_data, wiki_dscbs):
            sent = piece['sent']
            ents = piece['ents']
            labels = piece['labels']
            lprompt = ''
            for label in labels:
                lprompt += self.add_conj(label) + '$'
            mix_prompts = [(piece['start'], piece['end'], lprompt[:-1], 'label'),] \
                if lprompt else []
            dscbs = [] if wiki_dscb == '' else wiki_dscb.split('||')
            assert len(ents) == len(dscbs)
            for ent, dscb in zip(ents, dscbs):
                con_0 = ent[2] <= piece['start'] or ent[1] >= piece['end']
                con_1 = ent[3] > 0.3
                con_2 = dscb != 'No description defined'
                if con_0 and con_1 and con_2:
                    if '(' in dscb and ')' in dscb:
                        dscb = self.rm_comment(dscb)
                    kprompt = self.add_conj(dscb)
                    mix_prompts.append((ent[1], ent[2], kprompt, 'wiki'))
            mix_prompts.sort(key=self.sortByFirstElem)
            promptSent = ''
            start = 0
            seg_mark = {'label': ('||#', '#||'), 'wiki': ('||', '||')}
            for mix_prompt in mix_prompts:
                s = mix_prompt[0]
                e = mix_prompt[1]
                prompt = mix_prompt[2]
                flag = mix_prompt[3]
                promptSent += sent[start:s] + seg_mark[flag][0] + \
                    sent[s:e] + seg_mark[flag][1] + prompt + '||'
                start = e
            promptSent += sent[start:]
            promptSents.append(promptSent)
        return promptSents

class Gen_FIGER(Gen_Typing):

    # fewshot = 100
    # assert that if '/xxx/yyy' in labels, '/xxx' in labels (in a train set)

    def __init__(self, mode):
        self.task_name = 'FIGER'
        self.dir_path = '../DATA/dataset/FIGER'
        super().__init__(mode)

    def gen_LPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt>||<suffix>",
                ...
            ]
        '''
        promptSents = []
        for piece in self.raw_data:
            sent = piece['sent']
            labels = piece['labels']
            s, e = piece['start'], piece['end']
            lprompt = self.add_conj(labels[0].split('/')[-1].replace('_', ' ')) 
            sent = sent[:s] + '||#' + sent[s:e] + '#||' + lprompt + '||' + sent[e:]
            promptSents.append(sent)
        return promptSents

    def gen_MLPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt_0>$<LPrompt_1>$...$<LPrompt_n>||<suffix>",
                ...
            ]
        '''
        promptSents = []
        for piece in self.raw_data:
            sent = piece['sent']
            labels = piece['labels']
            s, e = piece['start'], piece['end']
            lprompt = ''
            for label in labels:
                lprompt += self.add_conj(label.split('/')[-1].replace('_', ' ')) + '$'
            sent = sent[:s] + '||#' + sent[s:e] + '#||' + lprompt[:-1] + '||' + sent[e:]
            promptSents.append(sent)
        return promptSents

    def gen_KMLPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt>||<suffix>",
                "<prefix>||<entity>||<KPrompt>||<suffix>",
                ...
            ]
        '''
        wiki_dscbs = self.get_wiki_prompts()
        promptSents = []
        for piece, wiki_dscb in zip(self.raw_data, wiki_dscbs):
            sent = piece['sent']
            ents = piece['ents']
            labels = piece['labels']
            lprompt = ''
            for label in labels:
                lprompt += self.add_conj(label.split('/')[-1].replace('_', ' ')) + '$'
            mix_prompts = [(piece['start'], piece['end'], lprompt[:-1], 'label'),]
            dscbs = [] if wiki_dscb == '' else wiki_dscb.split('||')
            assert len(ents) == len(dscbs)
            for ent, dscb in zip(ents, dscbs):
                con_0 = ent[2] <= piece['start'] or ent[1] >= piece['end']
                con_1 = ent[3] > 0.3
                con_2 = dscb != 'No description defined'
                if con_0 and con_1 and con_2:
                    if '(' in dscb and ')' in dscb:
                        dscb = self.rm_comment(dscb)
                    kprompt = self.add_conj(dscb)
                    mix_prompts.append((ent[1], ent[2], kprompt, 'wiki'))
            mix_prompts.sort(key=self.sortByFirstElem)
            promptSent = ''
            start = 0
            seg_mark = {'label': ('||#', '#||'), 'wiki': ('||', '||')}
            for mix_prompt in mix_prompts:
                s = mix_prompt[0]
                e = mix_prompt[1]
                prompt = mix_prompt[2]
                flag = mix_prompt[3]
                promptSent += sent[start:s] + seg_mark[flag][0] + \
                    sent[s:e] + seg_mark[flag][1] + prompt + '||'
                start = e
            promptSent += sent[start:]
            promptSents.append(promptSent)
        return promptSents

    def gen_fewshot(self, shots=10, isCoarseGrained=False):
        assert self.mode == 'train'
        num_labels = len(self.get_label_set(isCoarseGrained))
        flag = False
        label_count = {}
        fewshot_rawdata = []
        for piece in tqdm(self.raw_data, desc='few shotting'):
            if flag:
                break
            append_flag = False
            for label in piece['labels']:
                if isCoarseGrained and len(label.split('/'))!=2:
                    continue
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
                for label in piece['labels']:
                    if (not isCoarseGrained) or (len(label.split('/'))==2):
                        label_count[label] -= 1
            if len(label_count.keys()) == num_labels:
                flag = True
                for k in label_count.keys():
                    if label_count[k] < shots:
                        flag = False
                        break
        return fewshot_rawdata

def main():
    gener = Gen_OpenEntity('train')
    
if __name__ == "__main__":
    main()
