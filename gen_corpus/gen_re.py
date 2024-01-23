import os
import sys
sys.path.append('.')
import json
import datetime
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer
from process.utils.corpus import Gen_Corpus, Wikidata

class Gen_RE(Gen_Corpus):

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
        # ----------fix a bug here-----------
        data_size = len(raw_data)
        for i in range(data_size):
            for j in range(2):
                if raw_data[i]['ents'][j][1] == 1:
                    raw_data[i]['ents'][j][1] = 0
        # ----------fix a bug here-----------
        return raw_data
    
    def gen_plain(self):
        plain = []
        for piece in self.raw_data:
            plain.append(piece['text'].replace('\n', ''))
        return plain

    def get_labelSet(self):
        labelSet = []
        for piece in self.raw_data:
            label = piece['label']
            if label not in labelSet:
                labelSet.append(label)
        return sorted(labelSet)
    
    def get_label_prompts(self):
        label_file = 'label-'+self.task_name+'.txt'
        label_dir = os.path.join(self.cache_dir, label_file)
        label_dscbs = self.txt2list(label_dir)
        labels = self.get_labelSet()
        labels2dscbs = {}
        assert len(labels) == len(label_dscbs)
        for label, label_dscb in zip(labels, label_dscbs):
            dscb_1, dscb_2 = label_dscb.split('||')
            labels2dscbs[label] = [dscb_1, dscb_2]
        return labels2dscbs

    def get_wiki_prompts(self):
        wiki_file = 'wiki-'+self.task_name+'.txt'
        wiki_dir = os.path.join(self.cache_dir, wiki_file)
        return self.txt2list(wiki_dir)
    
    def gen_LPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt>||<suffix>",
                ...
            ]
        '''
        labels2dscbs = self.get_label_prompts()
        promptSents = []
        for piece in self.raw_data:
            label = piece['label']
            sent = piece['text']
            ents = piece['ents']
            if label == 'NA':
                promptSents.append(sent)
                continue
            bias = 0 if ents[0][1] < ents[1][1] else 1
            promptSent = sent[:ents[bias][1]] + '||#' + sent[ents[bias][1]:ents[bias][2]] + '#||' \
                + labels2dscbs[label][bias] + '||' + sent[ents[bias][2]:ents[1-bias][1]] + '||#' \
                    + sent[ents[1-bias][1]:ents[1-bias][2]] + '#||' + labels2dscbs[label][1-bias] \
                        + '||' + sent[ents[1-bias][2]:]
            promptSents.append(promptSent)
        return promptSents

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
            anns = piece['ann']
            dscbs = [] if wiki_dscb == '' else wiki_dscb.split('||')
            assert len(anns) == len(dscbs)
            promptSent = ''
            start = 0
            for ann, dscb in zip(anns, dscbs):
                if ann[3] > 0.3 and dscb != 'No description defined':
                    if '(' in dscb and ')' in dscb:
                        dscb = self.rm_comment(dscb)
                    kprompt = self.add_conj(dscb)
                    promptSent += sent[start:ann[1]] + '||' + \
                        sent[ann[1]:ann[2]] + '||' + kprompt + '||'
                else:
                    promptSent += sent[start:ann[2]]
                start = ann[2]
            promptSent += sent[start:]
            promptSents.append(promptSent)
        return promptSents
    
class Gen_TACRED(Gen_RE):

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
    
    def __init__(self, mode):
        self.task_name = 'TACRED'
        self.dir_path = '../DATA/dataset/TACRED'
        super().__init__(mode)

    def gen_KLPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt>||<suffix>",
                "<prefix>||<entity>||<KPrompt>||<suffix>",
                ...
            ]
        '''
        labels2dscbs = self.get_label_prompts()
        wiki_dscbs = self.get_wiki_prompts()
        promptSents = []
        for piece, wiki_dscb in zip(self.raw_data, wiki_dscbs):
            sent = piece['text']
            ents = piece['ents']
            label = piece['label']
            anns = piece['ann']
            dscbs = [] if wiki_dscb == '' else wiki_dscb.split('||')
            assert len(anns) == len(dscbs)
            mix_prompts = [] if label == 'NA' else [
                (ents[0][1], ents[0][2], labels2dscbs[label][0], 'label'),
                (ents[1][1], ents[1][2], labels2dscbs[label][1], 'label')
            ]
            for ann, dscb in zip(anns, dscbs):
                con_0 = ann[2] <= ents[0][1] or ann[1] >= ents[0][2]
                con_1 = ann[2] <= ents[1][1] or ann[1] >= ents[1][2]
                con_2 = ann[3] > 0.3
                con_3 = dscb != 'No description defined'
                if con_0 and con_1 and con_2 and con_3:
                    if '(' in dscb and ')' in dscb:
                        dscb = self.rm_comment(dscb)
                    kprompt = self.add_conj(dscb)
                    mix_prompts.append((ann[1], ann[2], kprompt, 'wiki'))
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

class Gen_FewRel(Gen_RE):

    '''
        Task type: Relation Classification
        File type: json
        Download url: https://github.com/thunlp/ERNIE
        "label": "P931", 
        "text": "Flights arrive in ... Dulles International Airport .", 
        "ents": [["Q676576", 61, 102, 0.5], ["Q5092", 105, 114, 0.5]]
    '''
    '''warning: lots of \n\n\n... appear in text'''

    def __init__(self, mode):
        self.task_name = 'FewRel'
        self.dir_path = '../DATA/dataset/FewRel'
        super().__init__(mode)

    def get_label_id_2_label_name(self):
        label_ids = self.get_labelSet()
        label_name_file = 'labelSet(name).txt'
        label_name_dir = os.path.join(self.cache_dir, label_name_file)
        label_names = self.txt2list(label_name_dir)
        assert len(label_ids) == len(label_names)
        label_id_2_label_name = {}
        for label_id, label_name in zip(label_ids, label_names):
            label_id_2_label_name[label_id] = label_name
        return label_id_2_label_name

    def gen_LPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt>||<suffix>",
                ...
            ]
        '''
        labels2dscbs = self.get_label_prompts()
        promptSents = []
        for piece in self.raw_data:
            label = piece['label']
            sent = piece['text']
            ents = piece['ents']
            bias = 0 if ents[0][1] < ents[1][1] else 1
            promptSent = sent[:ents[bias][1]] + '||#' + sent[ents[bias][1]:ents[bias][2]] + '#||' \
                + labels2dscbs[label][bias].replace('$','') + '||' + sent[ents[bias][2]:ents[1-bias][1]] + '||#' \
                    + sent[ents[1-bias][1]:ents[1-bias][2]] + '#||' + labels2dscbs[label][1-bias].replace('$','') \
                        + '||' + sent[ents[1-bias][2]:]
            promptSents.append(promptSent.replace('\n', ''))
        return promptSents

    def gen_KPrompt(self):
        '''
            [
                "<prefix>||<entity>||<KPrompt>||<suffix>",
                ...
            ]
        '''
        wiki_dscbs = self.get_wiki_prompts()
        promptSents = []
        for piece, wiki_dscb in zip(self.raw_data, wiki_dscbs):
            sent = piece['text']
            ents = piece['ents']
            dscbs = wiki_dscb.split('||')
            assert len(ents) == len(dscbs) == 2
            promptSent = ''
            start = 0
            # ---- fix entity order here ----
            if ents[0][1] > ents[1][1]:
                ents = [ents[1], ents[0]]
                dscbs = [dscbs[1], dscbs[0]]
            # ---- --------------------- ----
            for ent, dscb in zip(ents, dscbs):
                if ent[3] > 0.3 and dscb != 'No description defined':
                    if '(' in dscb and ')' in dscb:
                        dscb = self.rm_comment(dscb)
                    kprompt = self.add_conj(dscb)
                    promptSent += sent[start:ent[1]] + '||' + \
                        sent[ent[1]:ent[2]] + '||' + kprompt + '||'
                else:
                    promptSent += sent[start:ent[2]]
                start = ent[2]
            promptSent += sent[start:]
            promptSents.append(promptSent.replace('\n', ''))
        return promptSents

class Gen_CHEMPROT(Gen_RE):

    def __init__(self, mode):
        self.task_name = 'CHEMPROT'
        self.dir_path = '../DATA/dataset/CHEMPROT'
        super().__init__(mode)
    
    def gen_noRepeat(self):
        prev = ''
        noRepeat = []
        for piece in self.raw_data:
            sent = piece['text']
            if sent.replace(' ', '') != prev:
                prev = sent.replace(' ', '')
                noRepeat.append(piece)
        return noRepeat

class Gen_ReTACRED(Gen_RE):

    def __init__(self, mode):
        self.task_name = 'ReTACRED'
        self.dir_path = '../DATA/dataset/ReTACRED'
        super().__init__(mode)
    
    def entityType2prompt(self, entity_type):
        entity_type = entity_type.lower().replace('_', ' ')
        article = 'an ' if entity_type[0] in ['a', 'e', 'i', 'o'] else 'a '
        return 'is ' + article + entity_type
    
    def gen_LPrompt(self):
        '''
            [
                "<prefix>||#<entity>#||<LPrompt>||<suffix>",
                ...
            ]
        '''
        promptSents = []
        for piece in self.raw_data:
            label = piece['label']
            sent = piece['text']
            ents = piece['ents']
            bias = 0 if ents[0][1] < ents[1][1] else 1
            entity_type = [piece['subj_type'], piece['obj_type']]
            promptSent = sent[:ents[bias][1]] + '||#' + sent[ents[bias][1]:ents[bias][2]] + '#||' \
                + self.entityType2prompt(entity_type[bias]) + '||' + sent[ents[bias][2]:ents[1-bias][1]] + '||#' \
                    + sent[ents[1-bias][1]:ents[1-bias][2]] + '#||' + self.entityType2prompt(entity_type[1-bias]) \
                        + '||' + sent[ents[1-bias][2]:]
            promptSents.append(promptSent)
        return promptSents

    def gen_KLPrompt(self):
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
            sent = piece['text']
            ents = piece['ents']
            anns = piece['ann']
            dscbs = [] if wiki_dscb == '' else wiki_dscb.split('||')
            assert len(anns) == len(dscbs)
            mix_prompts = [
                (ents[0][1], ents[0][2], self.entityType2prompt(piece['subj_type']), 'label'),
                (ents[1][1], ents[1][2], self.entityType2prompt(piece['obj_type']), 'label')
            ]
            for ann, dscb in zip(anns, dscbs):
                con_0 = ann[2] <= ents[0][1] or ann[1] >= ents[0][2]
                con_1 = ann[2] <= ents[1][1] or ann[1] >= ents[1][2]
                con_2 = ann[3] > 0.3
                con_3 = dscb != 'No description defined'
                if con_0 and con_1 and con_2 and con_3:
                    if '(' in dscb and ')' in dscb:
                        dscb = self.rm_comment(dscb)
                    kprompt = self.add_conj(dscb)
                    mix_prompts.append((ann[1], ann[2], kprompt, 'wiki'))
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


def main():
    gener = Gen_ReTACRED('train_ap')
    gener.list2txt(gener.gen_KLPrompt())
    
    
if __name__ == "__main__":
    main()