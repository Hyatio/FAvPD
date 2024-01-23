import json
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

class Gen_Corpus():

    def load_raw_data(self):
        raise NotImplementedError()
    
    def gen_plain(self):
        raise NotImplementedError()

    def list2txt(self, textList):
        with open(self.output_path +'.txt', 'w') as f:
            for text in textList:
                f.write(str(text)+'\n')
    
    def list2txt_(self, textList):
        with open(self.output_path+'.txt', 'a') as f:
            for text in textList:
                f.write(str(text)+'\n')

    def list2json(self, jsonList):
        jsonList = json.dumps(jsonList)
        with open(self.output_path+'.json', 'w') as f:
            f.write(jsonList)

    def txt2list(self, file_path):
        list_ = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                list_.append(line[:-1])
        return list_

    def add_conj(self, dscb):
        vowel = ['a', 'e', 'i', 'o', 'u']
        if dscb[0] in vowel:
            return 'is an ' + dscb
        else:
            return 'is a ' + dscb
    
    def rm_comment(self, dscb):
        # apply to one comment ( means the string contain single pair of '()' )
        if len(dscb.split('(')) != 2 or len(dscb.split(')')) != 2:
            return dscb
        return dscb.split('(')[0] + ' ' + dscb.split(')')[1]
    
    def sortByFirstElem(self, elem):
        return elem[0]

    def isLetter(self, x):
        return (ord(x) >= 65 and ord(x) <= 90) \
            or (ord(x) >= 97 and ord(x) <= 122)
    
    def subword_detect(self, sent, s, e):
        # return False when word=sent[s:e] is a sub word, else return True
        sent = ' ' + sent + ' '
        if self.isLetter(sent[s]) or self.isLetter(sent[e+1]):
            return False
        else:
            return True

    def get_wikidata_dscb(self):
        '''
            ["wiki-dscb-0-0||wiki-dscb-0-1||...", wiki-dscb-1-0||...", ..., "wiki-dscb-..."]
        '''
        dscbs = []
        wiki = Wikidata()
        for piece in tqdm(self.raw_data, desc="Downloading Wikidata"):
            dscb = ''
            for ent in piece['ents']:
                content = wiki.get_content(ent[0])
                dscb += wiki.get_describe(content) + '||'
            dscbs.append(dscb[:-2])
            # with open('wiki.txt', 'a') as f:
            #     f.write(dscb[:-2]+'\n')
        return dscbs
    
    def get_wikidata_title(self):
        '''
            ["wiki-title-0-0||wiki-title-0-1||...", wiki-title-1-0||...", ..., "wiki-title-..."]
        '''
        titles = []
        wiki = Wikidata()
        for piece in tqdm(self.raw_data[:100], desc="Downloading Wikidata"):
            title = ''
            for ent in piece['ents']:
                content = wiki.get_content(ent[0])
                title += wiki.get_title(content) + '||'
            titles.append(title[:-2])
        return titles

class TREx():

    def __init__(self, data_path):
        self.data_path = data_path
        self.wikiTitles = None

    def load_wikiTitles(self):
        file_path = '../DATA/dataset/T-REx/titleList.txt'
        titles = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            titles.append(line[:-1].lower())
        return titles

    def load_TREx_fileList(self):
        trexs = []
        prefix = '../DATA/dataset/T-REx/TREx/re-nlg_'
        suffix = '.json'
        for i in range(0, 465):
            trexs.append(prefix+str(i*10000)+'-'+str((i+1)*10000)+suffix)
        return trexs

    def load_knowledge(src, ord):
        with open(src, 'r') as f:
            data = f.read()
            data = json.loads(data)
        return data[ord]['text']

class Wikidata():

    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
        self.url_prefix = 'https://www.wikidata.org/wiki/'

    def get_content(self, ent):
        url = self.url_prefix + ent
        r = requests.get(url, headers=self.headers)
        if r.status_code == 200:
            return r.content
    
    def get_describe(self, content):
        try:
            soup = BeautifulSoup(content.decode('utf-8'), "html.parser")
        except AttributeError:
            return 'No description defined'
        return soup.find(class_='wikibase-entitytermsview-heading-description').get_text()
    
    def get_title(self, content):
        try:
            soup = BeautifulSoup(content.decode('utf-8'), "html.parser")
        except AttributeError:
            return '###'
        title = str(soup.title.string)
        assert title[-11:] == ' - Wikidata'
        return title[:-11]