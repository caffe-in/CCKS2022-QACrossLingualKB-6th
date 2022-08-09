from email.mime import base
import random
from sklearn.model_selection import train_test_split
import torch
import time
from torch.utils.data import Dataset
import pytorch_lightning as pl
from transformers import BertModel,BertConfig,BertTokenizer
import argparse
def next_rels(ner,kb):
    rels ={}
    try:
        for rel,tail in kb[ner]:
            if rel not in rels.keys():
                rels[rel]=[tail]
            else:
                rels[rel].append(tail)
    except KeyError:
        pass
    return rels

def search_all_path(topic_entity,kb,hop_num):
    rel_path,entity_path = search_dfs(topic_entity,kb,hop_num)
    return rel_path

def search_dfs(topic_entity,kb,hop_num):
    rel_path =[]
    entity_path = []
    if hop_num == 0:
        return [[]],[[topic_entity]]
    rels = next_rels(topic_entity,kb)
    for rel,tails in rels.items():
        for tail in tails:
            next_rel,tail = search_dfs(tail,kb,hop_num-1)
            for r in next_rel:
                rel_path.append([rel]+r)
            for t in tail:
                entity_path.append([topic_entity]+t)
    return rel_path,entity_path

    
def create_kb(path):
        kb = {}
        with open(path, encoding='utf8') as f:
            triples = f.readlines()
            for triple in triples:
                triple = triple.split("\t")
                head = triple[0].strip()
                rel = triple[1].strip()
                tail = triple[2].strip()
            
                if head not in kb.keys():
                    kb[head] = [(rel,tail)]
                else:
                    kb[head].append((rel,tail))
        return kb
def get_lines(data_path):
        data_file = open(data_path,'r')
        data_lines = data_file.readlines()
        data_lines = [line.strip().split("\t") for line in data_lines]
        return data_lines
def result_compare(uhop_path,bs_path,dump_path):
    wiki_path = "..//data/result_compare/source/BS/predict_data_wiki.txt"
    dump_file = open(dump_path,'w')

    uhop_lines =get_lines(uhop_path)
    bs_lines = get_lines(bs_path)
    wiki_lines = get_lines(wiki_path)
    count=0
    for idx in range(len(uhop_lines)):
        uhop_line = uhop_lines[idx]
        bs_line = bs_lines[idx]
        wiki_line = wiki_lines[idx]
        ques = uhop_line[0]
        entity = uhop_line[1]
        

        u_path = uhop_line[2:]
        b_path = bs_line[2:]
        w_path = wiki_line[2:]
        if u_path!=b_path:
            write_str = ques+"\t"+entity+"\n"+"\t".join(w_path)+"\n"+"\t".join(u_path)+"\n"+"\t".join(b_path)+"\n"
            dump_file.write(write_str+"\n")
            count+=1
    print(count)
class PreDataset(Dataset):
    def __init__(self,data_path,kb_path,pretrained_path,mode):
        super().__init__()
        self.data_path = data_path
        self.kb_path = kb_path
        self.neg_num = 3
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.kb = create_kb(self.kb_path)
        self.features,self.labels = self.get_objs(self.data_path)
    def get_objs(self,data_path):
        lines = get_lines(data_path)        

        features_list = []
        labels_list = []
        ques_list = []
        print(len(lines))
        for idx,line in enumerate(lines):
            start = time.time()
            ques = line[0]
            topic_entity = line[1]
            hop_num = len(line[2:])
            gold_path = line[2:]
            neg_path_list = self.search_neg_path(topic_entity,hop_num)
            if gold_path in neg_path_list:
                neg_path_list.remove(gold_path)
            gold_path = " ".join(gold_path)
            neg_path_list = [" ".join(neg_path) for neg_path in neg_path_list]
            # for neg in neg_path_list:
            #     neg_file.write(neg+"\n")
            features_list.append(gold_path)
            features_list+=neg_path_list
            ques_list+=[ques]+[ques]*len(neg_path_list)
            labels_list+=[[1,0]]+[[0,1]]*len(neg_path_list)     #[1,0] for positive [0,1] for negative

            end = time.time()
        print(len(features_list))
        print(len(ques_list))
        print(len(labels_list))
        features_list = self.tokenizer(ques_list,features_list,padding=True,truncation=True,return_tensors="pt")
        labels_list = torch.tensor(labels_list,dtype=torch.float)

        return features_list,labels_list

        
    def search_neg_path(self,topic_entity,hop_num):
        neg_path_list = []
        count=0
        while len(neg_path_list)<self.neg_num and count<5:
            try:
                neg_path = self.search_neg(topic_entity,hop_num)
            except IndexError:
                break
            if neg_path not in (neg_path_list):
                neg_path_list.append(neg_path)
            count+=1
        return neg_path_list

    def search_neg(self,topic_entity,hop_num):
        neg_path = []
        while hop_num>0:
            next_rel_list = self.next_rels([topic_entity]).keys()
            next_rel_list = list(next_rel_list)
            random.shuffle(next_rel_list)
            next_rel = next_rel_list[0]
            
            neg_path.append(next_rel)
            hop_num-=1
            topic_entity = [item[1] for item in self.kb[topic_entity] if item[0]==next_rel][0]
        return neg_path

    def next_rels(self,ners):
        rels = {}
        for ner in ners:
            try:
                for rel,tail in self.kb[ner]:
                    if rel not in rels.keys():
                        rels[rel]=[tail]
                    else:
                        rels[rel].append(tail)
            except KeyError:
                pass
        return rels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        input_ids = self.features['input_ids'][index].long()
        token_ids =  self.features['token_type_ids'][index].long()
        attention_mask = self.features['attention_mask'][index].long()
        label = self.labels[index]
        return(input_ids,token_ids,attention_mask,label) 


    
def test_batch():
    data_path = "..//data//ccks_wiki//train_ques.txt"
    kb_path = "..//data//ccks_wiki//triples.txt"
    pretrianed_path = "/home/pretrains/pt/bert-base-uncased"
    dm = PreDataset(data_path,kb_path,pretrianed_path,mode="train")
    print("one sample example")
    print(dm.features["input_ids"][0])
    print(dm.features["token_type_ids"][0])
    print(dm.features["attention_mask"][0])

def gen_ques_classifier(data_path,dump_path):
    datafile = open(data_path,'r',encoding='utf8')
    dumpfile = open(dump_path,'w',encoding='utf8')
    lines = datafile.readlines()
    for line in lines:
        line = line.strip().split("\t")
        ques = line[0]
        if len(line)==4:
            dumpfile.write(ques+'\t'+str(0)+'\n')
        elif len(line)==5:
            dumpfile.write(ques+'\t'+str(1)+'\n')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",type=str,default="../data//ccks_wiki",help="the data path included the kb")
    args = parser.parse_args()
    base_path = args.base_path
    train_path = base_path+"//train_ques.txt"
    valid_path = base_path+"//valid_ques.txt"
    with open(train_path,'r') as train_file:
        trian_lines = train_file.readlines()    
    valid_file = open(valid_path,'w')
    train,valid = train_test_split(trian_lines,test_size=0.2,shuffle=True)
    with open(train_path,'w') as train_file:
        train_file.writelines(train)
        valid_file.writelines(valid)
    
    data_path = base_path+"//train_ques.txt"
    dump_path = base_path+"//hop_ques.txt"

    gen_ques_classifier(data_path,dump_path)
    # no_answer_path = base_path+"//no_answer.txt"
    # predict_data_path = base_path+"//predict_data.txt"
    # with open(no_answer_path,'r') as f:
    #     idx_lines = f.readlines()
    # idx_lines = [int(idx.split("\t")[0]) for idx in idx_lines]
    # with open(predict_data_path,'r') as f:
    #     predict_lines = f.readlines()
    # with open(no_answer_path,'w') as f:
    #     for idx in idx_lines:
    #         f.write(str(idx)+"\t"+predict_lines[idx-1])       