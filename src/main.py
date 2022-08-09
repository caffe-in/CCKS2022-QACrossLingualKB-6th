from gc import callbacks
from operator import mod
import pytorch_lightning as pl
import transformers
import torch
from classifier import classifier_baseline
from data_utility import PreDataset, create_kb,search_all_path
from transformers import BertModel,BertConfig,BertTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode",type=str,default="train")
parser.add_argument("--base_path",type=str,default="../data//ccks_wiki",help="the data path included the kb")
parser.add_argument("--pretrained_path",type=str,default="/home/pretrains/pt/bert-base-uncased",help="the dir included the pretrain model")
parser.add_argument("--model_path",type=str,default="../lightning_logs/ccks_wiki/version_0/checkpoints/epoch=0-step=477.ckpt")
parser.add_argument("--save_path",type=str,default="ccks_wiki")

args = parser.parse_args()

class BertSim(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = transformers.BertModel.from_pretrained(args.pretrained_path)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768,2)
    def forward(self,ids,mask,token_type_ids):
        _,output_1 = self.l1(ids,attention_mask = mask,token_type_ids = token_type_ids,return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    def training_step(self,batch,batch_idx):

        ids = batch[0]
        mask = batch[2] 
        token_type_ids = batch[1] 
        targets = batch[3] 
        outputs = self.forward(ids,mask,token_type_ids)
        loss = self.loss_fn(outputs,targets)
        return loss
    def validation_step(self,batch,batch_idx):
        ids = batch[0]
        mask = batch[2]
        token_type_ids = batch[1]
        targets = batch[3]
        outputs = self.forward(ids,mask,token_type_ids)

        loss = self.loss_fn(outputs,targets)
        self.log("val_loss",loss,prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(),lr=1e-05)
        return optimizer
    def loss_fn(self,outputs,targets):
        return torch.nn.BCEWithLogitsLoss()(outputs,targets)

def get_features(ques,all_path_list:list,tokenizer)->list:
    
    ques_list = [ques]*len(all_path_list)
    for idx, line in enumerate(all_path_list):
        line = "\t".join(line)
        all_path_list[idx] = line
    try:
        feature_list = tokenizer(ques_list,all_path_list,padding=True,truncation=True,return_tensors="pt")
    except ValueError:
        return None
    return feature_list

def predict(base_path,model_path,kb_path):
    predict_path = base_path+"//predict_ques.txt"
    dump_path = base_path+"//predict_data.txt"
    corpus_path = base_path+"//hop_ques.txt"
    hop_predict_path = base_path+"//hop_predict.txt"

    predict_file = open(predict_path,'r')
    dump_file  = open(dump_path,'w')
    hop_predict_file = open(hop_predict_path,'w')
    predcit_lines = predict_file.readlines()
    predcit_lines = [line.strip().split("\t") for line in predcit_lines]

    # predict the ques hop num
    classi = classifier_baseline()
    classi.train(corpus_path)
    texts = [line[0] for line in predcit_lines]
    hop_lines = classi.predict(texts)
    for line in hop_lines:
        hop_predict_file.write(str(int(line))+"\n")

    pretrianed_path = args.pretrained_path
    model = BertSim.load_from_checkpoint(model_path)
    tokenizer = BertTokenizer.from_pretrained(pretrianed_path)
    #predict start
    print("===================predict start===========================")
    kb = create_kb(kb_path)
    for idx,line in enumerate(predcit_lines):
        if idx%100==0:
            print(idx)
        question = line[0]
        topic_entity = line[1]
        hop_num = hop_lines[idx]
        hop_num=2 if hop_num==0 else 3

        all_rel_path = search_all_path(topic_entity,kb,hop_num)
        all_rel_path = list(set([tuple(p) for p in all_rel_path]))
        # print("the idx {},the length {} ".format(idx,len(all_rel_path)))
        features_list = get_features(question,all_rel_path,tokenizer)
        if features_list==None:
            answer_rel=""
        else:
            input_ids = features_list["input_ids"].long()
            token_type_ids = features_list["token_type_ids"].long()
            attention_mask = features_list["attention_mask"].long()

            outputs = model.forward(input_ids,attention_mask,token_type_ids)
        
            answer_idx = torch.max(outputs,0).indices[0]
            answer_rel = all_rel_path[answer_idx]

        write_str = question+"\t"+topic_entity+"\t"+answer_rel+"\n"
        dump_file.write(write_str)


    
    print("==================predict end===========================")







if __name__=="__main__":
    base_path = args.base_path
    mode = args.mode
    if mode=="train":
        batch_size=32
        logger = TensorBoardLogger("..//lightning_logs",name=args.save_path)
       

        train_data_path = base_path+"//train_ques.txt"
        valid_data_path = base_path+"//valid_ques.txt"
        kb_path = base_path+"//triples.txt"
        pretrianed_path = args.pretrained_path

        train_dataset = PreDataset(train_data_path,kb_path,pretrianed_path,mode="train")
        valid_dataset = PreDataset(valid_data_path,kb_path,pretrianed_path,mode="train")
        train_dataloader = DataLoader(train_dataset,batch_size,shuffle=False,num_workers=4)
        valid_dataloader = DataLoader(valid_dataset,batch_size,shuffle=False,num_workers=4)
        model = BertSim()
        trainer = pl.Trainer(gpus=[0],max_epochs=40,logger=logger,callbacks=[EarlyStopping(monitor="val_loss",mode="min")])
        trainer.fit(model,train_dataloader,valid_dataloader)
    elif mode=="predict":
        base_path = args.base_path
        kb_path = base_path+"//triples.txt"
        model_path = args.model_path

        predict(base_path,model_path,kb_path)
