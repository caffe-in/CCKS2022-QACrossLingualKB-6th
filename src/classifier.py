import json
import os
from os import truncate
from pickletools import optimize
import shutil
from tabnanny import check
from turtle import forward
from sklearn.utils import shuffle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from transformers import BertForSequenceClassification, BertModel,BertConfig,BertTokenizer
from torch.utils.data import Dataset,DataLoader,SequentialSampler
from transformers.optimization import AdamW
import transformers
import sklearn.metrics as metrics
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#1. train, valid, test generation
base_path = "..//data//ccks"
corpus_path = base_path+"//hop_ques.txt"

class classifier_baseline:
    def __init__(self):
        self.pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('lr', LogisticRegression(multi_class="ovr", solver="lbfgs"))
        ])
        self.parameters = {'lr__C': [0.1, 0.5, 1, 2, 5, 10, 100, 1000]}
        
    def train(self,corpus_path):
        with open(corpus_path,'r',encoding='utf8') as f:
            lines = f.readlines()
        texts,labels = [],[]
        for line in lines:
            line = line.strip().split('\t')
            texts.append(line[0])
            labels.append(int(line[1]))     # 0 for 2hop, 1 for 3hop
        rest_texts,test_texts,rest_labels,test_labels = train_test_split(texts,labels,test_size=0.1,random_state=1)
        train_texts,dev_texts,train_labels,dev_labels = train_test_split(rest_texts,rest_labels,test_size=0.1,random_state=1)

        self.best_classifier = GridSearchCV(self.pipeline, self.parameters, cv=5, verbose=1)
        self.best_classifier.fit(train_texts, train_labels)

        # best_predictions = self.best_classifier.predict(test_texts)
        # baseline_accuracy = np.mean(best_predictions == test_labels)
        # print("Baseline accuracy:", baseline_accuracy)
        # print(test_texts[0:10])
        # print(best_predictions[0:10])

    def predict(self,texts):
        return self.best_classifier.predict(texts)