# CCKS2022-QACrossLingualKB-6th

this is path reasoning code for CCKS2022: Question Answering over Cross-lingual Knowledge Graphs 6th





## Installation

```python
pip install -r requirements.txt
```

## Data Prepare

the data dir tree should be this. raw data can be get from https://github.com/RexWzh/CCKS-mKGQA

```
data
----ccks_wiki
        predict_ques.txt
        train_ques.txt
        triples.txt
```
attention: you may need rename the "valid_ques.txt" with "predict_ques.txt"
the valid_ques.txt will be generated automaticly in data prepare
then you should run the data_utility.py in src

```python
python data_utility.py --base_path ccks_wiki
```

you can change `ccks_wiki` to other name

the data dir tree now will be this

```
data
----ccks_wiki
        hop_ques.txt
        predict_ques.txt
        train_ques.txt
        triples.txt
        valid_ques.txt
```

## Training

```python
python main.py --mode train
```

**YOU CAN CHANGE THE PARAMETER in main.py**

```python
parser.add_argument("--base_path",type=str,default="../data//ccks_ILL11",help="the data path included the kb")
parser.add_argument("--pretrained_path",type=str,default="/home/pretrains/pt/bert-base-uncased",help="the dir included the pretrain model")
parser.add_argument("--model_path",type=str,default="lightning_logs/ccks_ILL11/version_0/checkpoints/epoch=11-step=11195.ckpt")
parser.add_argument("--save_path",type=str,default="ccks_ILL11")
parser.add_argument("--mode",type=str,default="train")
```

you can find the model at the`lightning_logs/*save_path*/version_x/checkpoints/epoch=xxxx.ckpt"`

## predict

```python
python main.py --mode predict
```

you can find the result in data//base_path//predict_data.txt



after path reasoning is the work: https://github.com/RexWzh/CCKS-mKGQA

if you have any problem, please post a issue.
