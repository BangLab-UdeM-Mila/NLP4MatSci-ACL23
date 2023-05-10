import pandas as pd
import numpy as np
import torch, json, os, time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoModel
from model import LanguageModel
from utils import *
import Levenshtein
from collections import defaultdict
import argparse
import datetime
import json
from logger import setup_logger
import logging

class Args(object):
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_length',default=512,type=float)
        parser.add_argument('--hidden_dim',default=768,type=int)
        parser.add_argument('--batch_size',default=24,type=int)
        parser.add_argument('--n_heads',default=8,type=int)
        parser.add_argument('--num_decoder_layer',default=3,type=int)
        parser.add_argument('--epoch',default=20,type=int)
        parser.add_argument('--lr',default=2e-5,type=float)
        parser.add_argument('--log_interval',default=100,type=int)
        parser.add_argument('--explanation',default=False,action='store_true')
        parser.add_argument('--setting',default='low_resource',type=str)
        parser.add_argument('--train_size',default=0.01,type=float)
        parser.add_argument('--basemodel',default='matscibert',type=str)
        parser.add_argument('--explanation_type',default='choice',type=str)
        parser.add_argument('--even_split',default=False,action='store_true')
        self.args  = parser.parse_args()
        print('parse args = ',self.args)

        def to_json(self):
            return json.dumps(self,default=lambda o: o.__dict__)

args = Args().args

MAX_LENGTH = args.max_length
HIDDEN_DIM = args.hidden_dim
BATCH_SIZE = args.batch_size
N_HEADS    = args.n_heads
NUM_DECODER_LAYER = args.num_decoder_layer
EPOCH = args.epoch
LR = args.lr
LOG_INTERVAL = args.log_interval
EXPLANATION = args.explanation
SETTING = args.setting
TRAIN_SIZE = args.train_size
BASE_MODEL = args.basemodel
EXPLANATION_TYPE = args.explanation_type
EVEN_SPLIT = args.even_split

param_list = [MAX_LENGTH, HIDDEN_DIM, BATCH_SIZE, N_HEADS, NUM_DECODER_LAYER, EPOCH, LR, LOG_INTERVAL, EXPLANATION, SETTING, TRAIN_SIZE, BASE_MODEL, EXPLANATION_TYPE, EVEN_SPLIT]


logger_name = '_'.join([str(x) for x in param_list])
time_now = setup_logger('MultiTaskPromptLearning', logger_name, level=logging.INFO, screen=False, tofile=True)
logger = logging.getLogger('MultiTaskPromptLearning')
logger.info('params = {}'.format('_'.join([str(x) for x in param_list])))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def most_similar_answer(a,answer_set):
    a = a.strip().replace(' ', '')
    if(a in answer_set):
        return a
    dis = [Levenshtein.distance(a,x) for x in answer_set]
    idx = np.argmin(dis)
    return answer_set[idx]

def decoding(true,pred,qtype,res_dict):
    y_true = defaultdict(list)
    y_pred = defaultdict(list)
    
    for x,y,t in zip(true,pred,qtype):
        x = x.lower()
        y = y.lower()
        t = int(t)
        if (t==0):
            answer_map = res_dict['t_type_dict']
            answer_set = res_dict['t_type_set']
            y_true['ner'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['ner'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==1):
            answer_map = res_dict['pc_type_dict']
            answer_set = res_dict['pc_type_set']
            y_true['pc'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['pc'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==2):
            answer_map = res_dict['r_type_dict']
            answer_set = res_dict['r_type_set']
            y_true['re'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['re'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==3):
            x = x.strip().replace(' ', '')
            y = y.strip().replace(' ', '')
            if (len(x)==0 and len(y)==0):
                y_pred['arg'].append(1)
            elif (len(x)==0):
                y_pred['arg'].append(0)
            elif (len(y)==0):
                answer_map = res_dict['e_role_dict']
                answer_set = res_dict['e_role_set']
                tmp_x = x.split(',')
                for a in tmp_x:
                    true_role = a.split(':')[1]
                    y_pred['arg'].append(0)
                    y_true['ee'].append(answer_map[true_role.strip().replace(' ', '')])
                    y_pred['ee'].append(answer_map[most_similar_answer(' ',answer_set)])
            else:
                tmp_x = x.split(',')
                tmp_y = y.split(',')
                answer_map = res_dict['e_role_dict']
                answer_set = res_dict['e_role_set']
                if(len(tmp_x)==len(tmp_y)):
                    pass
                elif(len(tmp_x)<len(tmp_y)):
                    tmp_y = tmp_y[0:len(tmp_x)]
                else:
                    tmp_y = tmp_y + [':']*(len(tmp_x)-len(tmp_y))
                for a,b in zip(tmp_x,tmp_y):
                    try:
                        true_arg,true_role = a.split(':')
                        pred_arg,pred_role = b.split(':')
                        if (true_arg==pred_arg):
                            y_pred['arg'].append(1)
                        else:
                            y_pred['arg'].append(0)
                        y_true['ee'].append(answer_map[true_role.strip().replace(' ', '')])
                        y_pred['ee'].append(answer_map[most_similar_answer(pred_role,answer_set)])
                    except:
                        true_arg,true_role = a.split(':')
                        y_pred['arg'].append(0)
                        y_true['ee'].append(answer_map[true_role.strip().replace(' ', '')])
                        y_pred['ee'].append(answer_map[most_similar_answer(' ',answer_set)])
        if (t==4):
            answer_map = res_dict['sar_dict']
            answer_set = res_dict['sar_set']
            y_true['sar'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['sar'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==5):
            answer_map = res_dict['sc_type_dict']
            answer_set = res_dict['sc_type_set']
            y_true['sc'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['sc'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==6):
            answer_map = res_dict['sf_type_dict']
            answer_set = res_dict['sf_type_set']
            y_true['sf'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['sf'].append(answer_map[most_similar_answer(y,answer_set)])
    return y_true,y_pred

def calc_loss(criterion,output,tgt,vocab_size):
    output = output.contiguous().view(-1, vocab_size)
    tgt = tgt.contiguous().view(-1)
    loss = criterion(output, tgt)
    return loss

def metric(labels,preds):
    assert len(labels)==len(preds)
    if (len(labels)==0):
        return 0,0
    micro_f1 = f1_score(labels,preds,average='micro')
    macro_f1 = f1_score(labels,preds,average='macro')
    return micro_f1,macro_f1

def test(model, dataloader, criterion, vocab_size, res_dict):
    model.eval()
    total_loss = 0
    all_start_time = time.time()
    matscibert_tokenizer = res_dict['tokenizer']
    y_true = defaultdict(list)
    y_pred = defaultdict(list)
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            answers = batch['answers']['input_ids'].to(device)
            batch['answers']['input_ids'] = batch['answers']['input_ids'][:,:-1]
            output = model(batch)
            
            preds = torch.argmax(output,dim=-1)

            loss = calc_loss(criterion = criterion, output = output, tgt = answers[:,1:], vocab_size = vocab_size)

            total_loss += loss
            
            real_ans = matscibert_tokenizer.batch_decode(answers,skip_special_tokens=True)
            pred_ans = matscibert_tokenizer.batch_decode(preds,skip_special_tokens=True)

            qtype = batch['qtype'].numpy().tolist()
            decode_true,decode_pred = decoding(real_ans,pred_ans,qtype,res_dict)
            for key in decode_true:
                y_true[key] += decode_true[key]
            for key in decode_pred:
                y_pred[key] += decode_pred[key]

    avg_loss = total_loss/len(dataloader)
    ppl = np.exp(avg_loss.item())
    all_end_time = time.time()
    logger.info('test | loss = {} ppl = {} time = {}'.format(avg_loss.item(), ppl, all_end_time-all_start_time))
    for key in y_true:
        micro,macro = metric(y_true[key],y_pred[key])
        logger.info('task = {} micro-f1 = {} macro-f1 = {}'.format(key,micro,macro))

def train(model, dataloader, criterion, vocab_size, optimizer, res_dict, show = True):
    model.train()
    total_loss = 0
    cur_loss   = 0
    all_start_time = time.time()
    start_time = time.time()
    matscibert_tokenizer = res_dict['tokenizer']
    y_true = defaultdict(list)
    y_pred = defaultdict(list)
    for k, batch in enumerate(dataloader):
        answers = batch['answers']['input_ids'].to(device)
        batch['answers']['input_ids'] = batch['answers']['input_ids'][:,:-1]
        output = model(batch)
        
        preds = torch.argmax(output,dim=-1)
        
        loss = calc_loss(criterion = criterion, output = output, tgt = answers[:,1:], vocab_size = vocab_size)

        
        total_loss += loss

        cur_loss += loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        real_ans = matscibert_tokenizer.batch_decode(answers,skip_special_tokens=True)
        pred_ans = matscibert_tokenizer.batch_decode(preds,skip_special_tokens=True)

        qtype = batch['qtype'].numpy().tolist()
        decode_true,decode_pred = decoding(real_ans,pred_ans,qtype,res_dict)
        for key in decode_true:
            y_true[key] += decode_true[key]
        for key in decode_pred:
            y_pred[key] += decode_pred[key]

        
        if k % LOG_INTERVAL == 0 and k > 0:
            cur_loss = cur_loss / LOG_INTERVAL
            ppl = np.exp(cur_loss.item())
            end_time = time.time()
            logger.info('train | batch = {} loss = {} ppl = {} time = {}'.format(k,cur_loss.item(),ppl,end_time-start_time))
            for key in y_true:
                micro,macro = metric(y_true[key],y_pred[key])
                logger.info('task = {} micro-f1 = {} macro-f1 = {}'.format(key,micro,macro))
            if (show is True):
                logger.info('real_ans = ',real_ans[0])
                logger.info('pred_ans = ',pred_ans[0])
            cur_loss   = 0
            start_time = time.time()

    avg_loss = total_loss/len(dataloader)
    ppl = np.exp(avg_loss.item())
    all_end_time = time.time()
    logger.info('train | loss = {} ppl = {} time = {}'.format(avg_loss.item(),ppl,all_end_time-all_start_time))
    for key in y_true:
        micro,macro = metric(y_true[key],y_pred[key])
        logger.info('task = {} micro-f1 = {} macro-f1 = {}'.format(key,micro,macro))

if __name__ == "__main__":
    train_loader, test_loader, pad_idx, vocab_size, res_dict = build_dataloader(MAX_LENGTH,BATCH_SIZE,explanation=EXPLANATION,setting=SETTING,train_size=TRAIN_SIZE, base_model=BASE_MODEL, explanation_type=EXPLANATION_TYPE, even_split=EVEN_SPLIT)
    params = dict()
    params['max_seq_length'] = MAX_LENGTH
    params['vocab_size'] = vocab_size
    params['hidden_dim'] = HIDDEN_DIM
    if (BASE_MODEL=='matscibert'):
        params['encoder'] = AutoModel.from_pretrained("m3rg-iitd/matscibert") # maybe this is better, but in ComputeCanada, I need to load local model files
    elif (BASE_MODEL=='matbert'):
        # can be download from https://github.com/lbnlp/MatBERT
        params['encoder'] = AutoModel.from_pretrained("./matbert-base-uncased")
    elif (BASE_MODEL=='scibert'):
        params['encoder'] = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    elif (BASE_MODEL=='scholarbert'):
        params['encoder'] = AutoModel.from_pretrained('globuslabs/ScholarBERT_1')
    elif (BASE_MODEL=='biobert'):
        params['encoder'] = AutoModel.from_pretrained('seiya/oubiobert-base-uncased')
    elif (BASE_MODEL=='batterybert'):
        params['encoder'] = AutoModel.from_pretrained('batterydata/batterybert-uncased')
    else:
        raise ValueError('basemodel invalid!!!')
    decoder_layer = nn.TransformerDecoderLayer(d_model = HIDDEN_DIM, nhead = N_HEADS, batch_first = True)
    params['decoder'] = nn.TransformerDecoder(decoder_layer, num_layers = NUM_DECODER_LAYER)
    model = LanguageModel(params=params)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for e in range(1,EPOCH+1):
        logger.info('\n')
        logger.info('TRAINING EPOCH = {}'.format(e))
        show = True if e>1 else False
        train(model, train_loader, criterion, vocab_size, optimizer, res_dict, show)
        if (e==EPOCH):
            test(model, test_loader, criterion, vocab_size, res_dict)
        logger.info('ENDING EPOCH = {}'.format(e))