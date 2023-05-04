# -*- coding: utf-8 -*-
"""
Handles data loading.

@author: Geerthan
"""

import pandas as pd
import torch
import torchtext


labelMap = {'None': 0, 'Positive': 1, 'Negative': 2}
maxSenLen = 128
maxSubjLen = 2

def get_other(mode, subj, asp, tokenizer, vocab, afinn_dict):
    train_data = pd.read_csv('data/sentihood/' + subj + '_' + asp + '/' + mode + '.tsv', 
                             header=None, sep="\t").values
        
    return process_tsv(train_data, tokenizer, subj, vocab, afinn_dict)

def get_train(subj, asp, tokenizer, afinn_dict):
    train_data = pd.read_csv('data/sentihood/' + subj + '_' + asp + '/train.tsv', 
                             header=None, sep="\t").values
    
    print('Train len:', len(train_data))
    return process_train_tsv(train_data, tokenizer, subj, afinn_dict)
        
    
def process_tsv(dataset, tokenizer, subj, vocab, afinn_dict):
    sentences = []
    subj_locs = []
    sen_sents = []
    ids = []
    labels = []
    
    for line in dataset:
        sentence = str(line[1])
        sentence = sentence.replace('location - 1', 'loc1')
        sentence = sentence.replace('location - 2', 'loc2')
        sentences.append(sentence)
                         
        labels.append(str(line[2]))

    for i in range(len(sentences)):
        sentences[i] = tokenizer(sentences[i])
    
    for sentence in sentences:
        sen_id = []
        sen_sent = []
        subj_loc = []
        for i in range(len(sentence)):
            if i >= maxSenLen:
                break
            sen_id.append(vocab[sentence[i]])
            
            if sentence[i] in afinn_dict:
                sen_sent.append(afinn_dict[sentence[i]])
            else:
                sen_sent.append(0)
        
            if sentence[i] == subj and len(subj_loc) < maxSubjLen:
                subj_loc.append(i)
        
        # Uncomment to use AbsaGRU-Last
        '''
        if len(subj_loc) == 0:
            subj_loc.append(len(sen_id)-1)
        else:
            subj_loc[0] = len(sen_id)-1
        '''
            
        for i in range(len(sen_id), maxSenLen):
            sen_id.append(vocab['.'])
            sen_sent.append(0)
            
        for i in range(len(subj_loc), maxSubjLen):
            subj_loc.append(-1)
            
        subj_locs.append(subj_loc)
        sen_sents.append(sen_sent)
        ids.append(sen_id)
                
    for i in range(len(labels)):
        labels[i] = labelMap[labels[i]]
    
            
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    lbl_tensor = torch.tensor(labels, dtype=torch.long)
    sub_tensor = torch.tensor(subj_locs, dtype=torch.long)
    sent_tensor = torch.tensor(sen_sents, dtype=torch.long)
    
    data = torch.utils.data.TensorDataset(ids_tensor, lbl_tensor, sub_tensor, sent_tensor)
    
    return data
        
def process_train_tsv(dataset, tokenizer, subj, afinn_dict):
    sentences = []
    subj_locs = []
    sen_sents = []
    ids = []
    labels = []
    
    for line in dataset:
        sentence = str(line[1])
        sentence = sentence.replace('location - 1', 'loc1')
        sentence = sentence.replace('location - 2', 'loc2')
        sentence = sentence.replace('-', ' ')
        sentence = sentence.replace('  ', ' ')
        sentences.append(sentence)
                         
        labels.append(str(line[2]))

    for i in range(len(sentences)):
        sentences[i] = tokenizer(sentences[i])
         
    print("Loading GloVe")
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    vocab = torchtext.vocab.vocab(glove.stoi)

    vocab.insert_token('<unk>', 0)
    
    pretrain_embed = glove.vectors
    pretrain_embed = torch.cat((torch.zeros(1, pretrain_embed.shape[1]), pretrain_embed))

    vocab.set_default_index(0)
    print("GloVe loaded")
    
    for sentence in sentences:
        sen_id = []
        sen_sent = []
        subj_loc = []
        for i in range(len(sentence)):
            if i >= maxSenLen:
                break
                continue
            sen_id.append(vocab[sentence[i]])
            
            if sentence[i] in afinn_dict:
                sen_sent.append(afinn_dict[sentence[i]])
            else:
                sen_sent.append(0)
            
            if sentence[i] == subj and len(subj_loc) < maxSubjLen:
                subj_loc.append(i)
            
        # Uncomment to use AbsaGRU-Last
        '''
        if len(subj_loc) == 0:
            subj_loc.append(len(sen_id)-1)
        else:
            subj_loc[0] = len(sen_id)-1
        '''
            
        for i in range(len(sen_id), maxSenLen):
            sen_id.append(vocab['.'])
            sen_sent.append(0)

        for i in range(len(subj_loc), maxSubjLen):
            if i == 0:
                subj_loc.append(-1)
            else:
                subj_loc.append(subj_loc[0])

            
        subj_locs.append(subj_loc)
        sen_sents.append(sen_sent)
        ids.append(sen_id)

    for i in range(len(labels)):
        labels[i] = labelMap[labels[i]]

    
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    lbl_tensor = torch.tensor(labels, dtype=torch.long)
    sub_tensor = torch.tensor(subj_locs, dtype=torch.long)
    sent_tensor = torch.tensor(sen_sents, dtype=torch.long)
    
    train_data = torch.utils.data.TensorDataset(ids_tensor, lbl_tensor, sub_tensor, sent_tensor)
    
    freq = torch.bincount(lbl_tensor) / len(lbl_tensor)
    train_weights = [1/freq[lbl_tensor[i]] for i in range(len(lbl_tensor))]
    
    return train_data, train_weights, pretrain_embed, vocab
    