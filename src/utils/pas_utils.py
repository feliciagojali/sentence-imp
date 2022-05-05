import json
import string
import numpy as np
import tensorflow as tf
from anytree import LevelOrderIter
from models import ExtractedPAS, NewPAS
from spansrl.src.features import SRLData
from tensorflow.keras.models import load_model
from .variables import additional_predicates, core_labels, verb_labels, additional_pred_regex, identified_predicates
tf.random.set_seed(42)

def load_srl_model(config):
    srl_data = SRLData(config)
    print('loading model..')
    srl_model = load_model(config['srl_model'])
    return srl_model, srl_data

def append_pas(data, type):
    filename = 'data/results/'+type+'_srl.json'
    try:
        f = open(filename)
        current = json.load(f)['data']
    except:
        current = []
    
    all = []
    all.extend(current)
    all.extend(data)
    with open(filename, "w") as file:
        all = {'data':all}
        json.dump(all, file)
    
def predict_srl(doc, srl_data, srl_model, config):
    ## Convert features 
    srl_data.extract_features(doc)
    feature1 = srl_data.word_emb_ft if config['use_fasttext'] else srl_data.word_emb_w2v
    feature2 = srl_data.word_emb_2
    feature3 = srl_data.char_input

    input = [feature1, feature2, feature3]
    
    ## Predicting
    if (config['use_pruning']):
        pred, idx_pred, idx_arg = srl_model.predict(input, batch_size=config['batch_size'])
        res =  srl_data.convert_result_to_readable(pred, idx_arg, idx_pred)
    else:
        pred = srl_model.predict(input, batch_size=config['batch_size'])
        res =  srl_data.convert_result_to_readable(pred)
    
    return res

def filter_incomplete_pas(pas_list, pos_tag_sent, isTraining=False):
    filtered = []
    for pas in pas_list:
        arg_list = [p[2] for p in pas['args']]
        
        # check if predicate out of range
        pred_id = pas['id_pred'][0]
        try:
            pred = pos_tag_sent.tokens[pred_id]
        except:
            # index out of range
            continue
        
        if (not isTraining):
            # PAS Selection Rules
            # must have core arguments conditions
            if(not bool(set(arg_list) & set(core_labels))):
                continue
            
            # at least two tokens arguments that is NOT punctuation
            max_len = len(pos_tag_sent.tokens)
            tokens = [[x for x in range(arg[0], arg[1]+1) if x < max_len] for arg in pas['args']]
            tokens = list(set(get_flatten_arguments(tokens)))
            filtered_tokens = [x for x in tokens if pos_tag_sent.tokens[x].name.text not in string.punctuation ]
            
            if (len(filtered_tokens) < 2):
                continue
            
        filtered.append(pas)
    return filtered
            
def filter_pas(pas_list, pos_tag_sent):
    if len(pas_list) == 1:
        return pas_list
    pred_list = [node.name.position - 1 for node in LevelOrderIter(pos_tag_sent.root) if node.name.pos_tag == 'VERB' ]
    i = 0
    chosen = []
    while(len(chosen) == 0 and i < len(pred_list)):
        pred_id = pred_list[i]
        chosen = [pas for pas in pas_list if pas['id_pred'][0] == pred_id]
        i+=1
    # If not defined as verb in pos tag then consider all tokens
    if (len(chosen) == 0):
        token_list = [node.name.position - 1 for node in LevelOrderIter(pos_tag_sent.root)]
        while(len(chosen) == 0 and i < len(token_list)):
            pred_id = token_list[i]
            chosen = [pas for pas in pas_list if pas['id_pred'][0] == pred_id]
            i+=1
    return chosen

def convert_PAS(pas, pos):
    new_pas = NewPAS()
    new_pas.add_arg(pas, pos)
    return new_pas

def convert_to_PAS_models(pas_article, pos_tag):
    return [[convert_PAS(pas, pos_tag_sent) for pas in pas_sent] for pas_sent, pos_tag_sent in zip(pas_article, pos_tag) ]

def convert_extractedPAS(sentences, pas):
    return ExtractedPAS(None, sentences, pas)

def convert_to_extracted_PAS(pas_list, sent_list):
    return [convert_extractedPAS(sent, pas) for (sent, pas) in zip(pas_list, sent_list)]

def get_sentence(extracted_pas):
    tokens = [get_flatten_pas(pas) for pas in extracted_pas.pas]
    pas_tokens = [[extracted_pas.tokens[t].name.text for t in token] for token in tokens]
    return pas_tokens

def get_flatten_arguments(arguments):
    tokens = [item for sublist in arguments for item in sublist]
    return tokens

def get_flatten_pas(pas):
    tokens = []
    tokens.extend(pas.verb)
    args = pas.args
    for arg in args:
        tokens.extend(get_flatten_arguments(args[arg]))
    tokens = set(tokens)
    tokens = sorted(tokens)
    return tokens
