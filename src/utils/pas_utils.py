import tensorflow as tf
from models import ExtractedPAS, NewPAS
from spansrl.src.features import SRLData
from tensorflow.keras.models import load_model

tf.random.set_seed(42)

def load_srl_model(config):
    srl_data = SRLData(config)
    print('loading model..')
    srl_model = load_model(config['srl_model'])
    return srl_model, srl_data


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
