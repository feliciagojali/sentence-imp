import re 
import ast
import json
import pandas as pd
from collections import Counter
data = pd.read_csv('data/raw/val_summary_corpus.csv')

def check_pred_id(pred_id, pas_list):
    arg_list = [x['args'] for x in pas_list if x['id_pred'][0] == pred_id[0]]
    return arg_list

def is_overlapped(range_1, range_2):
    s_1, e_1 = range_1
    s_2, e_2 = range_2
    
    arr = [x for x in range(max(s_1, s_2), min(e_1, e_2)+1)]
    
    if (len(arr)!=0):
        return True
    else:
        return False
    
def filter_no_overlapping_spans(arr, value):
    overlapped_pas = []
    for i, (sent, val_sent) in enumerate(zip(arr, value)):
        for j, (pas, val_pas) in enumerate(zip(sent, val_sent)):
            args = pas['args']
            overlapped = []
            
            for id_1, arg_1 in enumerate(args):
                for id_2 in range(id_1+1, len(args)):
                    arg_2 = args[id_2]
                    if (is_overlapped(arg_1[:2], arg_2[:2])):
                        if (len(overlapped) == 0):
                            overlapped.append([id_1, id_2])
                        else:
                            for k, o in enumerate(overlapped):
                                if id_1 in o or id_2 in o:
                                    overlapped[k].extend([id_1, id_2])
                                    break
            if (len(overlapped) != 0):
                for overlap in overlapped:
                    overlap = list(set(overlap))
                    val = [val_pas[x] for x in overlap]
                    sorted_overlap = [x for _, x in sorted(zip(val, overlap), key=lambda pair:pair[0], reverse=True)]
                    chosen = [sorted_overlap[0]]
                    not_chosen = sorted_overlap[1:]
                    for not_id in not_chosen:
                        id = 0
                        isOverlapped = False
                        while(id < len(chosen) and not isOverlapped):
                            current_id = chosen[id]
                            isOverlapped = is_overlapped(args[not_id][:2], args[current_id][:2])
                            id +=1
                        if (not isOverlapped):
                            chosen.append(not_id)
                    not_chosen = sorted(list(set(sorted_overlap) - set(chosen)))
                    for o in not_chosen:
                        overlapped_pas.append([i, j, o])
    
    for id in reversed(overlapped_pas):
        i, j, o = id
        del arr[i][j]['args'][o]
    return arr     

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

def _print_f1(total_gold, total_predicted, total_matched, message=""):
    print(total_gold)
    print(total_predicted)
    print(total_matched)
    precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
    recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print ("{}: Precision: {}, Recall: {}, F1: {}".format(message, precision, recall, f1))
    return precision, recall, f1
# Evaluate prediction and real, input is readable arg list
def evaluate( y, pred):
    # Adopted from unisrl
    total_gold = 0
    total_pred = 0
    total_matched = 0
    total_unlabeled_matched = 0
    comp_sents = 0
    label_confusions = Counter()
    total_gold_label = Counter()
    total_pred_label = Counter()
    total_matched_label = Counter()
    for y_sent, pred_sent in zip(y, pred):
        gold_rels = 0
        pred_rels = 0
        matched = 0
        for gold_pas in y_sent:
            pred_id = gold_pas['id_pred']
            gold_args = gold_pas['args']
            total_gold += len(gold_args)
            gold_rels += len(gold_args)
            total_gold_label.update(['V'])
            
            arg_list_in_predict = check_pred_id(pred_id, pred_sent)
            if (len(arg_list_in_predict) == 0):
                continue
            total_matched_label.update(['V'])
            for arg0 in gold_args:
                label = arg0[-1]
                total_gold_label.update([label])
                for arg1 in arg_list_in_predict[0]:
                    if (arg0[:-1] == arg1[:-1]): # Right span
                        total_unlabeled_matched += 1
                        label_confusions.update([(arg0[2], arg1[2]),])
                        if (arg0[2] == arg1[2]): # Right label
                            total_matched_label.update([arg0[2]])
                            total_matched += 1
                            matched += 1
        for pred_pas in pred_sent:
            total_pred_label.update(['V'])
            pred_id = pred_pas['id_pred']
            pred_args = pred_pas['args']
            for arg1 in pred_args:
                label = arg1[-1]
                total_pred_label.update([label])
            total_pred += len(pred_args)
            pred_rels += len(pred_args)
        
        if (gold_rels == matched and pred_rels == matched):
            comp_sents += 1

    precision, recall, f1 = _print_f1(total_gold, total_pred, total_matched, "SRL")

    return

def filter_article(sent, isLast=False, isFirst=False):
    result = sent
    result = re.sub(r"(/[a-z]*)?&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6}) ;", "", result) # remove HTML entites
    if (isFirst):
        result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn] ?6? ?[.] ?[cC]om [.,] [a-zA-Z ]+ : ', '', result) # remove liputan6 .com with place
        result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn]( [.] )6? ?[.] ?[cC]om [.,] [a-zA-Z ]+ : ', '', result)
        result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn] ?6? ?[.] ?[cC]om [.,] ', '', result) # remove liputan6 .com only
        result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn]( [.] )6? ?[.] ?[cC]om [.,] ', '', result) # remove liputan6 .com only
        result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn] ?6? ?[.] ?[cC]om [a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
        result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn]( [.] )6? ?[.] ?[cC]om [a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
        result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn] ?6? ?, ?[a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
        result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn]( [.] )6? ?, ?[a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
        result = re.sub('^[a-zA-Z]+ : [Ll][iI][pP][Uu][Tt][Aa][Nn] ?6? ?. ?[Cc]om , [a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
        result = re.sub('^[a-zA-Z]+ : [Ll][iI][pP][Uu][Tt][Aa][Nn]( [.] )6? ?. ?[Cc]om ?, ?[a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
        result = re.sub('[Ll][iI][pP][Uu][Tt][Aa][Nn] ?6? ?. ?[cC]om ?, ?[a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
        result = re.sub('[Ll][iI][pP][Uu][Tt][Aa][Nn]( [.] )6? ?. ?[cC]om ?, ?[a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
        result = re.sub('^[a-zA-Z]+ : ', '', result) # remove place
    if (isLast):
        result = re.sub("[(].*[)] ?[.]? ?$", "", result) #remove author
    return result

def preprocess(sent):
    if (not isinstance(sent, list)):
        sent = ast.literal_eval(sent)
    sents = [' '.join(sen) for sen in sent]
    sents = [filter_article(s, id==len(sents)-1, id==0) for  id, s in enumerate(sents)]
    sentences = [s.split(' ') for s in sents if s != '.' ]
    sentences = [[x for x in sent if x] for sent in sentences]
    sentences = [x for x in sentences if len(x) > 1]
    return sentences

def print_tokens(i, j):
    doc = current_corpus[i]
    for id, word in enumerate(doc[j]):
        print('id '+str(id) + ' = ' + str(word))
    if j == len(doc) - 1:
        return False
    return True


# id =  [3006, 10353, 10289, 10334, 2030, 10243, 5549, 1047, 6359, 8723]

# current_corpus = []
# for i in id:
#     doc = data['clean_article'][i]
#     current_corpus.append(preprocess(doc))


pred = [[{"id_pred":[1, 1], "args":[[0, 0, "ARG0"], [2, 2, "ARG1"],[3, 9, "ARG3"], [6, 6, "AM-LVB"]]}, {"id_pred": [7, 7], "args":[[4, 4, "ARG1"], [6, 6, "AM-TMP"], [8, 9, "ARG2"]]}]]

gold = [[{"id_pred":[1, 1], "args":[[0, 0, "ARG0"], [2, 2, "ARG1"],[3, 9, "ARG2"]]}, {"id_pred": [7, 7], "args":[[4, 4, "ARG1"], [6, 6, "AM-LVB"], [8, 9, "ARG2"]]}]]




# print(pred_[0])
evaluate(gold, pred)