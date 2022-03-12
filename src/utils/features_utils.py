import math
import string
import os.path
import pandas as pd
from collections import Counter
from .pas_utils import get_flatten_pas, get_flatten_arguments
from gensim.models import Word2Vec, fasttext

verb_pos_tags = ["VERB"]
noun_pos_tags = ["NOUN"]
features_name = ["fst_feature", "p2p_feature", "length_feature", "num_feature", "noun_verb_feature", "pnoun_feature", "location_feature", "temporal_feature", "max_doc_similarity_feature", "avg_doc_similarity_feature", "min_doc_similarity_feature","position_feature", "title_feature", "target"]
features_path = 'data/features/'

def prepare_df(ext_pas_list, config, types, start_idx):
    dict_train = {}
    dict_train['idx_news'] = []
    if (types != 'train' and 'target' in features_name):
        features_name.remove('target')
    for f in features_name:
        dict_train[f] = []
    
    idx = start_idx
    for doc in ext_pas_list:
        for ext_pas in doc:
            flag = False
            for f in features_name:
                val = getattr(ext_pas, f)
                for i in val:
                    if (not flag):
                        dict_train['idx_news'].append(idx)
                    dict_train[f].append(i)
                flag = True
        idx += 1
  
    train = pd.DataFrame(data=dict_train)
    if (types=='train'):
        if (os.path.isfile(features_path + types+ '_' + config['features'])):
            current = load_train_df(config, types)
            train = pd.concat([current, train], ignore_index=True)    
        train.to_csv(features_path + types+ '_' + config['features'])
    return train

def prepare_features(config, types, train_df=None):
    if (train_df is None):
        train_df = load_train_df(config, types)
    features = train_df.drop(['max_doc_similarity_feature', 'idx_news'], axis =1)
    features_min = features.drop(['avg_doc_similarity_feature'], axis=1)
    features_avg = features.drop(['min_doc_similarity_feature'], axis=1)
    target = []
    if 'target' in train_df.columns:
        target = train_df['target']
        features_min = features_min.drop(['target'], axis=1)
        features_avg = features_avg.drop(['target'], axis=1)

    return features_min, features_avg, target

def load_train_df(config, types):
    current = pd.read_csv(features_path + types+ '_' + config['features'], index_col=0)
    return current

def load_sim_emb(config):
    w2v = Word2Vec.load(config['w2v_sim_path']).wv
    ft = fasttext.load_facebook_vectors(config['ft_sim_path'])
    return w2v, ft
     
def generate_features(ext_pas_list, similarity_table, corpus_title):
    for i, (doc, title) in enumerate(zip(ext_pas_list, corpus_title)):    
        corpus_vocabs, most_common_words = get_corpus_vocabs_and_most_common_words(doc)
        idx_pas = 0
        max_p2p = 1 #0
        max_fst = 1 #0
        max_length = 1
        for j, extracted_pas in enumerate(doc):
            extracted_pas.p2p_feature = [calculate_sim(similarity_table[i],idx_pas, id_p) for id_p in range(len(extracted_pas.pas))]
            extracted_pas.fst_feature = [calculate_fst_pas(extracted_pas.tokens, p, most_common_words) for p in extracted_pas.pas]
            extracted_pas.length_feature = [calculate_length_pas(extracted_pas.tokens, p) for p in extracted_pas.pas]
            extracted_pas.num_feature = [calculate_pos_pas(["NUM"],extracted_pas.tokens, p)/length for p, length in zip(extracted_pas.pas, extracted_pas.length_feature)]
            extracted_pas.noun_verb_feature = [calculate_pos_pas(["NOUN","VERB"],extracted_pas.tokens, p)/length for p, length in zip(extracted_pas.pas, extracted_pas.length_feature)]
            extracted_pas.pnoun_feature = [calculate_pos_pas(["PROPN"],extracted_pas.tokens, p)/length for p, length in zip(extracted_pas.pas, extracted_pas.length_feature)]
            extracted_pas.temporal_feature = [(calculate_arg_pas("AM-TMP", p)/length) for p, length in zip(extracted_pas.pas, extracted_pas.length_feature)]
            extracted_pas.location_feature = [calculate_arg_pas("AM-LOC", p)/length for p, length in zip(extracted_pas.pas, extracted_pas.length_feature)]
            extracted_pas.max_doc_similarity_feature = [calculate_max_similarity(similarity_table[i], idx_pas, id_p) for id_p in range(len(extracted_pas.pas))]        
            extracted_pas.min_doc_similarity_feature = [calculate_min_similarity(similarity_table[i], idx_pas, id_p) for id_p in range(len(extracted_pas.pas))]

            title_tokens = [get_flatten_pas(pas) for pas in extracted_pas.pas]
            pas_tokens = [[extracted_pas.tokens[t].name.text for t in token] for token in title_tokens]
            if (title):
                extracted_pas.title_feature = [calculate_title_word_occurence(title, pas_token) for pas_token in pas_tokens] 
            else:
                extracted_pas.title_feature = [-1 for _ in extracted_pas.pas]
            extracted_pas.sent_tokens = [pas_tokens[i] for i in range(len(extracted_pas.pas))]
            extracted_pas.title = [title for pas in extracted_pas.pas]
            idx_pas += len (extracted_pas.pas)
            max_p2p = max(max_p2p, max(extracted_pas.p2p_feature))
            max_length = max(max_length, max(extracted_pas.length_feature))
            max_fst = max(max_fst, max(extracted_pas.fst_feature))
            
        total = count_pas(doc) - 1
        for extracted_pas in doc:
            extracted_pas.avg_doc_similarity_feature = [val/total for val in extracted_pas.p2p_feature]
            extracted_pas.length_feature = [val/max_length for val in extracted_pas.length_feature]
            extracted_pas.p2p_feature = [val/max_p2p for val in extracted_pas.p2p_feature]
            extracted_pas.fst_feature = [val/max_fst for val in extracted_pas.fst_feature]
            pos = float(extracted_pas.num_sentences - extracted_pas.idx_sentence)/extracted_pas.num_sentences
            extracted_pas.position_feature = [pos for _ in extracted_pas.pas]

def generate_sim_table(ext_pas_list, ext_pas_flatten, emb):
    similarity_table = []
    for i, doc in enumerate(ext_pas_list):
        similarity_doc = []
        mask = create_mask_arr(doc)
        for j, pas in enumerate(ext_pas_flatten[i]):
            similarity_pas = []
            for k in range(j+1, len(ext_pas_flatten[i])):
                ext_pas_j = doc[mask[j]]
                ext_pas_k = doc[mask[k]]
                sim = calculate_pas_similarity(ext_pas_flatten[i][j], ext_pas_j.tokens, ext_pas_flatten[i][k], ext_pas_k.tokens, emb)
                sim = max(sim, 0)
                sim = min(sim, 1)
                similarity_pas.append(sim)
            similarity_doc.append(similarity_pas)
        similarity_table.append(similarity_doc)
    return similarity_table

def generate_target_features(ext_pas_list, target):
    for i, doc in enumerate(ext_pas_list):
        for j, extracted_pas in enumerate(doc):
            extracted_pas.target = target[i][j]
  
def create_mask_arr(pas_list):
    arr = []
    for i, pas in enumerate(pas_list):
        arr += [i for _ in range(len(pas.pas))]
    return arr

def calculate_argument_similarity(args1, tokens1, args2, tokens2, emb):
    embedding_model, embedding_model_oov = emb
    # search for shortest arguments
    first_loop_args = args1
    second_loop_args = args2
    switch_args = False
    
    if len(args1) > len(args2):
        first_loop_args = args2
        second_loop_args = args1
        switch_args = not switch_args

    total_sim = 0.0
    for arg1 in first_loop_args:
        max_sim_args = 0.0
        
        for arg2 in second_loop_args:
            # search for shortest agument
            first_loop_arg = arg1
            second_loop_arg = arg2
            switch_arg = switch_args
            if len(arg1) > len(arg2):
                first_loop_arg = arg2
                second_loop_arg = arg1
                switch_arg = not switch_arg
            
            first_ref_tokens = tokens1
            second_ref_tokens = tokens2
            if switch_arg:
                first_ref_tokens = tokens2
                second_ref_tokens = tokens1
            
            total_sim_arg = 0.0
            for word1 in first_loop_arg:
                max_sim_arg = 0.0

                for word2 in second_loop_arg:
                    first_word = first_ref_tokens[word1].name.text.lower()
                    second_word = second_ref_tokens[word2].name.text.lower()
                
                    if (embedding_model.has_index_for(first_word) and embedding_model.has_index_for(second_word)):
                        sim = embedding_model.similarity(first_word, second_word)
                    else:
                        sim = embedding_model_oov.similarity(first_word, second_word)

                    if (sim > max_sim_arg):
                        max_sim_arg = sim
                
                total_sim_arg += max_sim_arg

            if (len(first_loop_arg) > 0):
                total_sim_arg /= len(first_loop_arg)
        
            if (total_sim_arg > max_sim_args):
                max_sim_args = total_sim_arg
        
        total_sim += max_sim_args

    if (len(first_loop_args) > 0):
        total_sim /= len(first_loop_args)

    return total_sim

def calculate_pas_similarity(pas1, tokens1, pas2, tokens2, emb):
    sim = 0.0
    count = 0.0
    
    ## verb
    if (len(pas1.verb) > 0 or len(pas2.verb) > 0):
        count+=1
        
    arg1 = pas1.args
    arg2 = pas2.args
    key_list = list(set(list(arg1.keys()) + list(arg2.keys())))
    
    count += len(key_list)
  
    for key in key_list:
        a1 = arg1[key] if key in arg1 else []
        a2 = arg2[key] if key in arg2 else []
        sim += calculate_argument_similarity(a1, tokens1, a2, tokens2, emb)

    if count > 0:
        sim /= count
    
    return sim

def get_real_j_val(i, j):
    return 1 + i + j

def get_idx_j_val(i, real_j):
    return real_j - i - 1

def fulfill_terms(value):
    return (value > 0 and value <= 0.5)

def calculate_occurence(tokens, arguments, corpus_vocabs):
    for argument in arguments:
        for word in argument:
            if (tokens[word].name.pos_tag in (noun_pos_tags + verb_pos_tags)):
                if tokens[word].name.text.lower() in corpus_vocabs:
                    corpus_vocabs[tokens[word].name.text.lower()] += 1
                else:
                    corpus_vocabs[tokens[word].name.text.lower()] = 1

def get_corpus_vocabs_and_most_common_words(corpus_pas): # corpus_docs[corpus_name]["sentences"]
    corpus_vocabs = {}
    for extracted_pas in corpus_pas:
        pas = extracted_pas.pas
        for p in pas:
            #verb
            verb = p.verb
            calculate_occurence(extracted_pas.tokens, [p.verb], corpus_vocabs)
            args = p.args
            for _, val in args.items():
                calculate_occurence(extracted_pas.tokens, val, corpus_vocabs)
       
   
    most_common_words = []
    d = Counter(corpus_vocabs)
    for k, v in d.most_common(10):
        most_common_words.append(k)

    return corpus_vocabs, most_common_words

def calculate_frequent_semantic_term(tokens, arguments, most_common_words, arr):
    for argument in arguments:
        for word in argument:
            if tokens[word].name.text.lower() in most_common_words and word not in arr:
                arr.append(word)
    return arr

def calculate_fst_pas(tokens, pas, common_words):
    taken = []
    taken = calculate_frequent_semantic_term(tokens, [pas.verb],common_words, taken)
    for _, arg in pas.args.items():
        taken = calculate_frequent_semantic_term(tokens, arg, common_words, taken)
    return len(taken)

def calculate_sim(similarity_table, id_mask, id_pas):
    s = 0
    start = id_mask + id_pas
    for sim in similarity_table[start]:
        s += sim
    if (start != 0):
        i = start - 1
        for val in range(start):
            s += similarity_table[val][i]
            i -= 1
    return s   

def calculate_length_helper(tokens, arguments, arr):
    for argument in arguments:
        for word in argument:
            if (tokens[word].name.text not in string.punctuation and word not in arr):
                arr.append(word)
    return arr

def calculate_length_pas(tokens, pas):
    taken = []
    taken = calculate_length_helper(tokens, [pas.verb], taken)
    for _, arg in pas.args.items():
        taken = calculate_length_helper(tokens, arg, taken)
    return len(taken)

def calculate_pos_tag_helper(post_tag, tokens, arguments, arr):
    for argument in arguments:
        for word in argument:
            if (tokens[word].name.pos_tag in post_tag and word not in arr):
                arr.append(word)
    return arr

def calculate_pos_pas(tag, tokens, pas):
    taken = []
    taken = calculate_pos_tag_helper(tag, tokens, [pas.verb], taken)
    for _, arg in pas.args.items():
        taken = calculate_pos_tag_helper(tag, tokens, arg, taken)
    return len(taken)

def calculate_arg_pas(arg, pas):
    length = 0
    if (arg not in pas.args):
        return 0
    pas_list = get_flatten_arguments(pas.args[arg])
    pas_list = list(set(pas_list))
    length = len(pas_list)
    return length

def count_pas(doc):
    count = [len(ext_pas.pas) for ext_pas in doc]
    return sum(count)

def calculate_target(alpha, rouge):
  return alpha * (rouge[0] + rouge[1]) / 2

def compute_target(rouge, hyps, refs, empty_ids, articles):
    hypothesis = []
    for id_doc, doc in enumerate(articles):
        hyps_doc = []
        id = 0
        for id_sent, sent in enumerate(doc):
            # If sentence cannot be extracted to PAS
            if [id_doc, id_sent] in empty_ids:
                # Fill with original sentences from article
                hyps_doc.append(articles[id_doc][id_sent])
            else:
                hyps_doc.append(hyps[id_doc][id])
                id += 1
        hypothesis.append(hyps_doc)

    target_full = []

    for idx, ref in enumerate(refs):
        hyps_ = [' '.join(s) for s in flatten_summary(hypothesis[idx])]
        refs_ = [' '.join([' '.join(s) for s in ref]) for _ in range(len(hyps_))]
        scores = rouge.get_scores(hyps_, refs_, avg=False)          
        r = [[score['rouge-1']['r'],score['rouge-2']['r']]  for score in scores]
        target_all = [calculate_target(40, score) for score in r]
        sum_target = sum(target_all)
        target = []
        current_id = 0
        for idx_sent, sent in enumerate(hypothesis[idx]):
            if [idx, idx_sent] not in empty_ids:
                length = len(sent)
                target_sent = [target_all[current_id + i]/sum_target for i in range(length)]
                current_id += length
                target.append(target_sent)
        target_full.append(target)

    
    return target_full

def flatten_summary(arr):
    sum = []
    for summary in arr:
        if isinstance(summary, list):
            if isinstance(summary[0], list):
                for x in summary:
                    sum.append(x)
            else:
                sum.append([x for x in summary])
    return sum

def calculate_max_similarity(sim_table, id_mask, id_pas):
    s = 0
    start = id_mask + id_pas
    for sim in sim_table[start]:
        s = max(s,sim)
    if (start != 0):
        i = start - 1
        for val in range(start):
            s = max(s, sim_table[val][i])
            i -= 1
    return s

def calculate_min_similarity(sim_table, id_mask, id_pas):
    s = 999
    start = id_mask + id_pas
    for sim in sim_table[start]:
        s = min(s,sim)
    if (start != 0):
        i = start - 1
        for val in range(start):
            s = min(s, sim_table[val][i])
            i -= 1
    return s   

def calculate_title_word_occurence(title_tokens, pas_tokens):
    count = 0.0
    length = 0.0
    for word in pas_tokens:
        if word not in string.punctuation:
            length += 1
            if word.lower() in title_tokens:
                count += 1
    return count/length