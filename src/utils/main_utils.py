import ast
import json
import torch
import stanza
import os.path
import pickle, re
import pandas as pd
import networkx as nx
from tqdm import tqdm
from rouge import Rouge
from copy import deepcopy
from anytree import Node, RenderTree
from models import Sentence, Token, NewPAS
from .features_utils import create_mask_arr
from nltk.tokenize import sent_tokenize, word_tokenize
from .pas_utils import get_flatten_pas, get_flatten_arguments

tqdm.pandas()
stanza.download("id", model_dir='/raid/data/m13518101')
exception_pos_tags = ["PUNCT", "SYM", "X"]
rouge_metrics = ['rouge-1', 'rouge-2', 'rouge-l']
metrics = ['f', 'r', 'p']
raw_data_path = 'data/raw/'
models_path = 'models/linearRegression_spansrl_'
results_path = 'data/results/'

def initialize_nlp():
    nlp = stanza.Pipeline(lang="id", tokenize_pretokenized=True, dir='/raid/data/m13518101', processors='pos,tokenize', pos_batch_size=90)
    return nlp

def initialize_rouge():
    rouge = Rouge()
    return rouge

def load_reg_model(algorithm) :
    loaded_model = pickle.load(open(models_path + algorithm + ".sav", 'rb'))
    return loaded_model

def preprocess_title(url):
    url = ast.literal_eval(url)
    title = url.split('/')[-1]
    title = title.split['-']
    return title
    
def read_data(types, config):
    df = pd.read_csv(raw_data_path + types + '_' + config['data_path'], index_col=0)
    df['article'] = df['article'][0:1000].progress_apply(lambda x : ast.literal_eval(x))
    df['summary'] = df['summary'][0:1000].progress_apply(lambda x : ast.literal_eval(x))
    df['title'] = df['title'][0:1000].progress_apply(lambda x : ast.literal_eval(x))

    return df['article'], df['summary'], df['title']

def return_config(arg):
    config = arg[1]
    filename = './configurations.json'
    f = open(filename)
    all_config = json.load(f)

    try:
        config = all_config[config]
    except:
        config = all_config['default']

    return config

def filter_article(sent):
    result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn]( . )? ?6? . [cC]om [.,] [a-zA-Z ]+ : ', '', sent) # remove liputan6 .com with place
    result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn]( . )? ?6? . [cC]om [.,] ', '', result) # remove liputan6 .com only
    result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn]( . )? ?6? . [cC]om [a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
    result = re.sub('^.? ?[Ll][iI][pP][Uu][Tt][Aa][Nn]( . )? ?6? , [a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
    result = re.sub('^[a-zA-Z]+ : [Ll][iI][pP][Uu][Tt][Aa][Nn]( . )? ?6? . [Cc]om , [a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
    result = re.sub('[Ll][iI][pP][Uu][Tt][Aa][Nn]( . )? ?6? . [cC]om , [a-zA-Z ]+ : ', '', result) # remove liputan6 .com only
    result = re.sub(r"(/[a-z]*)?&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6}) ;", "", result) # remove HTML entites
    result = re.sub(r"\.\s+((\([^\)]+\)\s+\[[^\]]+\])|(\([^\)]+\)))", "", result) # delete author and recommendation at the end of paragraph   
    return result

def preprocess(sent):
    ## Clean Liputan 6 . com
    if (not isinstance(sent, list)):
        sent = ast.literal_eval(sent)
    sents = [' '.join(sen) for sen in sent]
    sents = [filter_article(s) for s in sents]
    sentences = [s.split(' ') for s in sents if s != '.' ]
    sentences = [[x for x in sent if x] for sent in sentences]
    sentences = [x for x in sentences if x != ['.']]
    return sentences

def generate_sentence(idx_topic, idx_news, idx_sentence, sentence, len):
    tokens = [Node(Token(int(word.id), word.text, word.upos, word.deprel, word.head)) for word in sentence.words]
    root = None
    return Sentence(idx_topic, idx_news, idx_sentence, sentence, tokens, root, len)

def tokenize(doc, idx_topic, idx_news):
    length = len(doc.sentences)
    sentences = [generate_sentence(idx_topic, idx_news, idx, sent, length) for idx, sent in enumerate(doc.sentences)]
    return sentences

def pos_tag(nlp, sent, idx):
    doc = nlp(sent)
    return tokenize(doc, idx, idx)

# Graph

def create_graph(corpus_pas_flatten, sim_table):
    graph_doc = nx.Graph()
    graph_doc.add_nodes_from(list(range(len(corpus_pas_flatten))))
    for i, row in enumerate(sim_table):
        for j, score in enumerate(row):
            score = 0.1
            if fulfill_terms(score):
                graph_doc.add_edge(i, get_real_j_val(i, j), initial_weight=score)
    return graph_doc

# new semantic graph modification
def semantic_graph_modification(graph_docs, label_pred):
    for node1, node2, data in graph_docs.edges(data=True):
        weight_features_node1 = label_pred[node1]
        weight_features_node2 = label_pred[node2]
        
        data['weight'] = data['initial_weight'] * ((0.5 * weight_features_node1) + (0.5 * weight_features_node2))
        # data['weight'] = (data['initial_weight'] * 0.5 * weight_features_node1) + (0.5 * weight_features_node2)

    # fill sum weight
    for node in graph_docs.nodes:
        graph_docs.nodes[node]['sum_weight'] = sum(graph_docs[node][link]['weight'] for link in graph_docs[node])

def get_argument_tokens_without_punctuation(pas_tokens, arguments):
    tokens = []
    for argument in arguments:
        for word in argument:
            if (pas_tokens[word].name.pos_tag not in exception_pos_tags):
                tokens.append(word)
    return tokens

def get_tokens_without_punctuation(ext_pas, idx_pas):
    tokens = []
    pas = ext_pas.pas[idx_pas]
    tokens.extend(get_argument_tokens_without_punctuation(ext_pas.tokens, [pas.verb]))
    for args in ext_pas.pas[idx_pas].args:  
        tokens.extend(get_argument_tokens_without_punctuation(ext_pas.tokens, ext_pas.pas[idx_pas].args[args]))
    tokens = set(tokens)
    tokens = sorted(token)
    
    return tokens

def get_pas_idx(mask, idx):
    pas_idx = 0
    for i in range(0, idx+1):
        if (mask[i] == mask[idx]):
            pas_idx+=1
    return pas_idx

def get_real_j_val(i, j):
    return 1 + i + j

def get_idx_j_val(i, real_j):
    return real_j - i - 1

def fulfill_terms(value):
    return (value > 0 and value <= 0.5)

def get_max_similarity(cand_elmt, summary, sentence_similarity_table):
    max_sim = 0.0
    for sum_elmt in summary:
        i = -1
        j = -1
        if (cand_elmt < sum_elmt):
            i = cand_elmt
            j = get_idx_j_val(i, sum_elmt)
        else:
            i = sum_elmt
            j = get_idx_j_val(i, cand_elmt)
        if sentence_similarity_table[i][j] > max_sim:
            max_sim = sentence_similarity_table[i][j]
    return max_sim

# MMR
def maximal_marginal_relevance(min_sum_length, min_sent, ext_pas_list, ext_pas_flatten, graph_sentences, num_iter, sentence_similarity_table, isOnlyOne=False):
    
    not_chosen = [i for i in range(len(ext_pas_flatten))]
    summary = []

    max_score = max(graph_sentences.nodes[i][num_iter] for i in not_chosen)
    for i in not_chosen:
        if graph_sentences.nodes[i][num_iter] == max_score:
            summary.append(i)
            not_chosen.remove(i)
            break
    mask = create_mask_arr(ext_pas_list)
    pas_idx = get_pas_idx(mask, i)
    sum_length = len(get_tokens_without_punctuation(ext_pas_list[mask[summary[0]]], pas_idx-1))
    get_summary = True
    idx_pas_chosen = {}
    while (get_summary and len(not_chosen) > 0):
        max_mmr = 0.0
        idx_max_mmr = -1
        for i in not_chosen:
            max_sim = get_max_similarity(i, summary, sentence_similarity_table)
            mmr = (0.5 * graph_sentences.nodes[i][num_iter]) - (0.5 * max_sim)
            if mmr > max_mmr:
                max_mmr = mmr
                idx_max_mmr = i
        
        if idx_max_mmr == -1:
            idx_max_mmr = not_chosen[0]
       
        corpus_pas_idx = mask[idx_max_mmr]
        pas_idx = get_pas_idx(mask, idx_max_mmr)
       
        tokens = get_tokens_without_punctuation(ext_pas_list[mask[idx_max_mmr]], pas_idx-1)
        length_new_summary = len(tokens)
        
        if (mask[idx_max_mmr] in idx_pas_chosen):
            if (isOnlyOne):
                not_chosen.remove(idx_max_mmr)
                continue
            current_sent = idx_pas_chosen[mask[idx_max_mmr]]
            combined = list(set(current_sent + tokens))
            length_new_summary = len(combined) - len(current_sent)
            idx_pas_chosen[mask[idx_max_mmr]] = combined
            
        else:
            idx_pas_chosen[mask[idx_max_mmr]] = tokens

        sum_length += length_new_summary # len(corpus_sentences[idx_max_mmr].sentence.words)
        summary.append(idx_max_mmr)
        not_chosen.remove(idx_max_mmr)
        if (sum_length >= min_sum_length and len(idx_pas_chosen) >= min_sent):
            get_summary = False

    return summary
# NLG

def get_first_subject_tokens(pas):
    tokens = []
    agent = pas.args['ARG0']
    if (len(agent) > 0):
        for subj in agent[0]:
            if pas.tokens[subj].name.pos_tag not in exception_pos_tags:
                tokens.append(subj)
    return tokens

def get_first_subject(pas):
    tokens = get_first_subject_tokens(pas)

    subjects = [pas.tokens[token].name.text for token in tokens]
    return " ".join(subjects)

def get_tokens_without_first_subject(pas):
    tokens = []
    tokens.extend(get_flatten_arguments([pas.verb]))
    
    for arg in pas.args:
        if (arg == 'ARG0' and len(pas.args[arg])> 1):
            tokens.extend(get_flatten_arguments(pas.args[arg][1:]))
        if (arg != 'ARG0'):
            tokens.extend(get_flatten_arguments(pas.args[arg]))
    tokens = sorted(set(tokens))
    return tokens

def levenshtein_distance(word_1, word_2):
    word_1_length = len(word_1)
    word_2_length = len(word_2)

    # create two work vectors of integer distances
    v0 = [0] * (word_2_length + 1)
    v1 = [0] * (word_2_length + 1)

    # initialize v0 (the previous row of distances)
    # this row is A[0][i]: edit distance for an empty s
    # the distance is just the number of characters to delete from t
    for i in range(word_2_length + 1):
        v0[i] = i

    for i in range(word_1_length):
        # calculate v1 (current row distances) from the previous row v0

        # first element of v1 is A[i+1][0]
        # edit distance is delete (i+1) chars from s to match empty t
        v1[0] = i + 1

        # use formula to fill in the rest of the row
        for j in range(word_2_length):
            # calculating costs for A[i+1][j+1]
            deletion_cost = v0[j + 1] + 1
            insertion_cost = v1[j] + 1
            if word_1[i] == word_2[j]:
                substitution_cost = v0[j]
            else:
                substitution_cost = v0[j] + 1

            v1[j + 1] = min(deletion_cost, insertion_cost, substitution_cost)

        # copy v1 (current row) to v0 (previous row) for next iteration
        # swap v0 with v1
        v0 = deepcopy(v1)

    # after the last swap, the results of v1 are now in v0
    max_length = max(word_1_length, word_2_length)
    if (max_length > 0):
        return float(v0[word_2_length])/max_length
    else:
        return float(v0[word_2_length])

def check_subject_pas(pas):
    label = 'ARG0'
    if (label not in pas.args):
        return []
    # first subject
    sub = pas.args[label][0]
    # check if exist in other arguments
    passed_sub = []
    for arg in pas.args:
        arg_id = set(get_flatten_arguments(pas.args[arg]))
        if (any(i in sub for i in arg_id)):
            break
        passed_sub.append(sub)
    return passed_sub

def combine_pas(pas_list, tokens):
    new_pas = NewPAS()
    new_pas.tokens = tokens
    new_pas.verb = []
    total_arg = []
    
    pas_list.sort(key=lambda x: x.verb[0], reverse=True)
    for pas in pas_list:
        new_pas.verb += pas.verb
        total_arg += list(pas.args.keys())
    total_arg = set(total_arg)
    for arg in total_arg:
        for pas in pas_list:
            if (arg not in pas.args):
                continue
            if(arg in new_pas.args):
                # sort
                if(new_pas.args[arg][0][0] < pas.args[arg][0][0]):
                    new_pas.args[arg] += pas.args[arg]
                else:
                    new_pas.args[arg] = pas.args[arg] + new_pas.args[arg]
            else:
                new_pas.args[arg] = pas.args[arg]
        
    return new_pas

def natural_language_generation(summary, corpus_pas, pas, isGrouped=True):
    mask = create_mask_arr(corpus_pas)
    summary_pas = []
    combined = {}
    for idx in summary:
        idx_ = mask[idx]
        if (idx_ not in combined):
            combined[idx_] = [idx]
        else:
            combined[idx_].append(idx)

    # sort and combine pas that originate from same sentence
    for idx in sorted(combined):
        combined_pas = [pas[x] for x in combined[idx]]
        summary_pas.append(combine_pas(combined_pas, corpus_pas[idx].tokens))
    
    if (not isGrouped):
        grouped_summary_pas = [[pas] for pas in summary_pas]
    else:
        # grouped with subjects
        grouped_summary_pas = []
        picked_pas = []
        for idx_1, pas_1 in enumerate(summary_pas):
            if (idx_1 not in picked_pas):
                a_group = [pas_1]
                picked_pas.append(idx_1)
                # filter pas that can be grouped with subject
                if (len(check_subject_pas(pas_1)) == 0):
                    grouped_summary_pas.append(a_group)
                    continue
                for idx_2, pas_2 in enumerate(summary_pas):
                    # filter pas that can be grouped with subject
                    if (len(check_subject_pas(pas_2)) == 0):
                        continue
                    if idx_1 != idx_2:
                        tokens_1 = get_flatten_pas(pas_1)
                        tokens_2 = get_flatten_pas(pas_2)
                        
                        subject_tokens_1 = get_first_subject_tokens(pas_1)
                        subject_tokens_2 = get_first_subject_tokens(pas_2)

                        subject_1 = get_first_subject(pas_1).lower()
                        subject_2 = get_first_subject(pas_2).lower()
                    
                        distance = levenshtein_distance(subject_1, subject_2)
                        if ((tokens_1[0] in subject_tokens_1) and (tokens_2[0] in subject_tokens_2)) and (distance >= 0 and distance <= 0.3):
                            a_group.append(pas_2)
                            picked_pas.append(idx_2)
                grouped_summary_pas.append(a_group)
    summary_paragraph = []
    idx_grouped_summary_pas = 0
    while idx_grouped_summary_pas < len(grouped_summary_pas):
        pases = grouped_summary_pas[idx_grouped_summary_pas]

        if (len(pases) > 1):
            pases.sort(key = lambda x: len(get_first_subject_tokens(x)), reverse = True)
            
            summary_sentence = []
            for idx, pas in enumerate(pases):
                if (idx == 0):
                    subject_tokens = get_first_subject_tokens(pas)
                    other_tokens = get_tokens_without_first_subject(pas)
                    summary_sentence.extend([pas.tokens[token].name.text for token in subject_tokens])
                    summary_sentence.extend([pas.tokens[token].name.text for token in other_tokens])
                elif (idx == len(pases) - 1):
                    other_tokens = get_tokens_without_first_subject(pas)
                    if len(pases) == 2:
                        summary_sentence.append("dan")
                    else:
                        summary_sentence.extend([",", "dan"])
                    summary_sentence.extend([pas.tokens[token].name.text for token in other_tokens])
                else:
                    other_tokens = get_tokens_without_first_subject(pas)
                    summary_sentence.append(",")
                    summary_sentence.extend([pas.tokens[token].name.text for token in other_tokens])
            # summary_sentence.append(".")
            summary_paragraph.append(summary_sentence)
        else:
            pas = pases[0]
            tokens = get_flatten_pas(pas)
            summary_sentence = [pas.tokens[token].name.text for token in tokens]
            summary_sentence.append(".")
            summary_paragraph.append(summary_sentence)

        idx_grouped_summary_pas += 1

    return summary_paragraph

# Evaluation
def transform_summary(summary_paragraph):
    reference_summary = []
    for sent in summary_paragraph:
        reference_summary.append(" ".join(sent))
    return " ".join(reference_summary)

def initialize_result():
    res = {}
    res['idx_news'] = []
    res['sum'] = []
    res['ref'] = []
    for i in rouge_metrics:
        for m in metrics:
            res[i+'-'+m] = []
    return res

def evaluate(rouge, refs, hyps, s, current=None):
    if (current is None):
        current = initialize_result()
    scores = calculate_rouge(rouge, refs, hyps)
    idx = s
    for r, h in zip(refs, hyps):
        current['sum'].append(h)
        current['ref'].append(r)
        current['idx_news'].append(idx)
        idx+=1
    for score in scores:
        for r in rouge_metrics:
            for m in metrics:
                current[r+'-'+m].append(score[r][m])
    return current

def prepare_df_result(result, types):
    if (os.path.isfile(results_path+types+'_results.csv')):
        current = pd.read_csv(results_path+types+'_results.csv', sep=';', index_col=0)
        result = pd.concat([current, result], ignore_index=True)    
    result.to_csv(results_path+types+'_results.csv', sep=';')
    return result


def calculate_rouge(rouge, refs, hpys):
    score = rouge.get_scores(hpys, refs, avg=False)
    return score

# Interactive input

def accept_input():
    while(True):
        sent = str(input('Please input filepath that contain the articles: '))
        try:
            f = open(sent)
        except:
            continue
        articles = [sent_tokenize(s) for s in f.readlines()]
        articles = [word_tokenize(t) for t in articles]
        break
    titles = [[] for _ in articles]
    while(True):
        sent = str(input('Please input filepath that contain the title of the articles (please type `-1` if there are no titles provided): '))
        try:
            if sent == '-1':
                break
            f = open(sent)
        except:
            continue
        titles = [word_tokenize(t) for t in f.readlines()]
        break
    return articles, titles