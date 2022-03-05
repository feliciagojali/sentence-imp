import ast
import json
import stanza
import pickle, re
import pandas as pd
import networkx as nx
from tqdm import tqdm
from ..models import Sentence, Token
from anytree import Node, RenderTree

tqdm.pandas()
stanza.download("id")
exception_pos_tags = ["PUNCT", "SYM", "X"]
raw_data_path = '/data/raw/'

def initialize_nlp():
    nlp = stanza.Pipeline(lang="id", tokenize_pretokenized=True)
    return nlp

def read_data(types, config):
    df = pd.read_csv(raw_data_path + types + '_' + config['data_path'])
    df['article'] = df['article'].progress_apply(lambda x : ast.literal_eval(x))
    df['summary'] = df['summary'].progress_apply(lambda x : ast.literal_eval(x))
    df['title'] = df['title'].progress_apply(lambda x : ast.literal_eval(x))

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

def print_tree(root):
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))


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
    sentences = [[x for x in sent if x and x != '.' ] for sent in sentences]
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

def save_object(file_dir, obj):
    file_dest = file_dir + ".dat" 
    with open(file_dest, "wb") as f:
        pickle.dump(obj, f)

def load_object(file_dir):
    file_src = file_dir + ".dat"
    with open(file_src, "rb") as f:
        obj = pickle.load(f)
        return obj

def find_whole_word(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def save_graph(file_dir, obj):
    nx.write_gpickle(obj, file_dir + ".gpickle")

def load_graph(file_dir):
    return nx.read_gpickle(file_dir + ".gpickle")

def load_corpus(corpus_topics, corpus_pas, sentence_similarity_table, corpus_summaries_100, corpus_summaries_200, graph_docs):
    for corpus_name in (corpus_topics["test"] + corpus_topics["train"] + corpus_topics["validation"]): 
        corpus_pas[corpus_name] = load_object("../../temporary_data/corpus_pas/" + corpus_name)
        sentence_similarity_table[corpus_name] = load_object("../../temporary_data/sentence_similarity_table/" + corpus_name)
        corpus_summaries_100[corpus_name] = load_object("../../temporary_data/corpus_summaries_100/" + corpus_name)
        # corpus_summaries_200[corpus_name] = load_object("../../temporary_data/corpus_summaries_200/" + corpus_name)
        graph_docs[corpus_name] = load_graph("../../temporary_data/graph_docs/" + corpus_name)

