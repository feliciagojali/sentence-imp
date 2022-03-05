import sys
from utils.pas_utils import load_pas_corpus, load_srl_model
from utils.main_utils import  initialize_nlp, read_data, return_config, preprocess, pos_tag

def main():
    nlp = initialize_nlp()
    config = return_config(sys.argv)
    corpus, summary, corpus_title = read_data('train', config)
    
    batch = config['batch'] if config['batch'] else len(corpus)


    # Load model
    srl_model, srl_data = load_srl_model(config)

    idx = 0
    while (idx < len(corpus)):
        s = idx
        e = idx + batch if idx + batch < len(corpus) else len(corpus)

        # Preprocess
        corpus = [preprocess(x) for x in preprocess[s:e]]

        # Pos Tag
        corpus_pos_tag = [pos_tag(nlp, sent, i+s) for i, sent in enumerate(corpus[s:e])]

        # SRL
        ## Convert features

