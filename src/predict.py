import sys
import torch
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from models import GraphAlgorithm
from utils.features_utils import  load_sim_emb, generate_sim_table, generate_features, prepare_df, prepare_features
from utils.pas_utils import filter_pas, convert_to_PAS_models, convert_to_extracted_PAS, load_srl_model, predict_srl, filter_incomplete_pas
from utils.main_utils import accept_input, create_graph, initialize_nlp, load_reg_model, maximal_marginal_relevance, natural_language_generation, preprocess, prepare_df_result, pos_tag, read_data, return_config, transform_summary, semantic_graph_modification


tf.random.set_seed(42)

results_path = 'data/results/'

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,7"


def main():
    config = return_config(sys.argv)
    
    corpus, corpus_title = accept_input()
    
    batch = config['batch_size'] if 'batch_size' in config else len(corpus)
    isOneOnly = config['one_pas_only']
    algorithm = config['algorithm']
    # Load model
    w2v, ft = load_sim_emb(config)

    final_result = []
    loaded = False
    idx = 0
    while (idx < len(corpus)):
        print('Current = '+ str(idx))
        s = idx
        e = idx + batch if idx + batch < len(corpus) else len(corpus)

        # Preprocess
        print('Preprocessing...')
        current_corpus = [preprocess(x) for x in corpus[s:e]]
        current_title = corpus_title[s:e]

        # Pos Tag
        print('POS Tagging...')
        if (not loaded):
            nlp = initialize_nlp()
        corpus_pos_tag = [pos_tag(nlp, sent, i+s) for i, sent in tqdm(enumerate(current_corpus))]
        # SRL
        print('Predicting SRL...')
        if (not loaded):
            srl_model, srl_data = load_srl_model(config)
        corpus_pas = [predict_srl(doc, srl_data, srl_model, config) for doc in tqdm(current_corpus)]

        ## Filter incomplete PAS
        corpus_pas = [[filter_incomplete_pas(pas) for pas in pas_doc]for pas_doc in corpus_pas]
        print(corpus_pas)
        
        ## Cleaning when there is no SRL
        print('Cleaning empty SRL...')
        empty_ids = []
        for i, doc in enumerate(corpus_pas):
            for j, srl in enumerate(doc):
                if srl == []:
                    empty_ids.append([i, j])

        for id in reversed(empty_ids):
            i, j = id
            print(id)
            del corpus_pas[i][j]
            del corpus_pos_tag[i][j]
            
         # Choose one pas
        if (isOneOnly):
            corpus_pas = [[filter_pas(pas, pos) for pas, pos in zip(pas_doc, pos_tag_doc)] for pas_doc, pos_tag_doc in zip(corpus_pas, corpus_pos_tag)]

        print('Extracting features...')
        # Convert to PAS Object
        pas_list = [convert_to_PAS_models(pas, pos) for pas, pos in zip(corpus_pas, corpus_pos_tag)]
        ext_pas_list = [convert_to_extracted_PAS(pas,sent) for pas, sent in zip(corpus_pos_tag, pas_list)]
        ext_pas_flatten = [np.concatenate([sent.pas for sent in doc]) for doc in ext_pas_list]


        # Semantic Graph Formation
        sim_table = generate_sim_table(ext_pas_list, ext_pas_flatten, [w2v, ft])
        graph_list = [create_graph(corpus_pas, sim) for corpus_pas, sim in zip(ext_pas_flatten, sim_table)]
        generate_features(ext_pas_list, sim_table, current_title)

        if (not loaded):
            reg = load_reg_model(config['algorithm'])
    
        total_sum = []
        for i, doc in enumerate(ext_pas_list):
            # Predicting
            df = prepare_df([doc], config, 'pred', s)
            features_min, features_avg, _ = prepare_features(config, 'pred', df)
            if config['algorithm'] == 'min':
                pred = reg.predict(features_min)
            else:
                pred = reg.predict(features_avg)
            
            semantic_graph_modification(graph_list[i], pred)
            # graph-based ranking algorithm
            graph_algorithm = GraphAlgorithm(graph_list[i], threshold=0.0001, dp=0.85, init=1.0, max_iter=100)
            graph_algorithm.run_algorithm()
            num_iter = graph_algorithm.get_num_iter()
            # maximal marginal relevance 100
            summary = maximal_marginal_relevance(15, 2, ext_pas_list[i], ext_pas_flatten[i], graph_list[i], num_iter, sim_table[i])
            summary_paragraph = natural_language_generation(summary, ext_pas_list[i], ext_pas_flatten[i], corpus_pos_tag[i], isOneOnly)

            hyps = transform_summary(summary_paragraph)
            total_sum.append(hyps)

        final_result.extend(total_sum)
        idx+=batch
        
    f = open(results_path+sys.argv[2], 'w+')
    for r in final_result:
        f.write(r)


if __name__ == "__main__":
    main()