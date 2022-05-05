import sys
import torch
import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from utils.features_utils import compute_target, load_sim_emb, generate_sim_table, generate_features, generate_target_features, load_train_df, prepare_df, prepare_features
from utils.pas_utils import convert_to_PAS_models, convert_to_extracted_PAS, get_sentence, load_srl_model, predict_srl, filter_incomplete_pas
from utils.main_utils import initialize_nlp, initialize_rouge, read_data, return_config, preprocess, preprocess_title, pos_tag

model_path = 'models/'
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

tf.random.set_seed(42)
def main():

    config = return_config(sys.argv)
    corpus, summary, corpus_title = read_data('train', config)
    
    batch = config['batch_size'] if 'batch_size' in config else len(corpus)

    # Load model
    w2v, ft = load_sim_emb(config)
    
    # Load current train
    try:
        current = load_train_df(config, 'train')
        idx = round(current.iloc[-1]['idx_news']) + 1
    except:
        idx = 0
    
    total_no_srl = 0
    loaded = False
    full_ext_pas_list = []
    full_start = idx
    while (idx < len(corpus)):
        start = time.time()

        print('Current = '+ str(idx))
        s = idx
        e = idx + batch if idx + batch < len(corpus) else len(corpus)

        # Preprocess
        print('Preprocessing...')
        current_corpus = [preprocess(x) for x in corpus[s:e]]
        current_summary = summary[s:e]
        current_title = [preprocess_title(x) for x in corpus_title[s:e]]

        # Pos Tag
        with torch.cuda.device(0):
            print('POS Tagging...')
            if (not loaded):
                nlp = initialize_nlp(True)
            corpus_pos_tag = [pos_tag(nlp, sent, i+s, True) for i, sent in tqdm(enumerate(current_corpus))]
            torch.cuda.empty_cache()

        # SRL
        print('Predicting SRL...')
        with tf.device('/gpu:1'):
            if (not loaded):
                srl_model, srl_data = load_srl_model(config)
            corpus_pas = [predict_srl(doc, srl_data, srl_model, config) for doc in tqdm(current_corpus)]
        
        ## Filter incomplete PAS
        corpus_pas = [[filter_incomplete_pas(pas) for pas in pas_doc]for pas_doc in corpus_pas]
    
        ## Cleaning when there is no SRL
        print('Cleaning empty SRL...')
        empty_ids = []
        for i, doc in enumerate(corpus_pas):
            for j, srl in enumerate(doc):
                if srl == []:
                    empty_ids.append([i, j])

        total_no_srl += len(empty_ids)
        no_found = []
        for id in reversed(empty_ids):
            i, j = id
            no_found.append(current_corpus[i][j])
            del corpus_pas[i][j]
            del corpus_pos_tag[i][j]
        print('Extracting features...')
        # Convert to PAS Object
        pas_list = [convert_to_PAS_models(pas, pos) for pas, pos in zip(corpus_pas, corpus_pos_tag)]
        ext_pas_list = [convert_to_extracted_PAS(pas,sent) for pas, sent in zip(corpus_pos_tag, pas_list)]
        ext_pas_flatten = [np.concatenate([sent.pas for sent in doc]) if doc != [] else [] for doc in ext_pas_list]
        
        
        # Get sentence
        pas_sentences = [[get_sentence(extracted_pas) for extracted_pas in doc] for doc in ext_pas_list]

        # Calculate target
        if (not loaded):
            r = initialize_rouge()
            loaded = True
        target = compute_target(r, pas_sentences, current_summary, empty_ids, current_corpus)

        # Extract features
        sim_table = generate_sim_table(ext_pas_list, ext_pas_flatten, [w2v, ft])
        generate_features(ext_pas_list, sim_table, current_title)
        generate_target_features(ext_pas_list, target)
        full_ext_pas_list.extend(ext_pas_list)
        print('Convert to DF...')
        if (idx % 500 == 0):
            prepare_df(full_ext_pas_list, config, 'train', full_start)
            full_start = idx+batch
            full_ext_pas_list = []
        print('waktu 10 data = '+str(time.time() - start))

        idx += batch
        for i in no_found:
            print(i)
    prepare_df(full_ext_pas_list, config, 'train', full_start)
    reg_min = LinearRegression()
    reg_avg = LinearRegression()

    features_min, features_avg, target = prepare_features(config, 'train')

    reg_min.fit(features_min, target)
    reg_avg.fit(features_avg, target)
    
    print('Dumping model')
    pickle.dump(reg_min, open(model_path+'linearRegression_spansrl_min_noincomplete.sav', 'wb'))
    pickle.dump(reg_avg, open(model_path+'linearRegression_spansrl_avg_noincomplete.sav', 'wb'))

if __name__ == "__main__":
    main()