verb_pos_tags = ["VERB"]
noun_pos_tags = ["NOUN"]
exception_pos_tags = ["PUNCT", "SYM", "X"]

rouge_metrics = ['rouge-1', 'rouge-2', 'rouge-l']
metrics = ['f', 'r', 'p']

raw_data_path = 'data/raw/'
results_path = 'data/results/new/'
features_path = 'data/features/'
models_path = 'models/new_linearRegression_spansrl_'

core_labels = ['ARG0','ARG1','ARG2', 'ARG3', 'ARG4', 'ARG5']
verb_labels = ['VERB']
features_name = ["fst_feature", "p2p_feature", "length_feature", "num_feature", "noun_verb_feature", "pnoun_feature", "location_feature", "temporal_feature", "max_doc_similarity_feature", "avg_doc_similarity_feature", "min_doc_similarity_feature","position_feature", "title_feature", "target"]