from traceback import print_list
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt

f = open('data/results/val_srl.json')
data = json.load(f)['data']
print(len(data))
# from sklearn.linear_model import LinearRegression
# def prepare_features(train_df):
#     features = train_df.drop(['max_doc_similarity_feature', 'idx_news'], axis =1)
#     features_min = features.drop(['avg_doc_similarity_feature'], axis=1)
#     features_avg = features.drop(['min_doc_similarity_feature'], axis=1)
#     target = []
#     if 'target' in train_df.columns:
#         target = train_df['target']
#         features_min = features_min.drop(['target'], axis=1)
#         features_avg = features_avg.drop(['target'], axis=1)

#     return features_min, features_avg, target

# data = pd.read_csv('data/features/train_final.csv', index_col=0)
# features_min, features_avg, target = prepare_features(data)

# reg_min = LinearRegression()
# reg_avg = LinearRegression()
# reg_min.fit(features_min, target)
# reg_avg.fit(features_avg, target)
# pickle.dump(reg_min, open('models/new_linearRegression_spansrl_min.sav', 'wb'))
# pickle.dump(reg_avg, open('models/new_linearRegression_spansrl_avg.sav', 'wb'))

    
