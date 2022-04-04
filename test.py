import pandas as pd
data = pd.read_csv('data/features/train_features.csv', index_col=0)
print(len(data))
print(data['fst_feature'][0])
idx = round(data.iloc[-1]['idx_news']) 
print(idx)