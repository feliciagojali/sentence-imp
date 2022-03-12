import os
import sys
import torch
sys.path.insert(0, 'src')
import unittest
import numpy as np
from copy import deepcopy
from utils.main_utils import initialize_nlp, pos_tag, return_config
from utils.features_utils import load_sim_emb, generate_sim_table, generate_features, calculate_sim, calculate_fst_pas, calculate_length_pas, calculate_max_similarity, calculate_min_similarity
from utils.pas_utils import convert_to_PAS_models, convert_to_extracted_PAS

current_title=[[ 'warga', 'sekitar', 'lebih', 'memilih'],['ini', 'petugas', 'menyita', 'beberapa']]
doc_1 = [['Akibatnya', ',', 'warga', 'sekitar', 'lebih', 'memilih', 'mengungsi', 'ke', 'tempat', 'lain', 'yang', 'dianggap', 'lebih',  'aman', 'daripada', 'bertahan', 'di', 'kota', 'tersebut', '.'], ['Ketidakberesan', 'sudah', 'muncul', 'sejak', 'Ustraindo', 'Petro', 'Gas', 'mengajukan', 'proposal', '.'], ['Jumlah', 'transaksi', 'ini', 'meningkat', 'bila', 'dibanding', 'tahun', 'sebelumnya', 'yang', 'hanya', 'Rp', '477', 'miliar', '.'], ['pakaian', 'kotor', 'dicuci', 'aku', 'di', 'kamar', 'mandi', '.']]
doc_2 = [['Dari', 'tangan', 'Siper', 'ini', ',', 'petugas', 'menyita', 'beberapa', 'barang', 'bukti', 'berupa', 'sepeda', 'motor', 'dari', 'berbagai', 'jenis', '.'], ['Saat', 'ini', ',', 'korban', 'yang', 'dirawat', 'di', 'rumah', 'sakit', 'tinggal', 'empat', 'orang', '.'], ['Dia', 'berharap', 'ini', 'bisa', 'menjadi', 'awal', 'yang', 'baik', 'untuk', 'pelaksanaan', 'tahapan', 'pemilu', 'berikutnya', '.', '.']]
dummy_srl = [[[{'id_pred': [5, 5], 'args': [[2, 3, 'ARG0'], [4, 4, 'AM-EXT'], [6, 13, 'ARG1'], [14, 18, 'ARG2']]}, {'id_pred': [6, 6], 'args': [[2, 3, 'ARG2'], [7, 13, 'AM-DIR']]}, {'id_pred': [11, 11], 'args': [[8, 9, 'ARG1'], [12, 13, 'ARG2']]}, {'id_pred': [15, 15], 'args': [[2, 3, 'ARG1'], [16, 18, 'ARG3']]}],
            [{'id_pred': [2, 2], 'args': [[0, 0, 'ARG1'], [1, 1, 'AM-LVB'], [3, 8, 'AM-TMP']]}, {'id_pred': [7, 7], 'args': [[4, 6, 'ARG0'], [8, 8, 'ARG1']]}],
            [{'id_pred': [3, 3], 'args': [[0, 2, 'ARG1'], [4, 12, 'AM-ADV']]}, {'id_pred': [5, 5], 'args': [[0, 3, 'ARG1'], [6, 12, 'ARG2']]}],
            [{'id_pred': [2, 2], 'args': [[0, 1, 'ARG1'], [3, 3, 'ARG0'], [4, 6, 'AM-LOC']]}]],
            [[{'id_pred': [6, 6], 'args': [[0, 3, 'ARG2'], [5, 5, 'ARG0'], [7, 15, 'ARG1']]}, {'id_pred': [10, 10], 'args': [[7, 9, 'ARG1'], [11, 15, 'ARG1']]}],
            [{'id_pred': [5, 5], 'args': [[0, 1, 'AM-TMP'], [3, 3, 'ARG1'], [6, 8, 'AM-LOC']]}, {'id_pred': [9, 9], 'args': [[0, 1, 'AM-TMP'], [3, 8, 'ARG3'], [10, 11, 'ARG1']]}],
            [{'id_pred': [1, 1], 'args': [[0, 0, 'ARG0'], [2, 12, 'ARG1']]}, {'id_pred': [4, 4], 'args': [[2, 2, 'ARG1'], [3, 3, 'AM-MOD'], [5, 7, 'ARG2'], [8, 12, 'AM-PRP']]}]]]
doc = [doc_1, doc_2]

os.environ["CUDA_VISIBLE_DEVICES"]="7"

with torch.cuda.device(0):
    nlp = initialize_nlp()
    corpus_pos_tag = [pos_tag(nlp, sent, i) for i, sent in (enumerate(doc))]
pas_list = [convert_to_PAS_models(pas) for pas in dummy_srl]
ext_pas_list = [convert_to_extracted_PAS(pas,sent) for pas, sent in zip(corpus_pos_tag, pas_list)]
ext_pas_flatten = [np.concatenate([sent.pas for sent in doc]) for doc in ext_pas_list]
config = return_config(['','default'])
w2v, ft = load_sim_emb(config)
sim_table = generate_sim_table(ext_pas_list, ext_pas_flatten, [w2v, ft])
generate_features(ext_pas_list, sim_table, current_title)
length_doc_1 = [17, 10, 5, 6, 9, 5, 13, 12, 7]

class FeaturesTest(unittest.TestCase):
    def test_sim_table(self):
        for i, sim_doc in enumerate(sim_table):
            id = 1
            for j, sim in enumerate(sim_doc):
                self.assertEqual(len(sim), len(ext_pas_flatten[i]) - id)
                id += 1
    
    def test_p2p_feature(self):
        dummy_sim_table = [
            [1, 2, 3, 4],
            [4, 1, 2],
            [4, 7],
            [0],
            []
        ]
        ext_pas = [[0, 0], [0, 0, 0]]
        idx_mask = 0
        p2p = []
        for i in range(len(ext_pas)):  
            p2p_feat = [calculate_sim(dummy_sim_table, idx_mask, id_pas) for id_pas in range(len(ext_pas[i]))]
            idx_mask += len(ext_pas[i])
            p2p.append(p2p_feat)
            
        self.assertEqual(p2p[0][0], 10)
        self.assertEqual(p2p[0][1], 8)
        self.assertEqual(p2p[1][0], 17)
        self.assertEqual(p2p[1][1], 8)
        self.assertEqual(p2p[1][2], 13)
    
    def test_fst_feature(self):
        common_words = ['warga', 'sekitar', 'memilih', 'ustraindo', 'petro','miliar']
        fst = [calculate_fst_pas(ext_pas.tokens, pas, common_words) for ext_pas in ext_pas_list[0] for pas in ext_pas.pas ]
        fst_real = [3, 2, 0, 2, 2, 2, 1, 1, 0]
        for x, y in zip(fst, fst_real):
            self.assertEqual(x, y)
    
    def test_length_feature(self):
        max_length = max(length_doc_1)
        length_real = [l/max_length for l in length_doc_1]
        length = [ext_pas.length_feature for ext_pas in ext_pas_list[0]]
        length = [items for sublist in length for items in sublist] 
        for x, y in zip(length, length_real):
            self.assertEqual(x, y)
    
    def test_num_features(self):
        num_doc_1 = [0, 0, 0, 0, 0, 0 , 1, 1, 0]
        num_real = [n/l for n, l in zip(num_doc_1, length_doc_1)]
        num = [ext_pas.num_feature for ext_pas in ext_pas_list[0]]
        num = [items for sublist in num for items in sublist] 
        for x, y in zip(num, num_real):
            self.assertEqual(x, y)
     
    def test_noun_verb_features(self):
        noun_verb_doc_1 = [7, 4, 2, 3, 4, 2, 6, 6, 4]
        noun_verb_real = [n/l for n, l in zip(noun_verb_doc_1, length_doc_1)]
        noun_verb = [ext_pas.noun_verb_feature for ext_pas in ext_pas_list[0]]
        noun_verb = [items for sublist in noun_verb for items in sublist] 
        for x, y in zip(noun_verb, noun_verb_real):
            self.assertEqual(x, y)
            
    def test_pnoun_features(self):
        pnoun_doc_1 = [0, 0, 0, 0, 3, 3, 1, 1, 0]
        pnoun_real = [n/l for n, l in zip(pnoun_doc_1, length_doc_1)]
        pnoun = [ext_pas.pnoun_feature for ext_pas in ext_pas_list[0]]
        pnoun = [items for sublist in pnoun for items in sublist] 
        for x, y in zip(pnoun, pnoun_real):
            self.assertEqual(x, y)
            
    def test_temporal_features(self):
        temporal_doc_1 = [0, 0, 0, 0, 6, 0, 0, 0, 0]
        temporal_real = [n/l for n, l in zip(temporal_doc_1, length_doc_1)]
        temporal = [ext_pas.temporal_feature for ext_pas in ext_pas_list[0]]
        temporal = [items for sublist in temporal for items in sublist] 
        for x, y in zip(temporal, temporal_real):
            self.assertEqual(x, y)
    
    def test_location_features(self):
        location_doc_1 = [0, 0, 0, 0, 0, 0, 0, 0, 3]
        location_real = [n/l for n, l in zip(location_doc_1, length_doc_1)]
        location = [ext_pas.location_feature for ext_pas in ext_pas_list[0]]
        location = [items for sublist in location for items in sublist] 
        for x, y in zip(location, location_real):
            self.assertEqual(x, y)
    
    def test_max_sim_feature(self):
        dummy_sim_table = [
            [1, 2, 3, 4],
            [4, 1, 2],
            [4, 7],
            [0],
            []
        ]
        ext_pas = [[0, 0], [0, 0, 0]]
        idx_mask = 0
        max_sim = []
        for i in range(len(ext_pas)):  
            max_ = [calculate_max_similarity(dummy_sim_table, idx_mask, id_pas) for id_pas in range(len(ext_pas[i]))]
            idx_mask += len(ext_pas[i])
            max_sim.append(max_)
        
        max_sim = [items for sublist in max_sim for items in sublist] 
        max_sim_real = [4, 4, 7, 4, 7]
        for x, y in zip(max_sim, max_sim_real):
            self.assertEqual(x, y)
    
    def test_min_sim_feature(self):
        dummy_sim_table = [
            [1, 2, 3, 4],
            [4, 1, 2],
            [4, 7],
            [0],
            []
        ]
        ext_pas = [[0, 0], [0, 0, 0]]
        idx_mask = 0
        min_sim = []
        for i in range(len(ext_pas)):  
            min_ = [calculate_min_similarity(dummy_sim_table, idx_mask, id_pas) for id_pas in range(len(ext_pas[i]))]
            idx_mask += len(ext_pas[i])
            min_sim.append(min_)
        
        min_sim = [items for sublist in min_sim for items in sublist] 
        min_sim_real = [1, 1, 2, 0, 0]
        for x, y in zip(min_sim, min_sim_real):
            self.assertEqual(x, y)
            
    def test_avg_sim_feature(self):
        p2p = []
        idx_mask = 0
        for i in range(len(ext_pas_list[0])):  
            p2p_feat = [calculate_sim(sim_table[0], idx_mask, id_pas) for id_pas in range(len(ext_pas_list[0][i].pas))]
            idx_mask += len(ext_pas_list[0][i].pas)
            p2p.append(p2p_feat)
        total = len(ext_pas_flatten[0]) - 1
        for i, ext_pas in enumerate(ext_pas_list[0]):
            for j in range(len(ext_pas.avg_doc_similarity_feature)):
                self.assertEqual(ext_pas.avg_doc_similarity_feature[j], p2p[i][j]/total)
    
    def test_title_feature(self):
        title_doc_1 = [5, 3, 1, 2, 0, 0, 0, 0, 0]
        title_real = [n/l for n, l in zip(title_doc_1, length_doc_1)]
        title = [ext_pas.title_feature for ext_pas in ext_pas_list[0]]
        title = [items for sublist in title for items in sublist] 
        for x, y in zip(title, title_real):
            self.assertEqual(x, y)         
if __name__ == '__main__':
    unittest.main()