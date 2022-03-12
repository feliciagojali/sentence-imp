import os
import sys
import torch
sys.path.insert(0, 'src')
import unittest
import numpy as np
from copy import deepcopy
from models import ExtractedPAS, NewPAS, Sentence
from utils.main_utils import initialize_nlp, pos_tag
from utils.pas_utils import convert_to_PAS_models, convert_to_extracted_PAS

    
doc_1 = [['Akibatnya', ',', 'warga', 'sekitar', 'lebih', 'memilih', 'mengungsi', 'ke', 'tempat', 'lain', 'yang', 'dianggap', 'lebih', 'aman', 'daripada', 'bertahan', 'di', 'kota', 'tersebut', '.'], ['Ketidakberesan', 'sudah', 'muncul', 'sejak', 'Ustraindo', 'Petro', 'Gas', 'mengajukan', 'proposal', '.'], ['Jumlah', 'transaksi', 'ini', 'meningkat', 'bila', 'dibanding', 'tahun', 'sebelumnya', 'yang', 'hanya', 'Rp', '477', 'miliar', '.'], ['pakaian', 'kotor', 'dicuci', 'aku', 'di', 'kamar', 'mandi', '.']]
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

class ModelsTest(unittest.TestCase):

    def test_sentence(self):
        for i, corpus in enumerate(corpus_pos_tag):
            for j, sent in enumerate(corpus):
                self.assertIsInstance(sent, Sentence)
                self.assertEqual(sent.idx_news, i)
                self.assertEqual(sent.idx_sentence, j)
                self.assertEqual(sent.num_sentences, len(corpus))
                for w in range(len(sent.tokens)):
                    self.assertEqual(sent.tokens[w].name.text, doc[i][j][w])

    def test_PAS(self):
        for i, pas_doc in enumerate(pas_list):
            for j, pas_sent in enumerate(pas_doc):
                self.assertEqual(len(pas_sent), len(dummy_srl[i][j]))   
                for k, pas in enumerate(pas_sent):
                    self.assertIsInstance(pas, NewPAS)
                    self.assertEqual(pas.verb, list(set(dummy_srl[i][j][k]['id_pred'])))
                    for arg in dummy_srl[i][j][k]['args']:
                        id_s, id_e, label = arg
                        self.assertIn(label, pas.args)
                        endpoints = [x for x in range(id_s,  id_e + 1)]
                        self.assertIn(endpoints, pas.args[label])
    
    def test_extractedPAS(self):
        for i, pas_doc in enumerate(ext_pas_list):
            self.assertEqual(len(pas_doc), len(dummy_srl[i]))
            for j, ext_pas in enumerate(pas_doc):
                self.assertEqual(len(ext_pas.pas), len(dummy_srl[i][j]))
                self.assertEqual(ext_pas.idx_news, corpus_pos_tag[i][j].idx_news)
                self.assertEqual(ext_pas.idx_sentence, corpus_pos_tag[i][j].idx_sentence)
                self.assertEqual(ext_pas.num_sentences, corpus_pos_tag[i][j].num_sentences)
                for pas in ext_pas.pas:
                    self.assertIsInstance(pas, NewPAS)
        
        length = [sum([len(srl_sent) for srl_sent in srl_doc]) for srl_doc in dummy_srl]
        
        for i, pas_flatten_doc in enumerate(ext_pas_flatten):
            self.assertEqual(len(pas_flatten_doc), length[i])
            for j, pas_flatten, in enumerate(pas_flatten_doc):
                self.assertIsInstance(pas_flatten, NewPAS)
                


    def test_extractedPAS_srl_not_found(self):
        empty_ids = [[0,1], [1,2]]
        pos_tag = deepcopy(corpus_pos_tag)
        srl = deepcopy(dummy_srl)

        for id in reversed(empty_ids):
            i, j = id
            del srl[i][j]
            del pos_tag[i][j]
            
        pas_list = [convert_to_PAS_models(pas) for pas in srl]
        ext_pas_list = [convert_to_extracted_PAS(pas,sent) for pas, sent in zip(pos_tag, pas_list)]

        for i, pas_doc in enumerate(ext_pas_list):
            self.assertEqual(len(pas_doc), len(srl[i]))
            for j, ext_pas in enumerate(pas_doc):
                self.assertEqual(len(ext_pas.pas), len(srl[i][j]))
                self.assertEqual(ext_pas.idx_news, pos_tag[i][j].idx_news)
                self.assertEqual(ext_pas.idx_sentence, pos_tag[i][j].idx_sentence)
                self.assertEqual(ext_pas.num_sentences, pos_tag[i][j].num_sentences)
                for pas in ext_pas.pas:
                    self.assertIsInstance(pas, NewPAS)
                    
        length = [sum([len(srl_sent) for srl_sent in srl_doc]) for srl_doc in srl]

        ext_pas_flatten = [np.concatenate([sent.pas for sent in doc]) for doc in ext_pas_list]
        for i, pas_flatten_doc in enumerate(ext_pas_flatten):
            self.assertEqual(len(pas_flatten_doc), length[i])
            for j, pas_flatten, in enumerate(pas_flatten_doc):
                self.assertIsInstance(pas_flatten, NewPAS)
                

if __name__ == '__main__':
    unittest.main()