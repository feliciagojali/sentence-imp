from copy import deepcopy

class ExtractedPAS:
    def __init__(self):
        self.idx_topic = -1
        self.idx_news = -1
        self.idx_sentence = -1
        self.idx_sentence_in_corpus = -1
        self.sentence = None
        self.tokens = []
        self.root = None

        self.parent = None
        self.delete = False

        self.pas = None
        self.clean_pas = None

        self.fst_feature = 0.0 # frequent semantic term
        self.tfidf_feature = 0.0 # TF IDF
        self.p2p_feature = 0.0 # PAS to PAS similarity
        self.position_feature = 0.0 # position
        self.length_feature = 0.0
        self.pnoun_feature = 0.0
        self.num_feature = 0.0
        self.noun_verb_feature = 0.0
        self.temporal_feature = 0.0
        self.location_feature = 0.0
        self.title_feature = 0.0
        self.doc_similarity_feature = 0.0

        self.target = 0
        
        self.num_sentences = 0 # number of sentences in a news
        self.sentence_score = 0 # sentence score for fuzzy inference system result

    def __init__(self, parent, idx_sentence_in_corpus, sentence, pas):
        self.idx_topic = sentence.idx_topic
        self.idx_news = sentence.idx_news
        self.idx_sentence = sentence.idx_sentence
        self.idx_sentence_in_corpus = idx_sentence_in_corpus
        self.sentence = sentence.sentence
        self.tokens = deepcopy(sentence.tokens)
        self.root = sentence.root
        
        self.parent = parent
        self.delete = False
        
        self.pas = pas
        self.clean_pas = None
        
        self.fst_feature = 0.0 # frequent semantic term
        self.tfidf_feature = 0.0 # TF IDF
        self.p2p_feature = 0.0 # PAS to PAS similarity
        self.position_feature = 0.0 # position
        self.length_feature = 0.0
        self.pnoun_feature = 0.0
        self.num_feature = 0.0
        self.noun_verb_feature = 0.0
        self.temporal_feature = 0.0
        self.location_feature = 0.0
        self.title_feature = 0.0
        self.doc_similarity_feature = 0.0
        self.docset_similarity_feature = 0.0
        
        self.target = sentence.target
        
        self.num_sentences = sentence.num_sentences # number of sentences in a news