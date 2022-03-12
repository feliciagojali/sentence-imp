from copy import deepcopy

class ExtractedPAS:

    def __init__(self, parent, sentence, pas):
        self.idx_topic = sentence.idx_topic
        self.idx_news = sentence.idx_news
        self.idx_sentence = sentence.idx_sentence
        self.sentence = sentence.sentence
        self.tokens = deepcopy(sentence.tokens)
        self.root = sentence.root
        
        self.parent = parent
        self.delete = False
        
        self.pas = pas        
        self.fst_feature = []
        self.p2p_feature = [] # PAS to PAS similarity
        self.position_feature = [] # position
        self.length_feature = []
        self.pnoun_feature = []
        self.num_feature = []
        self.noun_verb_feature = []
        self.temporal_feature = []
        self.location_feature = []
        self.title_feature = []
        self.max_doc_similarity_feature = []
        self.min_doc_similarity_feature = []
        self.avg_doc_similarity_feature = []

        self.target = sentence.target
        
        self.num_sentences = sentence.num_sentences # number of sentences in a news