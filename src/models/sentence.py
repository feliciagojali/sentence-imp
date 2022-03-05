class Sentence:
    def __init__(self):
        self.idx_topic = -1
        self.idx_news = -1
        self.idx_sentence = -1
        self.sentence = None
        self.tokens = []
        self.root = None
        self.num_sentences = 0 # number of sentences in a news
        self.target = 0

    def __init__(self, idx_topic, idx_news, idx_sentence, sentence, tokens, root, num_sentences):
        self.idx_topic = idx_topic
        self.idx_news = idx_news
        self.idx_sentence = idx_sentence
        self.sentence = sentence
        self.tokens = tokens
        self.root = root
        self.num_sentences = num_sentences # number of sentences in a news
        self.target = 0
        
    def __str__(self):
        return ("idx_topic=" + str(self.idx_topic) +
            ", idx_news=" + str(self.idx_news) + ", idx_sentence=" + str(self.idx_sentence) +
            ", root=" + str(self.root) + ", num_sentences=" + str(self.num_sentences))