class Article():
    def __init__(self, id, title, sentences, summary):
        self.id = id
        self.title = title
        self.sentences = sentences
        self.summary = summary
        self.pas = []

    
    def add_pas(self, pas):
        # self.pas = pas
        self.pas.append(pas)
