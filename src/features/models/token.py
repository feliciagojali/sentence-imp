class Token:
    def __init__(self):
        self.position = None
        self.text = None
        self.pos_tag = None
        self.relation = None
        self.governor = None
        self.occurence = 0
        self.tag = "O"
        
    def __init__(self, position, text, pos_tag, relation, governor):
        self.position = position
        self.text = text
        self.pos_tag = pos_tag
        self.relation = relation
        self.governor = governor
        self.occurence = 0
        self.tag = "O"

    def __str__(self):
        return ("position=" + str(self.position) + ", text=" + self.text + 
            ", pos_tag= " + self.pos_tag + ", relation=" + self.relation + ", governor=" + str(self.governor) +
            ", occurence=" + str(self.occurence) + ", tag=" + self.tag)