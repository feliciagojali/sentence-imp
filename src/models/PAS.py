class NewPAS:
    def __init__(self):
        self.args = {}
        self.verb = []
       
    
    def add_arg(self, labels, pos_tag): 
        """
        {
            "pred_id":
            "args": array
        }
        """
        max_len = len([1 for _ in (pos_tag.tokens)])
        verb = [x for x in labels['id_pred'] if x < max_len]
        self.verb = list(set(verb))
        for arg in labels['args']:
            endpoints = [x for x in range(arg[0], arg[1]+1) if x < max_len]
            label = arg[2]
            if (label in self.args):
                self.args[label].append(endpoints)
            else:
                endpoints = [endpoints]
                self.args[label] = endpoints

    def add_text_arg(self, labels, text):
        self.args[labels] = text

    # def __init__(self, agents, verbs, patient, locatives, temporals, goals, causes, extents, adverbials, modals, negations, others):
    #     self.agent = agents
    #     self.verb = verbs
    #     self.patient = patient
    #     self.location = locatives
    #     self.temporal = temporals
    #     self.goal = goals
    #     self.cause = causes
    #     self.extent = extents
    #     self.adverbial = adverbials 
    #     self.modal = modals 
    #     self.negation = negations
    #     self.other = others

    # def __str__(self):
    #     return ("agents=" + str(self.agent) + ", verb=" + str(self.verb) + 
    #         ", patient=" + str(self.patient) + ", location=" + str(self.location) + 
    #         ", temporal=" + str(self.temporal) + ", goal=" + str(self.goal) +
    #         ", cause=" + str(self.cause) + ", extent=" + str(self.extent) +
    #         ", adverbial=" + str(self.adverbial) + ", modal=" + str(self.modal) +
    #         ", negation=" + str(self.negation) + ", others=", str(self.other))
