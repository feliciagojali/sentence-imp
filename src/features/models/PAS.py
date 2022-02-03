
class PAS:
    def __init__(self):
        self.agent = []
        self.verb = []
        self.patient = []
        self.location = []
        self.temporal = []
        self.goal = []
        self.cause = []
        self.extent = []
        self.adverbial = []
        self.modal = []
        self.negation = []
        self.other = []

    def __init__(self, agents, verbs, patient, locatives, temporals, goals, causes, extents, adverbials, modals, negations, others):
        self.agent = agents
        self.verb = verbs
        self.patient = patient
        self.location = locatives
        self.temporal = temporals
        self.goal = goals
        self.cause = causes
        self.extent = extents
        self.adverbial = adverbials 
        self.modal = modals 
        self.negation = negations
        self.other = others

    def __str__(self):
        return ("agents=" + str(self.agent) + ", verb=" + str(self.verb) + 
            ", patient=" + str(self.patient) + ", location=" + str(self.location) + 
            ", temporal=" + str(self.temporal) + ", goal=" + str(self.goal) +
            ", cause=" + str(self.cause) + ", extent=" + str(self.extent) +
            ", adverbial=" + str(self.adverbial) + ", modal=" + str(self.modal) +
            ", negation=" + str(self.negation) + ", others=", str(self.other))

class NewPAS:
    def __init__(self):
        self.args = {}
        self.verb = []
        # self.agent = []
        # self.verb = []
        # self.patient = []
        # self.location = []
        # self.temporal = []
        # self.goal = []
        # self.cause = []
        # self.extent = []
        # self.adverbial = []
        # self.modal = []
        # self.negation = []
        # self.other = []
    
    def add_arg(self, labels): 
        """
        {
            "pred_id":
            "args": array
        }
        """
        self.verb = labels['id_pred']
        for arg in labels['args']:
            endpoints = [x for x in range(arg[0], arg[1]+1)]
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
