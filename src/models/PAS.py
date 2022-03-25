core_labels = ['ARG0','ARG1','ARG2', 'ARG3', 'ARG4', 'ARG5']
included_labels = ['AM-TMP' 'AM-LOC']
used_labels = core_labels + included_labels
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