class GraphAlgorithm:
    def __init__(self, graph, threshold=0.0001, dp=0.85, init=1.0, max_iter=100):
        self.__graph = graph
        self.__threshold = threshold
        self.__dp = dp
        self.__iteration = 0
        self.__init = init
        self.__max_iter = max_iter
        
    def init_graph(self):
        for node in self.__graph.nodes:
            self.__graph.nodes[node][self.__iteration] = self.__init

    def run_algorithm(self):
        keep_iteration = True
        self.__iteration = 0
        self.init_graph()
        
        for _ in range(self.__max_iter):
            self.__iteration += 1
            # print(self.__iteration)
            all_below_threshold = True
            for node in self.__graph.nodes:
                dp_multiplier = 0.0
                for neighbor in self.__graph[node]:
                    # neighbor's outgoing links
                    if (self.__graph.nodes[neighbor]['sum_weight'] > 0):
                        dp_multiplier += (self.__graph.nodes[neighbor][self.__iteration - 1] * self.__graph[node][neighbor]['weight'])/self.__graph.nodes[neighbor]['sum_weight']
                    else:
                        dp_multiplier += (self.__graph.nodes[neighbor][self.__iteration - 1] * self.__graph[node][neighbor]['weight'])
                self.__graph.nodes[node][self.__iteration] = (1 - self.__dp) + (self.__dp * dp_multiplier)
                # if (abs(self.__graph.node[node][self.__iteration] - self.__graph.node[node][self.__iteration - 1]) >= self.__threshold):
                #     all_below_threshold = False

            err = sum(abs(self.__graph.nodes[node][self.__iteration] - self.__graph.nodes[node][self.__iteration - 1]) for node in self.__graph.nodes) 
            if err < (len(self.__graph.nodes) * self.__threshold):
                break
                # keep_iteration = False
            # if (all_below_threshold):
            #    keep_iteration = False

    def get_num_iter(self):
        return self.__iteration
    
    def get_trained_graph(self):
        return self.__graph