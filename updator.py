class Updator:
    "update function class"

    def __init__(self):
        pass

    def compute_update(self, learning_rate, unit, outputs, errors, weight_updates, data, out_data):
        return None


#from Network import Network
import itertools

class UpdatorBackpropagation(Updator):
    "update backpropagation function class"

    def __init__(self):
        pass

    def compute_update(self, learning_rate, index, unit, outputs, errors, weight_update, data, out_data): 
        #inlearning_rate = 0.10
    
        weights = unit.get_weights()
        next_weights = []
        for node_weights, weight_update_node in itertools.izip(weights, weight_update): 
            next_node_weights = []
            for weight, weight_update in itertools.izip(node_weights, weight_update_node):
                next_node_weights.append(weight + learning_rate * weight_update)

            next_weights.append(next_node_weights)

        return next_weights


import itertools

class UpdatorTD(Updator):
    "update TD Reinforcement learning function class"

    def __init__(self):
        pass

    def compute_(self, Alpha, index, unit, outputs, errors, weight_update, data, out_data): 
        vgamma  = 0.001
        vlambda = 0.1
        valpha  = 0.001
    
        eprevs     = data[ 0 ][ index ]
        reward     = data[ 1 ]
        outputnext = data[ 2 ]

        output = outputs[ -1 ][ 0 ]

        es = []
        for eprev, weight_update in itertools.izip(eprevs, weight_update):
            es.append(vgamma * vlambda * eprev + weight_update * output)
        
        weights = unit.get_weights()
        next_weights = []
        for node_weights, e in itertools.izip(weights, es): 
            #print weight, outputnext, output, e
            next_node_weights = []
            for weight in node_weights:
                next_node_weights.append(weight + valpha *(reward + vgamma * outputnext[0] - output)*e)

            next_weights.append(next_node_weights)

        out_data.append(es)

        return next_weights
