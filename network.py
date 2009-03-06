import math
import itertools

import random
import math

class Node:
    "Neural network Node"
   
    def __init__(self, weight_nb):
        self.weights = []
       
        for i in range(weight_nb):
            self.weights.append(random.uniform(-1, 1)) 

#        print 'weights', self.weights

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        # TODO check the number of weights and raise exception if not equal
        self.weights = weights

    def compute_output(self, input, activator):
        "Compute output value using the current Node"
#        print 'Node', input
#        print 'weights', self.weights
        
        # check the number of inputs and raise exception if not equal
        return activator.compute_activation(input, self.weights)



class Unit:
    "Neural network unit class"

    def __init__(self, node_nb, previous_node_nb, activator, update):
        self.nodes = []
        self.update = update 
        self.activator = activator
        
        for i in range(node_nb):
            node = Node(previous_node_nb)
            self.nodes.append(node)

    def get_node_nb(self):
        return len(self.nodes) 

    def get_weights(self):
        weights = []
        
        for node in self.nodes:
            weights.append(node.get_weights())
        return weights

    def set_weights(self, weights):
        for node, weight in itertools.izip(self.nodes, weights):
            node.set_weights(weight)
        #self.Nodes[ index ].set_weights(weights)

    def set_activator(self, activator):
        self.activator = activator

    def get_activator(self):
        return self.activator

    def set_update(self, update):
        self.update = update

    def get_update(self):
        return self.update

    def load_values(self, values):
        "Load values  nodes of the current unit"
        def loadval(node, val): node.load_value(val)
        map(loadval, self.nodes, values)

    def compute_output(self, input):
        "Compute output value using the current unit"

        values = []
        for node in self.nodes: 
            values.append(node.compute_output(input, self.activator))
        
        return values 
    
    def compute_activation(self, inputs, weights):
        return self.activator.compute_activation(inputs, weights)
    
    def compute_errors(self, is_output_unit, next_unit_errors, desired_output, outputs, next_unit_weights):
        return self.activator.compute_errors(is_output_unit, next_unit_errors, desired_output, outputs, next_unit_weights)

    def compute_update(self, learning_rate, index, unit, outputs, errors, weight_update, data, out_data): 
        return self.update.compute_update(learning_rate, index, unit, outputs, errors, weight_update, data, out_data)



class Network:
    "Neural network class"

    def __init__(self, input_nb, learning_rate):
        self.units = []
        self.input_nb = input_nb
        self.learning_rate = learning_rate

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def add_unit(self, node_nb, activator, update):
        if len(self.units) == 0:
            previous_nodes_nb = self.input_nb
        else:
            previous_nodes_nb = self.units[ -1 ].get_node_nb()

        # Add 1  order to implement the bias
        previous_nodes_nb = previous_nodes_nb + 1

        self.units.append(Unit(node_nb, previous_nodes_nb, activator, update))

    def __saveweightsUnit(self, filename, Unit):
        "Save the weights from a unit to a file"
        weights = unit.get_weights()

        file = open(filename, 'a')
        for li in weights:
            for lj in li:
                file.write(str(lj))
                file.write(' ')
            file.write('\n')
        file.close()

    def saveweights(self, filename):
        "Save the weights  a file"
        file = open(filename, 'w')
        for unit in self.units:
            self.__saveweightsUnit(file, unit)
        file.close() 


    def __loadweightsUnit(self, file, Unit):
        "Save the weights from a unit to a file"
        weights = []
        for i in range(Unit.get_node_nb()):
            line = file.readline()
            elem = line.split(' ')
            weights.append([ float(j) for j in elem[:-1] ])
        Unit.set_weights(weights)
        
    def loadweights(self, filename):
        "Load the weights from a file"
        file = open(filename, 'r')
        for unit in self.units:
            self.__loadweightsUnit(file, unit)
        file.close()

    def compute_output(self, input):
        "Compute output using the neural network"
         
        values = [ input ] 
        for unit in self.units:
            # Add the bias value to the input
            values[ -1 ].append(1)
            values.append(unit.compute_output(values[ -1 ])) 

        return values
    

    def compute(self, input):

        values = self.compute_output(input)
        return values[ -1 ]


    def learn(self, input, desired_output, data, out_data):

        # Compute the outputs from the whole network
        outputs = self.compute_output(input) 

        # The 'None'  errors is just a dummy value
        errors = [ None ]
        previous_weights = None

        # Compute the error values: it has to be done backward 
        for unit, output, index in reversed(zip(self.units, outputs[ 1: ], range(len(self.units)))):
            if index == len(self.units) - 1:
                is_output_unit = True
            else:
                is_output_unit = False

            errors.append(unit.compute_errors(is_output_unit, errors[ -1 ], desired_output, output, previous_weights))
            previous_weights = unit.get_weights()
      
        # The dummy 'None' can be deleted
        del errors[ 0 ]

        # The right order is the converse
        errors.reverse()

        # The add_ing of the bias weights created useless error values
        # that have to be deleted 
        for i in range(len(errors) - 1):
            del errors[ i ][ -1 ] 

        # Compute the weight_update values
        weight_updates = []
        for lerror, linput in itertools.izip(errors, outputs[:-1]):
            unit_weight_updates = []
            for error in lerror:
                input_weight_updates = [] 
                for input in linput:
                    input_weight_updates.append(error * input)
                unit_weight_updates.append(input_weight_updates)
            weight_updates.append(unit_weight_updates)

        # Compute the new weights
        weights = []
        for unit, error, weight_update, index in itertools.izip(self.units, errors, weight_updates, range(len(self.units))):
            weights.append(unit.compute_update(self.learning_rate, index, unit, outputs, error, weight_update, data, out_data))
        
        # update the weights with the newly compute_d ones
        for unit, weight in itertools.izip(self.units, weights):
            unit.set_weights(weight)

        # print '---------------------------'

if __name__ == "__main__":
    print "npy"
