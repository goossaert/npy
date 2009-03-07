"""
Neural network module.
"""
__version__ = "$Revision: 68852 $"
__docformat__ = "restructuredtext en"
# $Source$

## Copyright (c) 2009 Emmanuel Goossaert 
##
## This file is part of npy.
##
## npy is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## npy is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with npy.  If not, see <http://www.gnu.org/licenses/>.


import math
import itertools
import random


class Node:
    """
    Neural network Node

    :IVariables:
        weights : sequence of floats
            The weights on the edges from each of the nodes from the previous
            layer to the current node.
    """

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
        """
        Compute output value using the current Node

        :Parameters:
            input : sequence 
                Data for the input unit of the network.
            activator : Activator
                Activator instance to be used to compute the output.

        :Returns:
            sequence : the output values of the network.
        """
                                                         

#        print 'Node', input
#        print 'weights', self.weights
        
        # check the number of inputs and raise exception if not equal
        return activator.compute_activation(input, self.weights)



class Unit:
    """
    Neural network unit class

    :IVariables:
        nodes : sequence of Node
            Nodes in the current unit. 
        activator : Activator
            Activator instance used to compute the activation function
            for the current unit.
        updator : Updator
            Updator instance used to compute the updates to the weights.
    """

    def __init__(self, node_nb, previous_node_nb, activator, updator):
        """
        Initializer

        :Parameters:
            node_nb : integer
                Number of nodes required in the unit. 
            previous_node_nb : integer
                Number of nodes in the previous unit.
            activator : Activator
                Activator instance used to compute the activation function
                for the current unit.
            updator : Updator
                Updator instance used to compute the updates to the weights.
        """

        self.nodes = []
        self.activator = activator
        self.updator = updator 
        
        for i in range(node_nb):
            node = Node(previous_node_nb)
            self.nodes.append(node)

    def get_node_nb(self):
        return len(self.nodes) 

    def get_weights(self):
        """
        Retrieve the weights of all the nodes of the current unit. 

        :Returns:
            sequence of floats : the weights of the nodes in the
            current unit. 
        """

        weights = []
        
        for node in self.nodes:
            weights.append(node.get_weights())
        return weights

    def set_weights(self, weights):
        """
        Set the weights of all the nodes of the current unit. 

        :Parameters:
            weights : sequence
                Weights to be loaded into the nodes of the current unit.
        """

        for node, weight in itertools.izip(self.nodes, weights):
            node.set_weights(weight)
        #self.Nodes[index].set_weights(weights)

    def set_activator(self, activator):
        self.activator = activator

    def get_activator(self):
        return self.activator

    def set_updator(self, updator):
        self.updator = updator

    def get_updator(self):
        return self.update

    def load_values(self, values):
        """
        Load values nodes of the current unit

        TODO Not sure what this does        

        :Parameters:
            values
                TODO

        :Returns:
        """
        def loadval(node, val): node.load_value(val)
        map(loadval, self.nodes, values)

    def compute_output(self, input):
        """
        Compute the output values for the current unit given
        the provided input data. 

        :Parameters:
            input : sequence of floats
                Data used by the current unit to compute its outputs.

        :Returns:
            sequence of floats : the output data for the current unit.
        """

        values = []
        for node in self.nodes: 
            values.append(node.compute_output(input, self.activator))
        
        return values 
    
    def compute_activation(self, inputs, weights):
        """
        Compute the value of the activation function.

        :Parameters:
            inputs : sequence of floats
                Input data to be treated by the activation function.
            weights
                Weights to be used by the activation function. 

        :Returns:
            Value of the activation function.
        """
        return self.activator.compute_activation(inputs, weights)
    
    def compute_errors(self, is_output_unit, next_unit_errors, desired_output, outputs, next_unit_weights):
        """
        Compute the error. 

        :Parameters:
            is_output_unit : boolean
                True if the unit is an output unit, False if not.
            next_unit_errors
                Errors of the next unit, sometimes necessary for the
                computation.
            desired_output : sequence of floats
                Output desired for the current instance. This is the
                output at the end of the process (TODO)
            outputs : sequence
                All the output of the different layers. *At the
                moment a layer receive this information, only the
                output of the NEXT layers have been filled.*
            next_unit_weights
                Weights of the next unit, that is to say on the
                edges between the nodes of the current unit and
                those of the next one.

        :Returns:
            The errors.
        """
        return self.activator.compute_errors(is_output_unit, next_unit_errors, desired_output, outputs, next_unit_weights)

    def compute_update(self, index, unit, outputs, errors, weight_updates, data, out_data): 
        """
        Compute the update to be applied, given the provided parameters. 

        :Parameters:
            index : integer
                Index of the unit in the network.
            unit : Unit
                Network unit to which the update has to be applied.
            outputs : sequence of sequences
                The outputs of each unit.
            errors : sequence
                Error values.
            weight_updates : sequence
                Update values for the weights.
            data
                Data input, to be filled by the user if necessary.
            out_data
                Data output, to be filled by the user if necessary.
                
        :Returns:
            The new values for the weights, after having applied the updates. 
        """
        return self.updator.compute_update(index, unit, outputs, errors, weight_updates, data, out_data)



class Network:
    """
    Neural network class
    
    :IVariables:
        units : sequence of Unit 
            Units of the network.
        input_nb : integer
            Number of nodes in the input unit.
        learning_rate : float
            Learning rate of the gradient descent process. 
    """

    def __init__(self, input_nb, learning_rate):
        """
        Initializer 

        :Parameters:
            input_nb : integer
                Number of nodes in the input unit.
            learning_rate : float
                Learning rate of the gradient descent process. 
        """
        self.units = []
        self.input_nb = input_nb
        self.learning_rate = learning_rate

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def add_unit(self, node_nb, activator, updator):
        """
        Adds a unit to the network as the new output unit. Takes care of
        making the connections with the previous unit.

        :Parameters:
            node_nb : integer
                Number of nodes required in the unit. 
            activator : Activator
                Activator instance used to compute the activation function
                for the current unit.
            updator : Updator
                Updator instance used to compute the updates to the weights.

        :Returns:
            The unit that has just been added to the network.
        """
        if len(self.units) == 0:
            previous_nodes_nb = self.input_nb
        else:
            previous_nodes_nb = self.units[-1].get_node_nb()

        # Add 1  order to implement the bias
        previous_nodes_nb = previous_nodes_nb + 1

        unit = Unit(node_nb, previous_nodes_nb, activator, updator)
        self.units.append(unit)
        return unit

    def __saveweightsUnit(self, filename, Unit):
        """
        TO BE DELETED 
        "Save the weights from a unit to a file"
        """
        weights = unit.get_weights()

        file = open(filename, 'a')
        for li in weights:
            for lj in li:
                file.write(str(lj))
                file.write(' ')
            file.write('\n')
        file.close()

    def saveweights(self, filename):
        """
        TO BE DELETED 
        "Save the weights  a file"
        """
        file = open(filename, 'w')
        for unit in self.units:
            self.__saveweightsUnit(file, unit)
        file.close() 


    def __loadweightsUnit(self, file, Unit):
        """
        TO BE DELETED 
        "Save the weights from a unit to a file"
        """
        weights = []
        for i in range(Unit.get_node_nb()):
            line = file.readline()
            elem = line.split(' ')
            weights.append([float(j) for j in elem[:-1]])
        Unit.set_weights(weights)
        
    def loadweights(self, filename):
        """
        TO BE DELETED 
        "Load the weights from a file"
        """
        file = open(filename, 'r')
        for unit in self.units:
            self.__loadweightsUnit(file, unit)
        file.close()

    def compute_output(self, input):
        """
        Compute the output values of all the units for the network.

        :Parameters:
            input : sequence of floats
                Data used by the network to compute the outputs.

        :Returns:
            sequence of sequences floats : the output data of all the units
            of the network.
        """
         
        values = [input] 
        for unit in self.units:
            # Add the bias value to the input
            values[-1].append(1)
            values.append(unit.compute_output(values[-1])) 

        return values

    
    def compute(self, input):
        """
        Compute the output values for the network.

        :Parameters:
            input : sequence of floats
                Data used by the network to compute the outputs.

        :Returns:
            sequence of floats : the output data of all output unit
            of the network.
        """
        values = self.compute_output(input)
        return values[-1]


    def learn(self, input, desired_output, data, out_data):
        """
        Makes the network learn based on the given input and desired output.

        :Parameters:
            input : sequence of floats
                Input instance to be learned.
            desired_output : sequence of floats
                Output desired for the current input instance.
            data
                Data input, to be filled by the user if necessary.
            out_data
                Data output, to be filled by the user if necessary,
                and can be retrieved when the function ends.
        """

        # Compute the outputs from the whole network
        outputs = self.compute_output(input) 

        # The 'None'  errors is just a dummy value
        errors = [None]
        previous_weights = None

        # Compute the error values: it has to be done backward 
        for unit, output, index in reversed(zip(self.units, outputs[1:], range(len(self.units)))):
            if index == len(self.units) - 1:
                is_output_unit = True
            else:
                is_output_unit = False

            errors.append(unit.compute_errors(is_output_unit, errors[-1], desired_output, output, previous_weights))
            previous_weights = unit.get_weights()
      
        # The dummy 'None' can be deleted
        del errors[0]

        # The right order is the converse
        errors.reverse()

        # The adding of the bias weights created useless error values
        # that have to be deleted 
        for i in range(len(errors) - 1):
            del errors[i][-1] 

        # Compute the weight_update values
        weight_updates = []
        for lerror, linput in itertools.izip(errors, outputs[:-1]):
            unit_weight_updates = []
            for error in lerror:
                input_weight_updates = [] 
                for input in linput:
                    input_weight_updates.append(self.learning_rate * error * input)
                unit_weight_updates.append(input_weight_updates)
            weight_updates.append(unit_weight_updates)

        # Compute the new weights
        weights = []
        for unit, error, weight_update, index in itertools.izip(self.units, errors, weight_updates, range(len(self.units))):
            weights.append(unit.compute_update(index, unit, outputs, error, weight_update, data, out_data))
        
        # update the weights with the newly compute_d ones
        for unit, weight in itertools.izip(self.units, weights):
            unit.set_weights(weight)

        # print '---------------------------'

if __name__ == "__main__":
    print "npy"
