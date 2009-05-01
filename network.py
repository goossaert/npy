"""
Neural network module.
"""
__docformat__ = "restructuredtext en" 
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

from npy.data import DataCollection
from npy.data import DataInstance
from npy.data import DataClassification
from npy.data import DataClassified

class Node:
    """
    Neural network Node.

    :IVariables:
        __weights : sequence of floats
            The weights on the edges from each of the nodes from the previous
            unit to the current node.
    """

    def __init__(self, previous_nb_node):
        """
        Initializer

        :Parameters:
            previous_nb_node : integer
                The number of the nodes in the previous unit.
        """
        self.__weights = []
       
        for i in range(previous_nb_node):
            self.__weights.append(random.uniform(-1, 1)) 

#        print 'weights', self.__weights

    def get_weights(self):
        return self.__weights

    def set_weights(self, weights):
        # TODO check the number of weights and raise exception if not equal
        self.__weights = weights

    def compute_output(self, input, activator):
        """
        Compute output value using the current Node

        :Parameters:
            input : sequence 
                Data for the input unit of the network.
            activator : Activator
                Activator instance to be used to compute the output.

        :Returns:
            sequence: the output values of the network.
        """
                                                         
        # check the number of inputs and raise exception if not equal
        return activator.compute_activation(input, self.__weights)



class Unit:
    """
    Neural network unit class.

    :IVariables:
        __nodes : sequence of Node
            Nodes in the current unit. 
        __activator : Activator
            Activator instance used to compute the activation function
            for the current unit.
        __updator : Updator
            Updator instance used to compute the updates to the weights.
    """

    def __init__(self, nb_node, previous_nb_node, activator, updator):
        """
        Initializer.

        :Parameters:
            nb_node : integer
                Number of nodes required in the unit. 
            previous_nb_node : integer
                Number of nodes in the previous unit.
            activator : Activator
                Activator instance used to compute the activation function
                for the current unit.
            updator : Updator
                Updator instance used to compute the updates to the weights.
        """

        self.__nodes = []
        self.__activator = activator
        self.__updator = updator 
        
        for i in range(nb_node):
            node = Node(previous_nb_node)
            self.__nodes.append(node)


    def get_nb_nodes(self):
        return len(self.__nodes) 


    def get_weights(self):
        """
        Retrieve the weights of all the nodes of the current unit. 

        :Returns:
            sequence of floats : the weights of the nodes in the
            current unit. 
        """

        weights = []
        
        for node in self.__nodes:
            weights.append(node.get_weights())
        return weights


    def set_weights(self, weights):
        """
        Set the weights of all the nodes of the current unit. 

        :Parameters:
            weights : sequence
                Weights to be loaded into the nodes of the current unit.
        """

        for node, weight in itertools.izip(self.__nodes, weights):
            node.set_weights(weight)
        #self.Nodes[index].set_weights(weights)


    def set_activator(self, activator):
        self.__activator = activator


    def get_activator(self):
        return self.__activator


    def set_updator(self, updator):
        self.__updator = updator


    def get_updator(self):
        return self.__updator


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
        map(loadval, self.__nodes, values)

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
        for node in self.__nodes: 
            values.append(node.compute_output(input, self.__activator))
        
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
        return self.__activator.compute_activation(inputs, weights)
    
    def compute_errors(self, next_unit_errors, desired_output, outputs, next_unit_weights, index_unit, nb_unit):
        """
        Compute the error. 

        :Parameters:
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
            index_unit : integer
                Index of the unit currently being handled.
            nb_unit : integer
                Total number of units in the network, without the
                input unit.

        :Returns:
            The error_network.
        """
        return self.__activator.compute_errors(next_unit_errors, desired_output, outputs, next_unit_weights, index_unit, nb_unit)

    def compute_update(self, index, unit, outputs, error_network, update_network, data, out_data): 
        """
        Compute the update to be applied, given the provided parameters. 

        :Parameters:
            index : integer
                Index of the unit in the network.
            unit : Unit
                Network unit to which the update has to be applied.
            outputs : sequence of sequences
                The outputs of each unit.
            error_network : sequence
                Error values.
            update_network : sequence
                Update values for the weights.
            data
                Data input, to be filled by the user if necessary.
            out_data
                Data output, to be filled by the user if necessary.
                
        :Returns:
            The new values for the weights, after having applied the updates. 
        """
        return self.__updator.compute_update(index, unit, outputs, error_network, update_network, data, out_data)
   

    def label_to_vector(self, label):
        """
        Convert a label into a vector a network is supposed to produce.

        :Parameters:
            label : number
                The label to convert.

        :Returns:
            sequence : the vector associated with the provided label.
        """
        return self.__activator.label_to_vector(label, len(self.__nodes))


    def vector_to_label(self, vector):
        """
        Convert a vector produced as an output by a network into a label. 
        The number of nodes in the output unit is not given as a parameter
        since this information can be derived from the length of the vector.

        :Parameters:
            vector : sequence
                The vector produced as an output by a network.

        :Returns:
            number : the label associated with the vector.
        """
        return self.__activator.vector_to_label(vector)




from npy.activator import Activator
from npy.updator import Updator


class Network:
    """
    Neural network class.

    :IVariables:
        __units : sequence of `Unit`
            Units of the network.
        __nb_input : integer
            Number of nodes in the input unit.
        __learning_rate : float
            Learning rate of the gradient descent process. 
    """

    def __init__(self,nb_input=-1,learning_rate=-1):
        """
        Initializer.
        """
        self.__units = []
        self.__nb_input = nb_input
        self.__learning_rate = learning_rate


    def reset(self):
        """
        Delete the internal structure of the network, making it ready
        to receive a new one.
        """
        self.__units = []
        self.__nb_input = -1
        self.__learning_rate = -1


    # TODO delete this, only used for debugging purposes
    def get_units(self):
        return self.__units

    
    def get_learning_rate(self):
        return self.__learning_rate


    def set_learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate


    def get_nb_input(self):
        return self.__nb_input


    def set_nb_input(self, nb_input):
        if nb_input <= 0:
            pass # TODO throw exception
        self.__nb_input = nb_input


    def add_unit(self, nb_nodes, name_activator, name_updator):
        """
        Adds a unit to the network as the new output unit. Takes care of
        making the connections with the previous unit.

        :Parameters:
            nb_nodes : integer
                Number of nodes required in the unit. 
            name_activator : string
                Name of the `Activator` to use to compute the activation
                function for the current unit.
            name_updator : string
                Name of the `Updator` to use to compute the updates to
                the weights.

        :Returns:
            The `Unit` that has just been added to the network.
        """
        if nb_nodes <= 0 or name_activator == None or name_updator == None:
            pass # TODO throw exception

        if len(self.__units) == 0:
            nb_previous_nodes = self.__nb_input
        else:
            nb_previous_nodes = self.__units[-1].get_nb_nodes()

        # Add 1 in order to implement the bias
        nb_previous_nodes = nb_previous_nodes + 1

        activator = Activator.build_instance_by_name(name_activator)
        updator = Updator.build_instance_by_name(name_updator)

        unit = Unit(nb_nodes, nb_previous_nodes, activator, updator)
        self.__units.append(unit)
        return unit


    def compute_output(self, instance):
        """
        Compute the output values of all the units for the network.

        :Parameters:
            instance : `DataInstance`
                `DataInstance` used by the network to compute the outputs.

        :Returns:
            sequence of sequences of floats : the output data of all the
            `Unit` of the network.
        """
        
        values = [list(instance.get_attributes())] 
        for unit in self.__units:
            # Add the bias value to the input
            values[-1].append(1)
            values.append(unit.compute_output(values[-1])) 

        return values

    
    #TODO rename in classify_instance
    def compute_label(self, instance):
        """
        Compute the output values for the network.

        :Parameters:
            instance : `DataInstance`
                `DataInstance` used by the network to compute the outputs.

        :Returns:
            integer : the label associated with the classification produced
            by the network for the given instance.
        """
        values = self.compute_output(instance)
        return self.vector_to_label(values[-1])


    def classify_data_collection(self, data_collection, filter):
        """
        Classify a `DataCollection`.

        :Parameters:
            data_collection: `DataCollection`
                `DataCollection` to classify.
            filter : Filter
                `Filter` to use to filter the data 

        :Returns:
            `DataClassification` : Classification of the `DataCollection`
            given in parameter.
        """

        data_classification = DataClassification()

        instances = data_collection.get_instances()

        for instance in instances:
            number_label = self.compute_label(instance)
            string_label = filter.number_to_label(number_label)
            data_classified = DataClassified(instance.get_index_number(), number_label, string_label)
            data_classification.add_data_classified(data_classified)

        return data_classification
        

    def learn_iteration(self, data_collection, nb_iteration):
        """
        Makes the network learn the instances of the given `DataCollection`.
        """

        for i in range(nb_iteration):
            for instance in data_collection.get_instances():
                self.learn_instance(instance)


    def learn_instance(self, instance, data=None, out_data=None):
        """
        Makes the network learn the given `Instance`.

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

        # Transform this method into a template mothod: it will increase the cohesion.

        desired_output = self.label_to_vector(instance.get_label())

        # Compute the outputs from the whole network
        outputs = self.compute_output(instance) 

        # The 'None'  error_network is just a dummy value
        error_network = [None]
        previous_weights = None

        # Compute the error values: it has to be done backward 
        for unit, output, index in reversed(zip(self.__units, outputs[1:], range(len(self.__units)))):
            error_network.append(unit.compute_errors(error_network[-1], desired_output, output, previous_weights, index, len(self.__units)))
            previous_weights = unit.get_weights()
      
        # The dummy 'None' can be deleted
        del error_network[0]

        # The right order is the converse
        error_network.reverse()

        # The adding of the bias weights created useless error values
        # that have to be deleted 
        for index_unit in range(len(error_network) - 1):
            del error_network[index_unit][-1] 

        # Compute the weight_update values
        update_network = []
        for error_unit, input_unit in itertools.izip(error_network, outputs[:-1]):
            update_unit = []
            for error_node in error_unit:
                update_node = [] 
                for input_node in input_unit:
                    update_node.append(self.__learning_rate * error_node * input_node)
                update_unit.append(update_node)
            update_network.append(update_unit)

        # Compute the new weights
        weights = []
        for unit, error_unit, weight_update, index in itertools.izip(self.__units, error_network, update_network, range(len(self.__units))):
            weights.append(unit.compute_update(index, unit, outputs, error_unit, weight_update, data, out_data))
       
        self.set_weights(weights)

    
    def label_to_vector(self, label):
        """
        Convert a label into a vector a network is supposed to produce.

        :Parameters:
            label : number
                The label to convert.

        :Returns:
            sequence : the vector associated with the provided label.
        """
        return self.__units[len(self.__units)-1].label_to_vector(label)


    def vector_to_label(self, vector):
        """
        Convert a vector produced as an output by a network into a label. 
        The number of nodes in the output unit is not given as a parameter
        since this information can be derived from the length of the vector.

        :Parameters:
            vector : sequence
                The vector produced as an output by a network.

        :Returns:
            number : the label associated with the vector.
        """
        return self.__units[len(self.__units) - 1].vector_to_label(vector)



    def get_structure(self):
        """
        Build a dictionary containing all the information related to
        the structure of the current neural network.

        :Returns:
            struct : dictionary
                Structure of the current neural network.
        """
        struct = {}
        struct["learning_rate"] = self.__learning_rate
        struct["nb_units"] = len(self.__units) + 1

        struct["unit1_nbnodes"] = self.__nb_input
        for index_unit, unit in zip(range(2,len(self.__units)+2), self.__units):
            name_unit = "unit" + str(index_unit)
            struct[name_unit + "_nbnodes"] = unit.get_nb_nodes()
            activator = unit.get_activator()
            struct[name_unit + "_activator"] = activator.get_name() 
            updator = unit.get_updator()
            struct[name_unit + "_updator"] = updator.get_name() 

        return struct 


    def set_structure(self, struct):
        """
        Set the internal structure of the network to the given information
        dictionary.

        :Parameters:
            struct : dictionary
                This dictionary must contain:
                    * learning_rate = the value of the learning rate
                    * nb_units = number of internal units
                    * unit1_nbnodes = number of nodes in the input unit
                And for each of the non-input units:
                    * unit#_nbnodes = number of nodes in the #-th unit
                    * unit#_activator = activator name in the #-th unit
                    * unit#_updator = updator name in the #-th unit
        """
        self.reset()
        self.__learning_rate = float(struct["learning_rate"])
        self.__nb_input = int(struct["unit1_nbnodes"])

        for index_unit in range(2, int(struct["nb_units"])+1):
            name_unit = "unit" + str(index_unit)

            name_activator = struct[name_unit + "_activator"]
            activator = Activator.build_instance_by_name(name_activator)

            name_updator = struct[name_unit + "_updator"]
            updator = Updator.build_instance_by_name(name_updator)

            nb_nodes = int(struct[name_unit + "_nbnodes"])
            self.add_unit(nb_nodes, activator, updator) 


    def get_weights(self):
        """
        Get the weights of the entire network as a sequence.

        :Returns:
            sequence : weights of the entire network
        """

        weight_network = []
        for unit in self.__units:
            weight_unit = unit.get_weights()
            weight_network.append(weight_unit)
        return weight_network


    def set_weights(self,weight_network):
        """
        Set the weights of the entire network from a sequence.

        :Parameters:
            weight_network : sequence
                weights of the entire network
        """

        for weight_unit, unit in zip(weight_network, self.__units):
            unit.set_weights(weight_unit)


if __name__ == "__main__":
    print "npy"
