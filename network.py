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

from npy.data import *
from npy.factory import Factory
from npy.activation import Activation
from npy.update import Update
from npy.error import ErrorLinear
from npy.error import ErrorOutputDifference
from npy.exception import *
from npy.label import *


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
        """
        :Raises NpyDataTypeError:
            If the number of weights given in parameters of different than
            the number already present in the network.
        """
        if len(weights) != len(self.__weights):
            raise NpyDataTypeError, 'The number of weights must be the same as the number already present in the network.'

        self.__weights = weights


    def compute_output(self, input, activation_function):
        """
        Compute output value using the current Node

        :Parameters:
            input : sequence 
                Data for the input unit of the network.
            activation_function : Activation
                Activation instance to be used to compute the output.

        :Returns:
            sequence: the output values of the network.
        """
                                                         
        # check the number of inputs and raise exception if not equal
        return activation_function.compute_activation(input, self.__weights)



class Unit:
    """
    Neural network unit class.

    :IVariables:
        __nodes : sequence of `Node`
            Nodes in the current unit. 
        __activation_function : `Activation`
            `Activation` instance used to compute the activation function
            for the current unit.
        __update_function : `Update`
            `Update` instance used to compute the updates to the weights.
        __error_function : `Error`
            `Error` instance used to compute the error of the unit.
    """

    def __init__(self, nb_nodes, previous_nb_nodes, activation_function, update_function, error_function):
        """
        Initializer.

        :Parameters:
            nb_nodes : integer
                Number of nodes required in the unit. 
            previous_nb_nodes : integer
                Number of nodes in the previous unit.
            activation_function : `Activation`
                `Activation` instance used to compute the activation function
                for the current unit.
            update_function : `Update`
                `Update` instance used to compute the updates to the weights.
            error_function : `Error`
                `Error` instance used to compute the error of the unit.
        """

        self.__nodes = []
        self.__activation_function = activation_function
        self.__update_function = update_function 
        self.__error_function = error_function
        
        for i in range(nb_nodes):
            node = Node(previous_nb_nodes)
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


    def set_activation_function(self, activation_function):
        self.__activation_function = activation_function


    def get_activation_function(self):
        return self.__activation_function


    def set_update_function(self, update_function):
        self.__update_function = update_function


    def get_update_function(self):
        return self.__update_function


    def set_error_function(self, error):
        self.__error_function = error


    def get_error_function(self):
        return self.__error_function


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
            values.append(node.compute_output(input, self.__activation_function))
        
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
        return self.__activation_function.compute_activation(inputs, weights)

    
    def compute_errors(self, next_unit_errors, desired_output, outputs, next_unit_weights, index_unit, nb_unit):
        """
        Compute the error

        :Parameters:
            next_unit_errors
                Errors of the next unit, sometimes necessary for the
                computation.
            desired_output : sequence of floats
                Output desired for the current data_instance. This is
                the output at the end of the process.
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
            The error for the unit.
        """

        if self.__error_function == None:
            if index_unit == nb_unit - 1:
                error_function = ErrorOutputDifference()
            else:
                error_function = ErrorLinear()
        else:
            error_function = self.__error_function 

        return error_function.compute_errors(next_unit_errors, desired_output, outputs, next_unit_weights, self.__activation_function.activation_derivative)

        #return self.__activation_function.compute_errors(next_unit_errors, desired_output, outputs, next_unit_weights, index_unit, nb_unit)


    def compute_update(self, index, unit, outputs, error_network, update_network, data, out_data): 
        """
        Compute the update to be applied, given the provided parameters. 

        :Parameters:
            index : integer
                Index of the unit in the network.
            unit : `Unit`
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
        return self.__update_function.compute_update(index, unit, outputs, error_network, update_network, data, out_data)
   

class UnitInput(Unit):
    """
    Neural network input unit class. Only holds the number of nodes
    in the input unit.

    :IVariables:
        __nb_nodes : integer 
            Number of nodes required in the unit. 
    """

    def __init__(self, nb_nodes):
        """
        Initializer.

        :Parameters:
            nb_node : integer
                Number of nodes required in the unit. 
        """
        Unit.__init__(self, nb_nodes, 0, None, None, None)
        self.__nb_nodes = nb_nodes


    def get_nb_nodes(self):
        return self.__nb_nodes




class Network:
    """
    Neural network class.

    :IVariables:
        __unit_input : `UnitInput`
            Input unit of the network
        __units : sequence of `Unit`
            Units of the network.
        __learning_rate : float
            Learning rate of the gradient descent process. 
        __label_function : `Label`
            Label function used to label output vectors.
    """

    def __init__(self, learning_rate=None):
        """
        Initializer.

        :Parameters:
            learning_rate : float
                Learning rate of the network.
        """
        self.__unit_input = None
        self.__units = []
        self.__learning_rate = learning_rate
        self.__label_function = None


    def reset(self):
        """
        Delete the internal structure of the network, making it ready
        to receive a new one.
        """
        self.__unit_input = None
        self.__units = []
        self.__learning_rate = None


    def get_units(self):
        units = [self.__unit_input]
        units.extend(self.__units[:])
        return units

    
    def get_learning_rate(self):
        return self.__learning_rate


    def set_learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate


    def set_label_function(self, name_label_function):
        """
        :Raises NpyTransferFunctionError:
            If name_label_function does not correspond to a label function.
        """
        Factory.check_prefix(name_label_function, Label.prefix)
        self.__label_function = Factory.build_instance_by_name(name_label_function)


    def get_label_function(self):
        return self.__label_function


    def add_unit(self, nb_nodes, name_activation_function=None, name_update_function=None, name_error_function=None):
        """
        Adds a unit to the network as the new output unit. Takes care of
        making the connections with the previous unit.

        :Parameters:
            nb_nodes : integer
                Number of nodes required in the unit. 
            name_activation_function : string
                Name of the `Activation` to use to compute the activation
                function for the current unit.
            name_update_function : string
                Name of the `Update` to use to compute the updates to
                the weights.
            name_error_function : string
                Name of the `Error` to use to compute the error of the `Unit`.
                If equal to None, then the error function is set
                automatically, depending on the unit position in the network.

        :Returns:
            The `Unit` that has just been added to the network. In the case
            of the input unit, None is returned.

        :Raises NpyTransferFunctionError:
            If the function names do not correspond to valid functions.

        :Raises NpyUnitError:
            If an error related to the unit topology is encountered.
        """
        # A positive number of nodes is required
        if nb_nodes <= 0:
            raise NpyUnitError, 'Number of nodes must be strictly positive.'

        # And for the non-input units, the activation and update functions
        # must be defined.
        if self.__unit_input != None \
          and (name_activation_function == None or name_update_function == None):
            raise NpyUnitError, 'Activation and update functions must be specified.'

        # Handle the input unit
        if self.__unit_input == None:
            unit = UnitInput(nb_nodes)
            self.__unit_input = unit
        else:
            # Handle the other units
            if len(self.__units) == 0:
                unit_previous = self.__unit_input
            else:
                unit_previous = self.__units[-1]

            # Add 1 in order to implement the bias
            nb_previous_nodes = unit_previous.get_nb_nodes() + 1

            # Retreive transfert function instances
            Factory.check_prefix(name_activation_function, Activation.prefix)
            activation_function = Factory.build_instance_by_name(name_activation_function)

            Factory.check_prefix(name_update_function, Update.prefix)
            update_function = Factory.build_instance_by_name(name_update_function)

            if name_error_function == None:
                error_function = None
            else:
                Factory.check_prefix(name_error_function, Error.prefix)
                error_function = Factory.build_instance_by_name(name_error_function)

            # Create the unit and add it to the network
            unit = Unit(nb_nodes, nb_previous_nodes, activation_function, update_function, error_function)
            self.__units.append(unit)

        return unit


    def compute_output(self, data_instance):
        """
        Compute the output values of all the units for the network.

        :Parameters:
            data_instance : `DataInstance`
                `DataInstance` used by the network to compute the outputs.

        :Returns:
            sequence of sequences of floats : the output data of all the
            `Unit` of the network.
        """
        
        values = [list(data_instance.get_attributes())] 
        for unit in self.__units:
            # Add the bias value to the input
            values[-1].append(1)
            values.append(unit.compute_output(values[-1])) 

        return values

    
    def classify_data_instance(self, data_instance):
        """
        Compute the output values for the network.

        :Parameters:
            data_instance : `DataInstance`
                `DataInstance` used by the network to compute the outputs.

        :Returns:
            integer : the label associated with the classification produced
            by the network for the given data_instance.
        """
        values = self.compute_output(data_instance)
        return self.vector_to_label(values[-1])


    def classify_data_set(self, data_set):
        """
        Classify a `DataSet`.

        :Parameters:
            data_set : `DataSet`
                `DataSet` to classify.

        :Returns:
            `DataClassification` : Classification of the `DataSet`
            given in parameter.
        """

        data_classification = DataClassification()

        data_instances = data_set.get_data_instances()
        for data_instance in data_instances:
            label_number = self.classify_data_instance(data_instance)
            data_classification.add_data_label(data_instance.get_index_number(), label_number)

        return data_classification
        

    def learn_cycles(self, data_set, nb_cycles):
        """
        Makes the network learn the data_instances of the given `DataSet`.
        """

        for i in range(nb_cycles):
            for data_instance in data_set.get_data_instances():
                self.learn_data_instance(data_instance)


    def learn_data_instance(self, data_instance, data=None, out_data=None):
        """
        Makes the network learn the given `DataInstance`.

        :Parameters:
            data_instance : `DataInstance`
                `DataInstance` to be learned.
            data
                Data input, to be filled by the user if necessary.
            out_data
                Data output, to be filled by the user if necessary,
                and can be retrieved when the function ends.
        """

        # Transform this method into a template mothod: it will increase the cohesion.

        desired_output = self.label_to_vector(data_instance.get_label_number())

        # Compute the outputs from the whole network
        outputs = self.compute_output(data_instance) 

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

        :Raises NpyTransferFunctionError:
            If no label function is defined for the network.
        """
        if self.__label_function == None:
            raise NpyTransferFunctionError, 'No label function is defined for the network.'

        nb_nodes_last_unit = self.__units[-1].get_nb_nodes()
        return self.__label_function.label_to_vector(label, nb_nodes_last_unit)


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

        :Raises NpyTransferFunctionError:
            If no label function is defined for the network.
        """
        if self.__label_function == None:
            raise NpyTransferFunctionError, 'No label function is defined for the network.'
        return self.__label_function.vector_to_label(vector)


    def get_structure(self):
        """
        Build a dictionary containing all the information related to
        the structure of the current neural network.

        :Returns:
            struct : dictionary
                Structure of the current neural network.
        """
        # General parameters
        struct = {}
        struct["learning_rate"] = self.__learning_rate
        struct["nb_units"] = len(self.__units) + 1

        # Input unit
        struct["unit1_nbnodes"] = self.__unit_input.get_nb_nodes()

        # For each hidden unit and the output unit
        for index_unit, unit in zip(range(2,len(self.__units)+2), self.__units):
            
            # Parameters for the current unit
            name_unit = "unit" + str(index_unit)
            struct[name_unit + "_nbnodes"] = unit.get_nb_nodes()

            # Parameters for the activation function
            activation_function = unit.get_activation_function()
            struct[name_unit + "_activation_function"] = activation_function.get_name() 

            # Parameters for the update function
            update_function = unit.get_update_function()
            struct[name_unit + "_update_function"] = update_function.get_name() 

            # Parameters for the error function
            error_function = unit.get_error_function()
            if error_function == None:
                name_error_function = 'None'
            else:
                name_error_function = error_function.get_name()
            struct[name_unit + "_error_function"] = name_error_function

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
                And for the hidden and output units:
                    * unit#_nbnodes = number of nodes in the #-th unit
                    * unit#_activation_function = activation_function name in the #-th unit
                    * unit#_update_function = update_function name in the #-th unit
                    * unit#_error_function = error_function name in the #-th unit
        """
        self.reset()

        # General parameters
        self.__learning_rate = float(struct["learning_rate"])

        # Input unit
        self.add_unit(int(struct["unit1_nbnodes"]))

        # For each hidden unit and the output unit
        for index_unit in range(2, int(struct["nb_units"]) + 1):
            name_unit = "unit" + str(index_unit)
            name_activation_function = struct[name_unit + "_activation_function"]
            name_update_function = struct[name_unit + "_update_function"]
            name_error_function = struct[name_unit + "_error_function"]
            if name_error_function == 'None': name_error_function = None
            nb_nodes = int(struct[name_unit + "_nbnodes"])
            self.add_unit(nb_nodes, name_activation_function, name_update_function, name_error_function) 


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
