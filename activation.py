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

from error import ErrorOutputDifference
from error import ErrorLinear
from label import LabelMax
from factory import FactoryMixin
from factory import Factory


class Activation(FactoryMixin):
    """
    Activation function class

    **The instance variables are not required when the object is
    created, but they MUST be initialized before it is used.**
    
    :IVariables:
        __error_hidden_unit : `Error`
            Error class instance used in the case of a hidden unit.
        __error_output_unit : `Error`
            Error class instance used in the case of an output unit.
        __label_function : `Label`
            Label associated with the current activation_function.
    """
    

    def __init__(self):
        """
        Initializer.
        """
        FactoryMixin.__init__(self)
        self.__error_hidden_unit = None
        self.__error_output_unit = None
        self.__label_function = None


    def set_error_hidden_unit(self, error):
        self.__error_hidden_unit = error


    def set_error_output_unit(self, error):
        self.__error_output_unit = error


    def set_label_function(self, label_function):
        self.__label_function = label_function


    def compute_activation(self, inputs, weights):
        """
        Compute the value of the activation function
        for the given sequences of inputs and weights.
        **This method MUST NOT be overridden by subclasses.**

        :Parameters:
            inputs : sequence of floats
                Input data to be treated by the activation function.
            weights : sequence of floats
                Weights to be used by the activation function. 

        :Returns:
            Value of the activation function.
        """
        value = 0
        for input, weight in zip(inputs, weights):
            value = value + input * weight

        return self.activation_function(value)


    def activation_function(self, x):
        """
        Activation function. 

        :Parameters:
            x : float
                input value

        :Returns:
            float : the value of the activation function for the given x value.
        """
        pass 


    #def compute_derivative(self, x):
    def activation_derivative(self, x):
        """
        Derivative of the activation function. 

        :Parameters:
            x : float
                input value

        :Returns:
            float : the value of the activation derivative for the given
            x value.
        """
        pass


    def compute_errors(self, next_unit_errors, desired_output, outputs, next_unit_weights, index_unit, nb_unit):
        """
        Compute the error, by choosing a different function whether the unit
        is a hidden unit or an output unit. **This method MUST NOT be
        overridden by subclasses.**

        :Parameters:
            next_unit_errors
                Errors of the next unit, sometimes necessary for the
                computation.
            desired_output : sequence of floats
                Output desired for the current instance. This is the
                output at the end of the process (TODO)
            outputs : sequence
                All the output of the different units. *At the
                moment a unit receive this information, only the
                output of the NEXT units have been filled.*
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
            The errors. 
        """
        if index_unit == nb_unit - 1:
            error = self.__error_output_unit
        else:
            error = self.__error_hidden_unit

        return error.compute_errors(next_unit_errors, desired_output, outputs, next_unit_weights, self.activation_derivative)
   

    def label_to_vector(self, label, nb_node):
        """
        Convert a label into a vector a network is supposed to produce.

        :Parameters:
            label : number
                The label to convert.
            nb_node : integer
                The number of nodes in the output unit of the network.

        :Returns:
            sequence : the vector associated with the provided label.
        """
        return self.__label_function.label_to_vector(label, nb_node)


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
        return self.__label_function.vector_to_label(vector)



class ActivationLinear(Activation):
    """
    Linear activation function
    """

    def __init__(self):
        Activation.__init__(self)
        self.set_error_hidden_unit(ErrorOutputDifference())
        self.set_error_output_unit(ErrorOutputDifference())
        self.set_label_function(LabelMax())
        self._set_name("ac_linear")
        pass


    def activation_function(self, x):
        return x 


    def activation_derivative(self, x):
        return 1


    @staticmethod
    def build_instance():
        return ActivationLinear()



class ActivationPerceptron(Activation):
    """
    Perceptron activation function
    """

    def __init__(self):
        Activation.__init__(self)
        self.set_error_hidden_unit(ErrorOutputDifference())
        self.set_error_output_unit(ErrorOutputDifference())
        self.set_label_function(LabelMax())
        self._set_name("ac_perceptron")
        pass


    def activation_function(self, x):
        if x > 0:
            return 1
        else:
            return -1


    def activation_derivative(self, x):
        return 1


    @staticmethod
    def build_instance():
        return ActivationPerceptron()



class ActivationSigmoid(Activation):
    """
    Sigmoid activation function
    """

    def __init__(self):
        """
        Uses ErrorLinear for the hidden unit and ErrorOutputDifference for
        the output unit.
        """
        Activation.__init__(self)
        self.set_error_hidden_unit(ErrorLinear())
        self.set_error_output_unit(ErrorOutputDifference())
        self.set_label_function(LabelMax())
        self._set_name("ac_sigmoid")


    def activation_function(self, x):
        return 1 / (1 + math.exp(-x))


    def activation_derivative(self, x):
        return x * (1 - x)


    @staticmethod
    def build_instance():
        return ActivationSigmoid()


# Declare the activation functions to the Activation class
Factory.declare_instance(ActivationLinear())
Factory.declare_instance(ActivationPerceptron())
Factory.declare_instance(ActivationSigmoid())
