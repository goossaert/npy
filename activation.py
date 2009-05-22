"""
Activation function module.
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
from exception import *


class Activation(FactoryMixin):
    """
    Activation function class.
    """

    prefix = 'ac_'

    def __init__(self):
        """
        Initializer.
        """
        FactoryMixin.__init__(self)


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


class ActivationLinear(Activation):
    """
    Linear activation function
    """

    def __init__(self):
        Activation.__init__(self)
        self._set_name("ac_linear")


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
        self._set_name("ac_perceptron")


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
        Activation.__init__(self)
        self._set_name("ac_sigmoid")


    def activation_function(self, x):
        return 1 / (1 + math.exp(-x))


    def activation_derivative(self, x):
        return x * (1 - x)


    @staticmethod
    def build_instance():
        return ActivationSigmoid()


# Declare the activation functions to the Factory
Factory.declare_instance(ActivationLinear())
Factory.declare_instance(ActivationPerceptron())
Factory.declare_instance(ActivationSigmoid())
