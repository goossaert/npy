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


class Activator:
    """
    Activator function class

    **The two instance variables are not required when the object is
    created, but they MUST be initialized before it is used.**
    
    :IVariables:
        error_hidden_unit : Error
            Error class instance used in the case of a hidden unit.
        error_output_unit : Error
            Error class instance used in the case of an output unit.
    """

    def __init__(self, error_hidden_unit=None, error_output_unit=None):
        """
        Initializer.
    
        :Parameters:
            error_hidden_unit : Error
                Error class instance used in the case of a hidden unit.
            error_output_unit : Error
                Error class instance used in the case of an output unit.
        """
        self.error_hidden_unit = error_hidden_unit
        self.error_output_unit = error_output_unit


    def set_error_hidden_unit(self, error):
        self.error_hidden_unit = error


    def set_error_output_unit(self, error):
        self.error_output_unit = error


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
        return None


    def compute_derivative(self, x):
        """
        Compute the derivative of the current activation function. 

        :Parameters:
            x : float
                TODO value of the variable for the derivative

        :Returns:
            float : the value of the derivative for the given x value.
        """
        return None 

    def compute_errors(self, is_output_unit, next_unit_errors, desired_output, outputs, next_unit_weights):
        """
        Compute the error, by choosing a different function whether the unit
        is a hidden unit or an output unit. **This method MUST NOT be
        overridden by subclasses.**

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
        if is_output_unit == True:
            error = self.error_output_unit
        else:
            error = self.error_hidden_unit

        return error.compute_errors(next_unit_errors, desired_output, outputs, next_unit_weights, self.compute_derivative)


class ActivatorLinear(Activator):
    """
    Linear activation function
    """

    def __init__(self):
        error = ErrorDirectOutput()
        self.error_hidden_unit = error 
        self.error_output_unit = error
        pass


    def compute_activation(self, inputs, weights):
        def mul(x, y): return x * y
        def add_(x, y): return x + y
        return reduce(add_, map(mul, inputs, weights))


    def compute_derivative(self, x):
        return 1


from error import ErrorDirectOutput

class ActivatorPerceptron(Activator):
    """
    Perceptron activation function
    """

    def __init__(self):
        error = ErrorDirectOutput()
        self.error_hidden_unit = error 
        self.error_output_unit = error
        pass


    def compute_activation(self, inputs, weights):
        value = 0
        for input, weight in zip(inputs, weights):
            value = value + input * weight

        if value > 0:   return 1
        else:           return -1


    def compute_derivative(self, x):
        return 1

from error import ErrorWeightedSum
from error import ErrorDirectOutput
import math

class ActivatorSigmoid(Activator):
    """
    Sigmoid activation function
    """

    def __init__(self):
        """
        Uses ErrorWeightedSum for the hidden unit and ErrorDirectOutput for
        the output unit.
        """
        self.error_hidden_unit = ErrorWeightedSum()
        self.error_output_unit = ErrorDirectOutput()


    def compute_activation(self, inputs, weights):
        def mul(x, y): return x * y
        def add_(x, y): return x + y
        def sigmoid(x): return 1 /(1 + math.exp(-x))

        value = 0
        for input, weight in zip(inputs, weights):
            value = value + input * weight

        # value = reduce(add_, map(mul, inputs, weights))
        value = sigmoid(value) 
        return value 


    def compute_derivative(self, x):
        return x * (1 - x)

