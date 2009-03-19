"""
Error computation module.
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


import itertools


class Error:
    """
    Abstract class for the error computation in the gradient descent process.
    """

    def __init__(self):
        """Initializer"""
        pass

    def compute_errors(self, next_unit_errors, desired_output, outputs, next_unit_weights, activation_derivative):
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
            activation_derivative
                Derivative of the activation function.

        :Returns:
            The errors.
        """
        return None



class ErrorDirectOutput(Error):
    """Output unit weight_update function class"""

    def __init__(self):
        Error.__init__(self)

    def compute_errors(self, errors, desired_output, outputs, next_unit_weights, activation_derivative):
    
        errors = [] 
        for desired, computed in itertools.izip(desired_output, outputs):
            errors.append(activation_derivative(computed) * (desired - computed))

        return errors


class ErrorWeightedSum(Error):
    """Weighted sum error function class"""

    def __init__(self):
        Error.__init__(self)

    def compute_errors(self, next_unit_errors, desired_output, outputs, next_unit_weights, activation_derivative):
        
        # Pre-allocate the error_sum list so that we can loop on it
        error_sum = []
        for i in range(len(next_unit_weights[0])):
            error_sum.append(0)

        # Compute the error_sum values
        for nexterror, weights in itertools.izip(next_unit_errors, next_unit_weights):
            for weight, error_sum_id in itertools.izip(weights, range(len(error_sum))): 
                error_sum[error_sum_id] = error_sum[error_sum_id] + nexterror * weight 

        # Multiply by the derivative of the activation function
        # to compute the final value
        errors = [] 
        for currenterror, computed in itertools.izip(error_sum, outputs):
            errors.append(activation_derivative(computed) * currenterror)

        return errors
