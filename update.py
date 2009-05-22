"""
Update function module.
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
from factory import FactoryMixin
from factory import Factory


class Update(FactoryMixin):
    """
    Abstract class for the gradient descent updating process
    """

    prefix = 'up_'

    def __init__(self):
        """
        Initializer
        """
        FactoryMixin.__init__(self)


    def compute_update(self, unit, outputs, errors, weight_updates, data, out_data):
        """
        Compute the update to be applied, given the provided parameters. 

        :Parameters:
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
        pass



class UpdateBackpropagation(Update):
    """
    Backpropagation update class 
    """

    def __init__(self):
        Update.__init__(self)
        self._set_name("up_backpropagation")


    def compute_update(self, index, unit, outputs, errors, weight_update, data, out_data): 
        #inlearning_rate = 0.10
    
        weights = unit.get_weights()
        next_weights = []
        for node_weights, weight_update_node in itertools.izip(weights, weight_update): 
            next_node_weights = []
            for weight, weight_update in itertools.izip(node_weights, weight_update_node):
                next_node_weights.append(weight + weight_update)

            next_weights.append(next_node_weights)

        return next_weights


    @staticmethod
    def build_instance():
        return UpdateBackpropagation()



class UpdateTD(Update):
    """
    TD Reinforcement learning update class
    """

    def __init__(self):
        Update.__init__(self)
        self._set_name("up_tdlearning")


    def compute_update(self, Alpha, index, unit, outputs, errors, weight_update, data, out_data): 

        vgamma  = 0.001
        vlambda = 0.1
        valpha  = 0.001
    
        eprevs     = data[0][index]
        reward     = data[1]
        outputnext = data[2]

        output = outputs[-1][0]

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
    

    @staticmethod
    def build_instance():
        return UpdateTD()


# Declare the activation functions to the Update class
Factory.declare_instance(UpdateBackpropagation())
Factory.declare_instance(UpdateTD())
