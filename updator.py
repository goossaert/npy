"""
Updator module.
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


class Updator:
    """
    Abstract class for the gradient descent updating process

    :CVariables:
        __updators : dictionary 
            This dictionary associate one instance of each possible subclass
            of Updator to a unique name. That way, one can create instances
            of any subclass just by passing the name to the factory method.

    :IVariables:
        __name : string
            Name of the subclass.
    """

    # dictionary of the updators
    __updators = {}
    
    def __init__(self):
        """
        Initializer
        """
        self.__name = None 


    def get_name(self):
        return self.__name


    def _set_name(self,name):
        self.__name = name


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
        return None


    def build_instance(self):
        """
        Build an instance of the current updator class.

        :Returns:
            An instance of the current updator class.
        """
        pass 


    @staticmethod
    def build_instance_by_name(name):
        """
        Build an instance of the updator given in parameter.

        :Parameters:
            name : string
                Name of the updator class to instanciate

        :Returns:
            An instance of the required updator.
        """
        # TODO What happens when the name is not in the dict?
        return Updator.__updators[name].build_instance()
    #build_instance_by_name = staticmethod(build_instance_by_name)
    
    
    @staticmethod
    def declare_updator(instance):
        """
        Add the name and an instance of a given activator in the general
        activator list. It will be used when a network will be built from
        a stream.

        :Parameters:
            instance : Updator
                Instance of the updator
        """
        name = instance.get_name()
        Updator.__updators[name] = instance 
    #declare_updator = staticmethod(declare_updator)


class UpdatorBackpropagation(Updator):
    """
    Backpropagation update class 
    """

    def __init__(self):
        Updator.__init__(self)
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


    def build_instance(self):
        return UpdatorBackpropagation()



class UpdatorTD(Updator):
    """
    TD Reinforcement learning update class
    """

    def __init__(self):
        Updator.__init__(self)
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
    

    def build_instance(self):
        return UpdatorTD()


# Declare the activators to the Updator class
Updator.declare_updator(UpdatorBackpropagation())
Updator.declare_updator(UpdatorTD())
