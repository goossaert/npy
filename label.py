"""
Label function module.
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


from factory import FactoryMixin
from factory import Factory


class Label(FactoryMixin):
    """
    This class is used to convert a label value, either a integer of a float,
    into the output vector a network is supposed to produce, and conversely.
    """
    
    prefix = 'la_'

    def __init__(self):
        """
        Initializer
        """
        pass


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
        pass


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
        pass



class LabelMax(Label):
    """
    This label_function simply converts a label into an unity vector,
    and give for label to any vector the index of its maximum value.

    For example with a network that has 4 nodes in the output unit:
       * the label 3 becomes the vector [0, 0, 1, 0]
       * the vector [.1, .3, .1, .5] becomes the label 4
    """

    def __init__(self):
        """
        Initializer
        """
        Label.__init__(self)
        self._set_name("la_max")


    def label_to_vector(self, label, nb_node):
        vector = [0 for i in range(nb_node)]
        if nb_node == 1:
            if label == 1:
                vector[0] = 0
            else:
                vector[0] = 1
        else:
            vector[label-1] = 1

        return vector


    def vector_to_label(self, vector):
        index_max = 0
        if len(vector) == 1:
            if(vector[0] >= .5):
                label = 2
            else:
                label = 1
        else:
            for index in range(1, len(vector)):
                if vector[index] > vector[index_max]:
                    index_max = index
            label = index_max + 1
        
        return label


    @staticmethod
    def build_instance():
        return LabelMax()


# Declare the label functions to the Factory
Factory.declare_instance(LabelMax())
