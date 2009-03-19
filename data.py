"""
Data manager module.
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


class DataInstance:
    """
    Represents an instance of a data set, and includes all the information
    that could be required: id, attributes and label.
    """

    def __init__(self, id_number, attributes, label):
        """
        Initializer
            
        :Parameters:
            id_number : integer
                Id number for this instance.
            attributes : sequence of floats
                Attributes to be used as inputs.
            label : integer 
                Value of the label given to the instance.
        """
        self.__id_number = id_number
        self.__attributes = attributes
        #self.__outputs = outputs
        self.__label = label


    def get_id_number(self):
        return self.__id_number


    def get_attributes(self):
        return self.__attributes[:]


    #def get_outputs(self):
    #    return self.__outputs


    def get_label(self):
        return self.__label



class DataCollection:
    """
    Organizes instances into a collection, so that they can be treated
    all together.
    """

    def __init__(self):
        """
        Initializer
        """

        self.__instances = {}


    def add_instance(self, instance):
        """
        Add a data instance into the data collection.

        :Parameters:
            instance : Instance
                Instance to add to the data collection.
        """

        # TODO throw an exception if the id is already in the dictionary, and update the doc to tell we throw an exception.
        self.__instances[instance.get_id_number()] = instance


    def get_instance_by_id(self, id_number):
        """
        Get an instance from the collection from its id_number

        :Parameters:
            id_number : integer
               Id number of the instance to be retrieved.

        :Returns:
            The instance of which the id number has been passed.
        """

        # TODO throw an exception if the id in not in the dictionary, and update the doc to tell we throw an exception.
        return self.__instances[id_number]


    def get_instances(self):
        """
        Get a sequence of the instances contained in this collection.

        :Returns:
            A sequence filled with the instances contained in this collection.
        """

        data = []
        for k, v in self.__instances.items():
            data.append(v)

        return data



class DataClassified:
    """
    This class contains the id of an instance in a data set, along with
    the label given by a network. That way, instance and classification
    are decoupled, and several classification of the same instance can be
    made.
    """

    def __init__(self, id_number, label):
        """
        Initializer
        
        :Parameters:
            id_number : integer
                Id number for the classified instance.
            label : integer 
                Value of the label given to the instance.
        """
        self.__id_number = id_number
        self.__label = label


    def get_id_number(self):
        return self.__id_number


    def get_label(self):
        return self.__label



class DataClassification:
    """
    Organizes classified data into a classification, so that
    they can be treated all together.
    """

    def __init__(self):
        """
        Initializer
        """

        self.__classified_instances = {}


    def add_data_classified(self, data_classified):
        """
        Add a classified data instance into the data classification.

        :Parameters:
            data_classified : DataClassified
                Classified data to add to the data classification.
        """

        # TODO throw an exception if the id is already in the dictionary, and update the doc to tell we throw an exception.

        self.__classified_instances[data_classified.get_id_number()] = data_classified


    def get_data_classified_by_id(self, id_number):
        """
        Get a classified data from the collection from its id_number

        :Parameters:
            id_number : integer
               Id number of the classified data to be retrieved.

        :Returns:
            The classified data of which the id number has been passed.
        """

        # TODO throw an exception if the id in not in the dictionary, and update the doc to tell we throw an exception.

        return self.__classified_instances[id_number]


    def get_data_classified(self):
        """
        Get a sequence of the classified data contained in this classification.

        :Returns:
            A sequence filled with the classified data contained in this
            classification.
        """

        data = []
        for k, v in self.__classified_instances.items():
            data.append(v)
            
        return data 
