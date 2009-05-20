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


from exception import *


class DataInstance:
    """
    Represents an instance of data, and includes all the information
    that could be required: id, attributes and label.
    """

    def __init__(self, index_number, attributes, label_number):
        """
        Initializer
            
        :Parameters:
            index_number : integer
                Id number for this instance.
            attributes : sequence of floats
                Attributes to be used as inputs.
            label_number : integer 
                Value of the label given to the instance.
        """

        self.__index_number = index_number
        self.__attributes = tuple(attributes)
        self.__label_number = label_number


    def get_index_number(self):
        return self.__index_number


    def get_attributes(self):
        return self.__attributes


    def get_label_number(self):
        return self.__label_number



class DataSet:
    """
    Organizes instances into a collection, so that they can be treated
    all together.

    :IVariables:
        __instances : dictionary
            Dictionary holding the all the `DataInstance`, associating
            instances to their id numbers.
        __name_attribute : tuple
            Sequence of the names of the attributes. Stored as a tuple
            because the order does matter.
    """

    def __init__(self):
        """
        Initializer.
        """

        self.__instances = {}
        self.__name_attribute = ()


    def add_instance(self, instance):
        """
        Add a data instance into the data set.

        :Parameters:
            instance : `Instance`
                Instance to add to the data set.

        :Raises NpyIndexError:
            If the instance index already exists in the `DataSet`.
        """

        if instance.get_index_number() in self.__instances:
            raise NpyIndexError, 'Index already exists in the DataSet'

        self.__instances[instance.get_index_number()] = instance


    def get_instance_by_id(self, index_number):
        """
        Get an instance from the collection from its index_number

        :Parameters:
            index_number : integer
               Id number of the instance to be retrieved.

        :Returns:
            The instance of which the id number has been passed.
            Returns None if no `Instance` has the given index_number in
            the `DataSet`.
        """

        if not index_number in self.__instances:
            return None

        return self.__instances[index_number]


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


    def set_name_attribute(self, name_attribute):
        self.__name_attribute = tuple(name_attribute)


    def get_name_attribute(self):
        return self.__name_attribute


    def get_nb_attributes(self):
        return len(self.__name_attribute)
        


class DataSetMixed(DataSet):
    """
    Data collection mixed, to hold un-numerized and un-normalized data.
    """
    
    def __init__(self):
        """
        Initializer
        """

        DataSet.__init__(self);



class DataSetNumeric(DataSet):
    """
    Data collection numeric, to hold numerized and normalized data.
    """
    
    def __init__(self):
        """
        Initializer.
        """

        DataSet.__init__(self);



class DataLabel:
    """
    This class contains the id of an instance in a data set, along with
    the label given by a network. That way, instance and classification
    are decoupled, and several classification of the same instance can be
    made.
    """

    def __init__(self, index_number, label_number):
        """
        Initializer
        
        :Parameters:
            index_number : integer
                Id number for the classified instance.
            label_number : integer 
                Numeric value of the label given to the instance.
        """
        self.__index_number = index_number
        self.__label_number = label_number


    def get_index_number(self):
        return self.__index_number


    def get_label_number(self):
        return self.__label_number



class DataClassification:
    """
    Organizes `DataLabel` into a classification, so that
    they can be treated all together.
    """

    def __init__(self):
        """
        Initializer
        """

        self.__data_labels = {}


    def add_data_label(self, data_label):
        """
        Add a classified data instance into the data classification.

        :Parameters:
            data_label : `DataLabel`
                Classified data to add to the data classification.

        :Raises NpyIndexError:
            If the instance index already exists in the `DataClassification`.
        """

        if data_label.get_index_number() in self.__data_labels:
            raise NpyIndexError, 'Index already exists in the DataClassification'

        self.__data_labels[data_label.get_index_number()] = data_label


    def get_data_label_by_id(self, index_number):
        """
        Get a classified data from the collection from its index_number.

        :Parameters:
            index_number : integer
               Id number of the classified data to be retrieved.

        :Returns:
            The classified data of which the id number has been passed.
            Returns None if no `DataLabel` has the given index_number in
            the `DataClassification`.
        """
        
        if not index_number in self.__data_labels:
            return None

        return self.__data_labels[index_number]


    def get_data_labels(self):
        """
        Get a sequence of the classified data contained in this classification.

        :Returns:
            A sequence filled with the classified data contained in this
            classification.
        """

        data = []
        for k, v in self.__data_labels.items():
            data.append(v)

        return data
