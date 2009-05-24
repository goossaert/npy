"""
Data filter module.
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

import sys

from npy.data import DataSetMixed
from npy.data import DataSetNumeric
from npy.data import DataInstance

class Numerizer:
    """
    Transforms a `DataSetMixed` into a `DataSetNumeric`
    by transforming all the ordinal and categorical attributes
    into numerical interval attributes.
    
    :IVariables:
        __attributes : dictionary
            Dictionary of the non-numeric attributes of the data set
            used to build this Numerizer. Each key is a dictionary itself,
            and associates a number to each string value of a given attribute.
        __label : dictionary
            Dictionary of the non-numeric labels of the `DataSet`.

    """

    def __init__(self, ds_source):
        """ 
        Builds a `Numerizer` based on the data provided in ds_source.

        :Parameters:
            ds_source : `DataSetMixed`
                Data to use in order to build the `Numerizer`.
        """

        self.__attributes = {}
        self.__label = {}

        data_instances = ds_source.get_data_instances()
        for data_instance in data_instances:
            # Process the attribute values
            for index, value in enumerate(data_instance.get_attributes()):
                try:
                    number = float(value)
                except ValueError:
                    # Every time a non-float attribute value is met,
                    # it is added to the numerizer
                    self.__add_value_for_attribute(value, index) 

            # Process the label value
            label = data_instance.get_label_number()
            try:
                number = float(label)
            except ValueError:
                # Every time a non-float label value is met,
                # it is added to the numerizer
                self.__add_value_for_label(label)


    def __add_value_for_attribute(self, value_attribute, index_attribute):
        """
        Add a possible value for a given ordinal or categorical attribute.
        
        :Parameters:
            index_attribute : integer 
                Id of the attribute in the `DataInstance` sequence
            value_attribute : string 
                Value of the attribute to store.

        :Raises NpyIndexError:
            If the given index is already in use.
        """

        if not index_attribute in self.__attributes:
            self.__attributes[index_attribute] = {}
        else:
            raise NpyIndexError, 'Index value already used'

        values = self.__attributes[index_attribute]
        if not value_attribute in values:
            values[value_attribute] = len(values) + 1


    def attribute_string_to_number(self, value_attribute, index_attribute):
        """
        Get the numeric value associated with the string value
        for a given attribute.

        :Parameters:
            index_attribute : integer 
                Id of the attribute in the `DataInstance` sequence
            value_attribute : string 
                Value of the attribute to convert to a number.

        :Returns:
            integer: the numeric value associated with a attribute string.
        """

        if not index_attribute in self.__attributes:
            return None

        values = self.__attributes[index_attribute]
        if not value_attribute in values:
            return None

        return values[value_attribute]


    def __add_value_for_label(self, value_label):
        """
        Add value for label.
        """

        if not value_label in self.__label:
            self.__label[value_label] = len(self.__label) + 1


    def label_string_to_number(self, label_string):
        """
        Get the numeric value associated with the string value
        for the label.

        :Parameters:
            label_string : string
                Label string to be converted into a label number.

        :Returns:
            integer : the label number.
            Returns None if the label string is not found.
        """

        if not label_string in self.__label:
            return None

        return self.__label[label_string]


    def label_number_to_string(self, label_number):
        """
        Get the string value associated with the numeric value
        for the label.

        :Parameters:
            label_number : integer
                Label number to be converted into a label string.

        :Returns:
            string : the label string.
            Returns None if the label index is not found.
        """

        label_string = None

        for string, number in self.__label.iteritems():
            if label_number == number:
                label_string = string
                break
            
        return label_string

             
    def numerize(self, ds_source):
        """
        Transforms a `DataSetMixed` into a `DataSetNumeric`
        by transforming all the ordinal and categorical attributes
        into numerical interval attributes.

        :Parameters:
            ds_source : `DataSetMixed`
                Data collection to numerize.

        :Returns:
            `DataSetNumeric` : Numerized data set.

        :Raises NpyDataTypeError:
            If ds_source is not of DataSetMixed type.
        """
        if not isinstance(ds_source, DataSetMixed):
            raise NpyDataTypeError, 'ds_source must be a DataSetMixed'

        ds_dest = DataSetNumeric()
        ds_dest.set_name_attribute(ds_source.get_name_attribute())

        data_instances = ds_source.get_data_instances()
        for data_instance_old in data_instances:

            attributes = []

            # Process the attribute values
            for index, value in enumerate(data_instance_old.get_attributes()):
                try:
                    number = float(value)
                except ValueError:
                    # Every time a non-float attribute value is met,
                    # it is added to the numerizer
                    number = self.attribute_string_to_number(value, index) 
                attributes.append(number)

            # Process the label value
            label_old = data_instance_old.get_label_number()
            try:
                label_new = float(label_old)
            except ValueError:
                # Every time a non-float label value is met,
                # it is added to the numerizer
                label_new = self.label_string_to_number(label_old)

            ds_dest.add_data_instance(data_instance_old.get_index_number(), attributes, label_new)

        return ds_dest 



class Normalizer:
    """
    Transforms a `DataSetNumeric` into a `DataSetNumeric` by transforming
    all the numerical values into values strictly contained into a given
    interval.

    :IVariables:
        __lower_bound : float
            Lower bound of the interval into which the values of the
            `DataSet` have to be translated. 
        __upper_bound : float
            Upper bound of the interval into which the values of the
            `DataSet` have to be translated. 
        __min : sequence
            Sequence of the smallest possible values for every attribute.
        __max : sequence
            Sequence of the highest possible values for every attribute.
    """

    def __init__(self, ds_source, lower_bound=0, upper_bound=1):
        """
        Builds a `Normalizer` based on the data provided in ds_source.

        :Parameters:
            ds_source : `DataSetNumeric`
                Data to use in order to build the normalizer.

        :Raises NpyDataTypeError:
            If ds_source is not of DataSetNumeric type.
        """
        if not isinstance(ds_source, DataSetNumeric):
            raise NpyDataTypeError, 'ds_source must be a DataSetNumeric'

        self.__lower_bound = float(lower_bound)
        self.__upper_bound = float(upper_bound)
        self.__min = None
        self.__max = None

        nb_attributes = ds_source.get_nb_attributes()
        value_min = [ float( sys.maxint) for i in range(nb_attributes) ]
        value_max = [ float(-sys.maxint) for i in range(nb_attributes) ]

        data_instances = ds_source.get_data_instances()
        for data_instance in data_instances:
            # Process the attribute values
            for index, value in enumerate(data_instance.get_attributes()):
                if value < value_min[index]:
                    value_min[index] = float(value)

                if value > value_max[index]:
                    value_max[index] = float(value)

        self.__set_min(value_min)
        self.__set_max(value_max)

             
    def set_lower_bound(self, value):
        self.__lower_bound = float(value)


    def set_upper_bound(self, value):
        self.__upper_bound = float(value)


    def __set_min(self, value_min):
        self.__min = value_min


    def __set_max(self, value_max):
        self.__max = value_max

             
    def normalize(self, ds_source):
        """
        Transforms a `DataSetNumeric` into a `DataSetNumeric`
        by normalizing the values of the attributes.

        :Parameters:
            ds_source : `DataSetNumeric` 
                Data collection to normalize.

        :Returns:
            `DataSetNumeric` : `DataSet` in which normalized
            `DataInstance` have to be places.
        """

        ds_dest = DataSetNumeric()
        ds_dest.set_name_attribute(ds_source.get_name_attribute())

        data_instances = ds_source.get_data_instances()
        for data_instance_old in data_instances:

            attributes_new = []

            # Normalize each attribute
            for index, value in enumerate(data_instance_old.get_attributes()):
                value_new = (value - self.__min[index]) * self.__max[index] * (self.__upper_bound - self.__lower_bound) + self.__lower_bound
                attributes_new.append(value_new)

            ds_dest.add_data_instance(data_instance_old.get_index_number(), attributes_new, data_instance_old.get_label_number())

        return ds_dest


class Filter:
    """
    Embeds a Numerizer and a Normalizer and allows to automatize the creation
    of those two filters, and also of their use.

    :IVariables:
        __numerizer : `Numerizer`
            `Numerizer` used by the filter.
        __normalizer : `Normalizer`
            `Normalizer` used by the filter.
    """
   
    def __init__(self,ds_source,normalizer_lower_bound=None,normalizer_upper_bound=None):
        """
        Initializer.

        :Parameters: 
            ds_source : `DataSet`
                `DataSet` used to create the filter.
            normalizer_lower_bound : float
                Lower bound used by the `Normalizer`.
            normalizer_upper_bound : float
                Upper bound used by the `Normalizer`.
        """

        self.__numerizer = Numerizer(ds_source)
        ds_numerized = self.__numerizer.numerize(ds_source)

        if normalizer_lower_bound != None or normalizer_upper_bound != None:
            self.__normalizer = Normalizer(ds_numerized, normalizer_lower_bound, normalizer_upper_bound)
        else:
            self.__normalizer = Normalizer(ds_numerized)
            

    def filter(self, ds_source):
        """
        Filter ds_source and produce and numerized and normalized
        data set.
        
        :Parameters:
            ds_source : `DataSetNumeric` 
                `DataSet` to filter.

        :Returns:
            `DataSetNumeric` : data set filtered
        """

        ds_numerized = self.__numerizer.numerize(ds_source)
        ds_normalized = self.__normalizer.normalize(ds_numerized)
        return ds_normalized


    def label_number_to_string(self, number):
        """
        Get the string value associated with the numeric value
        for the label.

        :Parameters:
            number : integer
                Label number to be converted into a label string.

        :Returns:
            string : the label string.
        """

        return self.__numerizer.label_number_to_string(number)
