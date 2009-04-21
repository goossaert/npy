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

from npy.data import DataCollectionRAW
from npy.data import DataCollectionPCD
from npy.data import DataInstance

class DataNumerizer:
    """
    Transforms a DataCollectionRAW into a DataCollectionPCD
    by transforming all the ordinal and categorical attributes
    into numerical interval attributes.
    """

    def __init__(self):
        """
        Initializer
        """

        self.__attributes = {}
        self.__label = {}


    def __add_value_for_attribute(self, value_attribute, index_attribute):
        """
        Add a possible value for a given ordinal or categorical attribute.
        
        :Parameters:
            index_attribute : integer 
                Id of the attribute in the instance sequence
            value_attribute : string 
                Value of the attribute to store.
        """

        if not index_attribute in self.__attributes:
            self.__attributes[index_attribute] = {}

        values = self.__attributes[index_attribute]
        if not value_attribute in values:
            values[value_attribute] = len(values) + 1


    def attribute_to_number(self, value_attribute, index_attribute):
        """
        Get the numeric value associated with the string value
        for a given attribute.
        """

        if not index_attribute in self.__attributes:
            return None #TODO throw exception instead

        values = self.__attributes[index_attribute]
        if not value_attribute in values:
            return None #TODO throw exception instead

        return values[value_attribute]


    def __add_value_for_label(self, value_label):
        """
        Add value for label.
        """

        if not value_label in self.__label:
            self.__label[value_label] = len(self.__label) + 1


    def label_to_number(self, string_label):
        """
        Get the numeric value associated with the string value
        for the label.
        """

        if not string_label in self.__label:
            return None #TODO throw exception instead

        return self.__label[string_label]


    def number_to_label(self, number_label):
        """
        Get the string value associated with the numeric value
        for the label.
        """

        string_label = None

        for string, number in self.__label.iteritems():
            if number_label == number:
                string_label = string
                break
            
        return string_label

    @staticmethod
    def build_numerizer(dc_source):
        """ 
        Builds a DataNumerizer based on the data provided in dc_source.

        :Parameters:
            dc_source : DataCollectionRAW 
                Data to use in order to build the numerizer.

        :Returns:
            DataNumerizer : the DataNumerizer built from dc_source.
        """

        numerizer = DataNumerizer()

        instances = dc_source.get_instances()
        for instance in instances:
            # Process the attribute values
            for index, value in enumerate(instance.get_attributes()):
                try:
                    number = float(value)
                except ValueError:
                    # Every time a non-float attribute value is met,
                    # it is added to the numerizer
                    numerizer.__add_value_for_attribute(value, index) 

            # Process the label value
            label = instance.get_label()
            try:
                number = float(label)
            except ValueError:
                # Every time a non-float label value is met,
                # it is added to the numerizer
                numerizer.__add_value_for_label(label)

        return numerizer

             
    def numerize(self, dc_source):
        """
        Transforms a DataCollectionRAW into a DataCollectionPCD
        by transforming all the ordinal and categorical attributes
        into numerical interval attributes.

        :Parameters:
            dc_source : DataCollectionRAW 
                Data collection to numerize.

        :Returns:
            DataCollectionPCD : Numerized data collection.
        """
        if not isinstance(dc_source, DataCollectionRAW):
            pass #TODO throw exception

        dc_dest = DataCollectionPCD()

        instances = dc_source.get_instances()
        for instance_old in instances:

            attributes = []

            # Process the attribute values
            for index, value in enumerate(instance_old.get_attributes()):
                try:
                    number = float(value)
                except ValueError:
                    # Every time a non-float attribute value is met,
                    # it is added to the numerizer
                    number = self.attribute_to_number(value, index) 
                attributes.append(number)

            # Process the label value
            label_old = instance_old.get_label()
            try:
                label_new = float(label_old)
            except ValueError:
                # Every time a non-float label value is met,
                # it is added to the numerizer
                label_new = self.label_to_number(label_old)

            instance_new = DataInstance(instance_old.get_id_number(), attributes, label_new)
            dc_dest.add_instance(instance_new)             

        return dc_dest 



class DataNormalizer:
    """
    Transforms a DataCollectionPCD into a DataCollectionPCD by transforming
    all the numerical values into values strictly contained into a given
    interval.
    """

    def __init__(self, inf=0, sup=1):
        """
        Initializer
        """
        self.__inf = inf
        self.__sup = sup
        self.__min = None
        self.__max = None

    def set_inf(self, value):
        self.__inf = value

    def set_sup(self, value):
        self.__sup = value

    def __set_min(self, value_min):
        self.__min = value_min

    def __set_max(self, value_max):
        self.__max = value_max

    @staticmethod
    def build_normalizer(dc_source):
        """ 
        Builds a DataNormalizer based on the data provided in dc_source.

        :Parameters:
            dc_source : DataCollectionPCD 
                Data to use in order to build the normalizer.

        :Returns:
            DataNormalizer : the DataNormalizer built from dc_source.
        """
        
        if not isinstance(dc_source, DataCollectionPCD):
            pass #TODO throw exception

        normalizer = DataNormalizer()

        nb_attributes = dc_source.get_nb_attributes()
        value_min = [ -sys.maxint for i in range(nb_attributes) ]
        value_max = [  sys.maxint for i in range(nb_attributes) ]

        instances = dc_source.get_instances()
        for instance in instances:
            # Process the attribute values
            for index, value in enumerate(instance.get_attributes()):
                if value < value_min[index]:
                    value_min[index] = value

                if value > value_max[index]:
                    value_max[index] = value

        normalizer.__set_min(value_min)
        normalizer.__set_max(value_max)

        return normalizer

             
    def normalize(self, dc_source):
        """
        Transforms a DataCollectionPCD into a DataCollectionPCD
        by normalizing the values of the attributes.

        :Parameters:
            dc_source : DataCollectionPCD 
                Data collection to normalize.

        :Returns:
            DataCollectionPCD : Data collection in which normalized intances have to be places.
        """
        dc_dest = DataCollectionPCD()

        instances = dc_source.get_instances()
        for instance_old in instances:

            attributes_new = []

            # Normalize each attribute
            for index, value in enumerate(instance_old.get_attributes()):
                value_new = (value - self.__min[index]) * self.__max[index] * (self.__sup - self.__inf) + self.__inf
                attributes_new.append(value_new)

            instance_new = DataInstance(instance_old.get_id_number(), attributes_new, instance_old.get_label())
            dc_dest.add_instance(instance_new)             

        return dc_dest



class Filter:
    """
    Embed a Numerizer and a Normalizer.
    """
   
    def __init__(self):
        """
        Initializer.
        """

        self.__numerizer = Numerizer()
        self.__numerizer = Normalizer()
        

    def filter(self, dc_source):
        """
        Filter dc_source and produce and numerized and normalized
        data collection.
        
        :Parameters:
            dc_source : DataCollectionPCD 
                Data collection to filter.

        :Returns:
            DataCollectionPCD : data collection filtered
        """
        dc_numerized = self.__numerizer.numerize(dc_source)
        dc_normalized = self.__normalizer.normalize(dc_numerized)
        return dc_normalized


    def number_to_label(self, number):
        """
        Get the string value associated with the numeric value
        for the label.
        """
        return numerizer.number_to_label(number)
