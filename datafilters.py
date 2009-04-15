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

from npy.data import DataCollectionRAW
from npy.data import DataCollectionPCD

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


    def __add_value_for_attribute(self, value_attribute, index_attribute)
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


    def get_number_from_attribute(self, value_attribute, index_attribute)
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


    def __add_value_for_label(self, value_label)
        """
        Add value for label.
        """

        if not value_label in self.__label:
            self.__labels[value_label] = len(self.__label) + 1


    def get_number_from_label(self, value_label)
        """
        Get the numeric value associated with the string value
        for the label.
        """

        if not value_label in self.__label:
            return None #TODO throw exception instead

        return self.__label[value_attribute]


    @staticmethod
    def build_numerizer(self, dc_source)
        """ 
        Builds a DataNumerizer based on the data provided in dc_source.

        :Parameters:
            dc_source : DataCollectionRAW 
                Data to use in order to build the numerizer.
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

             
    def numerize(self, dc_source)
        """
        Transforms a DataCollectionRAW into a DataCollectionPCD
        by transforming all the ordinal and categorical attributes
        into numerical interval attributes.

        :Parameters:
            dc_source : DataCollectionRAW 
                Data collection to numerize.
        """
        if not isinstance(data_collection, DataNumerizer)
            pass TODO #throw exception

        dc_dest = DataCollectionPCD()

        instances = dc_sources.get_instances()
        for instance_old in instances:

            attributes = []

            # Process the attribute values
            for index, value in enumerate(instance_old.get_attributes()):
                try:
                    number = float(value)
                except ValueError:
                    # Every time a non-float attribute value is met,
                    # it is added to the numerizer
                    number = self.get_number_from_attribute(value, index) 
                attributes.append(number)
            

            # Process the label value
            label_old = instance_old.get_label()
            try:
                label_new = float(label_old)
            except ValueError:
                # Every time a non-float label value is met,
                # it is added to the numerizer
                label_new = self.get_number_from_label(label_old)

            instance_new = DataInstance(instance_old.get_id_number(), attributes, label_new)
            dc_dest.add_instance(instance_new)             

          


class DataNormalizer:
    """
    Transforms a DataCollectionPCD into a DataCollectionPCD by transforming
    all the numerical values into values strictly contained into a given
    interval.
    """

    def __init__(self):
        """
        Initializer
        """
