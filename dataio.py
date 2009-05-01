"""
Data input/ouput module.
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


import csv
import sys
from npy.data import DataInstance

class DataIO_CSV:
    """
    Data CSV input/output class 
    
    :IVariables:
        __stream : Stream 
            Stream instance used for the I/O operations. In the case of this
            CSV module, the stream is the prefix of the CSV file to be used.
        __attribute_id : string
            Name of the attribute that contains the index field in the
            stream.
        __attribute_label : string
            Name of the attribute that contains the label field in the
            stream.
        __null_values : sequence of strings
            Strings that are to be considered as representing invalid
            and/or missing values in the data set.
    """

    def __init__(self, stream=None, attribute_id=None, attribute_label=None, null_values=[]):
        """
        Initializer
        
        :Parameters:
            stream : string 
                Prefix of the CSV file to be used.
            attribute_id : string
                Name of the attribute that contains the index field in the
                stream.
            attribute_label : string
                Name of the attribute that contains the label field in the
                stream.
            null_values : sequence of strings
                Strings that are to be considered as representing invalid
                and/or missing values in the data set.
        """
        self.__stream = stream
        self.__attribute_id = attribute_id
        self.__attribute_label = attribute_label 

        if not isinstance(null_values, list):
            null_values = [null_values]
        self.__null_values = null_values


    def set_stream(self,stream):
        """
        Setter for the stream

        :Parameters:
            stream : string 
                Prefix of the CSV file to be used.
        """
        self.__stream = stream


    def get_stream(self):
        """
        Getter for the stream

        :Returns:
            string : the prefix of the used CSV file.
        """
        return self.__stream


    def add_null_value(self, value):
        """
        Add a possible value for the null values, which will be replaced
        by None in the reading process.
        """
        self.__null_values.append(value)
        pass


    def set_attribute_id(self, name):
        """
        Set the name of the id attribute

        :Parameters:
            name : string 
                Name of the id attribute 
        """
        self.__attribute_id = name


    def set_attribute_label(self, name):
        """
        Set the label of the id attribute

        :Parameters:
            name : string 
                Name of the label attribute 
        """
        self.__attribute_label = name


    def read(self, data_collection):
        """
        Read a CSV file and fill the provided DataCollectionRAW with instances.

        :Parameters:
            data_collection : DataCollectionRAW
                Data Collection to be filled with the file content.
        """
        # TODO check that data_collection is indeed a DataCollectionRAW
        
        # TODO exception file exists
        # Store the file content into a sequence
        filename = self.__stream + ".csv" 
        reader = csv.reader(open(filename, "rb"))
        rows = []
        for row in reader:
            rows.append(row)

        # Check that we can find the label index
        index_label = -1
        index_label_found = True 
        try:
            index_label = rows[0].index(self.__attribute_label)
        except ValueError:
            index_label_found = False

        if not index_label_found:
            pass #TODO send exception: we cannot work without labels

        # Check that we can find the id index
        index_id = -1
        index_id_found = True 
        try:
            index_id = rows[0].index(self.__attribute_id)
        except ValueError:
            index_id_found = False

        # Read the first row to get the attribute names
        name_attribute = rows[0][:]
        name_attribute.remove(self.__attribute_label)
        if index_id_found:
            name_attribute.remove(self.__attribute_id)

        #for name in rows[0]:
        #    if name != self.__attribute_label and name != self.__attribute_id:
        #        name_attribute.append(name)
        data_collection.set_name_attribute(name_attribute) 

        # Create instances with the remaining lines
        for index_row, row in enumerate(rows[1:]):
            value_attribute = []
            for index_attribute, value in enumerate(row):
                if index_attribute == index_label \
                  or (index_id_found and index_attribute == index_id):
                    continue
                if value in self.__null_values:
                    value = None
                value_attribute.append(value)

            if not index_id_found:
                # If the file does not have instance indices, we simply
                # make them based on the instance row index
                index_instance = index_row
            else:
                index_instance = row[index_id]

            label = row[index_label]

            instance = DataInstance(index_instance, value_attribute, label)
            data_collection.add_instance(instance)
