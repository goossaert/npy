"""
Neural Network input/ouput module.
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


class NetworkIO_CSV:
    """
    Neural network CSV input/output class 

    :IVariables:
        __stream : Stream 
            Stream instance used for the I/O operations. In the case of this
            CSV module, the stream is the prefix of the CSV file to be used.
    """

    def __init__(self, stream=None):
        """
        Initializer
        
        :Parameters:
            stream : string 
                Prefix of the CSV file to be used.
        """
        self.__stream = stream


    def set_stream(self, stream):
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


    def write_table(self, stream_info, table):
        """
        Write a table to a CSV file.

        :Parameters:
            stream_info : string
                Additional information regarding the stream
            table : sequence of sequence
                Table to write

        :Raises NpyStreamError:
            If a problem occurs while writing the file.
        """
        filename = self.__stream + stream_info 
        string_error = 'Unable to write the file: ' + filename
        try:            writer = csv.writer(file(filename, "w"))
        except IOError: raise NpyStreamError, string_error

        for row in table:
            try:            writer.writerow(row)
            except IOError: raise NpyStreamError, string_error


    def read_table(self, stream_info):
        """
        Read a table from a CSV file.

        :Parameters:
            stream_info : string
                Additional information regarding the stream
            table : sequence of sequence
                Table to write

        :Raises NpyStreamError:
            If a problem occurs while reading the file.

        :Returns:
            sequence of sequence: the table read from the file
        """
        try:
            filename = self.__stream + stream_info
            reader = csv.reader(open(filename, "rb"))
        except IOError:
            string_error = 'Unable to read the file: ' + filename
            raise NpyStreamError, string_error

        table = []
        for row in reader:
            table.append(row)

        return table


    def read_structure(self, network):
        """
        Read a structure and load it in a given neural network

        :Parameters:
            network : Network
                Network where to put the structure.

        :Raises NpyStreamError:
            If a problem occurs while reading the file.
        """

        table = self.read_table('_str.csv')
   
        # build the information dictionary as we know it
        struct = {}
        for field, value in zip(table[0], table[1]):
            struct[field] = value 

        network.set_structure(struct)


    def write_structure(self, network):
        """
        Write the structure of a neural network to the stream

        :Parameters:
            network : Network
                Network of which the structure has to be written.

        :Raises NpyStreamError:
            If a problem occurs while reading the file.
        """

        # Retrieve the network structure
        struct = network.get_structure()

        # Build the data table with the fieldnames in the first row 
        fields = []
        values = []
        for k,v in struct.items():
            fields.append(k)
            values.append(v)
        table = [ fields, values ]

        self.write_table('_str.csv', table) 


    def read_weights(self,network):
        """
        Read weights from the stream and load them in a given neural network.

        :Parameters:
            network : Network
                Network of which weights need to be read.

        :Raises NpyStreamError:
            If a problem occurs while reading the file.
        """
        # Build the sequence that will be filled with weights
        weight_network = network.get_weights()
       
        table = self.read_table('_wgt.csv')
        row_fields = table[0]

        # Build the fieldname reference
        fields = {}
        for index_field, field in zip(range(len(row_fields)), row_fields):
            fields[field] = index_field
       
        # Fill the weight table
        for row in table[1:]:
            index_unit = int(row[fields["index_unit"]]) - 2
            index_node = int(row[fields["index_node"]]) - 1
            index_weight = int(row[fields["index_weight"]]) - 1
            weight = float(row[fields["weight"]])
            weight_network[index_unit][index_node][index_weight] = weight

        # Set the new weights to the network
        network.set_weights(weight_network) 


    def write_weights(self,network):
        """
        Write weights from the network to the stream.

        :Parameters:
            network : Network
                Network of which weights need to be written.

        :Raises NpyStreamError:
            If a problem occurs while reading the file.
        """
        # Retrieve the weights and prepare the table
        weight_network = network.get_weights()
        table = []

        # Fill the first row with the field names
        table.append(["index_unit", "index_node", "index_weight", "weight"])

        # Since the input unit is #1, the index_unit range has to be shifted
        # from a value of 2 and not 1 as this is the case for the other ids.
        for index_unit, weight_unit in zip(range(2,len(weight_network)+2), weight_network):
            for index_node, weight_node in zip(range(1,len(weight_unit)+1), weight_unit):
                for index_weight, weight in zip(range(1,len(weight_node)+1), weight_node):
                    table.append([index_unit, index_node, index_weight, weight])

        self.write_table('_wgt.csv', table) 
