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


    def read_structure(self, network):
        """
        Read a structure and load it in a given neural network

        :Parameters:
            network : Network
                Network where to put the structure.
        """

        try:
            filename = self.__stream + "_str.csv" 
            reader = csv.reader(open(filename, "rb"))
        except IOError:
            print "Unable to read the file:", filename
            exit()

        rows = []
        for row in reader:
            rows.append(row)
   
        # build the information dictionary as we know it
        struct = {}
        for field, value in zip(rows[0], rows[1]):
            struct[field] = value 

        #print struct

        network.set_structure(struct)


    def write_structure(self, network):
        """
        Write the structure of a neural network to the stream

        :Parameters:
            network : Network
                Network of which the structure has to be written.
        """

        # TODO check exceptions on the write call

        # Retrieve the network structure
        struct = network.get_structure()

        # Build the data table with the fieldnames in the first row 
        fields = []
        values = []
        for k,v in struct.items():
            fields.append(k)
            values.append(v)
        table = [ fields, values ]

        # Write the table to the file
        # TODO exception file exists
        filename = self.__stream + "_str.csv" 
        csvwriter = csv.writer(file(filename, "w"))
        for row in table:
            csvwriter.writerow(row)


    def read_weights(self,network):
        """
        Read weights from the stream and load them in a given neural network.

        :Parameters:
            network : Network
                Network of which weights need to be read.
        """
        # TODO should check if the structure is valid, but that could create
        # problems if the users use this as a semantic assumption in their code.

        # Build the sequence that will be filled with weights
        weight_network = network.get_weights()
       
        # TODO exception file exists
        filename = self.__stream + "_wgt.csv" 
        reader = csv.reader(open(filename, "rb"))

        # Retrieve the field row 
        for row in reader:
            row_fields = row
            break 

        # Build the fieldname reference
        fields = {}
        for index_field, field in zip(range(len(row_fields)), row_fields):
            fields[field] = index_field
       
        # Fill the weight table
        for row in reader:
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
        """
        # TODO check exceptions on the write call
       
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

        # TODO exception file exists
        # TODO check whether this code can be factorized with write_structure()
        filename = self.__stream + "_wgt.csv" 
        csvwriter = csv.writer(file(filename, "w"))
        for row in table:
            csvwriter.writerow(row)

