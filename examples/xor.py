"""
XOR example for the npy package.
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


import sys, string, itertools, random

sys.path.append('..')
from network import Network
from networkio import NetworkIO_CSV
from dataio import DataIO_CSV
from datafilter import Filter
from metric import MetricAccuracy
from train import TrainSimple
from data import *



def get_data_and_filter():
    # Create the CSV data file reader
    dataio_csv = DataIO_CSV(stream='xor.csv', attribute_id='index', attribute_label='xor', null_values='None')

    # Read the data file from the file
    ds_raw = DataSet()
    dataio_csv.read(ds_raw)

    # Create the filter based on the dataset
    data_filter = Filter(ds_source=ds_raw, normalizer_lower_bound=-1,normalizer_upper_bound=1)

    # Apply the filter on the data
    ds_filtered = data_filter.filter( ds_raw )
    
    return ds_filtered, data_filter


def create_network():
    # Create the network
    network = Network()
    network.learning_rate = 0.1

    # Pick 'LabelMax' as a labelling function, which means that when
    # classifying a data instance, the label given to the instance is the one
    # for which the value of the output function is the highest
    network.label_function = 'la_max'

    # Add units (layers) to the network:
    # Input unit: no need for activation and update functions
    network.add_unit( 2 )
    # Hidden unit
    network.add_unit( 3, 'ac_sigmoid', 'up_backpropagation' )
    # Output unit
    network.add_unit( 1, 'ac_sigmoid', 'up_backpropagation' )

    return network


def train_network(network, ds_filtered):
    trainer = TrainSimple()

    # Train the network:
    # - 'me_accuracy' is the metric function for accuracy, which corresponds to
    #   the number of instances correctly classied over the total number of
    #   instances
    # - 0.95 is the metric value at which the training can be stopped when
    #   reached, so in the case of the accuracy, the network is stopped when
    #   it gets to 95% of accuracy 
    # - 10000 is the maximum number of iterations 
    # - 100 is the time step at which the network is tested against the metric
    #   value constraint, so every 100 iterations, the network is tested
    nb_iterations = trainer.train_network( network, ds_filtered, 'me_accuracy', 0.95, 10000, 100 )

    print 'Iterations:', nb_iterations


def show_data_set(ds_filtered):
    print 'Data set:'
    for index, instance in enumerate(ds_filtered.get_data_instances()):
        print index + 1, instance.get_attributes(), instance.get_label_number()


def show_classification(classification, data_filter):
    print 'Classification:'
    for data_label in classification.get_data_labels():
        data_instance = data_label.get_data_instance()
        index_instance = data_instance.get_index_number()
        label_number = data_label.get_label_number()
        label_string = data_filter.label_number_to_string(label_number)
        print index_instance, label_string


if __name__ == '__main__':

    # Load the data set and create the filter based on it
    (ds_filtered, data_filter) = get_data_and_filter()
    show_data_set(ds_filtered)

    # Create and train the network
    network = create_network()
    train_network(network, ds_filtered)

    # Classify the dataset
    classification = network.classify_data_set(ds_filtered)
    show_classification(classification, data_filter)

    # Compute the metric value
    metric = MetricAccuracy()
    print 'Accuracy:', metric.compute_metric(ds_filtered, classification)

    # Save the network topology and its content
    csv_stream = NetworkIO_CSV('csvfile.csv')
    csv_stream.write_topology(network)
    csv_stream.write_weights(network)

    # Create a new network identical to the one that has just been trained
    network_new = Network()
    csv_stream.read_topology(network_new)
    csv_stream.read_weights(network_new)

