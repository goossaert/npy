"""
Training module.
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
from metric import Metric
from exception import *

class Train(FactoryMixin):
    """
    Training class.
    """
    
    prefix = 'tr_'

    def __init__(self):
        """
        Initializer.
        """
        FactoryMixin.__init__(self)


    def train_network(self, network, data_set, name_metric_function, metric_value_min, nb_iterations_max=10000, interval_check=100):
        """
        Apply a training process upon a `DataSet`.

        :Parameters:
            network : `Network`
                Network to train. 
            data_set : `DataSet`
                Data set on which to train the network.
            name_metric_function : string
                Name of the `Metric` function to be used to test the `Network`.
            metric_value_min
                Value minimum of the metric
            nb_iterations_max : integer
                Maximum number of iterations. If reached, the training is
                stopped prematurely in order to avoid infinite loops due
                to unreachable `Metric` values. Set to 10000 by default.
                If set to None, then the number of iterations is infinite.
            interval_check : integer
                Interval of learning cycles at which the `Network` has to be
                tested with the `Metric` function. Set to 100 by default.

        :Return:
            integer : number of iterations that has been necessary for the
            `Network` to reach the minimum value with the `Metric` function.
            If equal to nb_iterations_max, then the minimum value has not
            been reached and the training has been stopped prematurely.

        :Raises NpyValueError:
            If interval_check is lower than 1, or nb_iterations_max is lower
            than 1 and different than None.

        :Raises NpyTransferFunctionError:
            If name_metric_function does not correspond to a metric function.

        :Raises NpyDataTypeError:
            If the given `DataSet` has not been numerized.
        """
        pass



class TrainSimple(Train):
    """
    Make the network learns until it reaches a given value
    for a given `Metric`.
    """

    def __init__(self):
        """
        Initializer.
        """
        Train.__init__(self)
        self._set_name("tr_metric")


    def train_network(self, network, data_set, name_metric_function, metric_value_min, nb_iterations_max, interval_check):
        """
        Apply the training process on a `DataSet`, until the `Metric`
        value computed using metric_function *equals or is greater than*
        metric_value_min. This makes the assumption that the metric
        functions gives higher values for higher network performances.
        The training is stopped after nb_iterations_max to avoid infinite
        loops due to unreachable `Metric` values.
        """
        
        if interval_check < 1:
            raise NpyValueError, 'interval_check has to be greater or equal to 1.'

        if nb_iterations_max != None and nb_iterations_max < 1:
            raise NpyValueError, 'nb_iterations_max has to be greater or equal to 1, or equal to None.'

        try:
            Factory.check_prefix(name_metric_function, Metric.prefix)
            metric_function = Factory.build_instance_by_name(name_metric_function)
        except NpyTransferFunctionError, e:
            raise NpyTransferFunctionError, e.msg

        nb_iterations_current = 0
        metric_value_computed = metric_value_min - 1
        while (nb_iterations_max == None or nb_iterations_current < nb_iterations_max) \
           and metric_value_computed < metric_value_min:
            try:
                network.learn_cycles(data_set, interval_check)
                data_classification = network.classify_data_set(data_set)
            except NpyDataTypeError, e:
                raise NpyDataTypeError, e.msg
            metric_value_computed = metric_function.compute_metric(data_set, data_classification)
            nb_iterations_current += interval_check
            
        return nb_iterations_current


    @staticmethod
    def build_instance():
        return TrainSimple()



# Declare the learning functions to the Factory
Factory.declare_instance(TrainSimple())
