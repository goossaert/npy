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


    def train(self, network, data_set, name_metric_function, metric_value_min, nb_iterations_max, interval_check=100):
        """
        Apply a training process upon a `DataSet`.

        Makes the assumption that the metric functions gives higher values
        for higher network performances.
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


    def train(self, network, data_set, name_metric_function, metric_value_min, nb_iterations_max, interval_check=100):
        """
        Apply the training process on a `DataSet`, until the metric
        value computed using metric_function *equals or is greater than*
        metric_value_min.

        :Returns:
            integer : the number of iterations in the learning.

        :Raises NpyValueError:
            If interval_check is lower than 1.

        :Raises NpyTransferFunctionError:
            If name_metric_function does not correspond to a metric function.
        """
        
        if interval_check < 1:
            raise NpyValueError, 'interval_check has to be greater or equal to 1'

        Factory.check_prefix(name_metric_function, Metric.prefix)
        metric_function = Factory.build_instance_by_name(name_metric_function)

        nb_iterations_current = 0
        metric_value_computed = metric_value_min - 1
        while nb_iterations_current < nb_iterations_max and metric_value_computed < metric_value_min:
            network.learn_cycles(data_set, interval_check)
            data_classification = network.classify_data_set(data_set)
            metric_value_computed = metric_function.compute_metric(data_set, data_classification)
            nb_iterations_current += interval_check
            
        return nb_iterations_current


    @staticmethod
    def build_instance():
        return TrainSimple()



# Declare the learning functions to the Factory
Factory.declare_instance(TrainSimple())
