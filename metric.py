"""
Metric function module.
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
from data import DataSet
from data import DataClassification


class Metric(FactoryMixin):
    """
    Metric function class.
    """
    
    prefix = 'me_'

    def __init__(self):
        """
        Initializer.
        """
        FactoryMixin.__init__(self)

    
    def compute_metric(self, data_set, data_classification):
        """
        Compute the metric by comparing the expected and computed labels.
        """ 
        pass



class MetricAccuracy(Metric):
    """
    Accuracy metric class.

    The accuracy is (1 - error rate), and the error rate is simply the number
    of correctly classified instances over the total number of instances.
    """


    def __init__(self):
        """
        Initializer.
        """
        Metric.__init__(self)
        self._set_name("me_accuracy")
         

    def compute_metric(self, data_set, data_classification):
        """
        Compute the accuracy.
        """ 

        nb_correctly_classified = 0

        data_instances = data_set.get_data_instances() 
        if len(data_instances) == 0:
            return 0

        for data_instance_original in data_instances:
            label_original = data_instance_original.get_label_number()
            data_instance_classified = data_classification.get_data_label_by_id(data_instance_original.get_index_number())
            label_classified = data_instance_classified.get_label_number()
            if label_classified == label_original:
                nb_correctly_classified += 1

        return nb_correctly_classified / len(data_instances)


    @staticmethod
    def build_instance():
        return MetricAccuracy()



# Declare the metric functions to the Factory
Factory.declare_instance(MetricAccuracy())
