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
from data import DataCollection
from data import DataClassification


class Metric(FactoryMixin):
    """
    Metric function class.
    """
    

    def __init__(self):
        """
        Initializer.
        """
        FactoryMixin.__init__(self)

    
    def compute_metric(self, data_collection, data_classification):
        """
        Compute the metric by comparing the expected and computed labels.
        """ 
        pass



class MetricErrorRate(Metric):
    """
    Error rate metric class.
    
    The error rate is simply the number of correctly classified instances
    over the total number of instances.
    """


    def __init__(self):
        """
        Initializer.
        """
        Metric.__init__(self)
        self._set_name("me_errorrate")
         

    def compute_metric(self, data_collection, data_classification):
        """
        Compute the error rate.
        """ 

        nb_correctly_classified = 0

        instances = data_collection.get_instances() 
        if len(instances) == 0:
            return 0

        for instance_original in instances:
            label_original = instance_original.get_label_number()
            instance_classified = data_classification.get_classified_instance_by_id(instance_original.get_index_number())
            label_classified = instance_classified.get_label_number()
            if label_classified == label_original:
                nb_correctly_classified += 1

        return 1 - (nb_correctly_classified / len(instances))


    @staticmethod
    def build_instance():
        return MetricErrorRate()
