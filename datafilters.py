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
        pass

    @staticmethod
    def normalize(self, data_collection)
        """
        Transforms a DataCollectionRAW into a DataCollectionPCD
        by transforming all the ordinal and categorical attributes
        into numerical interval attributes.

        :Parameters:
            data_collection : DataCollectionRAW 
                Data collection to numerize.
        """
         


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
