"""
Factory module.
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


class Factory:
    """
    Factory that creates instances of classes just by reading a class name.

    :CVariables:
        __subclasses : dictionary 
            This dictionary associate one instance of each possible subclass
            of a base class to a unique name. That way, one can create
            instances of any subclass just by passing the name to the
            Factory Method function.
    """

    # dictionary of the subclasses
    __subclasses = {}


    def __init__(self):
        """
        Initializer
        """
        pass


    @staticmethod
    def build_instance_by_name(name):
        """
        Build an instance of the class given in parameter.

        :Parameters:
            name : string
                Name of the class to instanciate

        :Returns:
            An instance of the required updator.
        """
        # TODO What happens when the name is not in the dict?
        return Factory.__subclasses[name].build_instance()
    
    
    @staticmethod
    def declare_instance(instance):
        """
        Add the name and an instance of a given activator in the general
        activator list. It will be used when a network will be built from
        a stream.

        :Parameters:
            instance : implements `FactoryMixin`
                Instance of the class
        """
        # TODO First check that the name is not already present in the dict
        name = instance.get_name()
        Factory.__subclasses[name] = instance 



class FactoryMixin: 
    """
    Mixin class that provides classes with features related with their
    Factory Method capabilities.
    """
    
    def __init__(self):
        """
        Initializer
        """
        self.__name = None 


    def get_name(self):
        return self.__name


    def _set_name(self,name):
        self.__name = name


    def build_instance(self):
        """
        Build an instance of the implementing class.

        :Returns:
            An instance of the implementing class.
        """
        pass 
