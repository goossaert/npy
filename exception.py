"""
Exception module.
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


class NpyException(Exception):
    """
    Exception base class.
    
    The base Exception class is overridden, so that the __init__() function
    can be redefined.
    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class NpyStreamError(NpyException):
    """
    Raised when an error is met while working with streams.
    """
    pass


class NpyTransferFunctionError(NpyException):
    """
    Raised when a critical error regarding transfer function is encountered.
    """
    pass


class NpyDataTypeError(NpyException):
    """
    Raised when the type of a parameter is not the one that is expected.
    """
    pass


class NpyIndexError(NpyException):
    """
    Raised when an error regarding internal data structure indices occurs.
    """
    pass


class NpyValueError(NpyException):
    """
    Raised when the value of a parameter is critical.
    """
    pass


class NpyUnitError(NpyException):
    """
    Raised when a problem occurs during the creation of a `Unit`.
    """
    pass
