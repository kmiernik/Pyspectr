#!/usr/bin/env python3
"""K. Miernik 2012
k.a.miernik@gmail.com
Distributed under GNU General Public Licence v3

"""


class GeneralError(Exception):
    """General error class

    """
    def __init__(self, msg = ''):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)
