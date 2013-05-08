#!/usr/bin/env python3

class GeneralError(Exception):
    """General error class

    """
    def __init__(self, msg = ''):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)
