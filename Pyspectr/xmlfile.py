#!/usr/bin/env python3
"""K. Miernik 2018
k.a.miernik@gmail.com
Distributed under GNU General Public Licence v3

This module handles the ascii spectra defined in a xml file

The structure of the xml file is as follows 

<?xml version="1.0"?>
<spectra>
    <data_file name="kurczak0001.lst" id="1" header="10" 
               columns="1" comments="#" /> 
    <data_file name="kurczak0002.lst" id="2" header="10" 
               columns="1" comments="#" /> 
...
</spectra>

where 
name - Data file name
id - Unique histogram number assigned to the file
header - File header length (these lines are ignored)
columns - If equals 2, it is asumed that first column is the energy, 
          and the second is the number of counts. If "1" then it is understood
          as number of counts, while the x-axis is automatically generated
comments - symbol used for comment lines (ignored while reading data)

"""

import datetime
import numpy
import os
import struct
import sys
from array import array
import xml.dom.minidom

from Pyspectr.exceptions import GeneralError as GeneralError

class XmlFile:
    """Class for handling .xml and .dat (ASCII) spectra files.
    
    """

    def __init__(self, file_name, rounding='mid'):
        """Variable 'histograms' will hold a list of avaiable
        histograms and their parameters, see 'load' for details.
        """
        self.histograms = {}
        self.load(file_name)


    def load(self, file_name):
        """ Loads xml definition of histogram files

        """
        try:
            dom = xml.dom.minidom.parse(file_name)
            spectra = dom.getElementsByTagName('spectra')[0]


            data_files = spectra.getElementsByTagName('data_file')
            for data in data_files:
                histogram = {}
                his_id = int(data.getAttribute('id'))
                histogram['file_name'] = data.getAttribute('name')
                histogram['columns'] = int(data.getAttribute('columns'))
                histogram['header'] = int(data.getAttribute('header'))
                histogram['comments'] = data.getAttribute('comments')
                # For compability with hisfile class
                histogram['dimension'] = 1
                histogram['half_words_per_ch'] = 0
                histogram['title'] = histogram['file_name']
                histogram['offset'] = 0
                histogram['scaled'] = 0
                histogram['minc'] = 0
                histogram['maxc'] = 0
                self.histograms[his_id] = histogram

        except ValueError:
            raise GeneralError('Error loading spectra definition from {}'.format(file_name))


    def load_histogram(self, his_id):
        """Loads histogram with given id from the ascii file as specified
        in a xml config file

        Works only with 1D histograms for now
        
        """
        if self.histograms.get(his_id) is None:
            raise GeneralError("Histogram {} not found".format(his_id))
        histogram = self.histograms[his_id]

        data_file = open(histogram['file_name'])

        line_number = 0
        data_number = 0
        x_axis = []
        weights = []
        for line in data_file:
            line_number += 1
            if line_number <= histogram['header']:
                continue
            line = line.strip()
            if line.startswith(histogram['comments']):
                continue
            if histogram['columns'] == 1:
                try:
                    data_point = int(line)
                except ValueError:
                    raise GeneralError('Error loading data point, line {} file {}'.format(line_number,  histogram['file_name']))
                weights.append(data_point)
                x_axis.append(data_number)
                data_number += 1
            elif histogram['columns'] == 2:
                line = line.split()
                try:
                    x_point = int(line[0])
                    data_point = int(line[1])
                except ValueError:
                    raise GeneralError('Error loading data point, line {} file {}'.format(line_number,  histogram['file_name']))
                x_axis.append(x_point)
                weights.append(data_point)

        x_axis = numpy.array(x_axis)
        weights = numpy.array(weights)
        return [1, x_axis, None, weights]



if __name__ == "__main__":
    pass
