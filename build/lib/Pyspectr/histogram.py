"""K. Miernik 2012
k.a.miernik@gmail.com
Distributed under GNU General Public Licence v3

This module holds Histogram class which is a main data keeping
object in Pyspectr. Data are loaded from experimental files
containing histograms and loaded into the Histogram class that
adds also information about display (limits, binning, ...).

"""

import numpy
from Pyspectr.exceptions import GeneralError as GeneralError


class Histogram:
    """Histogram class holds data for 2 and 3 dimensional histograms.
    The 'weights' holds array of shape matching the shape of x and y axis
    (weights is a 1D or 2D array). The 'errors' keeps uncertainty value for
    each data point and is of the same shape as weights array.

    """


    def __init__(self, dim=1):
        """Initialize empty Histogram """
        self.dim = dim
        self.x_axis = None
        self.y_axis = None
        self.weights = None
        self.errors = None
        self.title = None


    def save_to_txt(self, file_name):
        """Save the histogram to ascii text file"""
        #numpy.savetxt(file_name, data.transpose(),
        #              fmt=['%8.2f', '%8.2f', '%8.2f']
        pass


    def rebin1d(self, bin_size):
        """Bin 1D histogram, bin_size must be an integer larger than 1.

        Returns a new, rebinned histogram. Be careful with errors, as
        they are calculated as sqrt(N), where N is the number of counts
        after rebinning. If the errors before rebinning are different 
        than sqrt(N) are therefore it is not the correct value!

        """
        if self.dim != 1:
            raise GeneralError('This function rebins 1D histograms only')
        # Drop the end of the histogram if lenght of histogram % bin_size
        # is not 0
        drop = len(self.weights) % bin_size
        if drop != 0:
            weights = self.weights[0:-drop]
            x_axis = self.x_axis[0:-drop]
        else:
            weights = self.weights[:]
            x_axis = self.x_axis[:]
        weights = weights.reshape((-1, bin_size)).sum(axis=1)
        x_axis = x_axis.reshape((-1, bin_size)).mean(axis=1)
        errors = numpy.sqrt(abs(weights))

        histo = Histogram(dim=self.dim)
        histo.x_axis = x_axis
        histo.weights = weights
        histo.title = '{}, bin {}'.format(self.title, bin_size)
        histo.errors = errors
        return histo


    def normalize1d(self, norm, bin_size=1, xmin=None, xmax=None):
        """Normalize 1D histogram using density,
        norm must be an int, float or
        'area' string - indicating normalization of the density to 1,
        using a range of xmin to xmax (or whole range by default).

        Each bin is divided by the normalization factor
        n_i = n_i / norm / bin_size

        """
        if self.dim != 1:
            raise GeneralError('This function normalizes 1D histograms only')

        histo = Histogram(dim=self.dim)
        histo.x_axis = self.x_axis
        histo.weights = self.weights
        histo.errors = self.errors

        if bin_size == 0:
            print('Warning: bin_size 0 overridden,' + 
                    ' using 1 instead')
            bin_size = 1

        if isinstance(norm, str):
            if norm.lower() == 'area':
                normalization = histo.weights[xmin:xmax].sum()
                if normalization == 0:
                    print('Warning: normalization 0 overridden,' + 
                          ' using 1 instead')
                    normalization = 1
            else:
                raise GeneralError("Normalization must be int," + 
                                " float or 'area' string")
        elif isinstance(norm, float) or isinstance(norm, int):
            if norm == 0:
                normalization = 1
                print('Warning: normalization 0 overridden, using 1 instead')
            else:
                normalization = norm
        else:
            raise GeneralError("Normalization must be int," + 
                               " float or 'area' string")

        histo.title = '{}, / {:.2e}'.format(self.title, 
                                           normalization * bin_size)
        histo.weights = histo.weights / normalization / bin_size
        histo.errors = histo.errors / normalization / bin_size
        return histo

