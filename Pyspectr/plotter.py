#!/usr/bin/env python3
"""K. Miernik 2012
k.a.miernik@gmail.com
Distributed under GNU General Public Licence v3

This module provides simple front-end to matplotlib

"""

import math
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

from Pyspectr.exceptions import GeneralError as GeneralError

class Plotter:
    """ This class comunicates with the matplotlib library
    and plot the data

    """

    def __init__(self, size):
        """Initialize the plot window, size defines the shape and size
        of the figure
        0 - None, 
        1 - 8x6, 
        11 (default) - 12x8,
        2 - 2 figs 8x8,
        12 - 2 figs 12x8

        """

        # Max bins in 2d histogram
        self.max_2d_bin = 256
        # Font size of labels and ticks
        self.font_size = 20
        # Set this variable to False if you want to disable the legend
        self.legend = True
        # Change this variable to another cmap if you need different colors
        self.cmap = cm.RdYlGn_r
        # Some selected color maps, you can toggle with toggle_color_map
        self.color_maps = [cm.RdYlGn_r, cm.binary, cm.hot, cm.spectral]

        if size == 0:
            pass
        if size == 1:
            plt.figure(1, (8, 6))
        elif size == 11:
            plt.figure(1, (12, 8))
        elif size == 2:
            plt.figure(1, (8, 6))
            plt.figure(2, (8, 6))
        elif size == 12:
            plt.figure(1, (12, 8))
            plt.figure(2, (12, 8))
        else:
            plt.figure(1, (8, 6))

        if size != 0:
            plt.tick_params(axis='both', labelsize=self.font_size)
            plt.grid()
            plt.ion()
            plt.show()


    def clear(self):
        """Clear current plotting area"""
        plt.clf()
        plt.tick_params(axis='both', labelsize=self.font_size)
        plt.grid()


    def xlim(self, x_range):
        """Change X range of a current plot"""
        plt.xlim(x_range)


    def ylim(self, y_range):
        """Change Y range of a current plot"""
        plt.ylim(y_range)


    def ylog(self):
        """Change y scale to log"""
        plt.yscale('log')


    def ylin(self):
        """Change y scale to linear"""
        plt.yscale('linear')


    def plot1d(self, plot, xlim=None, ylim=None):
        """ Plot 1D histogram
            The mode defines the way the data are presented,
            'histogram' is displayed with steps
            'function' with continuus line
            'errorbar' with yerrorbars

            The norm (normalization factor) and bin_size are given
            for the display purposes only. The histogram is not altered.

        """
        histo = plot.histogram

        if plot.mode == 'histogram':
            plt.plot(histo.x_axis, histo.weights,
                     ls='steps-mid', label=histo.title)
        elif plot.mode == 'function':
            plt.plot(histo.x_axis, histo.weights,
                     ls='-', label=histo.title)
        elif plot.mode == 'errorbar':
            plt.errorbar(histo.x_axis, histo.weights,
                          yerr=histo.errors,
                          marker='o', ls='None', label=histo.title)
        else:
            raise GeneralError('Unknown plot mode {}'.format(plot.mode))

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if self.legend:
            plt.legend(loc=0, numpoints=1, fontsize='small')


    def plot1d_4panel(self, plot, ranges):
        """
        Special 1D histogram plot. The plot is broken into 4 panels (stacked verically)
        the ranges variable should be given in a (x0, x1, x2, x3, x4) format, where
        xi defines the ranges of the subplots (x0-x1, x1-x2, x2-x3, x3-x4)

        """
        for i, r in enumerate(ranges[:-1]):
            x0 = r // plot.bin_size 
            x1 = ranges[i + 1] // plot.bin_size + 1
            ax = plt.subplot(4, 1, i + 1)
            ax.plot(plot.histogram.x_axis[x0:x1], 
                    plot.histogram.weights[x0:x1], 
                    ls='steps-mid')
            ax.set_xlim((r, ranges[i + 1]))
        ax.set_xlabel('E (keV)')
        plt.tight_layout()


    def plot2d(self, plot, xc=None, yc=None, logz=False):
        """Plot 2D histogram 
        xc is x range, yc is y range 
        
        """

        if plot.histogram.dim != 2:
            raise GeneralError('plot2d function needs a 2D histogram!')

        x = plot.histogram.x_axis
        y = plot.histogram.y_axis
        w = plot.histogram.weights

        if xc is not None:
            x = x[int(xc[0]):int(xc[1])]
            w = w[int(xc[0]):int(xc[1]),:]

        if yc is not None:
            y = y[int(yc[0]):int(yc[1])]
            w = w[:, int(yc[0]):int(yc[1])]

        initial_nx = len(x)
        initial_ny = len(y)
        nx = len(x)
        ny = len(y)

        binx = 1
        biny = 1
        # Rebin data if larger than defined number of bins (max_2d_bin)
        # This is needed due to the performance of matplotlib with large arrays
        if nx > self.max_2d_bin:
            binx = math.ceil(nx / self.max_2d_bin)
            missing = binx * self.max_2d_bin - nx
            if missing > 0:
                addx = numpy.arange(plot.histogram.x_axis[-1] + 1, 
                                    plot.histogram.x_axis[-1] + missing + 1)
                x = numpy.concatenate((x, addx))
                nx = len(x)
                z = numpy.zeros((missing, ny))
                w = numpy.concatenate((w, z), axis=0)
            x = numpy.reshape(x, (-1, binx))
            x = x.mean(axis=1)
        if ny > self.max_2d_bin:
            biny = math.ceil(ny / self.max_2d_bin)
            missing = biny * self.max_2d_bin - ny
            if missing > 0:
                addy = numpy.arange(plot.histogram.y_axis[-1] + 1, 
                                    plot.histogram.y_axis[-1] + missing + 1)
                y = numpy.concatenate((y, addy))
                z = numpy.zeros((nx, missing))
                w = numpy.concatenate((w, z), axis=1)
            y = numpy.reshape(y, (-1, biny))
            y = y.mean(axis=1)

        nx = len(x)
        ny = len(y)

        if nx != initial_nx or ny != initial_ny:
            w = numpy.reshape(w, (nx, binx, ny, biny)).mean(3).mean(1)
        w = numpy.transpose(w)

        title = plot.histogram.title
        # If logaritmic scale is used, mask values <= 0
        if logz:
            w = numpy.ma.masked_where(w <= 0, numpy.log10(w))
            title += ' (log10)'
        plt.title(title)
        CS = plt.pcolormesh(x, y, w, cmap=self.cmap)
        plt.xlim(xc)
        plt.ylim(yc)
        plt.colorbar()


    def color_map(self, cmap=None):
        """
        Change the color map to the cmap object, or toggle to the 
        next one from the preselected set,

        """
        if cmap is None:
            try:
                self.cmap = self.color_maps[(self.color_maps.\
                                            index(self.cmap) + 1) %
                                            len(self.color_maps)]
            except ValueError:
                self.cmap = self_color_maps[0]
        else:
            self.cmap = cmap
