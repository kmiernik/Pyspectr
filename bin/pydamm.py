#!/usr/bin/python3
"""
Krzysztof Miernik 2012
k.a.miernik@gmail.com

Damm-like analysis program for his/drr experiment spectra files.

"""

import numpy
import math
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import sys

sys.path.append('/home/krm/Documents/Programs/Python/Pyspectr')
from Pyspectr import hisfile as hisfile

class Experiment:

    def __init__(self, file_name, size=11):
        """Initialize plot and open data file (file_name)"""
        self.hisfile = None
        if size == 1:
            shape = (8, 6)
        elif size == 11:
            shape = (12, 8)
        else:
            shape = (12, 8)
        if size != 0:
            plt.figure(1, shape)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.ion()
        plt.show()
        # Max bins in 2d histogram
        self.MAX_2D_BIN = 256
        self.file_name = file_name
        self.load(file_name)
        self.current = {'x' : None,
                        'y' : None,
                        'z' : None,
                        'id' : None}


    def load(self, file_name):
        """Load his file (also tar gzipped files)"""
        self.hisfile = hisfile.HisFile(file_name)


    def rebin(histogram, bin_size, add=True, zeros=True):
        """Bin histogram. If add is True, the bins are sum of bins,
        otherwise the mean number of counts is used. 
        If zeros is true, in case the histogram must be extended
        (len(histogram) % bin_size != 0) is extended with zeros,
        otherwise an extrapolation of last two counts is used.

        Example
        y1 = binned(y1, bin1y)
        x1 = binned(x1, bin1y, False, False)

        """
        if len(histogram) % bin_size != 0:
            if zeros:
                addh = numpy.zeros((bin_size - len(histogram) % bin_size))
                histogram = numpy.concatenate((histogram, addh))
            else:
                d = histogram[-1] - histogram[-2]
                l = histogram[-1]
                n = bin_size - len(histogram) % bin_size
                addh = numpy.arange(l, l + n * d, d)
                histogram = numpy.concatenate((histogram, addh))

        if add:
            return histogram.reshape((-1, bin_size)).sum(axis=1)
        else:
            return histogram.reshape((-1, bin_size)).mean(axis=1)


    def d(self, his_id, norm=1):
        """Plot histogram in current window/ax """

        if self.hisfile is None:
            print('Please load data file first')
            return None

        title = self.hisfile.histograms[his_id]['title']
        data = self.hisfile.load_histogram(his_id)
        if len(data) == 2:
            label = ('{}: {} - {}'.
                    format(self.file_name.strip('.his').strip('/'),
                           his_id, title))
            if norm != 1:
                label += 'x{: .3f}'.format(1 / norm)
            plt.plot(data[0], data[1] / norm, ls='steps-mid', label=label)
            plt.legend(loc=0, fontsize='small')
            self.current['id'] = his_id
            self.current['x'] = data[0]
            self.current['y'] = data[1]
            self.current['z'] = None
            return (data[0], data[1])
        elif len(data) == 3:
            print('{} is not a 1D histogram'.format(his_id))

    
    def dd(self, his_id, rx=None, ry=None):
        """Plot 2D histogram in current window/ax,
        rx is x range, ry is y range 
        
        """
        if self.hisfile is None:
            print('Please load data file first')
            return None

        title = '{}: {}'.format(his_id,
                                self.hisfile.histograms[his_id]['title'])
        data = self.hisfile.load_histogram(his_id)

        if len(data) == 2:
            print('{} is not a 2D histogram'.format(his_id))
        elif len(data) == 3:
            self.clear()

            x = data[0]
            y = data[1]
            w = data[2]
            if rx is not None:
                x = x[rx[0]:rx[1]]
                w = w[rx[0]:rx[1],:]

            if ry is not None:
                y = y[ry[0]:ry[1]]
                w = w[:, ry[0]:ry[1]]

            nx = len(x)
            ny = len(y)

            binx = 1
            biny = 1
            # First rebin data if too large
            if nx > self.MAX_2D_BIN:
                binx = math.ceil(nx / self.MAX_2D_BIN)
                missing = binx * self.MAX_2D_BIN - nx
                if missing > 0:
                    addx = numpy.arange(data[0][-1] + 1, 
                                        data[0][-1] + missing + 1)
                    x = numpy.concatenate((x, addx))
                    nx = len(x)
                    z = numpy.zeros((missing, ny))
                    w = numpy.concatenate((w, z), axis=0)
                x = numpy.reshape(x, (-1, binx))
                x = x.mean(axis=1)
            if ny > self.MAX_2D_BIN:
                biny = math.ceil(ny / self.MAX_2D_BIN)
                missing = biny * self.MAX_2D_BIN - ny
                if missing > 0:
                    addy = numpy.arange(data[1][-1] + 1, 
                                        data[1][-1] + missing + 1)
                    y = numpy.concatenate((y, addy))
                    z = numpy.zeros((nx, missing))
                    w = numpy.concatenate((w, z), axis=1)
                y = numpy.reshape(y, (-1, biny))
                y = y.mean(axis=1)

            nx = len(x)
            ny = len(y)

            if nx != len(data[1]) or ny != len(data[0]):
                w = numpy.reshape(w, (nx, binx, ny, biny)).mean(3).mean(1)
            w = numpy.transpose(w)

            plt.title(title)
            #CS = plt.contourf(x, y, numpy.transpose(w),  
            #                  locator=ticker.MaxNLocator(nbins=100))
            CS = plt.pcolormesh(x, y, w,
                                cmap=cm.RdYlGn_r)
            plt.xlim(rx)
            plt.ylim(ry)
            plt.colorbar()
            self.current['id'] = his_id
            self.current['x'] = x
            self.current['y'] = y
            self.current['z'] = w
            return (x, y, w)


    def gx(self, his_id, rx=None, ry=None, norm=1):
        """Make projection on Y axis of 2D histogram with gate
        set on X (rx) and possibly on Y (ry)
        
        """
        if self.hisfile is None:
            print('Please load data file first')
            return None

        if rx is None or len(rx) != 2:
            print('Please select gate on X in (min, max) format')
            return None
        if ry is not None and len(ry) != 2:
            print('Please select gate on Y in (min, max) format')
            return None

        data = self.hisfile.load_histogram(his_id)

        if len(data) == 2:
            print('{} is not a 2D histogram'.format(his_id))
        elif len(data) == 3:
            x = data[0]
            y = data[1]
            w = data[2]
            if ry is not None:
                y = y[ry[0]:ry[1]+1]
                w = w[rx[0]:rx[1]+1, ry[0]:ry[1]+1].sum(axis=0)
            else:
                w = w[rx[0]:rx[1]+1, :].sum(axis=0)
            label = 'gx({},{}) {}: {}'.format(rx[0], rx[1], his_id,
                                    self.hisfile.histograms[his_id]['title'])
            plt.plot(y, w/norm, ls='steps-mid', label=label)
            plt.legend(loc=0, fontsize='small')
            self.current['id'] = his_id
            self.current['x'] = y
            self.current['y'] = w
            self.current['z'] = None
            return (y, w)


    def gy(self, his_id, ry=None, rx=None, norm=1):
        """Make projection on X axis of 2D histogram with gate
        set on Y (ry) and possibly on X (rx)
        
        """
        if self.hisfile is None:
            print('Please load data file first')
            return None

        if ry is None or len(ry) != 2:
            print('Please select gate on Y in (min, max) format')
            return None
        if rx is not None and len(rx) != 2:
            print('Please select gate on X in (min, max) format')
            return None

        data = self.hisfile.load_histogram(his_id)

        if len(data) == 2:
            print('{} is not a 2D histogram'.format(his_id))
        elif len(data) == 3:
            x = data[0]
            y = data[1]
            w = data[2]
            if rx is not None:
                x = x[rx[0]:rx[1]+1]
                w = w[rx[0]:rx[1]+1, ry[0]:ry[1]+1].sum(axis=1)
            else:
                w = w[:, ry[0]:ry[1]+1].sum(axis=1)
            label = 'gy({},{}) {}: {}'.format(ry[0], ry[1], his_id,
                                    self.hisfile.histograms[his_id]['title'])
            plt.plot(x, w/norm, ls='steps-mid', label=label)
            plt.legend(loc=0, fontsize='small')
            self.current['id'] = his_id
            self.current['x'] = x
            self.current['y'] = w
            self.current['z'] = None
            return (x, w)


    def clear(self):
        """Clear current plot"""
        plt.clf()
        plt.xlabel('X')
        plt.ylabel('Y') 


    def dl(self, x0, x1):
        """Change xrange of 1D histogram"""
        plt.xlim(x0, x1)
        if self.current['y'] is not None:
            plt.ylim(min(self.current['y'][x0:x1]),
                     max(self.current['y'][x0:x1]))


    def dmm(self, y0, y1):
        """Change yrange of 1D histogram """
        plt.ylim(y0, y1)


    def log(self):
        """Change y scale to log"""
        plt.yscale('log')


    def lin(self):
        """Change y scale to linear"""
        plt.yscale('linear')


    def list(self, his_id=None):
        """List all histograms or details on selected histogram"""
        if his_id is None:
            for key in sorted(self.hisfile.histograms.keys()):
                print('{: <6} {}'.format(key, 
                                    self.hisfile.histograms[key]['title']))
        else:
            try:
                dim = self.hisfile.histograms[his_id]['dimension']
                xmin = []
                xmax = []
                for i in range(dim):
                    xmin.append(self.hisfile.histograms[his_id]['minc'][0])
                    xmax.append(self.hisfile.histograms[his_id]['maxc'][0])
                print('{: <10} : {}'.format('ID', his_id))
                print('{: <10} : {}'.format('Title', 
                                    self.hisfile.histograms[his_id]['title']))
                print('{: <10} : {}'.format('Dimensions', dim))
                print('{: <10} : ({}, {})'.format('X range', xmin[0], xmax[0]))
                if dim > 1:
                    print('{: <10} : ({}, {})'.format('Y range', 
                                                      xmin[1], xmax[1]))
            except KeyError:
                print('Histogram id = {} not found'.format(his_id))


    def mark(self, x_mark):
        """Put vertical line on plot to mark the peak (or guide the eye)"""
        plt.axvline(x_mark, ls='--', c='#aaaaaa')


    def set_eff_params(self, pars):
        """Sets efficiency calibration parameters, the efficiency is calculated
        as 
        eff = exp(a0 + a1 * log(E) + a2 * log(E)**2 + ...)
        """
        self.eff_pars = pars


if __name__ == "__main__":
    pass
