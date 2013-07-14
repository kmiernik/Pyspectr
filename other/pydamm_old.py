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
from Pyspectr.decay_fitter import DecayFitter as DecayFitter
from Pyspectr.peak_fitter import PeakFitter as PeakFitter

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
        self.peaks = []


    def _replace_chars(self, text):
        """Clear text from characters that are not accepted by latex"""
        replace_chars = [['_', '-'],
                            ['$', '\$'],
                            ['%', '\%'],
                            ['~', ' '],
                            ['"', "''"],
                            ['\\', ' ']]
        replaced_text = text
        for r_ch in replace_chars:
            replaced_text = replaced_text.replace(r_ch[0], r_ch[1])
        return replaced_text


    def load(self, file_name):
        """Load his file (also tar gzipped files)"""
        self.hisfile = hisfile.HisFile(file_name)


    def d(self, his_id, norm=1, clear=True, plot=True):
        """Plot histogram in current window/ax """

        if self.hisfile is None:
            print('Please load data file first')
            return None

        title = self.hisfile.histograms[his_id]['title']
        data = self.hisfile.load_histogram(his_id)
        if data[0] == 1:
            if clear and plot:
                self.clear()
            label = '{}: {}'.format(his_id,
                                    self.hisfile.histograms[his_id]['title'])
            label = self._replace_chars(label)
            label = ('{}: {}'.format(label, his_id, title))
            if norm != 1:
                label += 'x{: .3f}'.format(1 / norm)
            if plot:
                plt.plot(data[1], data[3] / norm, ls='steps-mid', label=label)
                if self.current.get('xlim') is None:
                    self.current['xlim'] = plt.xlim()
                else:
                    plt.xlim(self.current['xlim'])
                plt.legend(loc=0, fontsize='small')
            self.current['id'] = his_id
            self.current['x'] = data[1]
            self.current['y'] = data[3]/norm
            self.current['z'] = None
            return (data[1], data[3]/norm)
        elif len(data) == 3:
            print('{} is not a 1D histogram'.format(his_id))

    
    def dd(self, his_id, rx=None, ry=None, logz=False, clear=True, plot=True):
        """Plot 2D histogram in current window/ax,
        rx is x range, ry is y range 
        
        """
        if self.hisfile is None:
            print('Please load data file first')
            return None

        title = '{}: {}'.format(his_id, 
                                self.hisfile.histograms[his_id]['title'])
        title = self._replace_chars(title)
        data = self.hisfile.load_histogram(his_id)

        if data[0] != 2:
            print('{} is not a 2D histogram'.format(his_id))
        else:
            if clear and plot:
                self.clear()

            x = data[1]
            y = data[2]
            w = data[3]
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
                    addx = numpy.arange(data[1][-1] + 1, 
                                        data[1][-1] + missing + 1)
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
                    addy = numpy.arange(data[2][-1] + 1, 
                                        data[2][-1] + missing + 1)
                    y = numpy.concatenate((y, addy))
                    z = numpy.zeros((nx, missing))
                    w = numpy.concatenate((w, z), axis=1)
                y = numpy.reshape(y, (-1, biny))
                y = y.mean(axis=1)

            nx = len(x)
            ny = len(y)

            if nx != len(data[2]) or ny != len(data[1]):
                w = numpy.reshape(w, (nx, binx, ny, biny)).mean(3).mean(1)
            w = numpy.transpose(w)

            if plot:
                z = w
                if logz:
                    z = numpy.ma.masked_where(w <= 0, numpy.log10(w))
                    title += ' (log10)'
                plt.title(title)
                CS = plt.pcolormesh(x, y, z,
                                    cmap=cm.RdYlGn_r)
                plt.xlim(rx)
                plt.ylim(ry)
                plt.colorbar()
            self.current['id'] = his_id
            self.current['x'] = x
            self.current['y'] = y
            self.current['z'] = w
            return (x, y, w)


    def gx(self, his_id, rx=None, ry=None, bg=None, norm=1,
           clear=True, plot=True):
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

        if data[0] != 2:
            print('{} is not a 2D histogram'.format(his_id))
        else:
            if clear and plot:
                self.clear()
            x = data[1]
            y = data[2]
            w = data[3]
            if ry is None:
                ry = [0, len(y)-2]
            y = y[ry[0]:ry[1]+1]
            g = w[rx[0]:rx[1]+1, ry[0]:ry[1]+1].sum(axis=0)
            if bg is not None:
                if (bg[1] - bg[0]) != (rx[1] - rx[0]):
                    print('#Warning: background and gate of different widths')
                g = g - w[bg[0]:bg[1]+1, ry[0]:ry[1]+1].sum(axis=0)
            label = 'gx({},{}) {}: {}'.format(rx[0], rx[1], his_id,
                                    self.hisfile.histograms[his_id]['title'])
            label = self._replace_chars(label)
            if norm == 'sum':
                norm = g.sum()
            if bg is not None:
                label += ' bg ({}, {})'.format(bg[0], bg[1])
            if plot:
                plt.plot(y, g/norm, ls='steps-mid', label=label)
                plt.legend(loc=0, fontsize='small')
                if self.current.get('xlim') is None:
                    self.current['xlim'] = plt.xlim()
                else:
                    plt.xlim(self.current['xlim'])
            self.current['id'] = his_id
            self.current['x'] = y
            self.current['y'] = g/norm
            self.current['z'] = None
            return (y, g/norm)


    def gy(self, his_id, ry=None, rx=None, bg=None, norm=1,
           clear=True, plot=True):
        """Make projection on X axis of 2D histogram with gate
        set on Y (ry) and possibly on X (rx), the bg gate selects a 
        background region to be subtracted from data
        
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

        if data[0] != 2:
            print('{} is not a 2D histogram'.format(his_id))
        else:
            if clear:
                self.clear()
            x = data[1]
            y = data[2]
            w = data[3]
            if rx is None:
                rx = [0, len(x)-2]
            x = x[rx[0]:rx[1]+1]
            g = w[rx[0]:rx[1]+1, ry[0]:ry[1]+1].sum(axis=1)
            if bg is not None:
                if (bg[1] - bg[0]) != (ry[1] - ry[0]):
                    print('#Warning: background and gate of different widths')
                g = g - w[rx[0]:rx[1]+1, bg[0]:bg[1]+1].sum(axis=1)

            label = 'gy({},{}) {}: {}'.format(ry[0], ry[1], his_id,
                                    self.hisfile.histograms[his_id]['title'])
            label = self._replace_chars(label)
            if norm == 'sum':
                norm = g.sum()
            if bg is not None:
                label += ' bg ({}, {})'.format(bg[0], bg[1])
            if plot:
                plt.plot(x, g/norm, ls='steps-mid', label=label)
                plt.legend(loc=0, fontsize='small')
                if self.current.get('xlim') is None:
                    self.current['xlim'] = plt.xlim()
                else:
                    plt.xlim(self.current['xlim'])
            self.current['id'] = his_id
            self.current['x'] = x
            self.current['y'] = g/norm
            self.current['z'] = None
            return (x, g/norm)


    def clear(self):
        """Clear current plot"""
        plt.clf()
        plt.xlabel('X')
        plt.ylabel('Y') 
        self.current['xlim'] = None


    def dl(self, x0, x1):
        """Change xrange of 1D histogram"""
        self.current['xlim'] = (x0, x1)
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


    def rebin(self, bin_size, clear=True, plot=True):
        """Re-bin the current histogram"""

        if (self.current['x'] is not None and
            self.current['y'] is not None):
            x = self.rebin_histogram(self.current['x'], bin_size,
                                     False, False)
            y = self.rebin_histogram(self.current['y'], bin_size)
            if plot:
                xlim = plt.xlim()
                if clear:
                    self.clear()
                plt.plot(x, y, ls='steps-mid')
                plt.xlim(xlim)
            self.current['x'] = x
            self.current['y'] = y
            return (x, y)


    def rebin_histogram(self, histogram, bin_size, add=True, zeros=True):
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


    def mark(self, x_mark):
        """Put vertical line on plot to mark the peak (or guide the eye)"""
        plt.axvline(x_mark, ls='--', c='black')


    def set_efficiency_params(self, pars):
        """Sets efficiency calibration parameters, the efficiency is calculated
        as 
        eff = exp(a0 + a1 * log(E) + a2 * log(E)**2 + ...)
        """
        self.eff_pars = pars


    def apply_efficiency_calibration(self, his_id=None, clear=True, plot=True):
        if his_id is not None:
            data = self.hisfile.load_histogram(his_id)
            if data[0] != 1:
                print('{} is not a 1D histogram'.format(his_id))
                return None
            x_axis = data[1]
            data_y = data[3]
        else:
            x_axis = self.current['x']
            data_y = self.current['y']

        # eff(E) = exp(a0 + a1 * log(E) + a2 * log(E)**2 + ...
        # lx = log(E)
        # s = a0 + a1 * lx + ...
        # eff(E) = exp(s)
        for i, x in enumerate(x_axis):
            lx = math.log(x)
            s = 0
            for p, a in enumerate(self.eff_pars):
                s += a * lx**p
            data_y[i] = data_y[i] / math.exp(s)
        self.current['y'] = data_y
        if plot:
            if clear:
                self.clear()
            plt.plot(x_axis, data_y, ls='steps-mid')


    def gamma_gamma_spectra(self, gg_id, gate, clear=True):
        """ Plots gamma-gamma gate broken into 4 subplots (0-600, 600-1200,
        1200-2000, 2000-4000. 
        gg_id is 2D histogram id
        gate is in form ((x1, y1), (x2, y2)) where i=1 is gate on line, i=2
        is gate on background

        """
        self.clear()
        x, y = self.gy(gg_id, gate[0], bg=gate[1])
        ranges = ((0, 600), (600, 1200), (1200, 2000), (2000, 4000))
        for i, r in enumerate(ranges):
            ax = plt.subplot(4, 1, i + 1)
            ax.plot(x[r[0]:r[1]], y[r[0]:r[1]], ls='steps-mid')
            ax.set_xlim(r)
        ax.set_xlabel('E (keV)')
        plt.tight_layout()


    def annotate(self, x, text, shiftx=0, shifty=0):
        """ Add arrow with line energy and possible short text"""
        length = 0.07 * (plt.ylim()[1] - plt.ylim()[0])
        y = self.current['y'][x]
        plt.annotate(text, xy=(x, y),
                    xytext=(x + shiftx, y + length + shifty),
                    rotation=90.,
                    xycoords='data',
                    fontsize=9,
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    arrowprops=dict(width=1, facecolor='black', headwidth=5,
                                    shrink=0.1))


    def load_gates(self, filename):
        """Load gamma gates from text file, the format is:
        # Comment line
        Name    x0  x1  bg0 bg1
        Example:
        110     111 113 115 117

        """
        gatefile = open(filename, 'r')
        lineN = 0
        gates = {}
        for line in gatefile:
            lineN += 1
            line = line.strip()
            if line.startswith('#'):
                continue
            items = line.split()
            if len(items) < 5:
                print('Warning: line {} bad data'.format(lineN))
                continue
            gates[int(items[0])] = ((int(items[1]), int(items[2])),
                                   (int(items[3]), int(items[4])))
        return gates

    def gamma_time_profile(self, his_id, gate, t_bin=1, rt=None, clear=True):
        """Plots gamma time profile, gate should be given in format:
            ((x0, x1, (bg0, bg1))
            
            the rt is gate in time in (t0, t1) format"""

        xg, yg = self.gx(his_id, rx=gate[0], ry=rt, plot=False)
        xb, yb = self.gx(his_id, rx=gate[1], ry=rt, plot=False)
        if t_bin > 1:
            xg = self.rebin_histogram(xg, t_bin,
                                     False, False)
            yg = self.rebin_histogram(yg, t_bin)
            yb = self.rebin_histogram(yb, t_bin)
        dyg = numpy.sqrt(yg)
        dyb = numpy.sqrt(yb)
        y = yg - yb
        dy = numpy.sqrt(dyg**2 + dyb**2)
        if clear:
            self.clear()
        plt.errorbar(xg, y, yerr=dy, ls='None', marker='o')
        plt.axhline(0, ls='-', color='black')


    def fit_gamma_decay(self, his_id, gate, cycle, 
                        t_bin=1, rt=None,
                        model='grow_decay',
                        pars=None,
                        clear=True):
        """Fits gamma decay time profile,
        his_id is E-time histogram id
        gate should be given in format:
            ((x0, x1, (bg0, bg1))
        cycle is list of beam start, beam stop, cycle end, e.g.
        (0, 100, 300)

        t_bin is a binning parameter
            
        rt is a gate in time in (t0, t1) format

        model is model used for fit (see decay_fitter)

        pars is a list of dictionaries (one dict per each parameter)
            
        """
        if pars is None:
            T0 = {'name' : 'T0', 'value' : cycle[0], 'vary' : False}
            T1 = {'name' : 'T1', 'value' : cycle[1], 'vary' : False}
            T2 = {'name' : 'T2', 'value' : cycle[2], 'vary' : False}
            P1 = {'name' : 'P1', 'value' : 100.0}
            t1 = {'name' : 't1', 'value' : 100.0}
            parameters = [T0, T1, T2, P1, t1]
            if model == 'grow_decay2':
                P2 = {'name' : 'P2', 'value' : 1000.0}
                t2 = {'name' : 't2', 'value' : 1000.0}
                parameters.append(P2)
                parameters.append(t2)
        else:
            parameters = pars

        df = DecayFitter()

        xg, yg = self.gx(his_id, rx=gate[0], ry=rt, plot=False)
        xb, yb = self.gx(his_id, rx=gate[1], ry=rt, plot=False)
        if t_bin > 1:
            xg = self.rebin_histogram(xg, t_bin,
                                     False, False)
            yg = self.rebin_histogram(yg, t_bin)
            yb = self.rebin_histogram(yb, t_bin)
        dyg = numpy.sqrt(yg)
        dyb = numpy.sqrt(yb)
        y = yg - yb
        dy = numpy.sqrt(dyg**2 + dyb**2)

        t, n, parameters = df.fit(xg, y, dy, model, parameters)

        if clear:
            self.clear()
        plt.errorbar(xg, y, yerr=dy, ls='None', marker='o')
        plt.plot(t, n, ls='-', color='red')
        plt.axhline(0, ls='-', color='black')
        return (t, n, parameters)


    def fit_peaks(self, rx, his_id=None):
        """Fit gaussian peaks to current plot.
        Returns list of lists:
            [E, x0, dx, A, dA, s, Area]
        where E is name of the peak, x0, A and s are fitted parameters
        and d'something' is its uncertainity. Area is total calculated area.
            """
        peaks = []
        for p in self.peaks:
            if rx[0] <= p.get('E') <= rx[1]:
                peaks.append(p)

        PF = PeakFitter(peaks, 'linear', '')

        if his_id is not None:
            data = self.hisfile.load_histogram(his_id)
            if data[0] != 1:
                print('{} is not a 1D histogram'.format(his_id))
                return None
            x_axis = data[1][rx[0]:rx[1]]
            data_y = data[3][rx[0]:rx[1]]
        else:
            x_axis = self.current['x'][rx[0]:rx[1]]
            data_y = self.current['y'][rx[0]:rx[1]]

        data_dy = numpy.zeros(len(data_y))
        for iy, y in enumerate(data_y):
            if y > 0:
                data_dy[iy] = math.sqrt(y)
            else:
                data_dy[iy] = 1

        PF.fit(x_axis, data_y, data_dy)
        print('#{:^7} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}'
                .format('Peak', 'x0', 'dx', 'A', 'dA', 's', 'Area'))
        peak_data = []
        for i, peak in enumerate(peaks):
            if peak.get('ignore') == 'True':
                continue
            x0 = PF.params['x{}'.format(i)].value
            dx = PF.params['x{}'.format(i)].stderr
            A = PF.params['A{}'.format(i)].value
            dA = PF.params['A{}'.format(i)].stderr
            s = PF.params['s{}'.format(i)].value
            Area = PF.find_area(x_axis, i)
            print('{:>8} {:>8.2f} {:>8.2f} {:>8.1f} {:>8.1f} {:>8.3f} {:>8.1f}'
                    .format(peaks[i].get('E'), x0, dx, A, dA, s, Area))
            peak_data.append([peaks[i].get('E'), x0, dx, A, dA, s, Area])
        return peak_data


    def pk(self, E, **kwargs):
        p = {'E' : E}
        p.update(kwargs)
        self.peaks.append(p)


    def pzot(self):
        self.peaks.clear()

if __name__ == "__main__":
    pass
