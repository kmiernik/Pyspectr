
#!/usr/bin/env python3
"""
    K. Miernik 2013
    k.a.miernik@gmail.com
    GPL v3

    Peak fitting class
"""

import math
import numpy
import os
import sys
import time

import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_errors

from Pyspectr.exceptions import GeneralError as GeneralError

class PeakFitter:

    def __init__(self, peaks, baseline, plot_name):
        self.plot_name = plot_name
        self.params = Parameters()
        self.peaks = peaks
        self.baseline = baseline
        if baseline == 'linear':
            self.params.add('a0')
            self.params.add('a1')
        elif baseline == 'quadratic':
            self.params.add('a0')
            self.params.add('a1')
            self.params.add('a2', value=0.0)
        else:
            raise GeneralError("Unknown background type {}".format(background))

        for peak_index in range(len(self.peaks)):
            self.params.add('x{}'.format(peak_index))
            self.params.add('s{}'.format(peak_index))
            self.params.add('A{}'.format(peak_index))
            if self.peaks[peak_index].get('model') == 'gauss_l':
                self.params.add('sL{}'.format(peak_index))


    def _gauss(self, params, data_x, peak_index):
        """Gaussian function

        """
        s = params['s{}'.format(peak_index)].value
        mu = params['x{}'.format(peak_index)].value
        A = params['A{}'.format(peak_index)].value
        return ( A / (math.sqrt(2 * math.pi) * s) *
                numpy.exp(-0.5 * ((data_x - mu) * (data_x - mu))
                                / math.pow(s, 2)) )

    def _gauss_lskew(self, params, data_x, peak_index):
        """Left skewed gaussian 

        """
        s = params['s{}'.format(peak_index)].value
        mu = params['x{}'.format(peak_index)].value
        A = params['A{}'.format(peak_index)].value
        sL = params['sL{}'.format(peak_index)].value
        y = []
        for x in data_x:
            if x < mu:
                d = 2 * math.pow(s, 2) * (1 + sL / s * (mu - x))
            else:
                d = 2 * math.pow(s, 2) 
            y.append(A / (math.sqrt(2 * math.pi) * s) *
                    math.exp(-0.5 * math.pow(x - mu, 2) / d) )
        return numpy.array(y)


    def _linear(self, params, data_x):
        a0 = params['a0'].value
        a1 = params['a1'].value
        return a0 + a1 * data_x


    def _quadratic(self, params, data_x):
        a0 = params['a0'].value
        a1 = params['a1'].value
        a2 = params['a2'].value
        return a0 + a1 * data_x + a2 * data_x * data_x


    def restrict_width(self, smin, smax):
        for i, peak in enumerate(self.peaks):
            self.params['s{}'.format(i)].value = (smax + smin) / 2
            self.params['s{}'.format(i)].max = smax


    def fit_func(self, params, data_x):
        y = numpy.zeros((len(data_x)))
        if self.baseline == 'linear':
            y += self._linear(params, data_x)
        elif self.baseline == 'quadratic':
            y += self._quadratic(params, data_x)
        for peak_index in range(len(self.peaks)):
            if (self.peaks[peak_index].get('model') is None or
                self.peaks[peak_index].get('model') == 'gauss'):
                y += self._gauss(params, data_x, peak_index)
            elif self.peaks[peak_index].get('model') == 'gauss_l':
                y += self._gauss_lskew(params, data_x, peak_index)
        return y


    def residual(self, params, data_x, data_y, data_dy):
        """Residuals to minimize

        """
        model = self.fit_func(params, data_x)
        return (data_y - model) / data_dy


    def find_area(self, data_x, peak_index):
        if (self.peaks[peak_index].get('model') is None or
            self.peaks[peak_index].get('model') == 'gauss'):
            yp = self._gauss(self.params, data_x, peak_index)
        elif self.peaks[peak_index].get('model') == 'gauss_l':
            yp = self._gauss_lskew(self.params, data_x, peak_index)

        return(numpy.sum(yp))


    def _initialize(self, data_x, data_y):
        for i, peak in enumerate(self.peaks):
            E = float(peak.get('E'))
            model = peak.get('model')
            self.params['x{}'.format(i)].value = E
            self.params['x{}'.format(i)].min = data_x[0]
            self.params['x{}'.format(i)].max = data_x[-1]
            self.params['s{}'.format(i)].value = 0.85
            self.params['s{}'.format(i)].vary = False
            self.params['A{}'.format(i)].value = data_y[int(E - data_x[0])]
            if model == "gauss_l":
                self.params['sL{}'.format(i)].value = 0.1
                self.params['sL{}'.format(i)].min = 0.0
                self.params['sL{}'.format(i)].max = 2.0

        x0 = numpy.average(data_x[0:5])
        y0 = numpy.average(data_y[0:5])
        x1 = numpy.average(data_x[-6:-1])
        y1 = numpy.average(data_y[-6:-1])
        self.params['a1'].value = (y1 - y0) / (x1 - x0)
        self.params['a0'].value = y0 - x0 * self.params['a1'].value


    def fit(self, data_x, data_y, data_dy, show, pause):
        self._initialize(data_x, data_y)
        result = minimize(self.residual, self.params, 
                          args=(data_x, data_y, data_dy))

        x = numpy.linspace(data_x[0], data_x[-1], 1000)
        y0 = self.fit_func(self.params, x)

        if self.baseline == 'linear':
            yb = self._linear(self.params, data_x)
        elif self.baseline == 'quadratic':
            yb = self._quadratic(self.params, data_x)

        if show == 'plot' or show == 'svg':
            plt.clf()
            plt.xlabel('Channel')
            plt.ylabel('Counts')
            plt.plot(data_x, data_y, linestyle='steps-mid')
            plt.plot(data_x, yb, linestyle='--')
            plt.plot(x, y0, linewidth=1.0)
            if show == 'svg':
                svg_name = 'fit_{0}_{1}-{2}'.format(self.plot_name,
                                            int(data_x[0]), int(data_x[-1]))
                svg_name = svg_name.replace('.', '').replace('/', '') + '.svg'
                plt.savefig(svg_name)
            else:
                plt.draw()
                time.sleep(pause)
        elif show == 'quiet':
            pass

