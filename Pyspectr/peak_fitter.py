#!/usr/bin/env python3

import math
import numpy
from lmfit import minimize, Parameters, report_errors

from Pyspectr.exceptions import GeneralError as GeneralError

class PeakFitter:
    """Class for fitting gaussian shaped curves using lmfit library"""

    def __init__(self, model=None):
        self.params = Parameters()
        self.select_model(model)

    def select_model(self, model=None):
        """Selects fit model"""
        if model is None or model == 'gauss':
            self.fit_func = self.gauss_bg
            self.params.add('a0', value=0.0)
            self.params.add('a1', value=0.0)
            self.params.add('mu', value=1.0)
            self.params.add('s', value=1.0, min=0.01)
            self.params.add('A', value=1.0, min=0.0)
        elif model == 'gauss_l':
            self.fit_func = self.gauss_bg_lskew
            self.params.add('a0', value=0.0)
            self.params.add('a1', value=0.0)
            self.params.add('mu', value=1.0)
            self.params.add('s', value=1.0, min=0.01)
            self.params.add('sL', value=0.1, min=0.0)
            self.params.add('A', value=1.0, min=0.0)
        elif model == 'gauss_skew':
            self.fit_func = self.gauss_bg_skew
            self.params.add('a0', value=0.0)
            self.params.add('a1', value=0.0)
            self.params.add('mu', value=1.0)
            self.params.add('s', value=1.0, min=0.01)
            self.params.add('sL', value=0.1, min=0.0)
            self.params.add('sH', value=0.1, min=0.0)
            self.params.add('A', value=1.0, min=0.0)
        elif model == 'gauss_doublet':
            self.fit_func = self.gauss_bg_doublet
            self.params.add('a0', value=0.0)
            self.params.add('a1', value=0.0)
            self.params.add('mu', value=1.0)
            self.params.add('mu1', value=1.0)
            self.params.add('s', value=1.0, min=0.01)
            self.params.add('A', value=1.0, min=0.0)
            self.params.add('A1', value=1.0, min=0.0)
        elif model == 'gauss_step_bg':
            self.fit_func = self.gauss_step_bg
            self.params.add('a0', value=0.0)
            self.params.add('a1', value=0.0)
            self.params.add('mu', value=1.0)
            self.params.add('s', value=1.0, min=0.01)
            self.params.add('A', value=1.0, min=0.0)
        else:
            raise GeneralError('Unknown fit model {}'.format(model))

    def find_initial_estimation_(self, data_x, data_y, data_dy):
        """Finds initial estimation of mu, A and baseline"""
        baseline_x = data_x[0:10]
        baseline_y = data_y[0:10]
        baseline_dy = data_dy[0:10]
        temp_fit_func = self.fit_func
        self.fit_func = self.linear
        result = minimize(self.residual, self.params, 
                          args=(baseline_x, baseline_y, baseline_dy))
        self.fit_func = temp_fit_func

        baseline = numpy.linspace(data_x[0], data_x[-1], num=len(data_x))
        for x in baseline:
            x = self.params['a0'].value + x * self.params['a1'].value

        self.params['A'].value = numpy.sum(data_y - baseline)
        self.params['mu'].value = data_x[numpy.argmax(data_y)]
        self.params['mu'].min = data_x[0]
        self.params['mu'].max = data_x[-1]

        if self.fit_func == self.gauss_bg_doublet:
            self.params['A'].value = self.params['A'].value / 2
            self.params['A1'].value = self.params['A'].value
            self.params['mu'].value = self.params['mu'].value 
            self.params['mu1'].value = self.params['mu'].value - 1.0 
            self.params['mu1'].min = self.params['mu'].min
            self.params['mu1'].max = self.params['mu'].max


    def fit(self, data_x, data_y, data_dy):
        """Performes fit"""
        self.find_initial_estimation_(data_x, data_y, data_dy)
        result = minimize(self.residual, self.params, 
                          args=(data_x, data_y, data_dy))
        return result


    def residual(self, params, data_x, data_y, data_dy):
        """Residuals to minimize

        """
        model = self.fit_func(params, data_x)
        return (data_y - model) / data_dy


    def gauss_bg(self, params, data_x):
        """Gaussian curve plus linear background

        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        mu = params['mu'].value
        s = params['s'].value
        A = params['A'].value
        return ( a1 * data_x + a0 + A / (math.sqrt(2 * math.pi) * s) *
                numpy.exp(-0.5 * ((data_x - mu) * (data_x - mu))
                                / math.pow(s, 2)) )


    def gauss_bg_lskew(self, params, data_x):
        """Left skewed gaussian plus background

        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        mu = params['mu'].value
        s = params['s'].value
        A = params['A'].value
        sL = params['sL'].value
        y = []
        for x in data_x:
            if x < mu:
                d = 2 * math.pow(s, 2) * (1 + sL / s * (mu - x))
            else:
                d = 2 * math.pow(s, 2) 
            y.append( a1 * x + a0 + A / (math.sqrt(2 * math.pi) * s) *
                    math.exp(-0.5 * math.pow(x - mu, 2) / d) )
        return numpy.array(y)


    def gauss_bg_skew(self, params, data_x):
        """Left and right skewed gaussian plus background

        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        mu = params['mu'].value
        s = params['s'].value
        A = params['A'].value
        sL = params['sL'].value
        sH = params['sH'].value
        y = []
        for x in data_x:
            if x < mu:
                d = 2 * math.pow(s, 2) * (1 + sL / s * (mu - x))
            else:
                d = 2 * math.pow(s, 2) * (1 + sH / s * (x - mu))
            y.append( a1 * x + a0 + A / (math.sqrt(2 * math.pi) * s) *
                    math.exp(-0.5 * math.pow(x - mu, 2) / d) )
        return numpy.array(y)


    def gauss_bg_doublet(self, params, data_x):
        """Gaussian doublet plus linear background

        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        mu = params['mu'].value
        mu1 = params['mu1'].value
        s = params['s'].value
        A = params['A'].value
        A1 = params['A1'].value
        return ( a1 * data_x + a0 +
                A / (math.sqrt(2 * math.pi) * s) *
                numpy.exp(-0.5 * ((data_x - mu) * (data_x - mu))
                                / math.pow(s, 2)) +
                A1 / (math.sqrt(2 * math.pi) * s) *
                numpy.exp(-0.5 * ((data_x - mu1) * (data_x - mu1))
                                / math.pow(s, 2)) )


    def gauss_step_bg(self, params, data_x):
        """Gaussian plus step function (Woods-Saxon) as background

        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        mu = params['mu'].value
        s = params['s'].value
        A = params['A'].value
        return ( a1 * data_x + a0 / (1 + numpy.exp(2 * (mu - data_x) / s)) +
                A / (math.sqrt(2 * math.pi) * s) *
                numpy.exp(-0.5 * ((data_x - mu) * (data_x - mu))
                                / math.pow(s, 2))  )



    def linear(self, params, data_x):
        """Linear background

        """
        a0 = params['a0'].value
        a1 = params['a1'].value
        return (a1 * data_x + a0)

if __name__ == "__main__":
    pass
