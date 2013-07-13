#!/usr/bin/env python3
"""K. Miernik 2012
k.a.miernik@gmail.com
Distributed under GNU General Public Licence v3

This module provides class to fit grow-in/decay functions
(Bateman equation solution and more)

"""


import math
import numpy
from lmfit import minimize, Parameters, report_errors

class DecayFitter:

    def __init__(self):
        self.models = {'grow_decay' : self.grow_decay,
                       'grow_decay_flash' : self.grow_decay_flash,
                       'grow_decay2' : self.grow_decay2,
                       'grow_decay2_bg' : self.grow_decay2_bg,
                       'grow_decay_isomer' : self.grow_decay_isomer,
                       'grow_decay_diffusion' : self.grow_decay_diffusion,
                       'decay_only' : self.decay_only,
                       'decay_only2' : self.decay_only2,
                       'grow_decay_offset' : self.grow_decay_offset}


    def grow_decay(self, params, data_x):
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        P1 = params['P1'].value
        t1 = params['t1'].value
        y = []
        for t in data_x:
            if T0 < t < T1:
                y.append(P1 * (1 - numpy.exp(-t / t1)))
            elif T1 <= t < T2:
                y.append(P1 * (numpy.exp(T1 / t1) - 1) * numpy.exp(-t / t1))
            else:
                y.append(0)
        return numpy.array(y)


    def grow_decay_flash(self, params, data_x):
        """Sudden flash at t=0 of extra beam"""
        F = params['F'].value
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        P1 = params['P1'].value
        t1 = params['t1'].value
        y0 = params['y0'].value
        y = []
        for t in data_x:
            if T0 < t < T1:
                y.append(F * numpy.exp(-t / t1) + 
                        P1 * (1 - numpy.exp(-t / t1)) + y0)
            elif T1 <= t < T2:
                y.append(F * numpy.exp(-T1 / t1) + 
                        P1 * (1 - numpy.exp(-T1 / t1)) *
                        numpy.exp(-(t - T1) / t1) + y0)
            else:
                y.append(0)
        return numpy.array(y)


    def grow_decay_offset(self, params, data_x):
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        TOFF = params['TOFF'].value
        P1 = params['P1'].value
        t1 = params['t1'].value
        y = []
        for t in data_x:
            if T0 < t < T1:
                y.append(P1 * (1 - numpy.exp(-(t - TOFF) / t1)))
            elif T1 <= t < T2:
                y.append(P1 * (numpy.exp(T1 / t1) - 1) * numpy.exp(-t / t1))
            else:
                y.append(0)
        return numpy.array(y)


    def decay_only(self, params, data_x):
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        P1 = params['P1'].value
        t1 = params['t1'].value
        y = []
        for t in data_x:
            if T0 > t:
                y.append(0)
            elif T0 <= t < T2:
                y.append(P1 * (numpy.exp(T1 / t1) - 1) * numpy.exp(-t / t1))
            else:
                y.append(0)
        return numpy.array(y)


    def decay_only2(self, params, data_x):
        """Simplified bateman, decay part only, the second activity in the chain"""
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        N1 = params['P1'].value
        t1 = params['t1'].value
        N2 = params['P2'].value
        t2 = params['t2'].value
        y = []
        for t in data_x:
            if  t < T0:
                y.append(0)
            elif T0 <= t < T2:
                ts = t - T1
                y.append(N1 / (t1 - t2) * 
                        (numpy.exp(-ts / t1) - numpy.exp(-ts / t2)) +
                        N2 / t2 * numpy.exp(-ts/ t2))
            else:
                y.append(0)
        return numpy.array(y)


    def grow_decay2(self, params, data_x):
        """Bateman for second in the chain, both grow and decay part of the cycle"""
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        P1 = params['P1'].value
        P2 = params['P2'].value
        t1 = params['t1'].value
        t2 = params['t2'].value
        y = []
        for t in data_x:
            if T0 < t < T1:
                y.append( (P1 + P2) * (1 - numpy.exp(-t / t2)) +
                        P1 * t1 / (t1 - t2) *
                        (numpy.exp(-t / t2) - numpy.exp(-t / t1)) )
            elif T1 <= t < T2:
                y.append(P1 * t1 / (t1 - t2) * 
                        (numpy.exp(T1 / t1) - 1) * numpy.exp(-t / t1) +
                        (P2 - P1 * t2 / (t1 - t2)) *
                        (numpy.exp(T1 / t2) - 1) * numpy.exp(-t / t2))
            else:
                y.append(0)
        return numpy.array(y)


    def grow_decay2_bg(self, params, data_x):
        """Bateman for second in the chain, both grow and decay part of 
        the cycle plus background"""
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        P1 = params['P1'].value
        P2 = params['P2'].value
        t1 = params['t1'].value
        t2 = params['t2'].value
        y0 = params['y0'].value
        y = []
        for t in data_x:
            if T0 < t < T1:
                y.append( (P1 + P2) * (1 - numpy.exp(-t / t2)) +
                        P1 * t1 / (t1 - t2) *
                        (numpy.exp(-t / t2) - numpy.exp(-t / t1))
                        + y0)
            elif T1 <= t < T2:
                y.append(P1 * t1 / (t1 - t2) * 
                        (numpy.exp(T1 / t1) - 1) * numpy.exp(-t / t1) +
                        (P2 - P1 * t2 / (t1 - t2)) *
                        (numpy.exp(T1 / t2) - 1) * numpy.exp(-t / t2) 
                        + y0)
            else:
                y.append(0)
        return numpy.array(y)


    def grow_decay_isomer(self, params, data_x):
        """Two half-lives (isomer) of one line

        """
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        P1 = params['P1'].value
        P2 = params['P2'].value
        t1 = params['t1'].value
        t2 = params['t2'].value
        y = []
        for t in data_x:
            if T0 < t < T1:
                y.append( P1 * (1 - numpy.exp(-t / t1)) + 
                        P2 * (1 - numpy.exp(-t / t2)) )
            elif T1 <= t < T2:
                y.append(P1 * (numpy.exp(T1 / t1) - 1) * numpy.exp(-t / t1) + 
                        P2 * (numpy.exp(T1 / t2) - 1) * numpy.exp(-t / t2))
            else:
                y.append(0)
        return numpy.array(y)


    def grow_decay_diffusion(self, params, data_x):
        """Ions escape from tape (diffusion)
        t2 is diffusion time
        P2 is percentage of diffusable ions
        P1 is total number of ions

        """
        T0 = params['T0'].value
        T1 = params['T1'].value
        T2 = params['T2'].value
        P1 = params['P1'].value
        P2 = params['P2'].value
        t1 = params['t1'].value
        t2 = params['t2'].value

        teff = t1 * t2 / (t1 + t2)
        y = []
        for t in data_x:
            if T0 < t < T1:
                y.append( P1 * (1 - numpy.exp(-t / t1)) + 
                        P2 * (1 - numpy.exp(-t / teff)) )
            elif T1 <= t < T2:
                y.append(P1 * (1 - P2) * (numpy.exp(T1 / t1) - 1) *
                        numpy.exp(-t / t1) + 
                        P1 * P2 * (numpy.exp(T1 / teff) - 1) * 
                        numpy.exp(-t / teff))
            else:
                y.append(0)
        return numpy.array(y)



    def residual(self, params, data_x, data_y, data_dy):
        model = self.fitfunc(params, data_x)
        return (data_y - model) / data_dy

    
    def fit(self, data_x, data_y, data_dy, model, parameters):
        """Fit decay  to data,
        
        parameters should be a list of dictionaries:
        dict = {'name': required - parameter name
               'value': required - initial value
               'vary': optional - default is True
               'min' : optional
               'max' : optional}
        """
        params = Parameters()
        for p in parameters:
            params.add(p['name'], value=p['value'])
            if p.get('vary') is not None:
                params[p['name']].vary = p['vary']
            if p.get('min') is not None:
                params[p['name']].min = p['min']
            if p.get('max') is not None:
                params[p['name']].max = p['max']
        ix_low = 0
        ix_high = len(data_x)
        for i, x in enumerate(data_x):
            if x >= params['T0'].value:
                ix_low = i
                break
        for i, x in enumerate(data_x[ix_low:]):
            if x >= params['T2'].value:
                ix_high = i + ix_low
                break

        self.fitfunc = self.models.get(model)
        if self.fitfunc is None:
            print('Could not find model: {}'.format(model))
            return None

        result = minimize(self.residual, params, args=(data_x[ix_low:ix_high],
                                                  data_y[ix_low:ix_high],
                                                  data_dy[ix_low:ix_high]))
        print('Model {}'.format(model))
        print('Reduced Chi2:', '{:.3f}'.format(result.redchi))
        for key, par in result.params.items():
            print('{} {:.3f} +/- {:.3f}'.format(key, par.value, par.stderr))
        print()
        time = numpy.arange(params['T0'].value, params['T2'].value, 
                            (params['T2'].value) / 200)
        counts = self.fitfunc(result.params, time)
        return (time, counts, result.params)

