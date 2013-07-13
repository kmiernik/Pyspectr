#!/usr/bin/env python3
"""
K. Miernik 2012
k.a.miernik@gmail.com

Performes fit to the decay part for all channels in E vs time spectrum

"""

import sys
import argparse
import math
import numpy

from lmfit import minimize, Parameters, report_errors
import matplotlib.pyplot as plt

import Pyspectr.hisfile as hisfile

class GeneralError(Exception):
    """General error class

    """
    def __init__(self, msg = ''):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)


def decay(params, data_x):
    T1 = params['T1'].value
    A = params['A'].value
    tau = params['tau'].value
    return A * numpy.exp(-(data_x - T1) / tau)


def residual(params, data_x, data_y, data_dy):
    model = fitfunc(params, data_x)
    return (data_y - model) / data_dy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('in_file', help='Input file')
    args = parser.parse_args()

    hisId = 2681
    T1 = 200
    T2 = 300

    his = hisfile.HisFile(args.in_file)
    dim, xaxis, yaxis, data = his.load_histogram(hisId)
    #data = data.transpose()

    fitfunc = decay
    params = Parameters()
    params.add('T1', value=T1, vary=False)
    params.add('A', value=100.0, min=0.0)
    params.add('tau', value=100.0, min=0.0)


    sys.stderr.write('.')
    symbol = 0
    for E in range(2, data.shape[0] - 1, 3):
        symbol += 1
        data_slice = sum(data[E-1:E+2])[T1:T2]
        dy = numpy.sqrt(numpy.abs(data_slice))
        for i, v in enumerate(dy):
            if dy[i] == 0:
                dy[i] = 1.0
        data_sum_err = math.sqrt(dy.sum())
        if data_slice.sum() - data_sum_err <= 0:
            continue

        params['A'].value = 100.0                
        params['tau'].value = 100.0                
        result = minimize(residual, params,
                          args=(yaxis[T1:T2], data_slice, dy))
        scale = 0.01
        print(E, result.params['tau'].value * scale * math.log(2),
                 result.params['tau'].stderr * scale * math.log(2))

        sys.stderr.write('\r')
        if symbol % 3 == 0:
            sys.stderr.write('.')
        elif symbol % 3 == 1:
            sys.stderr.write('o')
        else:
            sys.stderr.write('*')
    sys.stderr.write('\n')
