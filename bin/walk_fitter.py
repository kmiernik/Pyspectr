#!/usr/bin/env python3
"""K. Miernik 2012
k.a.miernik@gmail.com
Distributed under GNU General Public Licence v3

Fits walk correction model function to data stored in his/drr files
Walk histograms should have uncalibrated energy (raw channels), a shift of
time is also expected.

"""

import argparse
import math
import numpy
import matplotlib.pyplot as pyplot
from lmfit import minimize, Parameters, report_errors

import Pyspectr.hisfile as hisfile
from Pyspectr.exceptions import GeneralError as GeneralError

def fit_funcA(params, data_x):
    """Walk correction model A

    """
    a0 = params['a0'].value
    a1 = params['a1'].value
    a2 = params['a2'].value
    a3 = params['a3'].value
    a4 = params['a4'].value
    return (a0 + a1 / (data_x + a2) + a3 * numpy.exp(-data_x / a4))


def fit_funcB1(params, data_x):
    """Walk correction model B1 (low part)

    """
    a0 = params['a0'].value
    a1 = params['a1'].value
    a2 = params['a2'].value
    a3 = params['a3'].value
    return (a0 + (a1  + a2 / (data_x + 1.0)) * numpy.exp(-data_x / a3))
            

def fit_funcB2(params, data_x):
    """Walk correction model B2

    """
    a0 = params['a0'].value
    a1 = params['a1'].value
    a3 = params['a3'].value
    return (a0 + a1 * numpy.exp(-data_x / a3))
            

def residual(params, data_x, data_y, data_dy):
    """Residuals to minimize

    """
    model = fit_func(params, data_x)
    return (data_y - model) / data_dy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='More verbose output.')
    parser.add_argument('infile', help='Input file')
    parser.add_argument('--his', nargs=1, type=int, required=True,
                        help='Histogram id')
    parser.add_argument('--shift', nargs=1, type=float, default=[10.0],
                        help='Time shift')
    parser.add_argument('--binx', nargs=1, type=int, default=[1],
                        help='X binning factor')
    parser.add_argument('--biny', nargs=1, type=int, default=[8],
                        help='Y binning factor')
    parser.add_argument('--cut', nargs=2, type=float, default=[60.0, 16834],
                        help='Energy cut')
    parser.add_argument('--dump', action='store_true', 
            help='Print data used for fitting, do not fit nor plot')
    args = parser.parse_args() 


    binx = args.binx[0] 
    biny = args.biny[0]
    if (not (binx != 0 and ((binx & (binx - 1)) == 0)) or
       not (biny != 0 and ((biny & (biny - 1)) == 0))):
        print('Binning factor must be a power of 2')
        exit()


    shift = args.shift[0]
    energy_low_cut = args.cut[0]
    energy_high_cut = args.cut[1]

    try:
        his = hisfile.HisFile(args.infile)
    except GeneralError as err:
        print(err.msg)
        exit()

    dim, x_axis, y_axis, data = his.load_histogram(args.his[0])
    if dim != 2:
        print('Data must be a 2D histogram!')
        exit()

    # Rebinng histogram using reshape trick
    x_axis = x_axis.reshape((x_axis.shape[0]//binx, binx)).mean(1)
    y_axis = y_axis.reshape((y_axis.shape[0]//biny, biny)).mean(1)
    data_rebinned = data.reshape((data.shape[0]//binx,
                                  binx, 
                                  data.shape[1]//biny,
                                  -1)).sum(3).sum(1)

    walk_x = []
    walk_y = []
    walk_dy = []
    for y in range(data_rebinned.shape[1]):
        if not(energy_low_cut < y_axis[y] < energy_high_cut):
            continue
        s = sum(data_rebinned[:, y])
        if s > 0:
            # Weighted average
            mean = sum(x_axis * data_rebinned[:, y] / s)

            # Weighet avarage std dev
            V1 = s
            V2 = sum(data_rebinned[:, y] * data_rebinned[:, y])
            if V1 * V1 != V2:
                std_dev = numpy.sqrt(
                            V1 / (V1 * V1 - V2) * 
                            sum(data_rebinned[:, y] * (x_axis - mean) *
                                (x_axis - mean))
                            )
            else:
                continue
            # Ommit point with large deviation (ugly plot)
            if std_dev > 20:
                continue
            # Ommit too large difference
            if abs(mean - shift) > 80:
                continue
            walk_x.append(y_axis[y])
            walk_y.append(mean - shift)
            walk_dy.append(std_dev)

    walk_x = numpy.array(walk_x)
    walk_y = numpy.array(walk_y)
    walk_dy = numpy.array(walk_dy)

    if args.dump:
        for x, y, dy in zip(walk_x, walk_y, walk_dy):
            print(x, y, dy)
    else:
        params = Parameters()
        fit_func = fit_funcB2
        # Model A
        #params.add('a0', value=2)
        #params.add('a1', value=8000)
        #params.add('a2', value=200)
        #params.add('a3', value=10)
        #params.add('a4', value=200)
        # Model B
        params.add('a0', value=20, min=0, max=100)
        params.add('a1', value=20)
        #params.add('a2', value=300)
        params.add('a3', value=1500, min=1, max=5000)
        result = minimize(residual, params, args=(walk_x, walk_y, walk_dy))

        print('Reduced Chi2:', '{:.3f}'.format(result.redchi))
        for key, par in result.params.items():
            print('{} {:.3f} +/- {:.3f}'.format(key, par.value, par.stderr))

        for key, par in result.params.items():
            print('{:.3f}'.format(par.value), end=' ')
        print()

        pyplot.subplots(2, 1, sharex=True)
        pyplot.subplot(2, 1, 1)
        pyplot.ylabel('Walk correction (10 ns)')
        pyplot.errorbar(walk_x, walk_y, walk_dy, fmt='bo')
        pyplot.plot(walk_x, fit_func(params, walk_x), 'k')

        pyplot.subplot(2, 1, 2)
        pyplot.ylabel('Model - data')
        pyplot.xlabel('E (channels)')
        pyplot.plot(walk_x, fit_func(params, walk_x) - walk_y, 'bo')
        pyplot.show()
