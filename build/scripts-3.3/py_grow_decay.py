#!/usr/bin/python3

import argparse
import xml.dom.minidom

import numpy
import scipy
from lmfit import minimize, Parameters, report_errors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(filename, units=1.0):
    data_x = []
    data_y = []
    data_dy = []
    with open(filename, 'r') as datafile:
        lineN = 0
        for line in datafile:
            lineN += 1
            # Comments are from '#' to the end of line
            if '#' in line:
                line = line.split('#')[0]
            if len(line) > 0:
                values = line.split()
                try:
                    data_x.append(float(values[0]) * units)
                    data_y.append(float(values[1]))
                    data_dy.append(float(values[2]))
                except (IndexError, ValueError):
                    print('# Ignored bad input data at line {0}: {1}'
                           .format(lineN, line.strip('\n')))
    return (numpy.array(data_x), numpy.array(data_y), numpy.array(data_dy))


def grow_decay(params, data_x):
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


def grow_decay_flash(params, data_x):
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

def grow_decay_offset(params, data_x):
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

def decay_only(params, data_x):
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


def decay_only2(params, data_x):
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


def grow_decay2(params, data_x):
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


def grow_decay_isomer(params, data_x):
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


def grow_decay_diffusion(params, data_x):
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



def residual(params, data_x, data_y, data_dy):
    model = fitfunc(params, data_x)
    return (data_y - model) / data_dy


MODELS = {'grow_decay' : grow_decay,
          'grow_decay_flash' : grow_decay_flash,
          'grow_decay2' : grow_decay2,
          'grow_decay_isomer' : grow_decay_isomer,
          'grow_decay_diffusion' : grow_decay_diffusion,
          'decay_only' : decay_only,
          'decay_only2' : decay_only2,
          'grow_decay_offset' : grow_decay_offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config_file',  help='')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='More verbose output')
    args = parser.parse_args()

    dom = xml.dom.minidom.parse(args.config_file)
    config = dom.getElementsByTagName('Config')[0]
    tot_cols = len(config.getElementsByTagName('model'))

    fig_index = 1
    for inputfile in config.getElementsByTagName('file'):
        data_x, data_y, data_dy = load(inputfile.getAttribute('name'),
                                       float(inputfile.getAttribute('scale')))
        for model in inputfile.getElementsByTagName('model'):
            fitfunc = MODELS.get(model.getAttribute('name'))
            params = Parameters()
            for p in model.getElementsByTagName('param'):
                p_attr = {'name': None, 'value': None, 'vary': None,
                          'min' : None, 'max' : None}
                for attr in p_attr.keys():
                    if p.hasAttribute(attr):
                        p_attr[attr] = p.getAttribute(attr)
                if p_attr['min'] is not None:
                    p_attr['min'] = float(p_attr['min'])
                if p_attr['max'] is not None:
                    p_attr['max'] = float(p_attr['max'])
                if p_attr['vary'] is not None:
                    if p_attr['vary'].lower() == 'false':
                        p_attr['vary'] = False
                    else:
                        p_attr['vary'] = True
                else:
                    p_attr['vary'] = True

                params.add(p_attr['name'],
                           vary=p_attr['vary'], 
                           value=float(p_attr['value']),
                           min=p_attr['min'],
                           max=p_attr['max'])

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

            result = minimize(residual, params, args=(data_x[ix_low:ix_high],
                                                      data_y[ix_low:ix_high],
                                                      data_dy[ix_low:ix_high]))

            print('File {}, model {}'.format(
                    inputfile.getAttribute('name'), model.getAttribute('name')))
            print('Reduced Chi2:', '{:.3f}'.format(result.redchi))
            for key, par in result.params.items():
                print('{} {:.3f} +/- {:.3f}'.format(key, par.value, par.stderr))
            print()

            time = numpy.arange(0, params['T2'].value, 
                                (params['T2'].value) / 200)

            fig = plt.figure(fig_index, (9, 6))
            fig_index += 1

            ax1 = plt.subplot(2, 1, 1)
            ax1.set_ylabel('Counts')
            ax1.set_xlim(0, params['T2'].value)
            ax1.errorbar(data_x, data_y, data_dy, fmt='o', markersize=4,
                        markerfacecolor='none', markeredgecolor='blue',
                        label="Data '{}'".format(inputfile.getAttribute('name').replace('_', '\_')))
            ax1.plot(time, fitfunc(result.params, time), linewidth=2,
                    color='red',
                    label=r"Fit '{}' $\chi^2 / N$ = {: .2f}".format(model.getAttribute('name').replace('_', '-'), result.redchi))
            ax1.legend(loc='best', fontsize='small')

            ax2 = plt.subplot(2, 1, 2)
            ax2.set_xlim(0, params['T2'].value)
            ax2.errorbar(data_x[0:ix_high], 
                    data_y[0:ix_high] - 
                    fitfunc(result.params, data_x[0:ix_high]),
                    data_dy[0:ix_high], fmt='o')
            ax2.axhline(xmin=0, xmax=params['T2'].value,
                       color="black")
            ax2.set_ylabel(r'$\Delta$(Data - Fit)')
            ax2.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()
