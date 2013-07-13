#!/usr/bin/env python3
"""
K. Miernik 2012
k.a.miernik@gmail.com

Performes fit to selected lines in selected spectra in his/drr files
Skew gaussian function is used for fit. Reads configuration from xml file

Configuration example:
<?xml version="1.0"?>
<config>
    <data_file name="Co56_Apr10"/>
        <spectrum id="113-116" min="2550" max="2750" line="846.77"/>
        <spectrum id="113-115,116" min="3100" max="3400" line="1037.843"/>
    </data_file>
</config>

"""

import argparse
import copy
import math
import numpy
import scipy
import sys
import time

from lmfit import minimize, Parameters, report_errors
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import Pyspectr.peak_fitter as peak_fitter
import Pyspectr.hisfile as hisfile 
from Pyspectr.exceptions import GeneralError


class Parser:
    """ Parses xml, result is in file_list dictionary of a structure:
        {file_name : { spectrum_id : {'line': line, 'min': xmin, 'max' : xmax}, 
                    ... }
        ...}

    """
    def __init__(self, config_file):
        """Aggregates fitter, creates empy file_name dict and loads xml file
        
        """
        self.file_list = {}
        self.parse_xml(config_file)


    def parse_xml_spectrum(self, child, spectra):
        """Parses single <spectrum> element"""
        line = child.get('line')
        xmin = int(child.get('min'))
        xmax = int(child.get('max'))
        model = child.get('model')
        if model is None:
            model = "gauss"
        spectra_ids = child.get('id')
        id_list = []
        for element in spectra_ids.split(','):
            element = element.split('-')
            if len(element) > 1:
                new_elements = []
                for i in range(int(element[0]), int(element[1]) + 1):
                    id_list.append(i)
            else:
                id_list.append(int(element[0]))

        for i in id_list:
            d = {'line': line, 'min': xmin, 'max' : xmax, 'model': model}
            if spectra.get(i) is None:
                spectra[i] = [d]
            else:
               spectra[i].append(d)


    def parse_xml(self, config_file):
        """Parses xml config file"""
        tree = ET.parse(config_file)
        root = tree.getroot()
        for data_file in root.findall('data_file'):
            name = data_file.get('name')
            self.file_list[name] = {}
            for child in data_file.findall('spectrum'):
                self.parse_xml_spectrum(child, self.file_list[name])

    def _find_area(self, fitter, x0, dx, xmin, data_x, data_y, r=5):
        """ r is range in sigmas around middle of the peak

        """
        amin = int(x0 - r * dx) - xmin
        amax = int(x0 + r * dx) - xmin

        best_a1 = fitter.params['a1'].value
        best_a0 = fitter.params['a0'].value
        low_a1 = best_a1 - fitter.params['a1'].stderr
        low_a0 = best_a0 - fitter.params['a0'].stderr
        high_a1 = best_a1 + fitter.params['a1'].stderr
        high_a0 = best_a0 + fitter.params['a0'].stderr

        best_area = numpy.sum(data_y[amin:amax] - 
                              (best_a1 * data_x[amin:amax] + best_a0))
        low_area = numpy.sum(data_y[amin:amax] - 
                             (high_a1 * data_x[amin:amax] + high_a0))
        high_area = numpy.sum(data_y[amin:amax] - 
                              (low_a1 * data_x[amin:amax] + low_a0))
        dA = (high_area - low_area) / 2.0
        return (best_area, dA)


    def process(self, pause, show, verbose, rounding):
        """For each file in file_list, for each spectrum in a given files
        performes fit and adds result to entry in the dict

        """
        for file_name, spectra in self.file_list.items():
            his = hisfile.HisFile(file_name, rounding)
            for hisId, lines in spectra.items():
                data = his.load_histogram(hisId)
                for line in lines:
                    xmin = line['min']
                    xmax = line['max']
                    if data[0] != 1:
                        raise GeneralError('Only 1D histograms are suitable' +
                                'for this calibration')
                    data_x, data_y = data[1][xmin:xmax], data[3][xmin:xmax]
                    data_dy = numpy.sqrt(numpy.abs(data[3][xmin:xmax]))

                    # 0 counts have error 1 (poisson!)
                    for i, dy in enumerate(data_dy):
                        if dy == 0:
                            data_dy[i] = 1

                    try:
                        fitter = peak_fitter.PeakFitter(line['model'])
                        result = fitter.fit(data_x, data_y, data_dy)
                    except ValueError:
                        msg = ('Fit problems with spectrum {} line {}: {}')
                        raise GeneralError(msg.format(hisId, line['line'],
                                           'numerical issue encountered'))
                    except peak_fitter.GeneralError as err:
                        msg = ('Fit problems with spectrum {} line {}: {}')
                        raise GeneralError(msg.format(hisId, line['line'],
                                           err.msg))

                    if line['model'].startswith("gauss_doublet"):
                        line['dx'] = result.params['mu'].stderr
                        line['x0'] = result.params['mu'].value
                        line['x1'] = result.params['mu1'].value
                        line['Area'] = result.params['A'].value
                        line['Area1'] = result.params['A1'].value
                        line['dA'] = result.params['A'].stderr
                        line['dA1'] = result.params['A1'].stderr
                    else:
                        x0 = result.params['mu'].value
                        dx = result.params['mu'].stderr
                        line['x0'] = x0 
                        line['dx'] = dx 
                        line['Area'], line['dA'] = self._find_area(fitter, x0,
                                                                   dx, xmin,
                                                                   data_x,
                                                                   data_y)
                    line['redchi'] = result.redchi

                    if result.params['mu'].stderr == 0:
                        msg = ('Warning, line {} in spectrum {}:' + 
                              ' could not determine uncertainity\n')
                        sys.stderr.write(msg.format(line['line'], hisId))

                    if show == 'plot' or show == 'png':
                        x = numpy.linspace(data_x[0], data_x[-1], 1000)
                        y0 = fitter.fit_func(result.params, x)

                        plt.clf()
                        plt.xlabel('Channel')
                        plt.ylabel('Counts')
                        plt.title('Spectrum {} line {}'.format(hisId,
                                                               line['line']))
                        plt.plot(x, y0)
                        plt.errorbar(data_x, data_y, data_dy, fmt='o')
                        xpos = (plt.xlim()[0] + 
                                (plt.xlim()[1] - plt.xlim()[0]) * 0.1)
                        ypos = (plt.ylim()[1] - 
                                (plt.ylim()[1] - plt.ylim()[0]) * 0.1)
                        text = ('$\mu$ = {0:.2f}\n' + 
                               '$\chi^2/\mathcal{{N}}$' + 
                               ' = {1:.2f}').format(result.params['mu'].value,
                                                   result.redchi)
                        plt.text(xpos, ypos, r'{}'.format(text))
                        if show == 'png':
                            png_name = '{}_{}_{:.0f}.png'.format(
                                                            file_name,
                                                            hisId,
                                                            float(line['line']))
                            plt.savefig(png_name)
                            print('File', png_name, 'saved')
                        elif show == 'plot':
                            plt.draw()
                            time.sleep(pause)
                    elif show == 'text':
                        msg = ('Line {} in spectrum {},' + 
                               ' x0 = {:.2f},' + 
                               ' redchi = {:.2f}\n').format(
                                                        line['line'],
                                                        hisId,
                                                    result.params['mu'].value,
                                                        result.redchi)
                        sys.stderr.write(msg)
                    elif show == 'quiet':
                        pass
                    else:
                        raise GeneralException('Unknown show method {}'.format(
                                                show))

                    if verbose:
                        sys.stderr.write('{}\n'.format(20*'-'))
                        sys.stderr.write(
                            'Line {} spectrum {}\n'.format(line['line'], hisId))
                        sys.stderr.write(
                            'Reduced Chi2: {:.3f}\n'.format(result.redchi))
                        for key, par in result.params.items():
                            sys.stderr.write(
                                '{} {:.3f} +/- {:.3f}\n'.format(key,
                                                         par.value, par.stderr))
                        fwhm_factor = 2 * math.sqrt(math.log(2) * 2)
                        sys.stderr.write('{} {:.3f} +/- {:.3f}\n'.format('FWHM',
                                fwhm_factor * result.params['s'].value,
                                fwhm_factor * result.params['s'].stderr))


class Dumper:
    """Class for saving and analysing results

    """
    def __init__(self, target):
        """target should a writable object"""
        self.target = target

    def dump(self, file_list):
        """Dumps results to a target object"""
        for name, spectra in file_list.items():
            self.target.write('#In file: {}\n'.format(name))
            self.target.write('{0: <5} {1: ^8} {2: ^8} {3: ^6} {4: ^10} {5: ^10} {6:^7}\n'.format("#ID", 'E', 'X0', 'dX', 'Area', 'dA', 'RedChi'))
            for hisId, lines in spectra.items():
                for line in lines:
                    self.target.write('{0: <5} {1: ^8} {2: ^8.2f} {3: ^6.2f} {4: ^10.1f} {5: ^10.1f} {6: ^7.1f}\n'.format(hisId, line['line'], line['x0'], line['dx'], line['Area'], line['dA'], line['redchi']))
                    if line['model'].startswith("gauss_doublet"):
                        self.target.write('{0: <5} {1: ^8} {2: ^8.2f} {3: ^6.2f} {4: ^10.1f} {5: ^10.1f} {6: ^7.1f}\n'.format(hisId, line['line'], line['x1'], line['dx'], line['Area1'], line['dA1'], line['redchi']))
                self.target.write('\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='+', 
                        help='Config files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='More verbose output, prints detailed fit results\
to stderr.')
    parser.add_argument('--output', '-o', nargs=1, default=[sys.stdout], 
                        type=argparse.FileType('w'), help='Output file')
    parser.add_argument('--pause', '-p', nargs=1, type=float, default=[0.5],
                        help='Pause time in seconds')
    parser.add_argument('--rounding', '-r', nargs=1, default=['mid'],
                        help="Defines round method for axis values, available\
 are 'low', 'mid', 'high'.")

    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument('--text', action='store_true',
                        help='Text output during fitting')
    out_group.add_argument('--plot', action='store_true',
                        help='Plot window during fitting')
    out_group.add_argument('--png', action='store_true',
                        help='Png files saved during fitting')
    out_group.add_argument('--quiet', action='store_true',
                        help='No output during fitting')

    args = parser.parse_args()

    show = 'plot'
    if args.text:
        show = 'text'
    elif args.png:
        show = 'png'
    elif args.quiet:
        show = 'quiet'

    plt.ion()

    try:
        dumper = Dumper(args.output[0])
        for input_file in args.config:
            analyzer = Parser(input_file)
            analyzer.process(args.pause[0], show, args.verbose,
                             args.rounding[0])
            dumper.dump(analyzer.file_list)
    except GeneralError as err:
        print(err.msg)
        exit()

