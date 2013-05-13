#!/usr/bin/env python3
"""
    K. Miernik 2013
    k.a.miernik@gmail.com
    GPL v3

    Spectrum fitting code

"""

import argparse
import math
import numpy
import os
import sys
import time

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_errors

sys.path.append('/home/krm/Documents/Programs/Python/Pyspectr')
from Pyspectr.hisfile import HisFile as HisFile
from Pyspectr.peak_fitter import PeakFitter as PeakFitter
from Pyspectr.exceptions import GeneralError as GeneralError


class SpectrumParser:
    def __init__(self, file_name):
        self.base_name, ext = os.path.splitext(file_name)
        if len(ext) > 0 and ext in (".gz", ".his", ".tgz"):
            self.file_type = 'his'
            self.data_file = HisFile(file_name)
        elif len(ext) > 0 and ext in ".txt":
            self.file_type = 'txt'
            self.data_file = numpy.loadtxt(file_name)
        else:
            raise GeneralError(
                    'Files other than txt, his, tgz and gz are not supported')

    def parse(self, spectrum, show, pause):
        spectra_ids = spectrum.get('id')
        id_list = []
        if self.file_type == 'his':
            for element in spectra_ids.split(','):
                element = element.split('-')
                if len(element) > 1:
                    new_elements = []
                    for i in range(int(element[0]), int(element[1]) + 1):
                        id_list.append(i)
                else:
                    id_list.append(int(element[0]))
        elif self.file_type == 'txt':
            if spectra_ids != '':
                raise GeneralError('Spectrum id not supported for txt files')
            else:
                id_list.append('')

        peaks = spectrum.findall('peak')
        x_min = int(spectrum.get('min'))
        x_max = int(spectrum.get('max'))
        smin = spectrum.get('smin')
        smax = spectrum.get('smax')

        for spectrum_id in id_list:
            plot_name = '{}_{}'.format(self.base_name, spectrum_id)
            PF = PeakFitter(peaks, spectrum.get('baseline'), plot_name)
            if self.file_type == 'txt':
                data_x = self.data_file[x_min:x_max, 0]
                data_y = self.data_file[x_min:x_max, 1]
                data_dy = self.data_file[x_min:x_max, 2]
            elif self.file_type == 'his':
                data_x, data_y = self.data_file.load_histogram(spectrum_id)
                data_x = data_x[x_min:x_max]
                data_y = data_y[x_min:x_max]
                data_dy = []
                for y in data_y:
                    dy = numpy.sqrt(y) if y > 0 else 1.0
                    data_dy.append(dy)
                data_dy = numpy.array(data_dy)

            if smin is not None and smax is not None:
                PF.restrict_width(float(smin), float(smax))
            PF.fit(data_x, data_y, data_dy, show, pause)
            for i, peak in enumerate(peaks):
                if peak.get('ignore') == 'True':
                    continue
                x0 = PF.params['x{}'.format(i)].value
                dx = PF.params['x{}'.format(i)].stderr
                A = PF.params['A{}'.format(i)].value
                dA = PF.params['A{}'.format(i)].stderr
                s = PF.params['s{}'.format(i)].value
                Area = PF.find_area(data_x, i)
                print('{:>8} {:>8.2f} {:>8.2f} {:>8.1f} {:>8.1f} {:>8.3f} {:>8.1f}'
                      .format(peaks[i].get('E'), x0, dx, A, dA, s, Area))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs=1, 
                        help='Config files')
    parser.add_argument('--pause', '-p', nargs=1, type=float, default=[0.5],
                        help='Pause time in seconds')

    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument('--plot', action='store_true',
                        help='Plot window during fitting')
    out_group.add_argument('--svg', action='store_true',
                        help='SVG files saved during fitting')
    out_group.add_argument('--quiet', action='store_true',
                        help='No output during fitting')

    args = parser.parse_args()

    show = 'plot'
    if args.svg:
        show = 'svg'
    elif args.quiet:
        show = 'quiet'

    try:
        tree = ET.parse(args.config[0])
    except (xml.parsers.expat.ExpatError,
            xml.etree.ElementTree.ParseError) as err:
        print("File '{0}' parsing error: {1}".format(
               args.config[0], err))
        exit()

    plt.ion()
    plt.show()
    root = tree.getroot()
    for data_file in root.findall('data_file'):
        SP = SpectrumParser(data_file.get('name'))
        print('# File: ', data_file.get('name'))
        print('#{:^7} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}'
                .format('Peak', 'x0', 'dx', 'A', 'dA', 's', 'Area'))
        for spectrum in data_file.findall('spectrum'):
            SP.parse(spectrum, show, args.pause[0])
