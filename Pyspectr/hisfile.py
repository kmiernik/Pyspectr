#!/usr/bin/env python3
"""K. Miernik 2012
k.a.miernik@gmail.com
Distributed under GNU General Public Licence v3

This module handles the HIS/DRR data files.

"""

import numpy
import os
import struct
import sys
import tarfile

from array import array

from Pyspectr.exceptions import GeneralError as GeneralError

class HisFile:
    """Class for handling .his file.
    
    """

    def __init__(self, file_name, rounding='mid'):
        """Variable 'histograms' will hold a list of avaiable
        histograms and their parameters, see 'load' for details.
        'rounding' parameter descripes a method
        to be used to interpret the X bins values:
        'low' - bins have values (i)
        'mid' - bins have values (i + 0.5)
        'high'- bins have values (i + 1.0)

        """
        self.rounding = rounding
        self.histograms = {}
        self._tmp_files = []
        self.load(file_name)
        self.file_name = file_name


    def __del__(self):
        """Clears temporary files (if tar gz archive was opened)

        """
        for name in self._tmp_files:
            sys.stderr.write('Removing temp file: {}\n'.format(name))
            os.remove(name)

    
    def load(self, file_name):
        """ Load files, either a 'normal' files or tar gzipped archive

        """
        try:
            tar = tarfile.open(file_name, "r:gz")
        except (tarfile.ReadError, IOError):
            self._load_normal(file_name)
        else:
            self._untar(tar)
            self._load_normal(self._tmp_files[0])


    def _untar(self, tar):
        """Untar and ungzip opened tar archive.
        Loaded files are written to self._tmp_files list

        """

        sys.stderr.write('{:.<30}'.format('Checking tar archive'))
        sys.stderr.flush()
        archive_files = []
        for tarinfo in tar:
            if os.path.splitext(tarinfo.name)[1] not in (".drr", ".his",
                                                        ".list", ".log"):
                raise GeneralError('File {} does not seem to belong to gzipped HIS/DRR histograms'.format(tarinfo.name))
            if os.path.exists(tarinfo.name):
                raise GeneralError('File {} already exists'.format(tarinfo.name))
            archive_files.append(tarinfo.name)
        sys.stderr.write('[Done]\n')

        self._tmp_files += archive_files

        sys.stderr.write('{:.<30}'.format('Extracting tar archive'))
        sys.stderr.flush()
        tar.extractall()
        sys.stderr.write('[Done]\n')

        tar.close()


    def _load_normal(self, file_name):
        """Loads .drr file to get list of available histograms

        """
        name_without_ext = os.path.splitext(file_name)[0]
        with open('{}{}'.format(name_without_ext, '.drr'), 'rb') as drr_file:
            # ID text, number of histograms, number of half-words
            header = drr_file.read(20)
            initid, n_histograms, n_halfwords = struct.unpack('<12sII', header)

            # date (array of 6 ints)
            date = array('I')
            date.fromfile(drr_file, 6)

            # Description and garbage
            header = drr_file.read(84)

            his_list = []
            for i in range(n_histograms):
                raw_entry = drr_file.read(128)
                dimension, half_words_per_ch, offset, title = \
                                    struct.unpack('<hhI40s', (raw_entry[0:4] +
                                                            raw_entry[44:48] +
                                                            raw_entry[88:128]))
                scaled = array('h')
                scaled.frombytes(raw_entry[20:28])
                minc = array('h')
                minc.frombytes(raw_entry[28:36])
                maxc = array('h')
                maxc.frombytes(raw_entry[36:44])

                # These data are currently not used
                # Uncomment if needed
                # 
                #params = array('h')
                #params.frombytes(raw_entry[4:12])
                #raw = array('h')
                #raw.frombytes(raw_entry[12:20])
                #xlabel = struct.unpack('<12s', raw_entry[48:60])
                #ylabel = struct.unpack('<12s', raw_entry[60:72])
                #calcon = array('f')
                #calcon.frombytes(raw_entry[72:88])

                histogram = {}
                histogram['dimension'] = dimension
                histogram['half_words_per_ch'] = half_words_per_ch
                histogram['title'] = title.decode()
                histogram['offset'] = offset
                histogram['scaled'] = scaled
                histogram['minc'] = minc
                histogram['maxc'] = maxc
                his_list.append(histogram)

            for block_index in range(n_histograms // 32 + 1):
                raw_entry = drr_file.read(128)
                his_ids = array('I')
                his_ids.frombytes(raw_entry)
                for index, his_id in enumerate(his_ids):
                    if his_id != 0:
                        his_list[block_index * 32 + index] = \
                                    [his_id, his_list[block_index * 32 + index]]

            self.histograms = {x[0] : x[1] for x in his_list}
            self.base_name = name_without_ext


    @property
    def rounding(self):
        """Return rounding method

        """
        return self._rounding


    @rounding.setter
    def rounding(self, method):
        """Set method of rounding of axis values (down, middle or up)

        """
        if method == 'low':
            self._dx = 0
        elif method == 'mid':
            self._dx = 0.5
        elif method == 'high':
            self._dx = 1.0
        else:
            raise GeneralError('Unknown round method {}'.format(method))
        self._rounding = method


    def load_histogram(self, his_id):
        """Loads histogram with given id from the file.

        Returns array of:
        [dim, data_x, data_y, weights]
        where dim is 1 or 2 (dimension of the histogram)
        data_x is the X axis data
        data_y is the Y axis data (for 2D histograms) or None (for 1D)
        weights is the histograms data, a 1D array (for 1D histogram) or
                2D array for (2D histogram) matching the shapes of 
                data_x, data_y
        
        """
        if self.histograms.get(his_id) is None:
            raise GeneralError("Histogram {} not found".format(his_id))

        offset = self.histograms[his_id]['offset']
        his_name = '{}{}'.format(self.base_name, '.his')
        with open(his_name, 'rb') as his_file:
            length = 1
            dim = self.histograms[his_id]['dimension']
            if dim > 2:
                raise GeneralError('Histograms with dimensions >2 not supported')
            for d in range(self.histograms[his_id]['dimension']):
                length *= self.histograms[his_id]['scaled'][d]

            if self.histograms[his_id]['half_words_per_ch'] == 1:
                data = array('h')
            elif self.histograms[his_id]['half_words_per_ch'] == 2:
                data = array('i')
            else:
                msg = 'half-words per channel histograms are not supported'
                raise GeneralError('{} {}'.format(
                                self.histograms[his_id]['half_words_per_ch']),
                                    msg) 

            his_file.seek(offset * 2)
            data.fromfile(his_file, length)

        x_axis = numpy.arange(self.histograms[his_id]['minc'][0] + self._dx,
                            self.histograms[his_id]['maxc'][0] + self._dx + 1.0)
        if self.histograms[his_id]['dimension'] == 2:
            y_axis = numpy.arange(self.histograms[his_id]['minc'][1] + self._dx,
                            self.histograms[his_id]['maxc'][1] + self._dx + 1.0)
            data = numpy.reshape(data, (self.histograms[his_id]['scaled'][1],
                                        self.histograms[his_id]['scaled'][0]))
            data = numpy.transpose(data)

        if self.histograms[his_id]['dimension'] == 1:
            return [1, x_axis, None, numpy.array(data)]
        else:
            return [2, x_axis, y_axis, data]


if __name__ == "__main__":
    pass
