#!/usr/bin/env python3
"""K. Miernik 2012
k.a.miernik@gmail.com
Distributed under GNU General Public Licence v3

This module handles the HIS/DRR data files.

"""

import datetime
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
        self.base_name = os.path.splitext(file_name)[0]
        if os.path.exists(file_name):
            self.load(file_name)
            self.exists = True
        else:
            self.exists = False


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
            self._load_normal()
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


    def _load_normal(self):
        """Loads .drr file to get list of available histograms

        """
        with open('{}{}'.format(self.base_name, '.drr'), 'rb') as drr_file:
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
                data = array('H')
            elif self.histograms[his_id]['half_words_per_ch'] == 2:
                data = array('I')
            else:
                msg = 'half-words per channel histograms are not supported'
                raise GeneralError('{} {}'.format(
                                self.histograms[his_id]['half_words_per_ch']),
                                    msg) 

            his_file.seek(offset * 2)
            data.fromfile(his_file, length)

        data = numpy.array(data, dtype=numpy.int32)
        x_axis = numpy.arange(self.histograms[his_id]['minc'][0] + self._dx,
                            self.histograms[his_id]['maxc'][0] + self._dx + 1.0)
        if self.histograms[his_id]['dimension'] == 2:
            y_axis = numpy.arange(self.histograms[his_id]['minc'][1] + self._dx,
                            self.histograms[his_id]['maxc'][1] + self._dx + 1.0)
            data = numpy.reshape(data, (self.histograms[his_id]['scaled'][1],
                                        self.histograms[his_id]['scaled'][0]))
            data = numpy.transpose(data)

        if self.histograms[his_id]['dimension'] == 1:
            return [1, x_axis, None, data]
        else:
            return [2, x_axis, y_axis, data]


    def declare_histogram_1d(self, his_id, x_size, title):
        """Declaration of a new 1D histogram"""
        self._has_drr_changed = True
        self.histograms[his_id] = {
                'dimension' : 1,
                'half_words_per_ch' : 2,
                'title' : title,
                'offset' : -1,
                'scaled' : [x_size, 0, 0, 0],
                'minc' : [0, 0, 0, 0],
                'maxc' : [x_size - 1, 0, 0, 0]
                }


    def declare_histogram_2d(self, his_id, x_size, y_size, title):
        """Declaration of a new 2D histogram"""
        self._has_drr_changed = True
        self.histograms[his_id] = {
                'dimension' : 2,
                'half_words_per_ch' : 2,
                'title' : title,
                'offset' : -1,
                'scaled' : [x_size, y_size, 0, 0],
                'minc' : [0, 0, 0, 0],
                'maxc' : [x_size - 1, y_size - 1, 0, 0]
                }


    def plot(self, his_id, x, y=None, n=1):
        """Adds n to bin x (x, y) to histogram his_id"""
        his_file = open('{}.his'.format(self.base_name), 'r+b')
        half_words =  self.histograms[his_id]['half_words_per_ch']

        if self.histograms[his_id]['dimension'] == 1:
            x_size = self.histograms[his_id]['scaled'][0]
            ix = int(x + self._dx)
            if ix < 0 or ix >= x_size:
                raise GeneralError('His ID: {}, value {} out of range'.
                        format(his_id, x))

            his_file.seek(self.histograms[his_id]['offset'] * 2 + 
                        ix * half_words * 2, 0)
            if half_words == 1:
                data_type = 'H'
            elif half_words == 2:
                data_type = 'I'
            else:
                raise GeneralError(
                        'His ID: {}, unsupported half words per ch = {}'.
                        format(his_id, half_words)
                        )
            value = array(data_type)
            value.fromfile(his_file, 1)
            his_file.seek(self.histograms[his_id]['offset'] * 2 + 
                        ix * half_words * 2, 0)
            his_file.write(struct.pack(data_type, value[0] + n))

        else:
            x_size = self.histograms[his_id]['scaled'][0]
            y_size = self.histograms[his_id]['scaled'][1]
            ix = int(x + self._dx)
            iy = int(y + self._dx)
            if (iy < 0 or iy >= y_size or ix < 0 or ix > x_size):
                raise GeneralError('His ID: {}, value {} out of range'.
                        format(his_id, y))

            his_file.seek(self.histograms[his_id]['offset'] * 2 + 
                        ix * half_words * 2 +
                        iy * self.histograms[his_id]['scaled'][0] *
                        half_words * 2, 0)
            if half_words == 1:
                data_type = 'H'
            elif half_words == 2:
                data_type = 'I'
            else:
                raise GeneralError(
                        'His ID: {}, unsupported half words per ch = {}'.
                        format(his_id, half_words)
                        )
            value = array(data_type)
            value.fromfile(his_file, 1)
            his_file.seek(self.histograms[his_id]['offset'] * 2 + 
                        ix * half_words * 2 +
                        iy * self.histograms[his_id]['scaled'][0] *
                        half_words * 2, 0)
            his_file.write(struct.pack(data_type, value[0] + n))

        his_file.close()


    def create(self):
        self._create_drr()
        self._create_empty_his()
   

    def _create_drr(self):
        """Create drr file based on current list of histograms"""
        drr_file = open('{}.drr'.format(self.base_name), 'wb')

        initid = array('B')
        for l in 'HHIRFDIR0001':
            initid.append(ord(l))
        initid.tofile(drr_file)

        n_histograms = len(self.histograms)
        n_halfwords = 0
        histograms_id = []
        for his_id, his in self.histograms.items():
            if his['dimension'] == 1:
                n_halfwords += (his['scaled'][0] * his['half_words_per_ch'])
            else:
                n_halfwords += (his['scaled'][0] * his['scaled'][1] *
                                his['half_words_per_ch'])
            histograms_id.append(his_id)
        drr_file.write(struct.pack('<II', n_histograms, n_halfwords))

        now = datetime.datetime.now()
        date = array('I')
        date.append(0)
        date.append(now.year)
        date.append(now.month)
        date.append(now.day)
        date.append(now.hour)
        date.append(now.minute)
        date.tofile(drr_file)

        garbage = array('b')
        for i in range(84):
            garbage.append(0)
        garbage.tofile(drr_file)

        offset = 0
        for his_id, his in self.histograms.items():
            record = array('H')
            # Dimension
            record.append(his['dimension'])
            # Half-words (2-bytes long) per channel - we use uint32 all the way
            record.append(2)
            # Parameter id numbers - we use all 0
            record.append(0)
            record.append(0)
            record.append(0)
            record.append(0)
            # Raw length
            record.append(his['scaled'][0])
            record.append(his['scaled'][1])
            record.append(his['scaled'][2])
            record.append(his['scaled'][3])
            # Scaled length - we use same as raw
            record.append(his['scaled'][0])
            record.append(his['scaled'][1])
            record.append(his['scaled'][2])
            record.append(his['scaled'][3])
            # Min channel number
            record.append(his['minc'][0])
            record.append(his['minc'][1])
            record.append(his['minc'][2])
            record.append(his['minc'][3])
            # Max channel number
            record.append(his['maxc'][0])
            record.append(his['maxc'][1])
            record.append(his['maxc'][2])
            record.append(his['maxc'][3])

            # Write the whole record to file
            record.tofile(drr_file)

            # Location in his file (in 2-bytes units)
            drr_file.write(struct.pack('<I', offset))
            his['offset'] = offset

            # X axis label (empty)
            label = array('B')
            for i in range(12):
                label.append(0)
            label.tofile(drr_file)

            # Y axis label (empty)
            label.tofile(drr_file)

            # X axis calibration (empty)
            calibration = array('f')
            calibration.append(0)
            calibration.append(0)
            calibration.append(0)
            calibration.append(0)
            calibration.tofile(drr_file)

            # Histogram title
            name = his['title']
            title = array('B')
            l = 0
            while l < 40:
                if l < len(name):
                    title.append(ord(name[l]))
                else:
                    title.append(0)
                l += 1
            title.tofile(drr_file)

            # Calculate offset
            if his['dimension'] == 1:
                offset += his['scaled'][0] * his['half_words_per_ch']
            elif his['dimension'] == 2:
                offset += (his['scaled'][0] * his['scaled'][1] *
                           his['half_words_per_ch'])
            else:
                raise GeneralError('Dimension larger than 2 not supported')
        
        # Write a list of histograms ID in 128 bytes long block
        # each build of 32 records of 4 bytes long
        index = 0
        while index < len(histograms_id):
            block = array('I', [0] * 32)
            for i in range(32):
                if index < len(histograms_id):
                    block[i] = histograms_id[index]
                index += 1
            block.tofile(drr_file)

        drr_file.close()
   

    def _create_empty_his(self):
        """Create empty his file based on current list of histograms"""
        his_file = open('{}.his'.format(self.base_name), 'wb')

        for his_id, his in self.histograms.items():
            if his['half_words_per_ch'] == 1:
                array_type = 'H'
            elif his['half_words_per_ch'] == 2:
                array_type = 'I'

            if his['dimension'] == 1:
                data = array(array_type, [0] * his['scaled'][0])
                data.tofile(his_file)
            elif his['dimension'] == 2:
                data = array(array_type,
                        [0] * his['scaled'][0] * his['scaled'][1])
                data.tofile(his_file)

        his_file.close() 



if __name__ == "__main__":
    pass
