#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This module is useful to read the output of Matthieu Landes C++ code for
# calculating the covariance matrix spectral width.

import numpy as np
import datetime
import obspy

from scipy.signal import hilbert
from matplotlib import dates


def read_spectral_width(files, date_start='01-jan-2000', duration_day=1.0):

    # Get the number of files
    if type(files) is list:
        n_files = len(files)
    else:
        n_files = 1
        files = [files]

    # Get files metadata from first file
    with open(files[0], 'rb') as fid:
        n_times, n_frequencies = np.fromfile(fid, dtype=np.int32, count=2)
        frequencies = np.fromfile(fid, dtype=np.float32, count=n_frequencies)
        spectral_width = np.fromfile(fid, dtype=np.float32)

    # Initialization of coherence matrix from first read
    coherence = spectral_width.reshape(n_times, n_frequencies).T

    # Stack
    if n_files > 1:
        for file in files[1:]:
            with open(file, 'rb') as fid:
                np.fromfile(fid, dtype=np.int32, count=2)
                np.fromfile(fid, dtype=np.float32, count=n_frequencies)
                spectral_width = np.fromfile(fid, dtype=np.float32)
                spectral_width = spectral_width.reshape(n_times, n_frequencies)
                coherence = np.hstack((coherence, spectral_width.T))

    # Define time vector
    time_step = duration_day / n_times
    time_start = dates.datestr2num(date_start)
    times = np.arange(0, n_files, time_step) + time_start
    times = np.append(times, times[-1] + time_step)

    # Remove NaN
    coherence[np.isnan(coherence)] = 0
    coherence[coherence == -1] = 0

    return times, frequencies, coherence


class CovarianceReader():

    def __init__(self, filename, verbose=True):
        """ Check covariance file header and size of file.
        """
        self.file = filename
        with open(filename, 'rb') as flux:
            self.tsize, = np.fromfile(flux, dtype='int32', count=1)
            self.fsize, = np.fromfile(flux, dtype='int32', count=1)
            self.asize, = np.fromfile(flux, dtype='int32', count=1)
            self.f = np.fromfile(flux, dtype='float32', count=self.fsize)
            self.csize = 2 * self.asize * self.asize
            header_end = flux.tell()
            flux.seek(0, 2)
            covariance_size = flux.tell() / 4 - header_end / 4

        if verbose:
            print("Windows          : {:d}".format(self.tsize))
        f = (self.fsize, self.f[0], self.f[-1])
        if verbose:
            print(
                "Frequencies      : {:d} points [{:0.4f}, {:0.4f}] Hz".format(*f))
        if verbose:
            print("Stations         : {:d}".format(self.asize))
        file_size = self.tsize * self.fsize * self.asize**2 * 2
        if verbose:
            print("Check file size  :", covariance_size == file_size)
        if verbose:
            print("--")

    def read(self, tindex):
        """ Extracts the covariance matrix at given tindex (day number).
        """
        with open(self.file, 'rb') as flux:
            flux.seek(4 * (3 + self.fsize + self.csize *
                           self.fsize * (tindex - 1)), 0)
            C = np.fromfile(flux, dtype='float32',
                            count=self.csize * self.fsize)
            C = C.reshape(2, self.asize, self.asize, self.fsize, order='F')
        C = np.squeeze(C[0, :, :, :] + 1j * C[1, :, :, :])
        return C, self.f

    def read_full(self):
        """ Extracts the covariance matrix at given tindex (day number).
        """
        with open(self.file, 'rb') as flux:
            flux.seek(4 * (3 + self.fsize), 0)
            c_len = self.csize * self.fsize * self.tsize
            C = np.fromfile(flux, dtype='float32', count=c_len)
            c_shape = (2, self.asize, self.asize, self.fsize, self.tsize)
            C = C.reshape(*c_shape, order='F')
        C = np.squeeze(C[0, :, :, :, :] + 1j * C[1, :, :, :, :])
        return C, self.f
