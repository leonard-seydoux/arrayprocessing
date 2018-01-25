#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Gathers the data and methods to do antenna processing.
# Author: L. Seydoux (leonard.seydoux@gmail.com)

import h5py
import obspy
import numpy as np
import arrayprocessing as ap

from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import dates as mdt
from scipy.signal import stft, istft


def read(*args, **kwargs):
    """
    Top-level read function, returns Stream object.
    """

    data = ap.Stream()
    data.read(*args, **kwargs)
    return data


def h5read(*args, **kwargs):
    """
    Top-level read function, returns Stream object.
    """

    data = ap.Stream()
    data.h5read(*args, **kwargs)
    return data


class Stream(obspy.core.stream.Stream):

    def __init__(self):
        """
        Initialize the class with inherited.
        """

        super(Stream, self).__init__()

    def get_times(self):
        """
        Extract the times from the first stream object and
        convert it to a matplotlib datenum vector.
        """

        times = self[0].times()
        times /= 24 * 3600
        times -= times[0]
        start = self[0].stats.starttime.datetime
        times += mdt.date2num(start)

        return times

    def read(self, data_path, sort=False):
        """
        Read the data files specified in the datapath.

        Arguments:
        ----------
        :datapath (str or list): datapath with a single data file or with
        UNIX regexp, or a list of files.

        Keyword arguments:
        ------------------

        :sort (bool): whether or not the different traces are sorted in
        alphabetic order with respect to the station codes.
        """

        waitbar = ap.logtable.waitbar('Read data')

        # If data_path is a str, then only a single trace or a bunch of
        # seismograms with regexp
        if type(data_path) is str:
            self += obspy.read(data_path)
            waitbar.progress(1)

        # If data_path is a list, read each fils in the list.
        elif type(data_path) is list:
            self += obspy.read(data_path[0])
            for index, path in enumerate(data_path[1:]):
                self += obspy.read(path)
                waitbar.progress((index + 2) / len(data_path))

        # Sort if needed
        if sort is True:
            self.sort()

        # Check homogeneity
        n_samples = [stream.stats.npts for stream in self]
        homogeneity = len(set(n_samples)) == 1
        if homogeneity is not True:
            ValueError(" Error : Traces are not homogeneous.")

    def h5read(self, data_path, name='PZ', underscore=True,
               force_start=None):
        """
        Read the data files specified in the datapath.

        Arguments:
        ----------
        :datapath (str or list): datapath with a single data file or with
        UNIX regexp, or a list of files.

        Keyword arguments:
        ------------------

        :sort (bool): whether or not the different traces are sorted in
        alphabetic order with respect to the station codes.
        """

        # Underscored metadata
        metadata = '_metadata' if underscore is True else 'metadata'

        # Read meta
        traces = h5py.File(data_path, 'r')
        starttime = np.array(traces[metadata]['t0_UNIX_timestamp'])
        sampling_rate = np.array(traces[metadata]['fe'])
        station_codes = [key for key in traces[name].keys()]

        # Sizes
        n_times = len(traces[name][station_codes[0]]['Z'])
        n_stations = len(station_codes)

        # Times
        duration_seconds = (n_times / sampling_rate)
        times = np.linspace(0, duration_seconds, n_times, endpoint=False)

        # Header
        stats = obspy.core.trace.Stats()
        stats.sampling_rate = sampling_rate

        # Start time
        stats.starttime = obspy.UTCDateTime(datetime.fromtimestamp(starttime))
        if force_start is not None:
            stats.starttime = obspy.UTCDateTime(force_start)

        # Collect data into data np.array
        waitbar = ap.logtable.waitbar('Read data')
        for station, station_code in enumerate(station_codes):
            waitbar.progress((station + 1) / n_stations)
            data = traces[name][station_code]['Z'][:]
            stats.npts = len(data)
            stats.station = station_code.split('.')[0]
            self += obspy.core.trace.Trace(data=data, header=stats)

    def set_data(self, data_matrix, starttime, sampling_rate):
        """
        Set the data from any external set of traces.
        """

        n_traces, n_times = data_matrix.shape

        # Header
        stats = obspy.core.trace.Stats()
        stats.sampling_rate = sampling_rate
        stats.starttime = obspy.UTCDateTime(starttime)
        stats.npts = n_times

        # Assign
        waitbar = ap.logtable.waitbar('Read data')
        for trace_id, trace in enumerate(data_matrix):
            waitbar.progress((trace_id + 1) / n_traces)
            self += obspy.core.trace.Trace(data=trace, header=stats)

    def cut(self, starttime=None, endtime=None, pad=True, fill_value=0):
        """
        A wrapper for not defining the UTCDateTime in the main file.

        Keyword arguments:
        ------------------

        :starttime (str): the starting time.

        :endtime (str): the starting time.

        :pad (bool): whether the data has to be padded if the starting and
        ending times are outside the time limits of the seismic traces

        :fill_value (int, float or str): specify the values to use in order ot
        fill gaps, or pad the data if PAD kwarg is set to True.
        """

        # Convert date strings to obspy UTCDateTime
        starttime = obspy.UTCDateTime(starttime)
        endtime = obspy.UTCDateTime(endtime)

        # Trim with
        self.trim(starttime=starttime, endtime=endtime, pad=pad,
                  fill_value=fill_value)

        # Get the new time vector
        self.times = self.get_times()

    def homogenize(self, sampling_rate=20.0, method='linear',
                   start='2010-01-01', npts=24 * 3600 * 20):
        """
        Same prototype than homogenize but allows for defining the date in str
        format (instead of UTCDateTime).
        Same idea than with the cut method.
        """
        start = obspy.UTCDateTime(start)
        self.interpolate(sampling_rate, method, start, npts)

    def binarize(self, epsilon=1e-10):
        """
        Trace binarization in the temporal domain.
        """

        # Waitbar initialization

        waitbar = ap.logtable.waitbar('Binarize')
        n_traces = len(self)

        # Binarize
        for index, trace in enumerate(self):
            waitbar.progress((index + 1) / n_traces)
            trace.data = trace.data / (np.abs(trace.data) + epsilon)

    def stationarize(self, window=11, order=1, epsilon=1e-10):
        """
        Trace stationarization with smoothing time enveloppe.
        """

        # Waitbar initialization

        waitbar = ap.logtable.waitbar('Stationarize')
        n_traces = len(self)

        # Binarize
        for index, trace in enumerate(self):
            waitbar.progress((index + 1) / n_traces)
            smooth = ap.maths.savitzky_golay(np.abs(trace.data), window, order)
            trace.data = trace.data / (smooth + epsilon)

    def whiten(self, segment_duration_sec, method='onebit', smooth=11):
        """
        Spectral one-bit normalization of any obspy stream that may
        contain several traces.
        """

        # Define method
        if method == 'onebit':
            whiten_method = ap.maths.phase
        elif method == 'smooth':
            whiten_method = ap.maths.detrend_spectrum

        # Initialize for waitbar
        waitbar = ap.logtable.waitbar('Whiten')
        n_traces = len(self)
        fft_size = int(segment_duration_sec * self[0].stats.sampling_rate)
        duration = self[0].times()[-1]

        # Whiten
        for index, trace in enumerate(self):
            waitbar.progress((index + 1) / n_traces)
            data = trace.data
            _, _, data_fft = stft(data, nperseg=fft_size)
            data_fft = whiten_method(data_fft, smooth=smooth)
            _, data = istft(data_fft, nperseg=fft_size)
            trace.data = data

        # Trim
        self.cut(pad=True, fill_value=0, starttime=self[0].stats.starttime,
                 endtime=self[0].stats.starttime + duration)

    def stft(self, segment_duration_sec, bandwidth=None, **kwargs):
        """
        Obain array spectra (short window Fourier transform) from complex
        spectrogram function (mlab).
        """

        # Short-time Fourier transform arguments
        kwargs.setdefault('fs', self[0].stats.sampling_rate)
        kwargs.setdefault('nperseg', int(segment_duration_sec * kwargs['fs']))
        kwargs.setdefault('noverlap', kwargs['nperseg'] // 2)
        kwargs.setdefault('nfft', int(2**np.ceil(np.log2(kwargs['nperseg']))))

        # Other default STFT keyword arguments
        kwargs.setdefault('window', 'hanning')
        kwargs.setdefault('return_onesided', True)
        kwargs.setdefault('boundary', None)

        # Calculate spectra of each trace
        spectra = list()
        waitbar = ap.logtable.waitbar('Spectra')
        n_traces = len(self)

        for trace_id, trace in enumerate(self):

            frequencies, times, spectrum = stft(trace.data, **kwargs)
            spectra.append(spectrum)
            waitbar.progress((trace_id + 1) / n_traces)

        # Reduces a list of spectra to an array of shape (n_stations,
        # n_frequencies, n_times)
        spectra = np.array(spectra)

        # Extract spectra within bandwidth
        if bandwidth is not None:
            in_band = (frequencies >= 0.9 * bandwidth[0])
            in_band = in_band & (frequencies <= bandwidth[1])
            frequencies = frequencies[in_band]
            spectra = spectra[:, in_band, :]

        # Calculate spectral times
        times -= times[0]  # remove center times
        times /= 24 * 3600  # convert into fraction of day
        times += mdt.date2num(self[0].stats.starttime.datetime)  # + absolute

        # Set attributes
        self.spectra = spectra
        self.frequencies = frequencies
        self.spectral_times = times

        return self.spectra, self.frequencies, self.spectral_times

    def bartlett(self, n_average=10):
        """Averages the spectra amplitude over a set of AVERAGE windows."""

        # Reshape
        n_traces, n_frequencies, n_times = self.spectra.shape
        n_times = n_times // n_average
        self.spectra = self.spectra[..., :n_times * n_average]
        self.spectral_times = self.spectral_times[::n_average]
        self.spectra = self.spectra.reshape(
            n_traces, n_frequencies, n_times, n_average)
        self.spectra = np.abs(self.spectra.mean(axis=-1))

    def show(self, code=None, ax=None, figure_file_name=None, scale=10,
             **kwargs):

        # Default parameters
        times = self.get_times()
        self.sort()
        kwargs.setdefault('rasterized', True)
        kwargs.setdefault('lw', 0.2)
        kwargs.setdefault('c', 'C0')

        # Canvas
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(8, 6))
        else:
            fig = ax.figure

        # Show all traces if code is None
        if code is None:

            # Plot traces
            for index, trace in enumerate(self):
                trace.data[np.isnan(trace.data)] = 0.0
                trace.data = trace.data / (scale * trace.data.std() + 1e-4)
                ax.plot(times, trace.data + index + 1, **kwargs)

            # Cosmetics
            ax.set_yticks(range(len(self) + 2))
            self.stations = [stream.stats.station for stream in self]
            ax.set_yticklabels([' '] + self.stations + [' '])
            ax.set_ylim([0, len(self) + 1])
            ax.set_ylabel('Station code')

        else:

            # Get trace index
            stations = [stream.stats.station for stream in self]
            index = stations.index(code)

            # Show trace at given index
            trace = self[index]
            trace.data[np.isnan(trace.data)] = 0.0
            trace.data = trace.data / (scale * trace.data.std() + 1e-4)
            ax.plot(times, trace.data, **kwargs)
            ax.set_ylim([-1, 1])

        # Cosmetics
        time_step = times[1] - times[0]
        ax.set_xlim(times[0], times[-1] + time_step)

        # Save
        if figure_file_name is not None:
            fig.savefig(figure_file_name, dpi=300, bbox_inches='tight')
        else:
            return fig, ax

    def spectrogram(self, code=None, ax=None, cax=None, figure_file_name=None,
                    **kwargs):

        # Generate canvas if no axes are given
        if ax is None:
            gs = dict(width_ratios=[50, 1])
            fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw=gs)
            ax = ax.ravel()
        else:
            fig = ax.figure

        # Get index from code
        stations = [stream.stats.station for stream in self]
        station_index = stations.index(code)

        # Image
        kwargs.setdefault('rasterized', True)
        kwargs.setdefault('cmap', 'RdYlBu_r')
        spectrum = np.abs(self.spectra[station_index, :, :])
        spectrum /= np.max(spectrum)
        spectrum = np.log10(spectrum)
        times = self.spectral_times
        extended_times = np.hstack((times, times[-1] + (times[1] - times[0])))
        img = ax[0].pcolormesh(extended_times, self.frequencies, spectrum,
                            **kwargs)

        # Colorbar
        if cax is not None:
            plt.colorbar(img, cax=cax)
            cax.set_ylabel('Spectral amplitude (dBA)')
        else:
            plt.colorbar(img, cax=ax[1])
            ax[1].set_ylabel('Spectral amplitude (dBA)')

        # Save
        if figure_file_name is not None:
            fig.savefig(figure_file_name, dpi=300, bbox_inches='tight')
        else:
            return fig, ax
