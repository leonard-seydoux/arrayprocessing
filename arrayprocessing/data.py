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
from matplotlib import dates as md
from scipy.signal import stft, istft
from statsmodels import robust


def read(*args):
    """ (Top-level). Read the data files specified in the datapath with
        arrayprocessing.Stream.read.

    This method uses the obspy's read method itself. A check for the
    homogeneity of the seismic traces (same number of samples) is done a
    the end. If the traces are not of same size, an warning message
    shows up.

    No homogeneity check is returned by the function.

    Arguments:
    ----------
        data_path (str or list): path to the data. Path to a single file
            (str), to several files using UNIX regexp (str), or a list of
            files (list). See obspy.read method for more details.
    """

    data = ap.Stream()
    data.read(*args)
    return data


def h5read(*args, **kwargs):
    """
    Top-level read function, returns Stream object.
    """

    data = ap.Stream()
    data.h5read(*args, **kwargs)
    return data


def matread(*args, **kwargs):
    """
    Top-level read function, returns Stream object.
    """

    data = ap.Stream()
    data.matread(*args, **kwargs)
    return data


class Stream(obspy.core.stream.Stream):

    def __init__(self):
        """
        Initialize the class with inherited.
        """
        super(Stream, self).__init__()

    def get_times(self):
        """ Extract the times from the first stream object and convert it to
            a matplotlib datenum vector.
        """

        times = self[0].times()
        times /= 24 * 3600
        times -= times[0]
        start = self[0].stats.starttime.datetime
        times += md.date2num(start)

        return times

    def read(self, data_path):
        """ Read the data files specified in the datapath with obspy.

        This method uses the obspy's read method itself. A check for the
        homogeneity of the seismic traces (same number of samples) is done a
        the end. If the traces are not of same size, an warning message
        shows up.

        Arguments:
        ----------
            data_path (str or list): path to the data. Path to a single file
                (str), to several files using UNIX regexp (str), or a list of
                files (list). See obspy.read method for more details.

        Return:
        -------
            homogeneity (bool): True if the traces are all of the same size.

        """

        # Waitbar in both cases.
        waitbar = ap.logtable.waitbar('Read seismograms')

        # If data_path is a str, then only a single trace or a bunch of
        # seismograms with regexp
        if isinstance(data_path, str):
            self += obspy.read(data_path)
            waitbar.progress(1)

        # If data_path is a list, read each fils in the list.
        elif isinstance(data_path, list):
            for index, path in enumerate(data_path):
                self += obspy.read(path)
                waitbar.progress((index + 1) / len(data_path))

    def h5read(self, data_path, name='PZ', underscore=True,
               force_start=None, stations=None, channel='Z'):
        """
        Read the data files specified in the datapath.

        Arguments
        ---------
        :datapath (str or list): datapath with a single data file or with
        UNIX regexp, or a list of files.

        Keyword arguments
        -----------------

        :sort (bool): whether or not the different traces are sorted in
        alphabetic order with respect to the station codes.
        """

        # Underscored metadata
        metadata = '_metadata' if underscore is True else 'metadata'

        # Read meta
        traces = h5py.File(data_path, 'r')
        starttime = np.array(traces[metadata]['t0_UNIX_timestamp'])
        sampling_rate = np.array(traces[metadata]['fe'])
        if stations is None:
            station_codes = [k for k in traces[name].keys()]
        else:
            station_codes = [k for k in traces[name].keys() if k in stations]

        # Sizes
        n_stations = len(station_codes)

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
            data = traces[name][station_code][channel][:]
            stats.npts = len(data)
            stats.station = station_code.split('.')[0]
            self += obspy.core.trace.Trace(data=data, header=stats)

    def matread(self, data_path, data_name='data', starttime=0,
                sampling_rate=25.0, decimate=1):
        """
        Read the data files specified in the datapath.

        Arguments
        ---------
        :datapath (str or list): datapath with a single data file or with
        UNIX regexp, or a list of files.

        Keyword arguments
        -----------------

        :sort (bool): whether or not the different traces are sorted in
        alphabetic order with respect to the station codes.
        """

        # Read meta
        traces = np.array(h5py.File(data_path, 'r')[data_name])
        n_stations, n_times = traces.shape

        # Header
        stats = obspy.core.trace.Stats()
        stats.sampling_rate = sampling_rate
        stats.npts = n_times

        # Start time
        stats.starttime = obspy.UTCDateTime(starttime)

        # Collect data into data np.array
        waitbar = ap.logtable.waitbar('Read data')
        for station in range(0, n_stations, decimate):
            waitbar.progress((station + 1) / n_stations)
            data = traces[station, :]
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
            self.data = self.trace

    def cut(self, starttime=None, endtime=None, pad=True, fill_value=0):
        """A wrapper to the trim function with string dates or datetimes.

        Arguments:
        ----------
            starttime (str or datetime, optional): the starting time. Default
                to None when the data is not trimed.

            endtime (str or datetime, optional): the ending time. Default
                to None when the data is not trimed.

            pad (bool, optional): whether the data has to be padded if the
                starting and ending times are outside the time limits of the
                times.

            fill_value (int, float or str): specify the values to use in order
                to fill gaps, or pad the data if the pad kwarg is set to True.
        """

        # Convert date strings to obspy UTCDateTime
        starttime = obspy.UTCDateTime(starttime)
        endtime = obspy.UTCDateTime(endtime)

        # Trim
        self.trim(starttime=starttime, endtime=endtime, pad=pad,
                  fill_value=fill_value)

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

    def stationarize(self, length=11, order=1, epsilon=1e-10):
        """ Trace stationarization with smoothing time enveloppe.

        Args
        ----
            length (int): length of the smoothing window.

        """

        # Waitbar initialization

        waitbar = ap.logtable.waitbar('Stationarize')
        n_traces = len(self)

        # Binarize
        for index, trace in enumerate(self):
            waitbar.progress((index + 1) / n_traces)
            smooth = ap.maths.savitzky_golay(np.abs(trace.data), length, order)
            trace.data = trace.data / (smooth + epsilon)

    def demad(self):
        """ Normalize traces by Mean Absolute Deviation.
        """

        # Waitbar initialization
        waitbar = ap.logtable.waitbar('Remove MAD')
        n_traces = len(self)

        # Binarize
        for index, trace in enumerate(self):
            waitbar.progress((index + 1) / n_traces)
            mad = robust.mad(trace.data)
            if mad > 0:
                trace.data /= mad
            else:
                trace.data /= (mad + 1e-5)

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

    def standardize(self, segment_duration_sec):
        """
        Spectral standardization.
        """

        # Initialize for waitbar
        waitbar = ap.logtable.waitbar('Standardize')
        n_traces = len(self)
        fft_size = int(segment_duration_sec * self[0].stats.sampling_rate)
        duration = self[0].times()[-1]

        # Whiten
        for index, trace in enumerate(self):
            waitbar.progress((index + 1) / n_traces)
            data = trace.data
            _, _, datafft = stft(data, nperseg=fft_size)
            var = np.var(datafft.real, axis=-1) + np.var(datafft.imag, axis=-1)
            datafft /= var[:, None] ** (1 / 2)
            _, data = istft(datafft, nperseg=fft_size)
            trace.data = data

        # Trim
        self.cut(pad=True, fill_value=0, starttime=self[0].stats.starttime,
                 endtime=self[0].stats.starttime + duration)

    def show(self, ax=None, scale=0.5, index=0, path_figure=None, **kwargs):
        """ Plot all seismic traces.

        The date axis is automatically defined with matplotlib.dates.

        Arguments:
        ----------

            ax (matplotlib.pyplot.Axes, optional) the axes for the traces.
                Default to None, and the axes are created.

            scale (float): scaling factor for trace amplitude.

            path_figure (str, optional): if set, then save the figure to the
                path. Default to None, then return fig, ax and cax.

            **kwargs (dict): other keyword arguments passed to
                matplotlib.pyplot.pcolormesh.

        Return:
        ------

            If the path_figure kwargs is set to None (default), the following
            objects are returned:

            fig (matplotlib.pyplot.Figure) the figure instance.
            ax (matplotlib.pyplot.Axes) axes of the spectrogram.

        """

        # Default parameters
        times = self.get_times()
        kwargs.setdefault('rasterized', True)

        # Canvas
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(7, 6))
        else:
            fig = ax.figure

        # Plot traces
        self.sort()
        traces = np.array([s.data for s in self])
        traces = traces / traces.max()
        if robust.mad(traces).max() > 0:
            traces /= 1.2 * robust.mad(traces).max()

        for index, trace in enumerate(traces):
            trace[np.isnan(trace)] = 0.0
            trace *= scale
            ax.plot(times, trace + index + 1, **kwargs)

        # Station codes
        self.stations = [stream.stats.station for stream in self]
        ax.set_yticks(range(len(self) + 2))
        ax.set_yticklabels([' '] + self.stations + [' '])
        ax.set_ylim([0, len(self) + 1])
        ax.set_ylabel('Station code')

        # Time axis
        ax.set_xlim(times[[0, -1]])
        xticks = md.AutoDateLocator()
        ax.xaxis.set_major_locator(xticks)
        ax.xaxis.set_major_formatter(md.AutoDateFormatter(xticks))

        # Save
        if path_figure is not None:
            fig.savefig(path_figure, dpi=300, bbox_inches='tight')
        else:
            return fig, ax

    def stft(self, segment_duration_sec, bandwidth=None, overlap=0.5,
             **kwargs):
        """ Perform short-time Fourier transform onto individual traces.

        Arguments:
        ----------

            segment_duration_sec (float): duration of the time segments
                analysed with Fourier transform

            bandwidth (tuple or list, optional): frequency limits onto which
                the spectra could be truncated. Default to None, i.e. all
                frequencies are kept.

            overlap (float, optional): overlap between time segments. Default
                to 0.5, meaning that segments are overlapping at 50%.

            **kwargs (dict): other kwargs are passed to the scipy.signal.stft
                function. Some default kwargs are defined:
                - fs (float): sampling_rate extracted from self.
                - nperseg (int): segments length int(segement_duration * fs)
                - noverlap (int): overlap length int(segment_length * overlap)
                - nfft (int): length of fft output (nextpow2(segment_lenght))
                - window (str): window function (default 'hann')
                - boundary (:obj: `str`): pad signal at boundaries

        Return:
        -------

            spectra (ndarray): the Fourier spectra of shape
                (n_stations, n_frequencies, n_times). The frequency axis is
                truncated if the bandwidth argument is not None.

            frequencies (array): the frequency axis (n_frequencies). The
                frequency axis is truncated if the bandwidth argument is not
                None.

            times (array): the starting time of each window.

        """

        # Short-time Fourier transform arguments
        kwargs.setdefault('fs', self[0].stats.sampling_rate)
        kwargs.setdefault('nperseg', int(segment_duration_sec * kwargs['fs']))
        kwargs.setdefault('noverlap', int(kwargs['nperseg'] * overlap))
        kwargs.setdefault('nfft', int(2**np.ceil(np.log2(kwargs['nperseg']))))
        kwargs.setdefault('window', 'hann')
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

        # Reduces a list of spectra to an array
        spectra = np.array(spectra)
        self.n_frequencies_extended = len(frequencies)

        # Extract spectra within bandwidth
        df = frequencies[1] - frequencies[0]
        if bandwidth is not None:
            in_band = (frequencies >= bandwidth[0] - 2 * df)
            in_band = in_band & (frequencies <= bandwidth[1] + 2 * df)
            frequencies = frequencies[in_band]
            spectra = spectra[:, in_band, :]

        # Calculate spectral times
        times -= times[0]  # remove center times
        times /= 24 * 3600  # convert into fraction of day
        times += md.date2num(self[0].stats.starttime.datetime)  # + absolute

        # Set attributes
        self.spectra = spectra
        self.frequencies = frequencies
        self.spectral_times = times

        return self.spectra, self.frequencies, self.spectral_times

    def spectrogram(self, code=None, ax=None, cax=None, path_figure=None,
                    **kwargs):
        """ Pcolormesh the spectrogram of a single seismic trace.

        The spectrogram (modulus of the short-time Fourier transform) is
        extracted from the complex spectrogram previously calculated from stft.

        The spectrogram is represented in log-scale amplitude normalized by
        the maximal amplitude (dB re max).

        The date axis is automatically defined with matplotlib.dates.

        Arguments:
        ----------

            code (str, optional): the station code from which the spectrogram
                is desired. Default to None. If code is None, then the first
                spectrogram is plotted.

            ax (matplotlib.pyplot.Axes, optional) the axes for the spectrogram.
                Default to None, and some axes are created.

            cax (matplotlib.pyplot.Axes, optional) the axes for the colorbar.
                Default to None, and the axes are created. These axes should be
                given if ax is not None.

            path_figure (str, optional): if set, then save the figure to the
                path. Default to None, then return fig, ax and cax.

            **kwargs (dict): other keyword arguments passed to
                matplotlib.pyplot.pcolormesh.

        Return:
        ------

            If the path_figure kwargs is set to None (default), the following
            objects are returned:

            fig (matplotlib.pyplot.Figure) the figure instance.
            ax (matplotlib.pyplot.Axes) axes of the spectrogram.
            cax (matplotlib.pyplot.Axes) axes of the colorbar.

        """

        # Check if the spectra have been calculated.
        if not hasattr(self, 'spectra'):
            errormsg = 'You should perform stft() first. Exiting.'
            ap.logtable.row('ERROR [spectrogram]', errormsg)
            exit()

        # Create canvas if no axes are given
        if ax is None:
            gs = dict(width_ratios=[50, 1])
            fig, (ax, cax) = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw=gs)

        # Else extract figure parent from ax
        else:
            fig = ax.figure

        # Get station index from code
        if code is None:
            spectrum = self.spectra[7, :, :]
        else:
            stations = [stream.stats.station for stream in self]
            station_index = [i for i, s in enumerate(stations) if code in s]
            spectrum = self.spectra[station_index, :, :]
        spectrum = np.squeeze(spectrum)

        # Spectrogram
        spectrum = np.log10(np.abs(spectrum) / np.abs(spectrum).max())

        # Times
        times = self.spectral_times
        times = np.hstack((times, self.get_times()[-1]))

        # Image
        kwargs.setdefault('rasterized', True)
        img = ax.pcolormesh(times, self.frequencies, spectrum, **kwargs)

        # Colorbar
        plt.colorbar(img, cax=cax)
        cax.set_ylabel('Spectral amplitude (dB re max)')

        # Date ticks
        ax.set_xlim(self.get_times()[[0, -1]])
        xticks = md.AutoDateLocator()
        ax.xaxis.set_major_locator(xticks)
        ax.xaxis.set_major_formatter(md.AutoDateFormatter(xticks))

        # Frequencies
        ax.set_yscale('log')
        ax.set_ylabel('Frequencies (Hz)')
        ax.set_ylim(self.frequencies[[0, -1]])

        # Save or return
        if path_figure is not None:
            fig.savefig(path_figure, dpi=300, bbox_inches='tight')
        else:
            return fig, ax, cax
