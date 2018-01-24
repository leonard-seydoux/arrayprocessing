#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.fftpack as fft
import pickle

from itertools import product
from scipy.signal import butter, filtfilt, hilbert, tukey, bessel, decimate


# def extend_frequency(f, n_pad):
#     """
#     Extend the inital frequency vector f:0...fmax to
#     f_pad:0...FMAX where FMAX = n_pad*df.
#     """
#     n_initial = len(f)
#     df = f[2] - f[1]
#     f_pad = np.zeros(n_pad)
#     f_pad[:n_initial] = f
#     for i in range(n_initial, n_pad):
#         f_pad[i] = f_pad[i - 1] + df
#     return f_pad

def correlation(covariance_matrix):
    """
    Covariance matrix is a np.ndarray of shape
    n_frequencies, n_stations, n_stations
    """

    correlation = fft.ifft(covariance_matrix, axis=0)
    correlation = np.real(fft.fftshift(correlation, axes=0))
    return correlation.view(CorrelationMatrix)


class CorrelationMatrix(np.ndarray):

    def bandpass(self, bandwidth, sampling_rate=24, triu=False):
        """Filter the signal within bandwidth."""

        nyquist = 0.5 * sampling_rate
        bandwidth = [f / nyquist for f in bandwidth]
        b, a = butter(4, bandwidth, btype='bandpass')

        if triu is False:
            n_sensors = self.shape[1]
            for i, j in product(range(n_sensors), repeat=2):
                self[:, i, j] = filtfilt(b, a, self[:, i, j])
                self[:, i, j] /= np.max(self[:, i, j])
        else:
            for i in range(self.shape[0]):
                self[i] = filtfilt(b, a, self[i])
                self[i] /= np.max(self[i])

    def plot(self, ax, lag, antenna, norm=1, k=0, **kwargs):
        """Sismological view."""

        # Triangular view
        triu_i, triu_j = np.triu_indices(self.shape[1], k=k)
        distances = antenna.get_distances()

        # Plot
        kwargs.setdefault("lw", 0.2)

        if len(self.shape) == 3:
            for i, j in zip(triu_i, triu_j):
                ax.plot(lag, norm * self[:, i, j] + distances[i, j], **kwargs)

        elif len(self.shape) == 2:

            # Triangular view
            trii, trij = np.triu_indices(antenna.dim, k=k)

            if distances is None:
                distances = antenna.get_distances()

            distances = np.array([distances[i, j] for i, j in zip(trii, trij)])
            distance_sort = distances.argsort()
            distances = distances[distance_sort]
            correlations = self[distance_sort, :]

            for i in range(self.shape[0]):
                ax.plot(lag, norm * correlations[i] + distances[i], **kwargs)

    def pcolormesh(self, ax, lag, antenna, k=0, distances=None, **kwargs):
        """Acoustic view."""

        # Triangular view
        trii, trij = np.triu_indices(self.shape[1], k=k)

        if distances is None:
            distances = antenna.get_distances()

        distances = np.array([distances[i, j] for i, j in zip(trii, trij)])
        distance_sort = distances.argsort()
        distances = distances[distance_sort]
        correlations = np.array([self[:, i, j] for i, j in zip(trii, trij)])
        correlations = correlations[distance_sort, ...]

        # Pcolormesh
        cmax = np.abs(correlations).max()
        kwargs.setdefault('vmin', -cmax)
        kwargs.setdefault('vmax', cmax)
        kwargs.setdefault('rasterized', True)
        return ax.pcolormesh(lag, distances, correlations, **kwargs)

    def pcolormesh_triu(self, ax, lag, antenna, k=0, distances=None, **kwargs):
        """Acoustic view."""

        # Triangular view
        trii, trij = np.triu_indices(antenna.dim, k=k)

        if distances is None:
            distances = antenna.get_distances()

        distances = np.array([distances[i, j] for i, j in zip(trii, trij)])
        distance_sort = distances.argsort()
        distances = distances[distance_sort]
        correlations = self[distance_sort, :]

        # Pcolormesh
        cmax = np.abs(correlations).max()
        kwargs.setdefault('vmin', -cmax)
        kwargs.setdefault('vmax', cmax)
        kwargs.setdefault('rasterized', True)
        return ax.pcolormesh(lag, distances, correlations, **kwargs)

    def get_maxima(self):
        """Extract travel times from enveloppe."""

        # Extract triangular indexes
        triu_i, triu_j = np.triu_indices(self.shape[1], k=0)
        n_triu = len(triu_i)
        maxima = np.zeros(n_triu, dtype=int)
        self_hilbert = hilbert(self)

        for i, j in zip(triu_i, triu_j):
            maxima[i] = np.abs(self_hilbert[:, i, j]).argmax()

        return maxima

    def calculate_envelope(self):

        if len(self.shape) == 3:
            return np.abs(hilbert(self, axis=0)).view(CorrelationMatrix)
        else:
            return np.abs(hilbert(self, axis=-1)).view(CorrelationMatrix)

    def get_triu(self, k=0):
        """
        Time on the last dimension because of common scipy operations.
        """
        trii, trij = np.triu_indices(self.shape[1], k=k)
        triu = np.array([self[:, i, j] for i, j in zip(trii, trij)])
        return triu.view(CorrelationMatrix)

    # def plot_from(self, ax, reference=0, **kwargs):
    #     correlationss = self.get_from(reference)
    #     argsort = self._antenna.get_argsort_from(reference)
    #     factor = self._antenna.get_distances()[reference, argsort]
    #     for j in range(self._dim):
    #         ax.plot(self._lag[:-1], factor[j] /
    #                 20.0 + correlations[j], **kwargs)

    # def mask(self, average_slowness, width=0.4):
    #     mask = np.zeros_like(self._data)
    #     expected_times = self._antenna.get_distances() * average_slowness
    #     lag = self.get_lag()
    #     dlag = lag[2] - lag[1]
    #     for i in range(self._antenna.dim):
    #         for j in range(self._antenna.dim):
    #             delta_time = lag - expected_times[i, j]
    #             expected_time_index = np.abs(delta_time).argmin() - 1
    #             mask[i, j, expected_time_index] = 1
    #             dist = self._antenna.get_distances()[i, j]
    #             gauss = tukey(int(1 + width * dist / dlag), 0.2)
    #             mask[i, j, :] = np.convolve(mask[i, j, :], gauss, mode='same')

    #     n_lag = self._data.shape[-1]
    #     mask[:, :, :n_lag // 2] = mask[:, :, n_lag:n_lag // 2:-1]
    #     self._data *= mask
    #     for i, j in product(range(self._dim), repeat=2):
    #         if max(abs(self._data[i, j, :])) != 0:
    #             self._data[i, j, :] /= np.amax(abs(self._data[i, j, :]))

    # def save(self, pickle_filename):
    #     with open(pickle_filename, 'wb') as pickle_file:
    #         pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)
    #     pass
