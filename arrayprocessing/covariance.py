#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Calculation of the covariance matrix from raw seismograms.

import arrayprocessing as ap
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

from numpy.linalg import eigvalsh, eig, svd
from copy import deepcopy
from arrayprocessing.maths import xcov, xcov_std


class CovarianceMatrix(np.ndarray):

    """
    Useful to define an array of covariance matrices.
    """

    def get_eigenvalues(self, normalization=None):
        """
        Returns the spectrum of the covariance matrix with the desired
        normalization
        """
        eigenvalues = eigvalsh(self)
        eigenvalues = eigenvalues[::-1]

        if normalization == 'max':
            return eigenvalues / eigenvalues.max()

        elif normalization == 'sum':
            return eigenvalues / eigenvalues.sum()

        elif normalization is None:
            return eigenvalues

    def get_spectralwidth(self):
        """
        Returns the width of distribution of the eigenvalue spectrum of the
        covariance matrix.
        """

        eigenvalues = self.get_eigenvalues(normalization='sum')
        rank_max = len(eigenvalues)

        # In case of 0-valued data
        if eigenvalues.sum() == 0:
            return 0

        else:
            spectral_width = np.sum(eigenvalues * np.arange(rank_max))
            return spectral_width

    def get_entropy(self, epsilon=1e-10):
        """
        Returns the entropy from the distribution of the eigenvalue spectrum
        of the covariance matrix.
        """

        eigenvalues = self.get_eigenvalues(normalization='sum')
        rank_max = len(eigenvalues)

        # In case of 0-valued data
        if eigenvalues.sum() == 0:
            return 0

        else:
            log_eigenvalues = np.log(eigenvalues + epsilon)
            entropy = - np.sum(eigenvalues * log_eigenvalues)
            return entropy

    def get_eigenvector(self, rank=0):
        """
        Returns the eigenvector of given RANK from the covariance matrix.
        """

        _, eigenvectors = eig(self)
        return eigenvectors[:, rank]

    def equalize(self, rank):
        """
        Returns the eigenvector of given RANK from the covariance matrix.
        """
        U, D, V = svd(self)
        D = np.zeros_like(D, dtype=np.float32)
        D[:rank] = 1.0
        D = np.diag(D)
        covariance = U.dot(D).dot(V)
        return covariance.view(ap.CovarianceMatrix)

    def get_triu(self, k=0):
        """
        Frequency on the last dimension because of common scipy operations.
        """
        trii, trij = np.triu_indices(self.shape[1], k=k)
        triu = np.array([self[:, i, j] for i, j in zip(trii, trij)])
        return triu.view(CovarianceMatrix)


class RealCovariance():

    def __init__(self, stream=None, pickle_file=None, cpp=None, start=None):
        """
        Reads the data from a list of path to seismograms files, or directly
        extracts a previoulsy computed coherence object from pickle.
        """

        # Get coherence saved object if pickle_file is not None. In that case,
        # pickle_file is a filename to load.

        if pickle_file is not None:
            with open(pickle_file, 'rb') as input_file:
                self.__dict__.update(pickle.load(input_file))

        # If pickle_file is None, get the stream data.

        elif cpp is not None:

            times, frequencies, coherence = ap.read_spectral_width(
                files=cpp, date_start=start)
            self.times = times
            self.frequencies = frequencies
            self.coherence = coherence.T

        else:

            self.stream = stream
            self.sampling_rate = self.stream[0].stats.sampling_rate
            self.start = self.stream[0].stats.starttime

    def __add__(self, other):
        """
        Possibility to stack several computed coherence objects over time.
        """

        covariance_times = (self.times, other.times)
        self.times = np.concatenate(covariance_times)[:-1]

        coherence = (self.coherence.T, other.coherence.T)
        self.coherence = np.hstack(coherence).T

        if self.eigenvectors is not None:
            self.eigenvectors = np.concatenate((
                self.eigenvectors, other.eigenvectors), axis=1)

        return self

    def calculate(self, average=20, overlap=None, standardize=False):
        """
        Calculation of the array covariance matrix from the array data vectors
        stored in the spectra matrix (should be n_traces x n_windows x n_freq).
        """

        # Parametrization
        overlap = average // 2 if overlap is None else overlap
        ratio = average // overlap
        self.stream.spectra = self.stream.spectra.transpose([0, 2, 1])
        n_traces, n_windows, n_frequencies = self.stream.spectra.shape
        n_average = ratio * n_windows // average - (ratio - 1)
        n_times = n_average + 1

        # Initialization
        covariance_shape = (n_average, n_traces, n_traces, n_frequencies)
        ci = complex
        self.covariance = CovarianceMatrix(shape=covariance_shape, dtype=ci)
        waitbar = ap.logtable.waitbar('Covariance')

        xc = xcov if standardize is False else xcov_std

        # Compute
        for wid in range(n_average):
            self.covariance[wid] = xc(
                wid, self.stream.spectra, overlap, average)
            waitbar.progress(wid / (n_average - 1))

        # Get times
        times = self.stream.spectral_times[::average // 2]
        times = times[:n_times]
        dtimes = times[1] - times[0]
        times = np.insert(times, len(times), times[-1] + dtimes)
        self.times = times
        self.frequencies = self.stream.frequencies

    def calculate_spectralwidth(self):
        """
        Coherence extracted from a set of covariance matrices (n_times, n_freq)
        """
        # Initialization
        n_windows, _, _, n_frequencies = self.covariance.shape
        coherence = np.zeros((n_windows, n_frequencies))

        # Computation
        waitbar = ap.logtable.waitbar('Coherence')
        for wid in range(n_windows):
            waitbar.progress(wid / (n_windows - 1))
            for fid in range(n_frequencies):
                sw = self.covariance[wid, :, :, fid].get_spectralwidth()
                coherence[wid, fid] = sw
        self.coherence = np.vstack([coherence, coherence[-1, :]])

    def calculate_eigenvectors(self, order=0, frequency=None):
        """
        Extracts eigenvectors of given order within bandwith in a
        (n_stations, n_times, n_frequencies) matrix
        """

        # Initialization
        n_windows, n_traces, _, n_frequencies = self.covariance.shape

        if frequency is not None:
            eigenvector_shape = (n_traces, n_windows)
        else:
            eigenvector_shape = (n_traces, n_windows, n_frequencies)

        self.eigenvectors = np.zeros(eigenvector_shape, dtype='complex')

        # Computation
        waitbar = ap.logtable.waitbar('Eigenwavefield')
        for wid in range(n_windows):
            waitbar.progress(wid / (n_windows - 1))

            if frequency is not None:
                fid = np.abs(self.frequencies - frequency).argmin()
                self.eigenvectors[:, wid] = \
                    self.covariance[wid, :, :, fid].get_eigenvector(rank=order)
            else:
                for fid in range(n_frequencies):
                    self.eigenvectors[:, wid, fid] = \
                        self.covariance[wid, :, :, fid].get_eigenvector(order)

        self.eigenvectors = np.concatenate(
            (self.eigenvectors, self.eigenvectors[:, None, -1]), axis=1)

    def save(self, filename='coherence.pck', coherence=True, covariance=False,
             spectra=False, stream=False, eigenvectors=False):
        """
        Save the covariance object into pickle file.
        """

        self.covariance = self.covariance if covariance is True else None
        self.eigenvectors = self.eigenvectors if eigenvectors is True else None
        self.coherence = self.coherence if coherence is True else None
        self.stream = self.stream if stream is True else None
        self.spectra = self.spectra if spectra is True else None

        ap.logtable.row('Save as', filename)
        pickle_protocol = 2
        with open(filename, 'wb') as outfile:
            pickle.dump(self.__dict__, outfile, pickle_protocol)

    def get_coherence(self):
        return self.times, self.frequencies, self.coherence.T

    def get_covariance(self):
        return self.times, self.frequencies, self.covariance

    def get_eigenvectors(self):
        return self.times, self.frequencies, self.eigenvectors

    def show_coherence(self, ax=None, cax=None, path_figure=None, **kwargs):

        if ax is None:

            # Create axis
            gs = dict(width_ratios=[50, 1], wspace=0.1)
            fig, (ax, cax) = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw=gs)

        # Default options
        kwargs.setdefault('cmap', 'RdYlBu')
        kwargs.setdefault('rasterized', True)

        # Image
        img = ax.pcolormesh(self.times, self.frequencies, self.coherence.T,
                            **kwargs)
        ax.set_yscale('log')

        # Colorbar
        plt.colorbar(img, cax=cax)
        cax.set_ylabel('Spectral width')

        # Save
        if path_figure is not None:
            fig.savefig(path_figure, dpi=300, bbox_inches='tight')
        else:
            return fig, (ax, cax)
