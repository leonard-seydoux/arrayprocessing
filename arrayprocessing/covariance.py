#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Calculation of the covariance matrix from raw seismograms.

import arrayprocessing as ap
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import dates as md

from numpy.linalg import eigvalsh, eig, svd
from arrayprocessing.maths import xcov


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

        # If try to read from cpp code
        elif cpp is not None:
            times, frequencies, coherence = ap.read_spectral_width(
                files=cpp, date_start=start)
            self.times = times
            self.frequencies = frequencies
            self.coherence = coherence.T

        # Otherwise, get stream
        else:
            self.stream = stream
            self.sampling_rate = self.stream[0].stats.sampling_rate
            self.start = self.stream[0].stats.starttime

    def __add__(self, other, eigenvectors=None):
        """
        Possibility to stack several computed coherence objects over time.
        """

        other_time = np.hstack((other.times, other.time_end))
        covariance_times = (self.times, other_time)
        self.times = np.concatenate(covariance_times)
        self.time_end = other.time_end

        coherence = (self.coherence.T, other.coherence.T)
        self.coherence = np.hstack(coherence).T

        if self.eigenvectors is not None:
            self.eigenvectors = np.concatenate((
                self.eigenvectors, other.eigenvectors), axis=1)

        if self.covariance is not None:
            self.covariance = np.concatenate((
                self.covariance, other.covariance), axis=0).view(
                CovarianceMatrix)

        return self

    def calculate(self, average, overlap=0.5, standardize=False):
        """ Calculate covariance matrix from the stream.spectra.

        Attributes:
        -----------
            covariance (CovarianceMatrix, dtype=complex)

        Arguments:
        ----------
            average (int): number of averaging spectral windows.

            overlap (float): overlaping ratio between consecutive spectral
                windows. Default to 0.5.

            standardize (bool): use of standart deviation normalization.
                Default to False (no normalization applied).

        """

        # Parametrization
        overlap = int(average * overlap)

        # Reshape spectra in order to (n_stations, n_times, n_frequencies)
        self.stream.spectra = self.stream.spectra.transpose([0, 2, 1])
        n_traces, n_windows, n_frequencies = self.stream.spectra.shape

        # Times
        times = self.stream.spectral_times[:-average:overlap]
        n_average = len(times)

        # Initialization
        cov_shape = (n_average, n_traces, n_traces, n_frequencies)
        self.covariance = CovarianceMatrix(shape=cov_shape, dtype=complex)
        waitbar = ap.logtable.waitbar('Covariance')

        # Standardize
        if standardize is True:
            spectra = self.stream.spectra
            for fid in range(n_frequencies):
                for sid in range(n_traces):

                    # Demean
                    spectra[sid, :, fid] -= np.mean(spectra[sid, :, fid])

                    # Devar
                    var = np.var(spectra[sid, :, fid].real) +\
                        np.var(spectra[sid, :, fid].imag)
                    spectra[sid, :, fid] /= var ** (1 / 4)

            self.stream.spectra = spectra

        # Compute
        for t in range(n_average):
            self.covariance[t] = xcov(t, self.stream.spectra, overlap, average)
            waitbar.progress(t / (n_average - 1))

        # Get times
        self.times = times
        self.time_end = self.stream.get_times()[-1]
        self.frequencies = self.stream.frequencies

    def calculate_spectralwidth(self):
        """ Spectral width of the covariance matrices (n_times, n_frequencies).

        Creates an attribute coherence.
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

    def save(self, path='coherence.pkl', coherence=True, covariance=False,
             spectra=False, stream=False, eigenvectors=False):
        """Save the covariance object into pickle file.

        Arguments:
        ----------

            path (str, optional) Path to the pickle file.
                Default to 'coherence.pkl'.

            coherence (bool, optional) Save the coherence. Default to True.
            covariance (bool, optional) Save the covariance. Default to False.
            spectra (bool, optional) Save the spectra. Default to False.
            stream (bool, optional) Save the stream. Default to False.
            eigenvectors (bool, optional) Save the eigenvectors. Default False.


        """

        self.covariance = self.covariance if covariance is True else None
        self.eigenvectors = self.eigenvectors if eigenvectors is True else None
        self.coherence = self.coherence if coherence is True else None
        self.stream = self.stream if stream is True else None
        self.spectra = self.spectra if spectra is True else None

        ap.logtable.row('Save as', path)
        with open(path, 'wb') as outfile:
            pickle.dump(self.__dict__, outfile, pickle.HIGHEST_PROTOCOL)

        pass

    def get_coherence(self):
        return self.times, self.frequencies, self.coherence.T

    def get_covariance(self):
        return self.times, self.frequencies, self.covariance

    def get_eigenvectors(self):
        return self.times, self.frequencies, self.eigenvectors

    def show_coherence(self, ax=None, cax=None, path_figure=None, **kwargs):
        """ Calculate covariance matrix from the stream.spectra.

        Arguments:
        ----------

            ax (matplotlib.pyplot.Axes, optional) the axes for the coherence.
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

        # If axes are not given
        if ax is None:
            gs = dict(width_ratios=[50, 1], wspace=0.1)
            fig, (ax, cax) = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw=gs)
        else:
            fig = ax.figure

        # Default options
        kwargs.setdefault('rasterized', True)
        time = self.times
        time = np.hstack((time, self.time_end))
        img = ax.pcolormesh(time, self.frequencies, self.coherence.T, **kwargs)

        # Time limits
        ax.set_xlim(self.times[0], self.time_end)
        xticks = md.AutoDateLocator()
        ax.xaxis.set_major_locator(xticks)
        ax.xaxis.set_major_formatter(md.AutoDateFormatter(xticks))

        # Frequency limits
        ax.set_ylim(self.frequencies[[0, -1]])
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (Hz)')

        # Colorbar
        plt.colorbar(img, cax=cax)
        cax.set_ylabel('Spectral width')

        # Save
        if path_figure is not None:
            fig.savefig(path_figure, dpi=300, bbox_inches='tight')
        else:
            return fig, ax, cax
