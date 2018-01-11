#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Calculation of the covariance matrix from raw seismograms.

import arrayprocessing as ap
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

from scipy.special import jv
from copy import deepcopy
from arrayprocessing.maths import xouter


def planewave(antenna, frequency, slowness, azimuth):

    x, y = antenna.get_xy()
    wavenumber = 2 * np.pi * frequency * slowness
    scalar_product = np.sin(azimuth) * x + np.cos(azimuth) * y
    wavefield = np.exp(-1j * wavenumber * scalar_product)
    covariance = xouter(planewave)
    return covariance.view(ap.CovarianceMatrix).astype(complex)


def surface_noise(antenna, frequency, slowness):

    x, y = antenna.get_xy()
    distances = antenna.get_distances()
    wavenumber = 2 * np.pi * frequency * slowness
    covariance = jv(0, wavenumber * distances)
    return covariance.view(ap.CovarianceMatrix).astype(complex)


def volume_noise(antenna, frequency, slowness):

    x, y = antenna.get_xy()
    distances = antenna.get_distances()
    wavenumber = 2 * np.pi * frequency * slowness
    covariance = np.sinc(wavenumber * distances)
    return covariance.view(ap.CovarianceMatrix())


def estimated_surface_noise(antenna, frequency, slowness, n_sources=200,
                            n_snapshots=100):
    """ Estimate surface noise with random plane waves"""

    azimuths = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    x, y = antenna.get_xy()
    covariance = np.zeros((antenna.dim, antenna.dim), dtype=complex)

    waitbar = ap.logtable.waitbar('Surface noise estimate')
    for snapshot in range(n_snapshots):

        waitbar.progress((snapshot + 1) / n_snapshots)
        wavenumber = 2 * np.pi * frequency * slowness
        snapshots = np.zeros(antenna.dim, dtype=complex)
        phases = 2 * np.pi * np.random.rand(n_sources)

        for azimuth_id, (azimuth, phase) in enumerate(zip(azimuths, phases)):
            scalar_product = np.sin(azimuth) * x + np.cos(azimuth) * y
            snapshots += np.exp(-1j * wavenumber * scalar_product - 1j * phase)

        snapshots /= n_sources
        covariance += snapshots * snapshots.conj()[:, None]
        covariance /= n_snapshots
    return covariance.view(ap.CovarianceMatrix).astype(complex)


def estimated_volume_noise(antenna, frequency, slowness, n_sources=200,
                           n_snapshots=100):
    """ Estimate volume noise with random plane waves"""

    azimuths = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    x, y = antenna.get_xy()
    covariance = np.zeros((antenna.dim, antenna.dim), dtype=complex)

    waitbar = ap.logtable.waitbar('Volume noise estimate')
    for snapshot in range(n_snapshots):

        waitbar.progress((snapshot + 1) / n_snapshots)
        wavenumber = 2 * np.pi * frequency * slowness
        snapshots = np.zeros(antenna.dim, dtype=complex)
        phases = 2 * np.pi * np.random.rand(n_sources)

        for azimuth_id, (azimuth, phase) in enumerate(zip(azimuths, phases)):
            scalar_product = np.sin(azimuth) * x + np.cos(azimuth) * y
            k = wavenumber * np.random.rand(1)
            snapshots += np.exp(-1j * k * scalar_product - 1j * phase)

        snapshots /= n_sources
        covariance += snapshots * snapshots.conj()[:, None]
        covariance /= n_snapshots
    return covariance.view(ap.CovarianceMatrix).astype(complex)


def cylindrical(antenna, frequency, slowness, coordinate=(0.0, 0.0)):

    x, y = antenna.get_xy()
    r = np.sqrt((x - coordinate[0]) ** 2 + (y - coordinate[1]) ** 2)
    wavenumber = 2 * np.pi * frequency * slowness
    focal = 1 / np.sqrt(r + 1e-6) * np.exp(-1j * wavenumber * r)
    covariance = xouter(focal)
    return covariance.view(ap.CovarianceMatrix).astype(complex)


def cylindrical_wave(antenna, frequency, slowness, coordinate=(0.0, 0.0)):

    x, y = antenna.get_xy()
    r = np.sqrt((x - coordinate[0]) ** 2 + (y - coordinate[1]) ** 2)
    wavenumber = 2 * np.pi * frequency * slowness
    focal = 1 / np.sqrt(r + 1e-6) * np.exp(-1j * wavenumber * r)
    return focal


def estimated_noise(antenna, n_snapshots=100):

    random = np.random.randn
    N = antenna.dim
    covariance = np.zeros((N, N), dtype=complex)

    waitbar = ap.logtable.waitbar('Noise estimate')
    for index in range(n_snapshots):
        waitbar.progress((index + 1) / n_snapshots)
        snapshot = random(N) + 1j * random(N)
        covariance += snapshot * snapshot.conj()[:, None]
    covariance /= n_snapshots
    return covariance.view(ap.CovarianceMatrix).astype(complex)
