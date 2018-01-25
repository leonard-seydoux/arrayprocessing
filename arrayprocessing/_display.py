#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A bunch of widely used display methods.

import numpy as np
import matplotlib.pyplot as plt




def beam(fig, ax, slowness, BEAM):
    """
    Displays the beamforming in the slowness space.
    Automatic cosmetics.
    """

    # Pcolomesh
    slowness = np.linspace(slowness[0], slowness[-1], len(slowness) + 1)
    Sx, Sy = np.meshgrid(slowness, slowness)
    gridsize = len(slowness)
    bsty = dict(cmap='magma', rasterized=True)
    img = ax.pcolormesh(Sx, Sy, BEAM, vmin=0., vmax=1., **bsty)

    # Cross (black)
    cross_style = dict(ls='-', lw=.2, color='k')
    ax.plot(slowness, np.zeros(gridsize), **cross_style)
    ax.plot(np.zeros(gridsize), slowness, **cross_style)
    ax.plot(slowness, slowness, **cross_style)
    ax.plot(slowness, -slowness, **cross_style)
    p = np.linspace(0, 2 * np.pi, 100)
    circle = (max(slowness) / 2.0 * np.cos(p), max(slowness) / 2.0 * np.sin(p))
    ax.plot(*circle, **cross_style)

    # Cross (white)
    cross_style = dict(ls='--', lw=.21, color='w', dashes=[1, 1])
    ax.plot(slowness, np.zeros(gridsize), **cross_style)
    ax.plot(np.zeros(gridsize), slowness, **cross_style)
    ax.plot(slowness, slowness, **cross_style)
    ax.plot(slowness, -slowness, **cross_style)
    p = np.linspace(0, 2 * np.pi, 100)
    circle = (max(slowness) / 2.0 * np.cos(p), max(slowness) / 2.0 * np.sin(p))
    ax.plot(*circle, **cross_style)

    # Axis and aspect
    ax.set_xlim([np.amin(Sx), np.amax(Sx)])
    ax.set_ylim([np.amin(Sy), np.amax(Sy)])
    slowmax = max(abs(slowness))
    ax.set_xticks([-slowmax, -slowmax / 2.0, 0, slowmax / 2.0, slowmax])
    ax.set_yticks([-slowmax, -slowmax / 2.0, 0, slowmax / 2.0, slowmax])
    ax.set_xlabel('Slowness (s/km)')
    ax.set_ylabel('Slowness (s/km)')
    ax.set_aspect('equal')

    return img
