#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This package contains useful tools for covariance matrix and correlation
# matrix analsysis of array data.

from arrayprocessing import logtable
from arrayprocessing.data import Stream, read, h5read
from arrayprocessing.reader import read_spectral_width
from arrayprocessing.covariance import CovarianceMatrix, RealCovariance
from arrayprocessing.correlation import CorrelationMatrix, correlation
from arrayprocessing.antenna import Antenna, Map
from arrayprocessing.beam import Beam

from arrayprocessing import synthetic

# Intialization
logtable.line()
