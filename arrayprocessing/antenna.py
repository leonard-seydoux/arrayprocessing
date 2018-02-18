#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Seismic array utilities for antenna processing.

This module contains useful utilities for antenna processing with seismic
arrays, mainly used in others modules of the arrayprocessing package.

Todo:
    * Implement 3D array that account for the station elevation.

Authors:
    L. Seydoux (leonard.seydoux@gmail.com)
    J. Soubestre

Last update:
    Feb. 2018

"""

import copy
import numpy as np
import csv

from itertools import product


def geo2xy(lon, lat, reference=None):
    """ Geographical to cartesian coordinates.

    The calculation is based on the approximate great-circle distance formulae,
    and a spherical Earth with a radius of 40000 km.

    Args
    ----
        lon (float or array): longitudes
        lat (float or array): latitudes
        reference (tuple, optional): reference geographical coordinates for
            great-circle distance calculation. Default is average of longitude
            and latitude (barycenter).

    Return
    ------
        x (same type as `lon`): east-west coordinates in km
        y (same type as `lat`): north-south coordinates in km

    """

    # Definition of the geographical reference point
    if reference is None:
        lon0 = np.mean(lon)
        lat0 = np.mean(lat)
    else:
        lon0, lat0 = reference

    # Application of the formulae
    earth_radius = 40000  # km
    x = (lon0 - lon) * earth_radius * np.cos((lat + lat0) * np.pi / 360) / 360
    y = -(lat - lat0) * earth_radius / 360

    return x, y


class Antenna():
    """ Utilities for seismic array position and simple operations.


    Attributes:
        name (list of str): stations codes
        lon (float or array): longitude of the seismic stations
        lat (float or array): latitude of the seismic stations
        z (float or array): depth of the seismic stations (increasing values
            with depth).
        x, y (float or array): cartesian coordinates of the stations in a given
            reference point (see `reference` kwarg)
        dim (int): number of seismic stations

    """

    def __init__(self, path, geographical_reference=None,
                 csv_filter=None, depth_factor=1.):
        """

        Args:
        -----
            path (str): path to the csv file containing the stations
                coordinates. The csv file must contain the following headers:
                - NAME: the station codes
                - LON: the stations longitudes
                - LAT: the stations latitudes
                - ALT (optional): the station elevation (in km). If none is
                    given, the attribute `z` is set to 0 or an array of 0.
                    The altitude is given in positive km from the sea
                    level. They are translated to depth in the `z` attribute
                    in negative km from the sea level.
                - FILTERS: the other columns are considered to be used for
                    selecting parts of the seismic array (optional, with
                    respect to the `csv_filter` kwarg).
                If any other fields are present in the header, they are
                disregarded.
                The cartesian coordinates x and y are automatically calculated
                with the great-circle distance from the
                `geographical_reference` kwargs, and are given in km.

            geographical_reference (tuple of float, optional): two-dimensional
                tuple that contains the geographical reference point (lon, lat)
                in degrees in order to calculate the cartesian coordinates of
                the seismic array from this reference point (i.e. the
                great-circle distance with respect to this point).
                Default is the array barycenter (None).

            csv_filter (str, optional): sub-array selection from the
                corresponding field. In that case, the values of the
                for each station should be 0 (discard) or 1 (keep).
                The csv_filter string is converted to uppercase.

            depth_factor (float): a factor that is used to multiply the depth.
                It can be basically set to 1e-3 if the elevation is given in
                meters, to 1 (default) if the elevation is already in km, or
                to 0 in order to consider no elevation.

        """

        # Read the csv file
        csv_data = csv.reader(open(path, 'r'))

        # Extract list of header (this removes the header from the csv data)
        headers = [data.strip() for data in next(csv_data)]

        # Extract non-optional headers ids
        id_name = headers.index('NAME')
        id_lon = headers.index('LON')
        id_lat = headers.index('LAT')

        # Extract optional headers ids (i.e. altitude)
        try:
            id_z = headers.index('ALT')
            is_z_defined = True
        except ValueError:
            is_z_defined = False
            pass

        # Initialization
        name = list()
        lon = list()
        lat = list()

        # The altitude is automatically set to 0 if undefined in the csv file
        z = list()

        # Read each row in csv file
        for row in csv_data:

            # Extract only the stations for which the filter value is 1
            if csv_filter:
                id_filter = headers.index(csv_filter.upper())
                if int(row[id_filter]) == 1:
                    name.append(row[id_name].strip())
                    lat.append(float(row[id_lat]))
                    lon.append(float(row[id_lon]))
                    if is_z_defined is True:
                        z.append(float(row[id_z]))
                    else:
                        z.append(0)

            # If the csv_filter is None, then keep every station
            else:
                name.append(row[id_name].strip())
                lat.append(float(row[id_lat]))
                lon.append(float(row[id_lon]))
                if is_z_defined is True:
                    z.append(float(row[id_z]))
                else:
                    z.append(0)

        # Attribution
        self.name = name
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.z = - depth_factor * np.array(z)
        self.dim = len(self.lon)

        # Convert geographical coordinates to cartesian coordinates
        if geographical_reference is None:
            lon_0 = self.lon.mean()
            lat_0 = self.lat.mean()
        else:
            lon_0, lat_0 = geographical_reference
        self.x, self.y = geo2xy(self.lon, self.lat, geographical_reference)

        # Define very useful upper triangular indexes
        self.triu = np.triu_indices(self.dim, k=1)

        pass

    def get_distances(self, triangular=False):
        """ Distance matrix between each array elements.

        If the user want to give no consideration to the station depths, the
        factor kwarg of the __init__ method should be set to 0.

        Args
        ----
            triangular (bool): if True, returns the upper triangular vector
                of the distance matrix (of shape N x (N - 1) / 2). If False
                (default), then returns the full N x N distance matrix, with
                N the number of stations.

        Return
        ------
            distances (array): square matrix of shape (self.dim, self.dim).
                The distances are given in km, and the distance between sensor
                i and sensor j is given by distance[i, j].

        """

        distances = (self.x - self.x[:, None]) ** 2 + \
            (self.y - self.y[:, None]) ** 2 +\
            (self.z - self.z[:, None]) ** 2
        if triangular is True:
            return distances[self.triu]
        else:
            return distances

    def get_xyz(self):
        """ Get stations coordinates in km.

        Return
        ------
            Coordinates (tuple of arrays): (x, y, z) cartesian coordinates
                of the seismic stations in km. If no altitude coordinate was
                given in the csv file, then the z coordinate is an array of
                zeros of the same dimension than x or y.
        """
        return self.x, self.y, self.z

    def get_llz(self):
        """ Get geographical coordinates. Lon/lat in degrees depth in km.

        Return
        ------
            Coordinates (tuple of arrays): (x, y, z) cartesian coordinates
                of the seismic stations in km. If no altitude coordinate was
                given in the csv file, then the z coordinate is an array of
                zeros of the same dimension than x or y.
        """
        return self.lon, self.lat, self.z

    def get_reference(self):
        """ Get array barycenter.

        Return
        ------
            Coordinates (tuple): (x, y, z) geographical coordinates of the
                barycenter. Depth in km.

        """
        return np.mean(self.lon), np.mean(self.lat), np.mean(self.z)

    def get_radius(self, method='integration'):
        """ Typical radius of the seismic array based on the 3D extent.

        Todo
        ----
            * Explicit the definition of the typical radius with analytic
            * Consider the analytic formulae with 3D arrays

        Args
        ----
            method (str, optional): which definition of the radius to consider:
                - "integration": summation over distances, normalize by shape
                - "analytic": use analytical expression of equal density

        Return
        ------
            typical_radius (int): the typical equivalent radius of the array
                in km. If 2D array, then consider a disk, otherwise a sphere.

        """
        if method == 'integration':
            distances = self.get_distances()
            typical_radius = np.sum(distances) / self.dim ** 2

        elif method == 'analytic':
            distance_max = np.max(self.get_distances())
            typical_radius = 0.5 * distance_max
            typical_radius = 2.0 * typical_radius / 3.0

        return typical_radius

    def get_eigenthreshold(self, frequency, slowness, shift=0,
                           method='analytic'):
        """ Eigenthreshold for covariance matrix whitening based on radius.

        Args
        ----
            frequency (float or array): frequencies (in Hz) of analysis.
            slowness (float): slowness (in s/km) of the homogeneous medium.
            shift (int): constant-value for eigenthreshold shift.
            method (str): "integration" or "analytic", please refer to the
                `get_radius` method.

        Return
        ------
            eigenthreshold (same shape than frequency): the frequency-dependant
                eigenthreshold.
        """

        radius = self.get_radius(method=method)
        wavenumber = np.array(2 * np.pi * frequency * slowness, ndmin=1)
        eigenthreshold = 2 * np.floor(wavenumber * radius) + 1
        eigenthreshold = eigenthreshold.astype(int)
        eigenthreshold[eigenthreshold > 0] -= shift
        eigenthreshold[eigenthreshold > self.dim] = self.dim
        return eigenthreshold

    def get_argsort_from(self, reference):
        """ Sorting indexes from a reference station index.

        Arg
        ---
            reference (int): index of the reference stations.

        Return
        ------
            sorting indexes with respect to the distance to the reference
                station.
        """

        distances = self.get_distances()
        distances = distances[reference]
        return distances.argsort()

    def get_argsort_oriented(self, angle, tolerance=10):
        """ Selecting station pairs within a given alignment.

        Args
        ----
            angle (float): angle of inclination from North (degree)
            tolerance (float): aperture of the angle around the selecting
                angle.
        Return
        ------
            The indexes of the oriented station pairs.
        """

        # Initialization
        ioriented = list()
        for i, j in product(range(self.dim), repeat=2):

            # Calculate inclination
            yij = (self.y[j] - self.y[i])
            xij = (self.x[j] - self.x[i])
            inclination = (180.0 / np.pi) * np.arctan2(xij, yij)

            # Check if inclination lies in agnle +/- tolerance
            aligned = np.abs(np.abs(angle) - inclination) < tolerance / 2.0
            if aligned:
                ioriented += [[i, j]]
            if i == j:
                ioriented += [[i, j]]

        return ioriented

    def select(self, indexes):
        """ Index-based selection of stations.

        Args
        ----
            indexes (list of ints): the indexes of selected seismic stations.

        Returns
        -------
            selected (:obj: `Antenna`): the selected antenna.
        """

        selected = copy.deepcopy(self)
        selected.name = [selected.name[i] for i in indexes]
        selected.lon = selected.lon[indexes]
        selected.lat = selected.lat[indexes]
        selected.x, selected.y = selected.x[indexes], selected.y[indexes]
        selected.dim = len(selected.x)
        selected.shape = (selected.dim, selected.dim)
        return selected
