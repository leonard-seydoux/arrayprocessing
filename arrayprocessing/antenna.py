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
import networkx as nx
import csv

from arrayprocessing import logtable
from matplotlib import pyplot as plt
from itertools import product
from cartopy import crs
from cartopy.mpl import geoaxes
from cartopy.io.ogc_clients import WMTSRasterSource
from copy import deepcopy
from cartopy.feature import NaturalEarthFeature as nef
from matplotlib import ticker
from matplotlib import patheffects
from cartopy.crs import TransverseMercator


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
                id_filter = header.index(filtre.upper())
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

        pass

    def get_distances(self):
        """ Distance matrix between each array elements.

        If the user want to give no consideration to the station depths, the
        factor kwarg of the __init__ method should be set to 0.

        Return
        ------
            distances (array): square matrix of shape (self.dim, self.dim).
                The distances are given in km, and the distance between sensor
                i and sensor j is given by distance[i, j].

        """

        return (self.x - self.x[:, None]) ** 2 + \
            (self.y - self.y[:, None]) ** 2 +\
            (self.z - self.z[:, None]) ** 2

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
        """
        Returns station coordinates.
        """
        return np.vstack((self.lon, self.lat))

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

    def get_eigenthreshold_oldshool(self, frequency, slowness, shift=0):
        distance_max = np.max(self.get_distances())
        radius = 0.5 * distance_max
        radius = 2.0 * radius / 3.0
        wavenumber = 2 * np.pi * frequency * slowness
        eigenthreshold = int(2 * np.floor(wavenumber * radius) + 1)
        if eigenthreshold > shift:
            eigenthreshold -= shift
        if eigenthreshold > self.dim:
            eigenthreshold = self.dim
        return eigenthreshold

    def get_argsort_from(self, reference):
        distances = self.get_distances()
        distances = distances[reference]
        return distances.argsort()

    def set_oriented(self, angle, tolerance):
        ioriented = []
        for i, j in product(range(self.dim), repeat=2):
            yij = (self.y[j] - self.y[i])
            xij = (self.x[j] - self.x[i])
            inclination = (180.0 / np.pi) * np.arctan2(xij, yij)
            aligned = abs(np.abs(angle) - inclination) < tolerance / 2.0
            if aligned:
                ioriented += [[i, j]]
            if i == j:
                ioriented += [[i, j]]
        return ioriented

    def select(self, indexes):

        selected = copy.deepcopy(self)
        selected.name = [selected.name[i] for i in indexes]
        selected.lon = selected.lon[indexes]
        selected.lat = selected.lat[indexes]
        selected.x, selected.y = selected.x[indexes], selected.y[indexes]
        selected.dim = len(selected.x)
        selected.shape = (selected.dim, selected.dim)
        return selected


class Map(geoaxes.GeoAxes):

    def __init__(self, figsize=[4, 5], ax=None, extent=None,
                 projection=crs.PlateCarree()):

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1], projection=projection)
            ax.outline_patch.set_lw(0.5)
            self.__dict__ = ax.__dict__

        else:
            self.__dict__ = ax.__dict__

        if extent is not None:
            self.set_extent(extent)

    def plot_stations(self, coordinates, **kwargs):
        kwargs.setdefault('color', 'k')
        kwargs.setdefault('ms', 6)
        kwargs.setdefault('mfc', '#F5962A')
        kwargs.setdefault('mew', 0.5)
        kwargs.setdefault('transform', crs.PlateCarree())
        self.plot(*coordinates, 'v', **kwargs)

    def add_world_ocean_basemap(self, target_resolution=[600, 600]):
        """
        This function needs to copy-paste the following lines of code into the
        file ogc_clients.py of the cartopy package, in order to read the WMTS:

        METERS_PER_UNIT = {
            'urn:ogc:def:crs:EPSG::27700': 1,
            'urn:ogc:def:crs:EPSG::900913': 1,
            'urn:ogc:def:crs:EPSG:6.18.3:3857': 1,
            'urn:ogc:def:crs:EPSG::4326': _WGS84_METERS_PER_UNIT,
            'urn:ogc:def:crs:OGC:1.3:CRS84': _WGS84_METERS_PER_UNIT,
            'urn:ogc:def:crs:EPSG::3031': 1,
            'urn:ogc:def:crs:EPSG::3413': 1
        }

        _URN_TO_CRS = collections.OrderedDict([
            ('urn:ogc:def:crs:OGC:1.3:CRS84', ccrs.PlateCarree()),
            ('urn:ogc:def:crs:EPSG::4326', ccrs.PlateCarree()),
            ('urn:ogc:def:crs:EPSG::900913', ccrs.GOOGLE_MERCATOR),
            ('urn:ogc:def:crs:EPSG:6.18.3:3857', ccrs.GOOGLE_MERCATOR),
            ('urn:ogc:def:crs:EPSG::27700', ccrs.OSGB()),
            ('urn:ogc:def:crs:EPSG::3031', ccrs.Stereographic(
                central_latitude=-90,
                true_scale_latitude=-71)),
            ('urn:ogc:def:crs:EPSG::3413', ccrs.Stereographic(
                central_longitude=-45,
                central_latitude=90,
                true_scale_latitude=70))
        ])
        """

        # Web map server
        wmts = 'http://services.arcgisonline.com/arcgis/rest/services/Ocean/'
        wmts += 'World_Ocean_Base/MapServer/WMTS/1.0.0/WMTSCapabilities.xml'
        wmts = WMTSRasterSource(wmts, 'Ocean_World_Ocean_Base')

        proj = self.projection
        extent = self.get_extent()
        img = wmts.fetch_raster(proj, extent, target_resolution)
        self.imshow(img[0][0], extent=img[0][1], transform=proj,
                    origin='upper')

    def fancy_ticks(self, thickness=0.03, n_lon=7, n_lat=5, size=9):

        # Extract map meta from ax
        inner = self.get_extent()
        proj = self.projection

        # Define outer limits
        outer = [dc for dc in deepcopy(inner)]
        width = inner[1] - inner[0]
        height = inner[3] - inner[2]
        ratio = height / width
        outer[0] -= width * thickness * ratio
        outer[1] += width * thickness * ratio
        outer[2] -= height * thickness
        outer[3] += height * thickness

        inner_lon = np.linspace(inner[0], inner[1], n_lon)
        inner_lat = np.linspace(inner[2], inner[3], n_lat)
        outer_lon = np.linspace(outer[0], outer[1], n_lon)
        outer_lat = np.linspace(outer[2], outer[3], n_lat)

        w = dict(lw=.5, edgecolor='k', facecolor='w', transform=proj, zorder=9)
        b = dict(lw=0, facecolor='k', clip_on=False, transform=proj, zorder=10)

        # White frame
        self.fill_between([outer[0], outer[1]], outer[2], inner[2], **w)
        self.fill_between([outer[0], outer[1]], outer[3], inner[3], **w)
        self.fill_between([outer[0], inner[0]], inner[2], inner[3], **w)
        self.fill_between([outer[1], inner[1]], inner[2], inner[3], **w)

        bottom_heigth = (outer[2], inner[2])
        top_heigth = (outer[3], inner[3])
        for index, limits in enumerate(zip(inner_lon[:-1], inner_lon[1:])):
            self.fill_between(limits, *bottom_heigth, **w)
            self.fill_between(limits, *top_heigth, **w)
            if index % 2 == 0:
                self.fill_between(limits, *bottom_heigth, **b)
                self.fill_between(limits, *top_heigth, **b)

        left_width = (outer[0], inner[0])
        right_width = (outer[1], inner[1])
        for index, height in enumerate(zip(inner_lat[:-1], inner_lat[1:])):
            self.fill_between(left_width, *height, **w)
            self.fill_between(right_width, *height, **w)
            if index % 2 == 0:
                self.fill_between(left_width, *height, **b)
                self.fill_between(right_width, *height, **b)

        self.set_xticks(inner_lon)
        self.set_yticks(inner_lat)
        self.xaxis.set_tick_params(length=0, labelsize=size)
        self.yaxis.set_tick_params(length=0, labelsize=size)

        degree = u'\N{DEGREE SIGN}'
        dms = '{:.0f}\N{DEGREE SIGN}{:.0f}'
        lons = [dms.format(np.floor(l), l % 1 * 60) for l in inner_lon]
        lats = [dms.format(np.floor(l), l % 1 * 60) for l in inner_lat]
        lons = [l.replace(u'\N{DEGREE SIGN}0', degree) for l in lons]
        lats = [l.replace(u'\N{DEGREE SIGN}0', degree) for l in lats]
        self.set_xticklabels(lons)
        self.set_yticklabels(lats)
        self.set_extent(outer)

    def ticks(self, n_lon=7, n_lat=5):

        # Extract map meta from ax
        extent = self.get_extent()
        extent_lon = np.linspace(extent[0], extent[1], n_lon)
        extent_lat = np.linspace(extent[2], extent[3], n_lat)
        proj = self.projection

        self.set_xticks(extent_lon)
        self.set_yticks(extent_lat)

        degree = u'\N{DEGREE SIGN}'
        dms = '{:.0f}\N{DEGREE SIGN}{:.0f}'
        lons = [dms.format(np.floor(l), l % 1 * 60) for l in extent_lon]
        lats = [dms.format(np.floor(l), l % 1 * 60) for l in extent_lat]
        lonlabels = [l.replace(u'\N{DEGREE SIGN}0', degree) for l in lons]
        latlabels = [l.replace(u'\N{DEGREE SIGN}0', degree) for l in lats]
        self.set_xticklabels(lonlabels)
        self.set_yticklabels(latlabels)

    def repel_labels(self, coordinates, labels, k=0.01, fontsize=6, color='k'):

        x, y = coordinates
        G = nx.DiGraph()
        data_nodes = []
        init_pos = {}
        for xi, yi, label in zip(x, y, labels):
            data_str = 'data_{0}'.format(label)
            G.add_node(data_str)
            G.add_node(label)
            G.add_edge(label, data_str)
            data_nodes.append(data_str)
            init_pos[data_str] = (xi, yi)
            init_pos[label] = (xi, yi)

        pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

        # undo spring_layout's rescaling
        pos_after = np.vstack([pos[d] for d in data_nodes])
        pos_before = np.vstack([init_pos[d] for d in data_nodes])
        scale, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
        scale, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)
        shift = np.array([shift_x, shift_y])
        for key, val in pos.items():
            pos[key] = (val * scale) + shift

        for label, data_str in G.edges():
            transparent = (0, 0, 0, 0)
            t = self.annotate(label,
                              xy=pos[data_str], xycoords='data',
                              xytext=pos[label], textcoords='data',
                              color=color,
                              arrowprops=dict(arrowstyle="-", linewidth=0.4,
                                              connectionstyle="arc",
                                              color='k'),
                              bbox=dict(boxstyle='square,pad=0',
                                        facecolor=(0, 0, 0, 0),
                                        linewidth=0))

            t.set_fontname('Consolas')
            t.set_fontsize(fontsize)
            t.set_fontweight('bold')
            t.set_path_effects([patheffects.Stroke(
                linewidth=0.6, foreground='w'), patheffects.Normal()])

    def add_global_location(self, position=[.64, .0, .45, .45],
                            land_color='0.6', land_line_color='k',
                            ocean_color='#efefef', marker_kw=dict()):

        extent = self.get_extent()

        globe = crs.Orthographic(central_longitude=extent[0],
                                 central_latitude=extent[2])

        ax = self.figure.add_axes(position, projection=globe)
        ax.outline_patch.set_lw(0.5)
        ax.outline_patch.set_edgecolor('0.5')
        ax.add_feature(nef('physical', 'land', '110m'), facecolor=land_color,
                       lw=0.3, edgecolor=land_line_color)
        ax.add_feature(nef('physical', 'ocean', '110m'),
                       facecolor=ocean_color, lw=0)

        ax.set_global()
        gl = ax.gridlines(linewidth=0.3, linestyle='-')
        gl.xlocator = ticker.FixedLocator(range(-180, 181, 20))
        gl.ylocator = ticker.FixedLocator(range(-90, 91, 15))

        globe = crs.Orthographic(central_longitude=extent[0],
                                 central_latitude=extent[2] - 40)
        ax = self.figure.add_axes(position, projection=globe)
        ax.background_patch.set_fill(False)
        ax.outline_patch.set_lw(0.5)
        ax.outline_patch.set_edgecolor('0.5')
        shift = 5
        night = np.arctan(np.linspace(-2 * shift, 2 * shift, 500) + shift) - \
            np.arctan(np.linspace(-2 * shift, 2 * shift, 500) - shift)
        night, _ = np.meshgrid(night, night)
        ax.imshow(night, extent=(-160, 200, -90, 90), zorder=10, cmap='Greys',
                  alpha=0.4, transform=crs.PlateCarree(),
                  interpolation='bicubic')

        ax.plot(extent[0], extent[2], 's', **marker_kw)

    def add_lands(self, res='10m', lw=0.3, c='k', fc=(0, 0, 0, 0)):
        """Add coastlines with scalable linewidth"""

        land = nef('physical', 'land', res)
        self.add_feature(land, facecolor=fc, linewidth=lw, edgecolor=c)

    def scale_bar(self, length=50, location=(0.5, 0.05), lw=3):
        """
        Adds a scale bar to the axes.
        """

        # Get the limits of the axis in lat long
        west, east, south, north = self.get_extent(self.projection)

        # Make tmc horizontally centred on the middle of the map,
        # vertically at scale bar location
        horizontal = (east + west) / 2
        vertical = south + (north - south) * location[1]
        tmc = TransverseMercator(horizontal, vertical)

        # Get the extent of the plotted area in coordinates in metres
        left, right, bottom, top = self.get_extent(tmc)

        # Turn the specified scalebar location into coordinates in metres
        bar_x = left + (right - left) * location[0]
        bar_y = bottom + (top - bottom) * location[1]

        # Generate the x coordinate for the ends of the scalebar
        left_x = [bar_x - length * 500, bar_x + length * 500]

        # Plot the scalebar
        self.plot(left_x, 2 * [bar_y], '|-',
                  transform=tmc, color='k', lw=lw, mew=lw)

        # Plot the scalebar label
        bar_text = str(length) + ' km'
        text_y = bottom + (top - bottom) * (location[1] + 0.01)
        self.text(bar_x, text_y, bar_text, transform=tmc, ha='center',
                  va='bottom', weight='normal')
