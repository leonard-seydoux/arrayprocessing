#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Tools for dealing with seismic array geometries and mapping.

import copy
import numpy as np
import networkx as nx
import csv

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
    """
    Transforms longitude and latitude vectors to cartesian
    coordinates (in km). Uses the mean of longitude
    and latitude as reference point.
    """
    if reference is None:
        lon0 = np.mean(lon)
        lat0 = np.mean(lat)
    else:
        lon0, lat0 = reference

    x = (lon0 - lon) * 40000 * np.cos((lat + lat0) * np.pi / 360) / 360
    y = -(lat - lat0) * 40000 / 360
    return x, y


class Antenna():
    """
    Defines the coordinates of the antenna.
    From an array of cartesian coordinates,
    or from a file of geographical coordinates.
    """

    def __init__(self, cartesian_coordinates=None, txt=None,
                 geographical_coordinates=None, reference=None, csvfile=None,
                 filtre=None):
        """
        Get coordinates of the array sensor between each sensors of the
        antenna. CARTESIAN_COORDINATES are defined in km.
        """

        if cartesian_coordinates is not None:
            self.x = cartesian_coordinates[0]
            self.y = cartesian_coordinates[1]
            self.dim = len(self.x)
            self.shape = (self.dim, self.dim)

        elif txt is not None:
            columns = open(txt, 'r').readlines()
            columns = np.transpose([c.split() for c in columns])
            self.name = [str(e) for e in list(columns[0])]
            self.lon = np.array(list(map(float, columns[1])))
            self.lat = np.array(list(map(float, columns[2])))
            self.x, self.y = geo2xy(self.lon, self.lat, reference)
            self.dim = len(self.x)
            self.shape = (self.dim, self.dim)

        elif csvfile is not None:
            out = csv.reader(open(csvfile, 'r'))
            header = [h.strip() for h in next(out)]
            station_name_id = header.index('STA')
            lon_id = header.index('LON')
            lat_id = header.index('LAT')

            name = list()
            lon = list()
            lat = list()

            for row in out:
                if filtre:
                    filtre_id = header.index(filtre.upper())
                    if int(row[filtre_id]) == 1:
                        name.append(row[station_name_id].strip())
                        lat.append(float(row[lat_id]))
                        lon.append(float(row[lon_id]))

                else:
                    name.append(row[station_name_id].strip())
                    lat.append(float(row[lat_id]))
                    lon.append(float(row[lon_id]))

            self.name = name
            self.lon = np.array(lon)
            self.lat = np.array(lat)
            self.x, self.y = geo2xy(self.lon, self.lat, reference)
            self.dim = len(self.x)
            self.shape = (self.dim, self.dim)

        elif geographical_coordinates is not None:
            self.lon = geographical_coordinates[0]
            self.lat = geographical_coordinates[1]
            self.x, self.y = geo2xy(self.lon, self.lat, reference)
            self.dim = len(self.x)
            self.shape = (self.dim, self.dim)

        self.xy = [self.x, self.y]

    def get_distances(self):
        """
        Returns a N x N distance matrix between each antenna elements.
        """
        distance = (self.x - self.x[:, None])**2 + \
            (self.y - self.y[:, None])**2
        return np.sqrt(distance)

    def get_xy(self):
        """
        Returns station coordinates in km.
        """
        return self.x, self.y

    def get_dim(self):
        """
        Returns station coordinates in km.
        """
        return self.dim

    def get_ll(self):
        """
        Returns station coordinates.
        """
        return np.vstack((self.lon, self.lat))

    def get_llz(self):
        """
        Returns station coordinates.
        """
        return np.vstack((self.lon, self.lat, np.zeros(self.dim)))

    def get_reference(self):
        """
        Returns barycenter.
        """
        return np.mean(self.lon), np.mean(self.lat)

    def get_names(self):
        """
        Returns station coordinates.
        """
        return self.name

    def get_radiustypical(self):
        """
        Returns a N x N distance matrix between each antenna elements.
        """
        # distances = np.triu(self.get_distances())
        # return np.sum(distances)/(self.dim*(self.dim-1)/2.0)
        distances = self.get_distances()
        return np.sum(distances) / self.dim**2

    def get_eigenthreshold(self, frequency, slowness, shift=0):

        # radius = self.get_radiustypical()

        distance_max = np.max(self.get_distances())
        radius = 0.5 * distance_max
        radius = 2.0 * radius / 3.0

        wavenumber = 2 * np.pi * frequency * slowness
        try:
            eigenthreshold = int(2 * np.floor(wavenumber * radius)) + 1
            if eigenthreshold > 0:
                eigenthreshold -= shift
            if eigenthreshold > self.dim:
                eigenthreshold = self.dim
        except:
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
        latlabels = [dms.format(np.floor(l), l % 1 * 60) for l in extent_lat]
        lonlabels = [l.replace(u'\N{DEGREE SIGN}0', degree) for l in lonlabels]
        latlabels = [l.replace(u'\N{DEGREE SIGN}0', degree) for l in latlabels]
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
