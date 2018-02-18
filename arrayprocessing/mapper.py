#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Array-processing mapping tools.

This modules defines two maps: the seismic array map, and the local beam map.
The maps classes are GeoAxes inherited from the Cartopy package.

Authors:
    L. Seydoux (leonard.seydoux@gmail.com)
    J. Soubestre

Last update:
    Feb. 2018

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib import patheffects
from cartopy import crs
from cartopy.mpl import geoaxes
from cartopy.io.ogc_clients import WMTSRasterSource
from copy import deepcopy
from cartopy.feature import NaturalEarthFeature as nef
from cartopy.crs import TransverseMercator


class Map(geoaxes.GeoAxes):
    """ Create a map based on cartopy GeoAxes.

    The GeoAxes are implemented with few additional methods, mostly based on
    the plot() and imshow() methods.

    """

    def __init__(self, figsize=[4, 5], ax=None, extent=None,
                 projection=crs.PlateCarree()):
        """
        Optional Args
        -------------
            figsize (list or tuple): the size of the figure in inches.

            ax (:obj: `plt.Axes`): previously created axes. This could be
                useful for instance if the map is to be included within
                subplots. In that case, GeoAxes with matching position are
                created, and the original axes destroyed.

            extent (tuple or list): the geographical extent of the axes in
                degrees. Uses the same convention that with Cartopy:
                e.g. extent = (west, east, south, north)

            projection (:obj: `cartopy.crs`): projection of the map.
                For more details about projections, please refer to:
                http://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html
        """

        # Create axes if none are given
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1], projection=projection)
            ax.outline_patch.set_lw(0.5)
            self.__dict__ = ax.__dict__

        else:
            self.__dict__ = ax.__dict__

        # Set extents
        if extent is not None:
            self.set_extent(extent)

    def plot_symbols(self, x, y, symbol='v', **kwargs):
        """ Plot symbols on the map with according projection.

        Args
        ----
            x (array or float): the x-axis values
            y (array or float): the y-axis values
            **kwargs: the kwargs passed to the plt.plot() function.
                Default kwargs are defined for seismic stations:
                - color: k
                - mfc: #F5962A
                - ms: 6
                - mew: 0.5

        """
        kwargs.setdefault('color', 'k')
        kwargs.setdefault('ms', 6)
        kwargs.setdefault('mfc', '#F5962A')
        kwargs.setdefault('mew', 0.5)
        kwargs.setdefault('transform', crs.PlateCarree())
        return self.plot(x, y, symbol, **kwargs)

    def plot_img(self, image, extent=None, **kwargs):
        """ Plot image with automatic parameters like projection.
        """

        extent = self.get_extent() if extent is None else extent
        kwargs.setdefault('transform', crs.PlateCarree())
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('extent', extent)
        return self.imshow(image, **kwargs)

    def add_world_ocean_basemap(self, target_resolution=[600, 600]):
        """ Adds ESRI's Ocean Basemap to the map background.

        Args
        ----
            target_resolution (tuple or list): the resolution in pixels of the
                basemap.

        Note
        ----
            This function required the following lines to be copy-pasted
            into the file ogc_clients.py of the cartopy package, in order to
            work:

            ** START COPY BELOW **

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

            ** STOP COPY ABOVE **
        """

        # Web map server
        wmts = 'http://services.arcgisonline.com/arcgis/rest/services/Ocean/'
        wmts += 'World_Ocean_Base/MapServer/WMTS/1.0.0/WMTSCapabilities.xml'
        wmts = WMTSRasterSource(wmts, 'Ocean_World_Ocean_Base')

        proj = self.projection
        extent = self.get_extent()
        img = wmts.fetch_raster(proj, extent, target_resolution)
        self.imshow(img[0][0], extent=img[0][1], transform=proj,
                    origin='upper', interpolation='bicubic')

    def fancy_ticks(self, thickness=0.03, n_lon=7, n_lat=5, size=9):
        """ Add fancy ticks to the map (similar and fancy GMT basemap).

        Optional args
        -------------
            thickness (float): the thickness of the fancy bar
                (in figure width ratio).

            n_lon (int): the number of ticks per longitudes
            n_lat (int): the number of ticks per latitudes
            size (int): font size for the labels
        """

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

        # Inner limits
        inner_lon = np.linspace(inner[0], inner[1], n_lon)
        inner_lat = np.linspace(inner[2], inner[3], n_lat)

        # Black and white styles
        w = dict(lw=.5, edgecolor='k', facecolor='w', transform=proj, zorder=9)
        b = dict(lw=0, facecolor='k', clip_on=False, transform=proj, zorder=10)

        # White frame
        self.fill_between([outer[0], outer[1]], outer[2], inner[2], **w)
        self.fill_between([outer[0], outer[1]], outer[3], inner[3], **w)
        self.fill_between([outer[0], inner[0]], inner[2], inner[3], **w)
        self.fill_between([outer[1], inner[1]], inner[2], inner[3], **w)

        # Create frame
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
        """ Add normal ticks to the map.

        Optional args
        -------------
            n_lon (int): the number of ticks per longitudes
            n_lat (int): the number of ticks per latitudes
        """

        # Extract map meta from ax
        extent = self.get_extent()
        extent_lon = np.linspace(extent[0], extent[1], n_lon)
        extent_lat = np.linspace(extent[2], extent[3], n_lat)

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

    def repel_labels(self, x, y, labels, k=0.01, fontsize=6, color='k'):
        """ Automatically find position of labels to be displayed.

        Args
        ----
            x, y (arrays or floats): the original coordinates for each label
            labels (list of str): the labels
            k (float, optional): an argument for the spreading of the labels.
                If close to 0, then the labels are collapsed to their original
                position, otherwise, it can expand very far.
            fontsize (int): size of the labels
            color (str or tuple): color of the labels
        """

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
                            ocean_color='#efefef', **kawrgs):
        """ Global position shown in a small globe inset.

        Optional args
        -------------
            position (tuple): the position of the globe axes in the figure
            land_color (str or tuple): land color
            land_line_color (str or tuple): land lines color
            ocean_color (str or tuple): ocean color
            **kwargs: other kwargs are passed to the plt.plot function.
        """

        # Get local location from extent (bottom-left corner)
        extent = self.get_extent()
        globe = crs.Orthographic(central_longitude=extent[0],
                                 central_latitude=extent[2])

        # Add axes
        ax = self.figure.add_axes(position, projection=globe)
        ax.outline_patch.set_lw(0.7)
        ax.outline_patch.set_edgecolor('0.5')
        ax.add_feature(nef('physical', 'land', '110m'), facecolor=land_color,
                       lw=0.3, edgecolor=land_line_color)
        ax.add_feature(nef('physical', 'ocean', '110m'),
                       facecolor=ocean_color, lw=0)
        ax.set_global()

        # Grid
        gl = ax.gridlines(linewidth=0.3, linestyle='-')
        gl.xlocator = ticker.FixedLocator(range(-180, 181, 20))
        gl.ylocator = ticker.FixedLocator(range(-90, 91, 15))

        # Add second for the shadow
        globe = crs.Orthographic(central_longitude=150, central_latitude=0)
        ax = self.figure.add_axes(position, projection=globe)
        ax.background_patch.set_fill(False)
        ax.outline_patch.set_lw(0.7)
        ax.outline_patch.set_edgecolor('0.5')
        shift = 5
        night = np.arctan(np.linspace(-2 * shift, 2 * shift, 500) + shift) - \
            np.arctan(np.linspace(-2 * shift, 2 * shift, 500) - shift)
        night, _ = np.meshgrid(night, night)
        ax.imshow(night, extent=(-180, 180, -90, 90),
                  zorder=10, cmap='Greys',
                  alpha=0.5, transform=crs.PlateCarree(),
                  interpolation='bicubic')

        ax.plot(extent[0], extent[2], 's', **kawrgs)

    def add_lands(self, res='10m', lw=0.3, c='k', fc=(0, 0, 0, 0)):
        """ Add coastlines with scalable linewidth

         Optional args
        -------------
            res (str): natural Earth resolution (e.g. '10m', '50m', '110m').
            lw (flaot): line width of the coastlines, default to 0.3.
            c (str or tuple): color for the coastlines, default to 'k'.
            fc (str or tuple): color for the lands. Default to transparent.
        """
        land = nef('physical', 'land', res)
        self.add_feature(land, facecolor=fc, linewidth=lw, edgecolor=c)

    def scale_bar(self, length=50, location=(0.5, 0.06), lw=1):
        """ Add a scale bar to the axes.

        Optional arguments
        ------------------

            length (float): length of the bar in km
            location (tuple): location of the bar in axes nomarlized
                coordinates.
            lw (float): width on the coastline in points.
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

    def save(self, path, **kwargs):
        """ Save the axes' figure to the given path.

        Args
        ----
            path (str): path to the file where to save the figure.
            **kwargs: keyword arguments directly passed to the
                `plt.savefig()` method.
        """
        fig = self.figure
        fig.savefig(path, **kwargs)
        pass
