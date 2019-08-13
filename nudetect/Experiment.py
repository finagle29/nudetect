'''
The NuDetect module contains an object-oriented framework for processing and
plotting NuSTAR detector test data. Specifically, it has the classes 
'GammaFlood' for analysis of gamma flood test data (including count 
distribution, gain correction, and generating a spectrum), 'Noise' for 
electronic noise data, and 'Leakage' for leakage current data. 

Each instance of one of these classes can represent a single experiment done 
in the detector test lab (i.e., the data collected between running 'start 
startscreening' and 'start endscreening' in ITOS, although each 'Leakage' 
instances can currently represent multiple experiments). Hence, each of these 
classes inherits from an abstract 'Experiment' subclass, which contains 
methods and attributes shared amongst subclasses.
'''

# Packages for making life easier
import os.path
import string
import argparse
import datetime

# Data analysis packages
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.table import Table
import astropy.io.ascii as asciio

# Plotting packages
import matplotlib.pyplot as plt
import matplotlib.cm # color map


class Experiment:
    '''
    A base class for classes representing various detector tests, like 
    GammaFlood and Noise. This houses some methods that all such classes share.

    Public Class Attributes:
        numericize: dict
            Maps all ascii letters to empty strings. When passed to the 
            'translate' method of strings, removes all letters from the string.
        full_detector: bool
            If True, the region being analyzed by this instance represents
            the full detector. If False, then the analyzed region is a strict
            subset of the full detector.
            (default: True)
        num_caps: int
            The number of sampling capacitors mediating the pixel readout
            (default: 16)

    Private Class Attributes:
        _full_det_shape: Tuple(int, int)
            Dimensions in pixels of the full detector
            (default: (32, 32))
        _num_rows: int
            number of rows in the analyzed region
            (default: 32)
        _num_cols: int
            number of columns in the analyzed region
            (default: 32)
        _det_shape: Tuple(int, int)
            Dimensions in pixels of the region being analyzed.
            (default: (32, 32))
        _det_shape_buff: Tuple(int, int)
            Shape of an array representing the detector pixels with a 1 pixel 
            buffer around all edges.
            (default: (34, 34))
        _row_iter: a 'range' object
            Iterates through row numbers (y coordinates) of pixels in the 
            analyzed region.
            (default: range(32))
        _col_iter: a 'range' object
            Iterates through col numbers (y coordinates) of pixels in the 
            analyzed region.
            (default: range(32))
        _col_slice: a 'slice' object
            For accessing the columns of the analyzed detector region from an 
            array representing pixels of the whole detector.
            (default: slice(0, 32))
        _row_slice: a 'slice' object
            For accessing the rows of the analyzed detector region from an 
            array representing pixels of the whole detector.
            (default: slice(0, 32))
        _start_row: int
            The first row (y coordinate) in the analyzed region.
            (default: 0)
        _end_row: int
            One greater than the last row in the analyzed region.
            (default: 0)
        _start_col: int
            The first column (y coordinate) in the analyzed region.
            (default: 32)
        _end_col: int
            One greater than the last column in the analyzed region.
            (default: 32)
    '''

    # A class attribute for removing letters from strings. Used in subclasses
    # when formatting units.
    numericize = str.maketrans('', '', string.ascii_letters)

    # Dimensions of the whole detector
    _full_det_shape = (32, 32)

    #
    # Class attributes related to the shape of the portion of the 
    # detector that will be analyzed. Initialized below to a region
    # representing the whole detector.
    #

    # A class attribute indicating the dimensions of the detector being tested.
    _num_rows = 32
    _num_cols = 32
    _det_shape = (_num_rows, _num_cols)
    # Shape of an array representing the detector pixels with a 1 pixel width
    # buffer around all edges.
    _det_shape_buff = (_num_rows + 2, _num_cols + 2)

    # Iterator (range) objects for iterating through rows and columns.
    _row_iter = range(_num_rows)
    _col_iter = range(_num_cols)

    # Slice objects for accessing a detector region from the whole detector
    _row_slice = slice(0, 32)
    _col_slice = slice(0, 32)

    _start_row = 0
    _start_col = 0
    _end_row = 32
    _end_col = 32

    full_detector = True

    # The number of sampling capacitors mediating the pixel readout
    num_caps = 16

    def select_detector_region(self, start_col, start_row, end_col, end_row):
        '''
        Selects a region of the detector to be analyzed, if not the full
        detector. Following python convention, the pixel coordinates are
        zero-indexed, and the starting coordinates are included in the region, 
        while the ending coordinates are excluded. 

        For example, to select the 3 x 5 square region with one corner at 
        (1, 2) and the opposite corner at (5, 4), as shown by the x's in the
        diagram below, one call this method like this:

            >>> from nudetect import Noise
            >>> noise = Noise('raw_data_path', [other initilaization args])
            >>> noise.select_detector_region(1, 2, 6, 5)

        y

        6 o o o o o o o
        5 o o o o o o o
        4 o x x x x x o
        3 o x x x x x o
        2 o x x x x x o
        1 o o o o o o o
        0 o o o o o o o
          0 1 2 3 4 5 6  x


        Arguments:
            start_col: int
                The first column in the region. Corresponds to the x-values
                of a pixel coordinate (RAWX as referenced in the FITS files).
            start_row: int
                The first row in the region. Corresponds to the y-values
                of a pixel coordinate (RAWY as referenced in the FITS files).
            end_col: int
                One greater than the last column in the region. Corresponds to 
                the x-values of a pixel coordinate (RAWX as referenced in the 
                FITS files).
            start_row: int
                One greater than the last row in the region. Corresponds to 
                the y-values of a pixel coordinate (RAWY as referenced in the 
                FITS files).
        '''
        # Checking parameter values
        check_non_negative(start_row=start_row, end_row=end_row, 
            start_col=start_col, end_col=end_col)

        if start_row >= end_row:
            raise ValueError("'start_row' must be < 'end_row'")

        if start_col >= end_col:
            raise ValueError("'start_col' must be < 'end_col'")

        # Simple iterators that will iterate through the pixel coordinates 
        # relative to the full detector.
        row_iter = range(start_row, end_row)
        col_iter = range(start_col, end_col)

        # Slice objects for accessing a detector region from the whole detector
        row_slice = slice(start_row, end_row)
        col_slice = slice(start_col, end_col)

        # Total number or rows and columns in the selected region.
        num_rows = end_row - start_row
        num_cols = end_col - start_col

        # Dimensions of the region in pixels.
        det_shape = (num_rows, num_cols)
        # Dimensions of the above region with a 1 pixel buffer on all sides.
        det_shape_buff = (num_rows + 2, num_cols + 2)

        self._start_row = start_row
        self._start_col = start_col
        self._end_col   = end_col
        self._end_row   = end_row

        self._row_iter = row_iter
        self._col_iter = col_iter

        self._num_rows = num_rows
        self._num_cols = num_cols

        self._row_slice = row_slice
        self._col_slice = col_slice

        self._det_shape = det_shape
        self._det_shape_buff = det_shape_buff

        # If the region selected is not the full detector, set
        # the 'full_detector' attribute to False.
        self.full_detector = self._det_shape == self._full_det_shape
        # Setting the 'etc' attribute so that information about the selected
        # region is at the end of all files saved about this analysis.
        if self.full_detector:
            self.etc += f'region({start_row},{start_col})({end_row},{end_col})'


    def detector_region_info(self):
        '''Returns a dict with information about the detector shape.'''

        private_keys = {'_row_iter', '_col_iter', '_det_shape', '_det_shape_buff',
            '_start_row', '_start_col', '_end_row', '_end_col'}

        info = {}

        for key, val in self.__dict__.items():
            if key in private_keys:
                info[key[1:]] = val

        info['full_detector'] = self.full_detector

        return info


    #
    # Small helper methods: 'title' and '_set_save_dir'.
    #

    def title(self, plot):
        '''
        Returns a plot title based on the instance's attributes and the 
        type of plot. 'plot' is a str indicating the type of plot.
        '''
        temp = r'$' + self.temp + r'^{\circ}$C'
        voltage = r'$' + self.voltage + r'$ V'

        analysis = type(self).__name__
        if type(self).__name__ == 'GammaFlood':
            analysis = r'$\gamma$ Flood '


        title = f'{analysis} {self.detector} {plot} ({voltage}, {temp})'

        if self.etc:
            title += f' -- {self.etc}'

        return title


    def _set_save_dir(self, save_dir, save_type=None):
        '''
        A helper method for initializing a 'save_dir' attribute. Must be called
        after the 'detector' attribute is initialized.
        Argument:
            save_dir: str
                The path to a directory where files will be saved be default.

        Keyword Argument:
            save_type: str
                If 'data', then all processed data outputs will be sent to
                the directory passed for 'save_dir' by default. If 'plot',
                then all plots will be sent to 'save_dir'. If None, then 
                all files will be sent to 'save_dir' unless paths are otherwise
                specified for data or plot files.
                (default: None)
        '''
        # If a directory was supplied, insert the detector ID where appropriate
        # and check that the resulting directory exists.
        if save_dir:
            save_dir = save_dir.format(self.detector)
            if not os.path.exists(save_dir):
                raise ValueError(f'The directory {save_dir} does not exist.')

        if save_type is None:     self.save_dir = save_dir
        elif save_type == 'data': self.data_dir = save_dir
        elif save_type == 'plot': self.plot_dir = save_dir


    #
    # Save path management method: construct_path.
    #

    def construct_path(self, save_type=None, description='', ext='', 
        save_dir='', subdir='', etc=''):
        '''
        Constructs a path for saving data and figures based on user input. 
        If the string passed to 'save_dir' has an empty pair of curly braces 
        '{}', they will be replaced by the detector ID 'self.detector'.

        Note to developers: This function is designed to throw a lot of 
        exceptions and be strict about formatting early on to avoid 
        complications later. Call it early in scripts to avoid losing the 
        results of a long computation to a mistyped directory.

        Keyword Arguments:
            save_type:
                A string specifying the type of data for which this method
                is generating a save path. Can be 'data' or 'plot'. This 
                determines whether the method looks in the instance's 
                'data_dir' or 'plot_dir' attribute for a save directory
                if 'save_dir' is not specified. If None, the method will
                look in the instance attribute 'save_dir'.
                (default: None)
            ext: str
                The file name extension.
            description: str
                A short description of what the file contains. This will be 
                prepended to the file name.
                (default: '')
            etc: str 
                Other important information, e.g., pixel coordinates. This  
                will be appended to the file name.
                (default: '')
            save_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'save_dir' attribute. If an empty string,
                will default to the attribute 'save_dir'.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            subdir: str
                A path to a sub-directory of 'save_dir' to which a file will
                be saved. Empty curly braces '{}' are formatted the same way
                as in 'save_dir'. 
                (default: '')

        Return:
            save_path: str
                A Unix/Linux/MacOS style path that can be used to save data
                and plots in an organized way.
        '''
        #
        # Handling exceptions and potential errors
        #

        # If 'ext' does not start with a '.', fix it.
        if ext and ext[0] != '.':
            ext = f'.{ext}'

        # If no 'save_dir' argument was supplied, take instead the value in 
        # the 'data_dir' or 'plot_dir' attributes, unless they also weren't 
        # supplied values, in which case we look in the 'save_dir' attribute.
        if not save_dir:
            if save_type == 'data':
                save_dir = self.data_dir
            elif save_type == 'plot':
                save_dir = self.plot_dir

        if not save_dir:
            save_dir = self.save_dir

        # Append the subdirectory 'subdir' to the path, if specified.
        if subdir and save_dir:
            save_dir += f'/{subdir}'
        elif subdir:
            save_dir = subdir

        # If the 'save_dir' argument was supplied, format it to include the 
        # detector ID in place of '{}' and check that the resulting directory
        # exists.
        if save_dir:
            save_dir = save_dir.format(self.detector)
            if not os.path.exists(save_dir):
                raise ValueError(f'The directory {save_dir} does not exist.')

        #
        # Constructing the path name
        #

        # Construct the file name from the file name in 'self.raw_data_path'.
        filename = os.path.basename(self.raw_data_path) # Extracts the filename
        save_path = os.path.splitext(filename)[0] # Removes the extension

        # Map all whitespace characters and '.' to underscores
        trans = str.maketrans(
            '.' + string.whitespace, 
            '_' * (len(string.whitespace) + 1)
        )
        save_path = save_path.translate(trans)
        
        # Prepend the description if specified
        if description:
            save_path = f'{description}_{save_path}'

        # Append extra info to the file name if specified
        if self.etc:
            save_path += f'_{self.etc}'
        if etc:
            save_path += f'_{etc}'

        # Append the file extension
        save_path += ext

        # Prepend the save directory if specified
        if save_dir:
            save_path = f'{save_dir}/{save_path}'

        return save_path


    #
    # Plotting methods: 'plot_pixel_hist' and 'plot_pixel_map'.
    #

    def plot_pixel_hist(self, value_label, values=None, bins=70, 
        hist_range=None, title='', text_pos='right', save_plot=True,
        plot_dir='', plot_subdir='', plot_ext='.pdf', etc='', **kwargs):
        '''
        Plots a histogram of some value for each pixel. If data is
        not supplied explicitly with 'values', an attribute chosen 
        based on 'value_label' will be used. For example, 

        if value_label == 'Gain':
            values = self.gain


        Arguments:
            value_label: str
                A short label denoting what data is supplied in 'values'.
                This is used to determine various default values, like the 
                attribute to pull data from, the title, and labels. Should be 
                'Count', 'FWHM', 'Mean', or 'Leakage' for best results.

        Keyword Arguments:
            values: array-like
                A array of numbers to make a histogram of. Required if anything
                other than 'Gain', 'Count', or 'FWHM' is supplied to 
                'value_label'.
            bins: int
                The number of bins in which to histogram the data. Passed 
                directly to plt.hist.
                (default: 70)
            hist_range: tuple(number, number)
                Indicated the range in which to bin data. Passed directly to
                plt.hist. If None, it is set to (0, 4) for gain-corrected data
                and to (0, 150) otherwise.
                (default: None)
            title: str
                The figure title. If 'auto', a title is generated using the
                'title' method. If an empty string is passed, no title
                is shown.
                (default: '')
            text_pos: str
                Indicates where information about mean and standard deviation
                appears on the plot. If 'right', appears in upper right. If 
                'left', appears in upper left.
                (default: 'right')
            save_plot: bool
                If True, saves the plot to 'save_dir'.
            plot_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'save_dir' attribute. If an empty string,
                will default to the attribute 'save_dir'.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            plot_subdir: str
                A path to a sub-directory of 'save_dir' to which the file will
                be saved. Empty curly braces '{}' are formatted the same way
                as in 'save_dir'. 
                (default: '')
            plot_ext: str
                The file extension to the saved file.
                (default: '.pdf')
            etc: str
                A string appended to the filename (before the extension).
                (default: '')
        '''
        if save_plot:
            description = (value_label.lower() + '_hist').replace(' ', '_')
            save_path = self.construct_path('plot', ext=plot_ext, 
                description=description, save_dir=plot_dir, subdir=plot_subdir,
                etc=etc)

        # Constructing the plot title, if none supplied


        # Default labels
        if title == 'auto':
            title = self.title(f'{value_label} Histogram')
        text_units = ''
        axis_units = ''
        xlabel = value_label

        if 'count' in value_label.lower():
            if values is None: 
                values = self.count_map
            if title == 'auto':
                title = self.title('Count Histogram')
            xlabel = 'Counts'
            mean, stdv = get_mean_stdv(values, 0, value_label)

        elif 'fwhm' in value_label.lower():
            xlabel = 'FWHM'
            if values is None: 
                values = self._fwhm_map
            if title == 'auto':
                title = self.title('FWHM Histogram')

            # Setting some plot parameters and converting units based on  
            # whether the supplied data is gain-corrected.
            if self._gain_corrected:
                if hist_range is None:
                    hist_range = (0, 4)
                mean, stdv = get_mean_stdv(values, 0, value_label)
                mean *= 1000
                stdv *= 1000
                text_units = ' eV'
                axis_units = ' (keV)'
            else:
                if hist_range is None:
                    hist_range = (0, 150)
                mean, stdv = get_mean_stdv(values, 0, value_label)
                text_units = ' channels'
                axis_units = ' (channels)'

        elif 'mean' in value_label.lower():
            xlabel = 'Mean'
            if values is None: 
                values = self._mean_map
            if title == 'auto':
                title = self.title('Mean Histogram')

            # Setting some plot parameters and converting units based on  
            # whether the supplied data is gain-corrected.
            if self._gain_corrected:
                mean, stdv = get_mean_stdv(values, 0, value_label)
                mean *= 1000
                stdv *= 1000
                text_units = ' eV'
                axis_units = ' (keV)'
            else:
                mean, stdv = get_mean_stdv(values, 0, value_label)
                text_units = ' channels'
                axis_units = ' (channels)'

        elif 'leak' in value_label.lower():
            if values is None:
                raise ValueError('Must manually supply data for leakage '
                    + 'current.')
            if title == 'auto':
                title = self.title('Leakage Current Histogram')

            xlabel = 'Leakage Current'
            mean, stdv = get_mean_stdv(values, 2, value_label)
            text_units = ' pA'
            axis_units = ' (pA)'

        else:
            if 'xlabel' in kwargs:
                xlabel = kwargs['xlabel']
            if 'text_units' in kwargs:
                text_units = kwargs['text_units']
            if 'axis_units' in kwargs:
                axis_units = kwargs['axis_units']
            mean, stdv = get_mean_stdv(values, 3, value_label)

        values = values.flatten()

        # Make the plot
        plt.figure()
        ax = plt.axes() # need axes object for text positioning
        try:
            plt.hist(values, bins=bins, range=hist_range, 
                histtype='stepfilled')
        except ValueError as ve:
            print(ve, "The 'plot_pixel_hist' method will exit wihtout "
                "raising an exception.")
            return

        # Setting text position based on user input. This will display the mean
        # and standard deviation of the fwhm data.
        if text_pos == 'right':
            left_side = 0.5
        elif text_pos == 'left':
            left_side = 0.05
        else:
            raise ValueError("'text_pos' can be either 'right' or 'left'. "
                + f"Instead {text_pos} was passed.")

        plt.text(left_side, 0.9, f'Mean = {mean:.2f}{text_units}',
            fontsize=14, transform=ax.transAxes)
        plt.text(left_side, 0.8, f'1-Sigma = {stdv:.2f}{text_units}',
            fontsize=14, transform=ax.transAxes)

        plt.xlabel(f'{xlabel}{axis_units}')
        plt.ylabel('Pixels') 
        plt.title(title)

        if save_plot:
            plt.savefig(save_path)


    def plot_pixel_map(self, value_label, values=None, cmap_name='inferno',  
        cb_label='', vmin=None, vmax=None, title='', save_plot=True, 
        plot_ext='.pdf', plot_dir='', plot_subdir='', etc=''):
        '''
        Construct a heatmap of counts across the detector using matplotlib. If
        data is not supplied explicitly with 'values', an attribute chosen 
        based on 'value_label' will be used. For example, 

        if value_label == 'Gain':
            values = self.gain

        Arguments:
            value_label: str
                A short label denoting what data is supplied in 'values'.
                The strings 'Gain', 'Count', 'FWHM', 'Mean', and 'Leakage', if 
                supplied, will trigger some presets regarding file name, plot 
                title, and plot label formatting. If 'Gain', 'Count', or 
                'FWHM' are supplied, 'values' will automatically be set to
                the value in the appropriate processed data attribute.

        Keyword Arguments:
            values: 2D array-like
                A array of numbers to make a heat map of. Required if anything
                other than 'Gain', 'Count', 'Mean', or 'FWHM' is supplied to 
                'value_label'.
            cmap_name: str
                The name of a matplotlib colormap. Passed to 'mpl.cm.get_cmap'.
                (default: 'inferno')
            cb_label: str
                This string becomes the color bar label. If the empty string,
                the color bar label is chosen based on 'value_label'.
                (default: '')
            vmin: float
                Passed directly to plt.imshow.
                (default: None)
            vmax: float
                Passed directly to plt.imshow.
                (default: None)
            title: str
                The figure title. If 'auto', a title is generated using the
                'title' method. If an empty string is passed, no title
                is shown.
                (default: '')
            save_plot: bool
                If True, saves the plot to a file.
            plot_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'save_dir' attribute. If an empty string,
                will default to the attribute 'save_dir'.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            plot_subdir: str
                A path to a sub-directory of 'save_dir' to which the file will
                be saved. Empty curly braces '{}' are formatted the same way
                as in 'save_dir'. 
                (default: '')
            etc: str
                A string appended to the filename (before the extension).
                (default: '')
        '''
        # Generate a save path, if needed.
        if save_plot:
            description = (value_label.lower() + '_map').replace(' ', '_')
            save_path = self.construct_path('plot', ext=plot_ext, 
                description=description, save_dir=plot_dir, subdir=plot_subdir,
                etc=etc)


        # Set the color bar label and 'values', if not supplied
        if 'gain' in value_label.lower():
            if not cb_label: 
                cb_label = 'Gain (eV/channel)'
            if values is None: 
                values = self.gain * 1000
            if title == 'auto':
                title = self.title('Gain Map')

        elif 'count' in value_label.lower():
            if not cb_label: 
                cb_label = 'Counts'
            if values is None: 
                values = self.count_map
            if title == 'auto':
                title = self.title('Count Map')

        elif 'fwhm' in value_label.lower():
            if not cb_label: 
                if self._gain_corrected:
                    cb_label = 'FWHM (keV)'
                else:
                    cb_label = 'FWHM (channels)'
            if values is None: 
                values = self._fwhm_map
            if title == 'auto':
                title = self.title('FWHM Map')

        elif 'mean' in value_label.lower():
            if not cb_label: 
                if self._gain_corrected:
                    cb_label = 'Mean (keV)'
                else:
                    cb_label = 'Mean (channels)'
            if values is None: 
                values = self._mean_map
            if title == 'auto':
                title = self.title('Mean Map')

        elif 'leak' in value_label.lower():
            if not cb_label:
                cb_label = 'Leakage Current (pA)'
            if values is None:
                raise ValueError('Must manually supply data for leakage '
                    + 'current.')
            if title == 'auto':
                title = self.title('Leakage Current Map')
                
        else: 
            # Setting the colorbar label, in none supplied
            if not cb_label: 
                cb_label = value_label
            # Constructing the plot title, if none supplied
            if title == 'auto':
                title = self.title(f'{value_label} Map')

        # Formatting the figure
        fig = plt.figure()
        cmap = matplotlib.cm.get_cmap(cmap_name)
        cmap.set_bad(color='gray')

        # The 'extent' kwarg is necessary to make axes flush to the image.
        extent = (self._start_col, self._end_col, 
            self._end_row, self._start_row)
        plt.imshow(values, vmin=vmin, vmax=vmax, extent=extent,
            origin='upper', cmap=cmap)

        c = plt.colorbar()
        c.set_label(cb_label, labelpad=10)

        # Making the axis ticks line up nicely with the detector edge.
        if self.full_detector and self._det_shape == (32, 32):
            ticks = np.arange(0, 36, 8)
            plt.xticks(ticks)
            plt.yticks(ticks)

        plt.gca().xaxis.tick_top()

        plt.title(title)

        if save_plot:
            plt.savefig(save_path)
