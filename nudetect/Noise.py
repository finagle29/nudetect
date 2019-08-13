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


class Noise(Experiment):
    '''
    A class containing important experiment parameters with methods to supply
    data analysis functions for noise data.

    Public attributes:
        raw_data_path: str
            A path to the noise data.
        detector: str
            The detector ID.
        voltage: str:
            The bias voltage in Volts.
        temp: str
            The temperature in degrees Celsius.
        pos: int
            The detector position.
        data_dir: str
            The default directory to which processed data files are saved.
            If supplied, this overrides the 'save_dir' kwarg, and uses the
            same formatting. If an empty string, defaults to 'save_dir'.
            (default: '')
        plot_dir: str
            The default directory to which plot files are saved. If 
            supplied, this overrides the 'save_dir' kwarg, and uses the
            same formatting. If an empty string, defaults to 'save_dir'.
            (default: '')
        save_dir: str
            A default directory to save file outputs to from this 
            instance's  methods. Method arguments let one choose a 
            subdirectory of this path, or override it altogether.

            If the string passed to 'save_dir' has an empty pair of curly 
            braces '{}', they will be replaced by the detector ID 
            'self.detector'. For example, if self.detector == 'H100' and 
            save_dir == 'figures/{}/pixels', then the directory that 
            'save_path' points to is 'figures/H100/pixels'.
            (default: '')

        etc: str
            Other important information to append to created files's names.
        gain: 32 x 32 numpy.ndarray
            Pixel-by-pixel gain data for the detector. This can be supplied
            after initialization though the 'gain' attribute. Do not supply
            a dummy value here if no gain is available. The methods of this
            class take care of that.
        count_map: 2D numpy.ndarray
            A 32 x 32 array with the number of events collected during the 
            noise test at each corresponding pixel.
            (initialized to None)

    Private attributes:
        _fwhm_map: 2D numpy.ndarray
            An array with the fwhm of the gaussian fit to the noise
            data collected at the corresponding pixel. Axes below:
                axis 0: row or y-coordinate
                axis 1: column or x-coordinate
            For example, to access the fwhm of the spectrum of the pixel
            at row 3 and column 4,
                >>> fwhm_map = get_fwhm_map()
                >>> fwhm_map[3, 4]
            (initialized to None)

        _mean_map: 2D numpy.ndarray
            An array with the mean of the gaussian fit to the noise
            data collected at the corresponding pixel. Axes below:
                axis 0: row or y-coordinate
                axis 1: column or x-coordinate
            For example, to access the mean of the spectrum of the pixel
            at row 3 and column 4,
                >>> fwhm_map = get_mean_map()
                >>> fwhm_map[3, 4]
            (initialized to None)

        _fwhm_maps: 3D numpy.ndarray
            An array with the fwhm of the gaussian fit to the noise
            data collected at the corresponding pixel. Axes below:
                axis 0: starting capacitor
                axis 1: row or y-coordinate
                axis 2: column or x-coordinate
            For example, to access the fwhm of the spectrum of the pixel
            at row 3 and column 4, and starting capacitor 0,
                >>> fwhm_map = get_fwhm_map()
                >>> fwhm_map[0, 3, 4]
            (initialized to None)

        _mean_maps: 3D numpy.ndarray
            An array with the mean of the gaussian fit to the noise
            data collected at the corresponding pixel. Axes below:
                axis 0: starting capacitor
                axis 1: row or y-coordinate
                axis 2: column or x-coordinate
            For example, to access the mean of the spectrum of the pixel
            at row 3 and column 4, and starting capacitor 0,
                >>> mean_map = get_fwhm_map()
                >>> mean_map[0, 3, 4]
            (initialized to None)

        quick_fit_data: pandas.DataFrame
            A MultiIndexed DataFrame containing the mean and FWHM of each
            Gaussian fit and their errors. Intended to help spot when
            fitting has gone poorly. 

            Columns:
                'mean', 'mean error', 'fwhm', 'fwhm error'
            Index:
                ('pixel row', 'pixel col')

            For example, to get the mean of the gaussian fit at the pixel 
            in row 10, column 11 (i.e., RAWY = 10, RAWX = 11), 
            one would type:

            >>> fit_data.loc[(10, 11), 'mean']

            For more, check out the pandas documentation for MultiIndexing
            at http://pandas.pydata.org/pandas-docs/stable/advanced.html
            and look at the MultiIndex heirarchy itself using

            >>> fit_data.index

        full_fit_data: pandas.DataFrame
            A MultiIndexed DataFrame containing the mean and FWHM of each
            Gaussian fit and their errors. Intended to help spot when
            fitting has gone poorly. 

            Columns:
                'mean', 'mean error', 'fwhm', 'fwhm error'
            Index:
                ('start cap', 'pixel row', 'pixel col')

            For example, to get the mean of the gaussian fit to the 4th 
            starting capactior at the pixel in row 10, column 11 
            (i.e., RAWY = 10, RAWX = 11), one would type:

            >>> fit_data.loc[(4, 10, 11), 'mean']

            All columns with data for starting capacitor 4 only would be:

            >>> fit_data.loc[4]

            All columns with data for the pixel at row 10, column 11:

            >>> fit_data.xs((10, 11), level=('pixel row', 'pixel col'))

            For more, check out the pandas documentation for MultiIndexing
            at http://pandas.pydata.org/pandas-docs/stable/advanced.html
            and look at the MultiIndex heirarchy itself using

            >>> fit_data.index0

        _gain_corrected: bool
            If True, indicates that all processed data attributes have been
            corrected for gain. If False, then none of them have.
            (initialized to None)
    '''
    def __init__(self, raw_data_path, detector, voltage, temp, pos=0, 
        gain=None, data_dir='', plot_dir='', save_dir='', etc=''):
        '''
        Initialized an instance of the 'Noise' class.

        Arguments:
            raw_data_path: str
                A path to the noise data.
            detector: str
                The detector ID.
            voltage: str:
                The bias voltage in Volts.
            temp: str
                The temperature in degrees Celsius.

        Keyword arguments:
            pos: int
                The detector position.
                (default: 0)
            gain: 32 x 32 numpy.ndarray
                Pixel-by-pixel gain data for the detector. This can be supplied
                after initialization though the 'gain' attribute. Do not supply
                a dummy value here if no gain is available. The methods of this
                class take care of that.
                (default: None)
            data_dir: str
                The default directory to which processed data files are saved.
                If supplied, this overrides the 'save_dir' kwarg, and uses the
                same formatting. If an empty string, defaults to 'save_dir'.
                (default: '')
            plot_dir: str
                The default directory to which plot files are saved. If 
                supplied, this overrides the 'save_dir' kwarg, and uses the
                same formatting. If an empty string, defaults to 'save_dir'.
                (default: '')
            save_dir: str
                A default directory to save file outputs to from this 
                instance's  methods. Method arguments let one choose a 
                subdirectory of this path, or override it altogether.

                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            etc: str
                Other important information to append to created files's names.
        '''
        
        temp = str(temp)
        voltage = str(voltage)

        # Remove any unit symbols from voltage and temperature
        temp = temp.translate(self.numericize)
        voltage = voltage.translate(self.numericize)

        # If gain is supplied, make sure it's a 32 x 32 array
        if gain is not None \
            and gain.shape != self._det_shape \
            and gain.shape != self._full_det_shape:

            raise ValueError("The array 'gain' should either have the shape "
                f"{self._det_shape} or {self._full_det_shape}. Instead, an "
                f"array of shape {gain.shape} was passed.")

        # Initialize '_gain_corrected' to None. This will be set to True or 
        # False when 'noise_map' is called, denoting whether the attribute
        # 'fwhm_map' is corrected for gain.
        self._gain_corrected = None
        self.gain = gain
        self._fwhm_map = None
        self.count_map = None

        self.raw_data = None
        self.raw_data_path = raw_data_path
        self.detector = detector
        self.temp = temp
        self.voltage = voltage
        self.pos = int(pos)
        self.etc = etc

        self._set_save_dir(save_dir)
        self._set_save_dir(plot_dir, save_type='plot')
        self._set_save_dir(data_dir, save_type='data')

    #
    # Methods for accessing private attributes
    #

    def load_raw_data(self):
        '''Loads raw data from FITS file into attributes of this instance.'''
        self.raw_data_1d, self.raw_data_2d = fits_to_df(self.raw_data_path,
            colnames={'RAWX', 'RAWY', 'PH_RAW', 'UP', 'S_CAP'},
            pos=self.pos)


    def load_fwhm_map(self, fwhm_map, gain_corrected=None):
        '''
        Sets the '_fwhm_map' and '_gain_corrected' attributes of this 
        instance based on a path to the fwhm map data file.

        Arguments:
            fwhm_map: str
                A path to an ascii file containing FWHM map data.

        Keyword Arguments:
            gain_corrected: bool
                If True, indicated that the supplied FWHM data was gain 
                corrected and is in units of keV. If False, then the data
                should still be in units of channels. If None, then the 
                value will be determined by the path (specifically
                whether the phrase 'nogain' is in the file name).
        '''
        # If 'gain_corrected' specified, set its value based on the 
        # path 'fwhm_map'.
        if gain_corrected is None:
            gain_corrected = 'nogain' not in fwhm_map
            if 'gain' not in fwhm_map:
                raise Exception('Could not determine from the file name '
                    + 'whether the FWHM map was corrected for gain. Please'
                    + "enter an appropriate value for 'gain_corrected'.")

        if type(gain_corrected) != bool:
            raise TypeError("'gain_corrected must be type 'bool'. Type "
                + f"{type(gain_corrected)} was given.")

        # Make sure we don't mix processed data that is gain corrected with 
        # processed data that isn't. This way, the _gain_corrected attribute
        # can represent the entire instance's processed data at once, and 
        # generally makes things simpler.
        if self._gain_corrected == False and gain_corrected == True:
            raise ValueError('It looks like the data being loaded is gain '
                "corrected, but there is data stored in this instance that "
                "isn't. Mixing the two is not allowed.")
        elif self._gain_corrected == True and gain_corrected == False:
            raise ValueError('It looks like the data being loaded is not gain '
                "corrected, but there is data stored in this instance that "
                "is. Mixing the two is not allowed.")
 
        self._gain_corrected = gain_corrected

        fwhm_map = np.loadtxt(fwhm_map)

        if fwhm_map.shape != self._det_shape and \
           fwhm_map.shape != self._full_det_shape:

            raise ValueError("The array 'fwhm_map' should either have the  "
                f"shape {self._det_shape} or {self._full_det_shape}. Instead, "
                f"an array of shape {fwhm_map.shape} was passed.")

        # Mask large values, taking into account whether fwhm is in units
        # of channels or of keV.
        if gain_corrected:
            fwhm_map = np.ma.masked_where(fwhm_map > 5, fwhm_map)
        else:
            fwhm_map = np.ma.masked_where(fwhm_map > 400, fwhm_map)

        self._fwhm_map = fwhm_map


    def set_fwhm_map(self, fwhm_map, gain_corrected):
        '''
        Sets the '_fwhm_map' and '_gain_corrected' attributes of this 
        instance using a numpy.ndarray object containing the data and 
        user input for whether it is gain corrected.

        Arguments:
            fwhm_map: numpy.ndarray
                A 2D numpy array containing FWHM map data.
            gain_corrected: bool
                If True, indicated that the supplied FWHM data was gain 
                corrected and is in units of keV. If False, then the data
                should still be in units of channels.
        '''
        if type(gain_corrected) != bool:
            raise TypeError("'gain_corrected must be type 'bool'. Type "
                + f"{type(gain_corrected)} was given.")

        self._gain_corrected = gain_corrected

        if fwhm_map.shape != self._det_shape and \
           fwhm_map.shape != self._full_det_shape:

            raise ValueError("The array 'fwhm_map' should either have the  "
                f"shape {self._det_shape} or {self._full_det_shape}. Instead, "
                f"an array of shape {fwhm_map.shape} was passed.")

        # Mask large values, taking into account whether fwhm is in units
        # of channels or of keV.
        if gain_corrected:
            fwhm_map = np.ma.masked_where(fwhm_map > 5, fwhm_map)
        else:
            fwhm_map = np.ma.masked_where(fwhm_map > 400, fwhm_map)

        self._fwhm_map = fwhm_map


    def get_fwhm_map(self):
        '''Returns a copy of the private attribute '_fwhm_map'.'''
        return self._fwhm_map


    def load_mean_map(self, mean_map, gain_corrected=None):
        '''
        Sets the '_mean_map' and '_gain_corrected' attributes of this 
        instance based on a path to the mean map data file.

        Arguments:
            mean_map: str
                A path to an ascii file containing mean map data.

        Keyword Arguments:
            gain_corrected: bool
                If True, indicated that the supplied mean data was gain 
                corrected and is in units of keV. If False, then the data
                should still be in units of channels. If None, then the 
                value will be determined by the path (specifically
                whether the phrase 'nogain' is in the file name).
        '''
        # If 'gain_corrected' specified, set its value based on the 
        # path 'mean_map'.
        if gain_corrected is None:
            gain_corrected = 'nogain' not in mean_map
            if 'gain' not in mean_map:
                raise Exception('Could not determine from the file name '
                    + 'whether the mean map was corrected for gain. Please'
                    + "enter an appropriate value for 'gain_corrected'.")

        if type(gain_corrected) != bool:
            raise TypeError("'gain_corrected must be type 'bool'. Type "
                + f"{type(gain_corrected)} was given.")
 
        self._gain_corrected = gain_corrected

        mean_map = np.loadtxt(mean_map)

        if mean_map.shape != self._det_shape and \
           mean_map.shape != self._full_det_shape:

            raise ValueError("The array 'mean_map' should either have the  "
                f"shape {self._det_shape} or {self._full_det_shape}. Instead, "
                f"an array of shape {mean_map.shape} was passed.")

        # Mask large values, taking into account whether mean is in units
        # of channels or of keV.
        if gain_corrected:
            mean_map = np.ma.masked_where(mean_map > 5, mean_map)
        else:
            mean_map = np.ma.masked_where(mean_map > 400, mean_map)

        self._mean_map = mean_map


    def set_mean_map(self, mean_map, gain_corrected):
        '''
        Sets the '_mean_map' and '_gain_corrected' attributes of this 
        instance using a numpy.ndarray object containing the data and 
        user input for whether it is gain corrected.

        Arguments:
            mean_map: numpy.ndarray
                A 2D numpy array containing mean map data.
            gain_corrected: bool
                If True, indicated that the supplied mean data was gain 
                corrected and is in units of keV. If False, then the data
                should still be in units of channels.
        '''
        if type(gain_corrected) != bool:
            raise TypeError("'gain_corrected must be type 'bool'. Type "
                + f"{type(gain_corrected)} was given.")

        self._gain_corrected = gain_corrected

        if mean_map.shape != self._det_shape and \
           mean_map.shape != self._full_det_shape:

            raise ValueError("The array 'mean_map' should either have the  "
                f"shape {self._det_shape} or {self._full_det_shape}. Instead, "
                f"an array of shape {mean_map.shape} was passed.")

        # Mask large values, taking into account whether mean is in units
        # of channels or of keV.
        if gain_corrected:
            mean_map = np.ma.masked_where(mean_map > 5, mean_map)
        else:
            mean_map = np.ma.masked_where(mean_map > 400, mean_map)

        self._mean_map = mean_map


    def get_mean_map(self):
        '''Returns a copy of the private attribute '_mean_map'.'''
        return self._mean_map


    def get_gain_corrected(self):
        '''Returns a copy of the private attribute '_gain_corrected'.'''
        return self._gain_corrected


    #
    # Heavy lifting data analysis methods: 'gen_quick_noise' and 
    # 'gen_full_noise'.
    #

    def gen_quick_noise(self, gain=None, save_plot=True, plot_dir='', 
        plot_subdir='', plot_ext='.pdf', save_data=True, data_dir='', 
        data_subdir='', data_ext='.txt'):
        '''
        For each combination of pixel coordinates and starting capacitor,
        plots a spectrum of the noise and fits it with a Gaussian. The 
        mean and FWHM of this Gaussian are recorded. A count map is also 
        generated.

        This method can be called with or without gain correction. It will
        try to find gain data stored in the 'Noise' instance if no gain
        data is supplied via the 'gain' parameter. If no gain data is found,
        it will give outputs in units of channels.

        Keyword Arguments:
            gain: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents its  
                respective pixel's gain, where channels * gain = energy. If 
                None, defaults to the array in 'self.gain'.
                (default: None)
            save_plot: bool
                If true, plots an energy spectrum for each pixel and saves
                the figure.
                (default: True)
            plot_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'save_dir' attribute. If an empty string,
                will default to the attribute 'save_dir'.
                If the string passed to 'plot_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                plot_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            plot_subdir: str
                A path to a sub-directory of 'plot_dir' to which the file will
                be saved. Empty curly braces '{}' are formatted the same way
                as in 'plot_dir'. 
                (default: '')
            plot_ext: str
                The file name extension for the plot file.
                (default: '.pdf')  
            save_data: bool 
                If True, saves gain data as an ascii file.
                (default: True)
            data_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'save_dir' attribute. If an empty string,
                will default to the attribute 'save_dir'.
                If the string passed to 'data_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                data_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            data_subdir: str
                A path to a sub-directory of 'data_dir' to which the file will
                be saved. Empty curly braces '{}' are formatted the same way
                as in 'data_dir'. 
                (default: '')
            data_ext: str
                The file name extension for the noise map data files. 
                (default: '.txt')

        Return:
            fit_data: pandas.DataFrame
                A MultiIndexed DataFrame containing the mean and FWHM of each
                Gaussian fit and their errors. Intended to help spot when
                fitting has gone poorly. 

                Columns:
                    'mean', 'mean error', 'fwhm', 'fwhm error'
                Index:
                    ('pixel row', 'pixel col')

                For example, to get the mean of the gaussian fit at the pixel 
                in row 10, column 11 (i.e., RAWY = 10, RAWX = 11), 
                one would type:

                >>> fit_data.loc[(10, 11), 'mean']

                For more, check out the pandas documentation for MultiIndexing
                at http://pandas.pydata.org/pandas-docs/stable/advanced.html
                and look at the MultiIndex heirarchy itself using

                >>> fit_data.index
        '''
        # 'etc' and 'etc_plot' will be appended to file names, denoting  
        # whether data/plots were gain-corrected.
        gain_bool = (self.gain is not None) or (gain is not None)
        if gain_bool:
            etc = 'gain'
        else:
            etc = 'nogain'

        # Check if gain data was supplied for the whole detector or just
        # the region being analyzed, if necessary.

        # 'etc_plot' will be formatted to have pixel coordinates, since a
        # spectrum is plotted for each pixel.
        etc_plot = etc + '_x{}_y{}'

        # Generating the save paths, if needed.
        if save_data:
            fwhm_path = self.construct_path('data', ext=data_ext, 
                save_dir=data_dir, subdir=data_subdir, 
                description='quick_fwhm_data',
                etc=etc)
            mean_path = self.construct_path('data', ext=data_ext, 
                save_dir=data_dir, subdir=data_subdir, 
                description='quick_mean_data',
                etc=etc)
            count_path = self.construct_path('data', ext=data_ext, 
                save_dir=data_dir, subdir=data_subdir, 
                description='quick_count_data', etc=etc)
            fit_data_path = self.construct_path('data', ext='.csv', 
                save_dir=data_dir, subdir=data_subdir, 
                description='quick_fit_data', etc=etc)

        if save_plot:
            plot_path = self.construct_path('plot', save_dir=plot_dir, 
                etc=etc_plot, subdir=plot_subdir, description='pix_spectrum', 
                ext=plot_ext)

        if not gain_bool:
            gain = np.ones(self._det_shape)
        # If gain data is not passed directly as a parameter, but is an 
        # attribute of this instance, use the attribute's gain data.
        elif gain is None:
            gain = self.gain

        # If we are not analyzing the full detector but are given gain data
        # for the full detector, slice out only the necessary gain data.
        if not self.full_detector and gain.shape == self._full_det_shape:
            gain = gain[self._row_slice, self._col_slice]

        maxchannel = 1000
        bins = np.arange(-maxchannel, maxchannel)

        # Shape of the arrays of processed data. For NuSTAR style detectors, 
        # should be (32, 32).
        output_shape = (self._num_rows, self._num_cols)

        # Below, 'np.full' with 'nan' is used so that fitting parameters are
        # left as nan if fitting fails.

        # Initilaizing pixel map of FWHM values
        fwhm_map = np.full(output_shape, np.nan)
        # Initilaizing pixel map of centroid values
        mean_map = np.full(output_shape, np.nan)
        # Initializing pixel map of counts
        count_map = np.empty(output_shape)

        # Initializing a DataFrame to store information about how the 
        # fitting went for each pixel.
        index = pd.MultiIndex.from_product([self._row_iter, self._col_iter],
            names=['pixel row', 'pixel col'])

        columns = ['mean', 'mean error', 'fwhm', 'fwhm error']

        fit_data = pd.DataFrame(
            np.empty((np.prod(output_shape), len(columns))),
            columns=columns, index=index)

        # Generate 'chan_map', a nested list representing an array 
        # of lists, each of which contains all the trigger readings for 
        # its corresponding pixel. A buffer is added on two of the sides 
        # because the raw data contains dummy values representing the 
        # imaginary pixels in the 3 x 3 grid surroudning a pixel on a
        # detector edge whose readout was triggered.
        chan_map = [[[] 
            for col in range(self._num_cols + 2)] 
            for row in range(self._num_rows + 2)]

        ph_raw = self.raw_data_2d['PH_RAW']

        # Iterating through pixels
        for col in self._col_iter:
            col_mask = self.raw_data_1d.loc[:, 'RAWX'] == col
            for row in self._row_iter:
                row_mask = self.raw_data_1d.loc[:, 'RAWY'] == row
                # Storing all readings for the current pixel in 'pulses'.
                pixel_mask = (col_mask) & (row_mask)
                pulses = ph_raw.loc[pixel_mask]
                for i in pulses.index:
                    # If this pulse was triggered by the experiment (by a 
                    # 'micro pulse'), then add the pulse data for the 3 x 3
                    # pixel grid centered on the triggered pixel to the 
                    # corresponding indices of 'chan_map'.
                    if self.raw_data_1d.at[i, 'UP']:
                        for j in range(9):
                            mapcol = (col - self._start_col) + (j % 3) - 1
                            maprow = (row - self._start_row) + (j // 3) - 1
                            chan_map[maprow][mapcol].append(pulses.at[i, j])

        del pulses, pixel_mask, col_mask, row_mask

        # Generate a count map of micropulse-triggered events from 
        # 'chan_map'.
        count_map = np.array([[len(chan_map[row][col]) 
            for col in range(1, self._num_cols + 1)] 
            for row in range(1, self._num_rows + 1)])
       
        # Generate a fwhm map of noise, and plot the gaussian fit to each 
        # pixel's spectrum.
        
        # Iterate through elements of chan_map
        for row in self._row_iter:
            for col in self._col_iter:
                maprow = row - self._start_row
                mapcol = col - self._start_col
                # If there were events at this pixel, bin them by channel
                if chan_map[maprow][mapcol]:
                    # Binning events by channel
                    spectrum, edges = np.histogram(chan_map[maprow][mapcol], 
                        bins=bins, range=(-maxchannel, maxchannel))

                    # Fitting the noise peak at/near zero channels
                    fit_channels = edges[:-1]
                    g_init = models.Gaussian1D(amplitude=np.max(spectrum), 
                        mean=0, stddev=75)
                    fit_g = fitting.LevMarLSQFitter()
                    g = fit_g(g_init, fit_channels, spectrum)

                    # Recording the gain-corrected FWHM and mean data
                    # for this pixel in the corresponding arrays.
                    fwhm_map[maprow, mapcol] = np.multiply(
                        g.fwhm, gain[maprow, mapcol])

                    mean_map[maprow, mapcol] = np.multiply(
                        g.mean, gain[maprow, mapcol])

                    # If the fit succeeded, record some of the fit information
                    # in the 'fit_data' DataFrame.
                    if fit_g.fit_info['param_cov'] is not None:
                        # 1 stardard deviation error for Gaussian parameters.
                        sigma_err = np.diag(fit_g.fit_info['param_cov'])[2]
                        fwhm_err = 2 * np.sqrt(2 * np.log(2)) * sigma_err
                        mean_err = np.diag(fit_g.fit_info['param_cov'])[1]

                        # Populating a row of fit_data with fit information
                        df_row = [g.mean.value, mean_err, g.fwhm, fwhm_err]
                        fit_data.loc[(row, col)] = df_row
                    else:
                        df_row = [g.mean.value, np.nan, g.fwhm, np.nan]
                        fit_data.loc[(row, col)] = df_row

                    if save_plot:
                        plt.hist(np.multiply(
                                chan_map[maprow][mapcol], 
                                gain[maprow, mapcol]),
                            bins=np.multiply(bins, gain[maprow, mapcol]), 
                            range=(-maxchannel * gain[maprow, mapcol], 
                                    maxchannel * gain[maprow, mapcol]), 
                            histtype='stepfilled')

                        plt.plot(np.multiply(
                            fit_channels, gain[maprow, mapcol]), 
                            g(fit_channels))

                        plt.ylabel('Counts')
                        if gain_bool:
                            plt.xlabel('Energy (keV)')
                        else:
                            plt.xlabel('Channel')

                        plt.tight_layout()
                        plt.savefig(plot_path.format(row, col))
                        plt.close()
        

        # Mask large values, taking into account whether fwhm is in units
        # of channels or of keV.
        if gain_bool:
            fwhm_map = np.ma.masked_where(fwhm_map > 5, fwhm_map)
        else:
            fwhm_map = np.ma.masked_where(fwhm_map > 400, fwhm_map)

        self._fwhm_map = fwhm_map
        self._mean_map = mean_map
        self.count_map = count_map
        self._quick_fit_data = fit_data
        self._chan_map = chan_map
        # Set '_gain_corrected' way down here to make sure the maps of 
        # FWHM and mean were successfully generated.
        self._gain_corrected = gain_bool

        if save_data:
            np.savetxt(fwhm_path, fwhm_map)
            np.savetxt(mean_path, mean_map)
            np.savetxt(count_path, count_map)
            fit_data.to_csv(fit_data_path)

        return fit_data


    def gen_full_noise(self, gain=None, save_plot=False, plot_dir='', 
        plot_subdir='', plot_ext='.pdf', save_data=True, data_dir='', 
        data_subdir=''):
        '''
        For each combination of pixel coordinates and starting capacitor,
        plots a spectrum of the noise and fits it with a Gaussian. The 
        mean and FWHM of this Gaussian are recorded. A count map is also 
        generated.

        This method can be called with or without gain correction. It will
        try to find gain data stored in the 'Noise' instance if no gain
        data is supplied via the 'gain' parameter. If no gain data is found,
        it will give outputs in units of channels.

        Keyword Arguments:
            gain: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents its  
                respective pixel's gain, where channels * gain = energy. If 
                None, defaults to the array in 'self.gain'.
                (default: None)
            save_plot: bool
                If true, plots and energy spectrum for each pixel and saves
                the figure.
                (default: False)
            plot_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'save_dir' attribute. If an empty string,
                will default to the attribute 'save_dir'.
                If the string passed to 'plot_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                plot_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            plot_subdir: str
                A path to a sub-directory of 'plot_dir' to which the file will
                be saved. Empty curly braces '{}' are formatted the same way
                as in 'plot_dir'. 
                (default: '')
            plot_ext: str
                The file name extension for the plot file.
                (default: '.pdf')  
            save_data: bool 
                If True, saves gain data as an ascii file.
                (default: True)
            data_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'save_dir' attribute. If an empty string,
                will default to the attribute 'save_dir'.
                If the string passed to 'data_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                data_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            data_subdir: str
                A path to a sub-directory of 'data_dir' to which the file will
                be saved. Empty curly braces '{}' are formatted the same way
                as in 'data_dir'. 
                (default: '')

        Return:
            fit_data: pandas.DataFrame
                A MultiIndexed DataFrame containing the mean and FWHM of each
                Gaussian fit and their errors. Intended to help spot when
                fitting has gone poorly. 

                Columns:
                    'mean', 'mean error', 'fwhm', 'fwhm error'
                Index:
                    ('start cap', 'pixel row', 'pixel col')

                For example, to get the mean of the gaussian fit to the 4th 
                starting capactior at the pixel in row 10, column 11 
                (i.e., RAWY = 10, RAWX = 11), one would type:

                >>> fit_data.loc[(4, 10, 11), 'mean']

                All columns with data for starting capacitor 4 only would be:

                >>> fit_data.loc[4]

                All columns with data for the pixel at row 10, column 11:

                >>> fit_data.xs((10, 11), level=('pixel row', 'pixel col'))

                For more, check out the pandas documentation for MultiIndexing
                at http://pandas.pydata.org/pandas-docs/stable/advanced.html
                and look at the MultiIndex heirarchy itself using

                >>> fit_data.index
        '''
        # 'etc' and 'etc_plot' will be appended to file names, denoting  
        # whether data/plots were gain-corrected.
        gain_bool = (self.gain is not None) or (gain is not None)
        if gain_bool:
            etc = 'gain'
        else:
            etc = 'nogain'

        # Check if gain data was supplied for the whole detector or just
        # the region being analyzed, if necessary.

        # 'etc_plot' will be formatted to have pixel coordinates, since a
        # spectrum is plotted for each pixel.
        etc_plot = etc + '_x{}_y{}_scap{}'

        # Generating the save paths, if needed.
        if save_data:
            fwhm_path = self.construct_path('data', ext='.npy', 
                save_dir=data_dir, subdir=data_subdir, 
                description='full_fwhm_data',
                etc=etc)
            mean_path = self.construct_path('data', ext='.npy', 
                save_dir=data_dir, subdir=data_subdir, 
                description='full_mean_data',
                etc=etc)
            count_path = self.construct_path('data', ext='.npy', 
                save_dir=data_dir, subdir=data_subdir, 
                description='full_count_data', etc=etc)
            fit_data_path = self.construct_path('data', ext='.csv', 
                save_dir=data_dir, subdir=data_subdir, 
                description='full_fit_data', etc=etc)

        if save_plot:
            plot_path = self.construct_path('plot', save_dir=plot_dir, 
                etc=etc_plot, subdir=plot_subdir, description='pix_spectrum', 
                ext=plot_ext)

        if not gain_bool:
            gain = np.ones(self._det_shape)
        # If gain data is not passed directly as a parameter, but is an 
        # attribute of this instance, use the attribute's gain data.
        elif gain is None:
            gain = self.gain

        # If we are not analyzing the full detector but are given gain data
        # for the full detector, slice out only the necessary gain data.
        if not self.full_detector and gain.shape == self._full_det_shape:
            gain = gain[self._row_slice, self._col_slice]

        maxchannel = 1000
        bins = np.arange(-maxchannel, maxchannel)

        # Shape of the arrays of processed data. For NuSTAR style detectors, 
        # should be (16, 32, 32) for number of sampling capacitors along the 
        # 0th axis and the detector dimensions along the 1st and 2nd axes.
        output_shape = (self.num_caps, self._num_rows, self._num_cols)

        # Below, 'np.full' with 'nan' is used so that fitting parameters are
        # left as nan if fitting fails.

        # Initilaizing map of FWHM values for the noise gaussian at each pixel 
        # and starting capacitor.
        fwhm_maps = np.full(output_shape, np.nan)
        # Initilaizing map of centroid values for the noise gaussian at each 
        # pixel and starting capacitor.
        mean_maps = np.full(output_shape, np.nan)
        # Initializing map of counts at each pixel and starting capacitor.
        count_maps = np.empty(output_shape)

        # Capacitor indices
        cap_inds = range(self.num_caps)
        # Pixel row indices
        row_inds = self._row_iter
        # Pixel column indices
        col_inds = self._col_iter

        chan_maps = [[[[]
            for col in range(self._num_cols + 2)]
            for row in range(self._num_rows + 2)]
            for cap in range(self.num_caps)]

        # Initializing a DataFrame to store information about how the 
        # fitting went for each pixel and starting capacitor.
        index = pd.MultiIndex.from_product([cap_inds, row_inds, col_inds],
            names=['start cap', 'pixel row', 'pixel col'])

        columns = ['mean', 'mean error', 'fwhm', 'fwhm error']

        fit_data = pd.DataFrame(
            np.empty((np.prod(output_shape), len(columns))),
            columns=columns, index=index)

        ph_raw = self.raw_data_2d['PH_RAW']

        # Iterating through starting capacitor values
        for start_cap in range(self.num_caps):
            start_cap_mask = self.raw_data_1d.loc[:, 'S_CAP'] == start_cap
            # Generate 'chan_map', a nested list representing an array 
            # of lists, each of which contains all the trigger readings for 
            # its corresponding pixel. A buffer is added on two of the sides 
            # because the raw data contains dummy values representing the 
            # imaginary pixels in the 3 x 3 grid surroudning a pixel on a
            # detector edge whose readout was triggered.
            chan_map = [[[] 
                for col in range(self._num_cols + 2)] 
                for row in range(self._num_rows + 2)]

            # Iterating through pixels
            for col in self._col_iter:
                col_mask = self.raw_data_1d.loc[:, 'RAWX'] == col
                for row in self._row_iter:
                    row_mask = self.raw_data_1d.loc[:, 'RAWY'] == row
                    # Storing all readings for the current pixel in 'pulses'.
                    mask = (col_mask) & (row_mask) & (start_cap_mask)
                    pulses = ph_raw.loc[mask]
                    for i in pulses.index:
                        # If this pulse was triggered by the experiment (by a 
                        # 'micro pulse'), then add the pulse data for the 3 x 3
                        # pixel grid centered on the triggered pixel to the 
                        # corresponding indices of 'chan_map'.
                        if self.raw_data_1d.at[i, 'UP']:
                            for j in range(9):
                                mapcol = (col - self._start_col) + (j % 3) - 1
                                maprow = (row - self._start_row) + (j // 3) - 1
                                chan_map[maprow][mapcol].append(
                                    pulses.at[i, j])

            del pulses, mask, row_mask, col_mask, start_cap_mask

            # Generate a count map of micropulse-triggered events from 
            # 'chan_map' and insert it into the appropriate slice of the 
            # 'count_maps' array
            count_map = np.array([[len(chan_map[row][col]) 
                for col in range(self._num_cols)] 
                for row in range(self._num_rows)])
            count_maps[start_cap] = count_map
            chan_maps[start_cap] = chan_map
           
            # Generate a fwhm map of noise, and plot the gaussian fit to each 
            # pixel's spectrum.
            
            # Iterate through pixels
            for row in self._row_iter:
                for col in self._col_iter:
                    maprow = row - self._start_row
                    mapcol = col - self._start_col
                    # If there were events at this pixel, bin them by channel
                    if chan_map[maprow][mapcol]:
                        # Binning events by channel
                        spectrum, edges = np.histogram(
                            chan_map[maprow][mapcol], 
                            bins=bins, range=(-maxchannel, maxchannel))

                        # Fitting the noise peak at/near zero channels
                        fit_channels = edges[:-1]
                        g_init = models.Gaussian1D(amplitude=np.max(spectrum), 
                            mean=0, stddev=75)
                        fit_g = fitting.LevMarLSQFitter()
                        g = fit_g(g_init, fit_channels, spectrum)

                        # Recording the gain-corrected FWHM and mean data
                        # for this pixel and starting capacitor in the
                        # corresponding arrays.
                        fwhm_maps[start_cap, maprow, mapcol] = np.multiply(
                            g.fwhm, gain[maprow, mapcol])

                        mean_maps[start_cap, maprow, mapcol] = np.multiply(
                            g.mean, gain[maprow, mapcol])

                        # If the fit succeeded, record some of the fit 
                        # information in the 'fit_data' DataFrame.
                        if fit_g.fit_info['param_cov'] is not None:
                            # 1 stardard deviation error for Gaussian 
                            # parameters.
                            sigma_err = np.diag(fit_g.fit_info['param_cov'])[2]
                            fwhm_err = 2 * np.sqrt(2 * np.log(2)) * sigma_err
                            mean_err = np.diag(fit_g.fit_info['param_cov'])[1]

                            # Populating a row of fit_data with fit information
                            df_row = [g.mean.value, mean_err, g.fwhm, fwhm_err]
                            fit_data.loc[(start_cap, row, col)] = df_row
                        else:
                            df_row = [g.mean.value, np.nan, g.fwhm, np.nan]
                            fit_data.loc[(start_cap, row, col)] = df_row

                        if save_plot:
                            plt.hist(np.multiply(
                                    chan_map[maprow][mapcol], 
                                    gain[maprow, mapcol]),
                                bins=np.multiply(bins, gain[maprow, mapcol]), 
                                range=(-maxchannel * gain[maprow, mapcol], 
                                        maxchannel * gain[maprow, mapcol]), 
                                histtype='stepfilled')

                            plt.plot(np.multiply(fit_channels, 
                                gain[maprow, mapcol]), g(fit_channels))

                            plt.ylabel('Counts')
                            if gain_bool:
                                plt.xlabel('Energy (keV)')
                            else:
                                plt.xlabel('Channel')

                            plt.tight_layout()
                            plt.savefig(plot_path.format(row, col, start_cap))
                            plt.close()
        

        # Mask large values, taking into account whether fwhm is in units
        # of channels or of keV.
        if gain_bool:
            fwhm_maps = np.ma.masked_where(fwhm_maps > 5, fwhm_maps)
        else:
            fwhm_maps = np.ma.masked_where(fwhm_maps > 400, fwhm_maps)

        self._fwhm_maps = fwhm_maps
        self._mean_maps = mean_maps
        self.count_maps = count_maps
        self._full_fit_data = fit_data
        self._chan_maps = chan_maps
        # Set '_gain_corrected' way down here to make sure the maps of 
        # FWHM and mean were successfully generated.
        self._gain_corrected = gain_bool

        if save_data:
            # We can't save the array mask because the feature isn't 
            # implemented in numpy's 'save' function yet. If you really
            # want to save the mask, you can pickle it, though I've read
            # that lead to larger file sizes.
            np.save(fwhm_path, fwhm_maps.data)
            np.save(mean_path, mean_maps.data)
            np.save(count_path, count_maps.data)
            fit_data.to_csv(fit_data_path)

        return fit_data


    def gain_correct_fwhm(self, gain=None, save_data=True, data_dir='', 
        data_subdir='', data_ext='.txt'):
        '''
        Apply gain corrections to processed noise data, if generated without 
        gain correction.
        '''
        pass
