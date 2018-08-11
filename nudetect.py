'''
Module with functions for analysis of gamma flood data. If running this 
file as a script, it will run a complete data analysis pipeline on the
supplied gamma flood data.
'''

# Packages for making life easier
import os.path
import string
import argparse

# Data analysis packages
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import models, fitting
import astropy.io.ascii as asciio

# Plotting packages
import matplotlib.pyplot as plt
import matplotlib as mpl


class Line:
    '''
    A class for spectral lines. The infomation in each instance will help
    supply parameters for fitting peaks. New instances can be created on 
    the fly and instances defined in the source code can be temporarily
    modified as needed. The data instances contain will usually be accessed
    by the 'line' method of the 'Experiment' class.

    Attributes:
        source: str
            The name of the radioactive source producing the line. This is
            case-sensitive, and should be formatted as in the example below.
            E.g., 'Am241'
        energy: float
            The energy of the line in keV.
        chan_low: int
            When searching the channel spectrum for peaks, channels below
            'chan_low' will be ignored.
        chan_high: int
            When searching the channel spectrum for peaks, channels above
            'chan_high' will be ignored.
    '''
    # A dict to contain all instances of 'Line'
    lines = {}

    def __init__(self, source, energy, chan_low, chan_high):
        self.source = source
        self.energy = energy
        self.chan_low = chan_low
        self.chan_high = chan_high

        # Formating 'source' as a LaTeX string and storing it in 'self.latex'
        sym, num = '', ''
        for char in source:
            if char in string.ascii_letters:
                sym += char
            elif char in string.digits:
                num += char

        self.latex = r'${}^{' + num + r'}$' + sym

        # Add this instance to 'lines' upon instantiation
        Line.lines[source] = self


class Experiment:
    '''
    A base class for classes representing various detector tests, like 
    GammaFlood and Noise. This houses some methods that all such classes share.
    '''
    def construct_path(self, description='', ext='', save_dir='', etc=''):
        '''
        Constructs a path for saving data and figures based on user input. 
        If the string passed to 'save_dir' has an empty pair of curly braces 
        '{}', they will be replaced by the detector ID 'self.detector'.

        Note to developers: This function is designed to throw a lot of 
        exceptions and be strict about formatting early on to avoid 
        complications later. Call it early in scripts to avoid losing the 
        results of a long computation to a mistyped directory.

        Keyword Arguments:
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
                The directory to which the file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')

        Return:
            save_path: str
                A Unix/Linux/MacOS style path that can be used to save data
                and plots in an organized way.
        '''
        ### Handling exceptions and potential errors

        # If 'ext' does not start with a '.', fix it.
        if ext and ext[0] != '.':
            ext = f'.{ext}'

        # Check that the save directory exists
        if save_dir:
            save_dir = save_dir.format(self.detector)
            if not os.path.exists(save_dir):
                raise ValueError(f'The directory {save_dir} does not exist.')

        ### Constructing the path name

        # Construct the file name from the file name in 'self.datapath'.
        filename = os.path.basename(self.datapath) # Extracts the filename
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


    def count_hist(self, count_map=None, bins=100, title=None, 
        save=True, ext='.pdf', save_dir=''):
        '''
        Plots a count histogram of 'count_map' data.

        Keyword Arguments:
            count_map: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents the number of
                counts read by the detector pixel at the corresponding index.
                If None, then 'self.count_map' is used instead.
            bins: int
                Number of bins for histogram.
                (default: 100)
            title: str
                Figure title. If None, defaults to a title constructed by the 
                'Experiment' class's 'title' method.
                (default: None)
            save:
                If True, 'spectrum' will be saved as an ascii file. Parameters 
                relevant to this saving are below
            save_dir: str
                The directory to which the  file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the file is saved to
                the directory 'figures/H100/pixels'.
                (default: '')
            ext: str
                The file name extension for the count_map file. 
                (default: '.pdf')
        '''

        # Generate a save path, if needed.
        if save:
            save_path = self.construct_path(ext=ext, description='count_hist', 
                save_dir=save_dir)

        if  count_map is None:
            count_map = self.count_map

        plt.figure()
        plt.hist(np.array(count_map).flatten(), bins=bins, 
            range=(0, np.max(count_map) + 1), 
            histtype='stepfilled')
        plt.ylabel('Pixels')
        plt.xlabel('Counts')
        plt.title(self.title('Count Histogram'))
        plt.tight_layout()

        if save:
            plt.savefig(save_path)


    def pixel_map(self, values, value_label, cb_label='', vmin=None, vmax=None,
        title=None, save=True, ext='.pdf', save_dir=''):
        '''
        Construct a heatmap of counts across the detector using matplotlib.

        Arguments:
            values: 2D array
                A 32 x 32 array of numbers.
            value_label: str
                A short label denoting what data is supplied in 'values'.
                In the hover tooltip, the pixel's value will be labelled with 
                this string. E.g., if value_label = 'Counts', the tooltip might
                read 'Counts: 12744'. The color bar title is also set to this 
                value. Also, value_label.lower() is prepended to the file name
                if saving the plot.

        Keyword Arguments:
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
                Figure title. If None, defaults to a title constructed by the 
                'title' method.
                (default: None)
            save: bool
                If True, saves the plot to a file.
            save_dir: str
                The directory to which the plot file will be saved. If 
                unspecified, the file will be saved to the current directory.
                (default: '')
        '''
        # Generate a save path, if needed.
        if save:
            description = (value_label.lower() + '_map').replace(' ', '_')
            save_path = self.construct_path(ext=ext, description=description, 
                save_dir=save_dir)

        # Constructing the plot title, if none supplied
        if not title:
            plot_type = f'{value_label} Map'
            title = self.title(plot_type)

        # Set the color bar label, if not supplied
        if not cb_label:
            if 'gain' in value_label.lower():
                cb_label = 'Gain (eV/channel)'
            elif 'count' in value_label.lower():
                cb_label = 'Counts'
            elif 'fwhm' in value_label.lower():
                cb_label = 'FWHM (keV)'

        # Formatting the figure
        fig = plt.figure()
        masked = np.ma.masked_values(values, 0.0)
        current_cmap = mpl.cm.get_cmap('inferno')
        current_cmap.set_bad(color='gray')
        # The 'extent' kwarg is necessary to make axes flush to the image.
        plt.imshow(masked, vmin=vmin, vmax=vmax, extent=(0, 32, 0 , 32),
            cmap='inferno')
        c = plt.colorbar()
        c.set_label(cb_label, labelpad=10)

        ticks = np.arange(0, 36, 8)
        plt.xticks(ticks)
        plt.yticks(ticks)

        plt.title(title)

        if save:
            plt.savefig(save_path)


class Noise(Experiment):
    '''
    A class containing important experiment parameters with methods to supply
    data analysis functions for noise data.

    Public attributes:
        datapath: str
            A path to the noise data.
        detector: str
            The detector ID.
        voltage: str:
            The bias voltage in Volts.
        temp: str
            The temperature in degrees Celsius.
        pos: int
            The detector position.
        gain: 32 x 32 numpy.ndarray
            Pixel-by-pixel gain data for the detector. This can be supplied
            after initialization though the 'gain' attribute. Do not supply
            a dummy value here if no gain is available. The methods of this
            class take care of that.
        etc: str
            Other important information to append to created files's names.
        count_map: 2D numpy.ndarray
            A 32 x 32 array with the number of events collected during the 
            noise test at each corresponding pixel.
            (initialized to None)

    Private attributes:
        _fwhm_map: 2D numpy.ndarray
            A 32 x 32 array with the fwhm of the gaussian fit to the noise
            data collected at the corresponding pixel.
            (initialized to None)
        _gain_corrected: bool
            If True, indicates that the '_fwhm_map' attribute has been gain-
            corrected. If False, it has not. If None, then the '_fwhm_data'
            attribute should not have been initialized yet.
            (initialized to None)
    '''
    def __init__(self, datapath, detector, voltage, temp, pos, gain=None, 
        etc=''):
        '''
        Initialized an instance of the 'Noise' class.

        Arguments:
            datapath: str
                A path to the noise data.
            detector: str
                The detector ID.
            voltage: str:
                The bias voltage in Volts.
            temp: str
                The temperature in degrees Celsius.
            pos: int
                The detector position.

        Keyword arguments:
            gain: 32 x 32 numpy.ndarray
                Pixel-by-pixel gain data for the detector. This can be supplied
                after initialization though the 'gain' attribute. Do not supply
                a dummy value here if no gain is available. The methods of this
                class take care of that.
            etc: str
                Other important information to append to created files's names.
        '''
        # Remove any unit symbols from voltage and temperature
        temp = str(temp)
        voltage = str(voltage)
        numericize = str.maketrans('', '', string.ascii_letters)
        temp = temp.translate(numericize)
        voltage = voltage.translate(numericize)

        # If gain is supplied, make sure it's a 32 x 32 array
        if gain is not None and gain.shape != (32, 32):
            raise ValueError("'gain' should be a 32 x 32 array. Instead an "
                + f"array of shape {gain.shape} was passed.")

        # Initialize '_gain_corrected' to None. This will be set to True or 
        # False when 'noise_map' is called, denoting whether the attribute
        # 'fwhm_map' is corrected for gain.
        self._gain_corrected = None
        self.gain = gain
        self._fwhm_map = None
        self.count_map = None

        self.datapath = datapath
        self.detector = detector
        self.temp = temp
        self.voltage = voltage
        self.pos = int(pos)
        self.etc = etc

    #
    # Small helper methods and such: 'title', 'load_fwhm_map', 'set_fwhm_map', 
    # 'get_fwhm_map', and 'get_gain_corrected'.
    #


    def title(self, plot):
        '''
        Returns a plot title based on the instance's attributes and the 
        type of plot. 'plot' is a str indicating the type of plot.
        '''
        temp = r'$' + self.temp + r'^{\circ}$C'
        voltage = r'$' + self.voltage + r'$ V'

        title = f'Noise {self.detector} {plot} ({voltage}, {temp})'

        if self.etc:
            title += f' -- {self.etc}'

        return title


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

        self._gain_corrected = gain_corrected

        fwhm_map = np.loadtxt(fwhm_map)

        if fwhm_map.shape != (32, 32):
            raise ValueError("'fwhm_map' should reference a 32 x 32 array."
                + f"Instead an array of shape {fwhm_map.shape} was given.")

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

        if fwhm_map.shape != (32, 32):
            raise ValueError("'fwhm_map' should be a 32 x 32 array."
                + f"Instead an array of shape {fwhm_map.shape} was given.")

        self._fwhm_map = fwhm_map


    def get_fwhm_map(self):
        '''Returns a copy of the private attribute '_fwhm_map'.'''
        return self._fwhm_map


    def get_gain_corrected(self):
        '''Returns a copy of the private attribute '_gain_corrected'.'''
        return self._gain_corrected


    #
    # Heavy lifting data analysis method: 'noise_map'
    #

    def noise_map(self, gain=None, save_plot=True, plot_dir='',
        plot_ext='.pdf', save_data=True, data_dir='', data_ext='.txt'):
        '''
        Calculates the noise FWHM for each pixel and generates a noise count
        map. Also plots a noise spectrum for each pixel.

        Keyword Arguments:
            gain: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents its  
                respective pixel's gain, where channels * gain = energy. If 
                None, defaults to the array in 'self.gain'.
                (default: None)
            save_plot: bool
                If true, plots and energy spectrum for each pixel and saves
                the figure.
                (default: True)
            plot_dir: str
                The directory to which the plot file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the file is saved to
                the directory 'figures/H100/pixels'.
                (default: '')
            plot_ext: str
                The file name extension for the plot file.
                (default: '.pdf')  
            save_data: bool 
                If True, saves gain data as a .txt file.
                (default: True)
            data_dir: str
                The directory to which the gain file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the file is saved to
                the directory 'figures/H100/pixels'.
                (default: '')
            data_ext: str
                The file name extension for the gain file. 
                (default: '.txt')

        Return: tuple(numpy.ndarray, numpy.ndarray)
            fwhm_map: 2D numpy.ndarray
                A 32 x 32 array with the fwhm of the gaussian fit to the noise
                data collected at the corresponding pixel.
            count_map: 2D numpy.ndarray
                A 32 x 32 array with the number of events collected during the 
                noise test at each corresponding pixel.
        '''
        # 'etc' and 'etc_plot' will be appended to file names, denoting  
        # whether data/plots were gain-corrected.
        gain_bool = (self.gain is not None) or (gain is not None)
        if gain_bool:
            etc = 'gain'
        else:
            etc = 'nogain'

        # 'etc_plot' will be formatted to have pixel coordinates, since a
        # spectrum is plotted for each pixel.
        etc_plot = etc + '_x{}_y{}'

        # Generating the save paths, if needed.
        if save_data:
            fwhm_path = self.construct_path(ext=data_ext, save_dir=data_dir, 
                description='fwhm_data', etc=etc)
            count_path = self.construct_path(ext=data_ext, save_dir=data_dir,
                description='count_data', etc=etc)

        if save_plot:
            plot_path = self.construct_path(save_dir=plot_dir, etc=etc_plot,
                description='pix_spectrum', ext=plot_ext)

        # Get data from noise FITS file
        with fits.open(self.datapath) as file:
            data = file[1].data

        if not gain_bool:
            gain = np.ones((32, 32))
        # If gain data is not passed directly as a parameter, but is an 
        # attribute of self, use the attribute's gain data.
        elif gain is None:
            gain = self.gain

        # 'start' and 'end' denote the indices between which 'data['TEMP']'
        # takes on a resonable value and the detector position is the desired 
        # position. start is the first index with a temperature greater than 
        # -20 C, and end is the last such index.
        mask = np.multiply((data['DET_ID'] == self.pos), (data['TEMP'] > -20))
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        del mask

        maxchannel = 1000
        bins = np.arange(-maxchannel, maxchannel)

        # Generate 'chan_map', a nested list representing a 33 x 33 array of 
        # list, each of which contains all the trigger readings for its 
        # corresponding pixel.
        chan_map = [[[] for col in range(33)] for row in range(33)]
        # Iterating through pixels
        for col in range(32):
            RAWXmask = np.array(data['RAWX'][start:end]) == col
            for row in range(32):
                RAWYmask = np.array(data['RAWY'][start:end]) == row
                # Storing all readings for the current pixel in 'pulses'.
                inds = np.nonzero(np.multiply(RAWXmask, RAWYmask))
                pulses = data.field('PH_RAW')[inds]
                for idx, pulse in enumerate(pulses):
                    # If this pulse was triggered by the experiment (by a 
                    # 'micro pulse'), then add the pulse data for the 3 x 3
                    # pixel grid centered on the triggered pixel to the 
                    # corresponding indices of 'chan_map'.
                    if data['UP'][idx]:
                        for i in range(9):
                            mapcol = col + (i % 3) - 1
                            maprow = row + (i // 3) - 1
                            chan_map[maprow][mapcol].append(pulse[i])

        del data, mapcol, maprow, pulses, inds, RAWYmask, RAWXmask

        # Generate a count map of micropulse-triggered events from 'chan_map'.
        count_map = np.array(
            [[len(chan_map[j][i]) for i in range(32)] for j in range(32)])
        self.count_map = count_map
       
        # Generate a fwhm map of noise, and plot the gaussian fit to each 
        # pixel's spectrum.
        fwhm_map = np.full((32, 32), np.nan)
        # Iterate through pixels
        for row in range(32):
            for col in range(32):
                # If there were events at this pixel, bin them by channel
                if chan_map[row][col]:
                    # Binning events by channel
                    spectrum, edges = np.histogram(chan_map[row][col], 
                        bins=bins, range=(-maxchannel, maxchannel))

                    # Fitting the noise peak at/near zero channels
                    fit_channels = edges[:-1]
                    g_init = models.Gaussian1D(amplitude=np.max(spectrum), 
                        mean=0, stddev=75)
                    fit_g = fitting.LevMarLSQFitter()
                    g = fit_g(g_init, fit_channels, spectrum)

                    # Recording the gain-corrected FWHM data for this pixel
                    # in fwhm_map.
                    fwhm_map[row][col] = g.fwhm * gain[row][col]
                    if save_plot:
                        plt.hist(np.multiply(
                                chan_map[row][col], gain[row][col]),
                            bins=np.multiply(bins, gain[row][col]), 
                            range=(-maxchannel * gain[row][col], 
                                    maxchannel * gain[row][col]), 
                            histtype='stepfilled')

                        plt.plot(np.multiply(fit_channels, gain[row][col]), 
                            g(fit_channels))

                        plt.ylabel('Counts')
                        if gain_bool:
                            plt.xlabel('Energy (keV)')
                        else:
                            plt.xlabel('Channel')

                        plt.tight_layout()
                        plt.savefig(plot_path.format(row, col))
                        plt.close()
        
        self._fwhm_map = fwhm_map
        # Set '_gain_corrected' way down here to make sure the 'fwhm_map'
        # attribute was successfully set.
        self._gain_corrected = gain_bool

        if save_data:
            np.savetxt(fwhm_path, fwhm_map)
            np.savetxt(count_path, count_map)

        return fwhm_map, count_map


    #
    # Plotting method: 'fwhm_hist'
    #

    def fwhm_hist(self, bins=50, hist_range=None, save=True, save_dir='', 
        ext='.pdf', text_pos='right'):
        '''
        Plots a histogram of the fwhms for the noise of each pixel.

        Keyword Arguments:
            bins: int
                The number of bins in which to histogram the data. Passed 
                directly to plt.hist.
                (default: 50)
            hist_range: tuple(number, number)
                Indicated the range in which to bin data. Passed directly to
                plt.hist. If None, it is set to (0, 4) for gain-corrected data
                and to (0, 150) otherwise.
                (default: None)
            text_pos: str
                Indicates where information about mean and standard deviation
                appears on the plot. If 'right', appears in upper right. If 
                'left', appears in upper left.
                (default: 'right')
            save: bool
                If True, saves the plot to 'save_dir'.
            save_dir: str
                The directory to which the file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the directory that 
                'save_path' points to is 'figures/H100/pixels'.
                (default: '')
            ext: str
                The file extension to the saved file.
                (default: '.pdf')
        '''
        # Setting some plot parameters and converting units based on whether 
        # the supplied data is gain-corrected.
        if self._gain_corrected:
            if hist_range is None:
                hist_range = (0, 4)
            mean_fwhm = str(int(round(np.mean(fwhm) * 1000, 0)))
            stdv_fwhm = str(int(round(np.std(fwhm) * 1000, 0)))
            fwhm_units = 'eV'
            axis_units = 'keV'
        else:
            if hist_range is None:
                hist_range = (0, 150)
            mean_fwhm = str(round(np.mean(fwhm), 0))
            stdv_fwhm = str(round(np.std(fwhm), 0))
            fwhm_units = 'channels'
            axis_units = 'channels'

        # Make the plot
        plt.figure()
        ax = plt.axes() # need axes object for text positioning
        fwhm = self._fwhm_map.flatten()
        plt.hist(fwhm, bins=bins, range=hist_range, histtype='stepfilled')

        # Setting text position based on user input. This will display the mean
        # and standard deviation of the fwhm data.
        if text_pos == 'right':
            left_side = 0.5
        elif text_pos == 'left':
            left_side = 0.05
        else:
            raise ValueError("'text_pos' can be either 'right' or 'left'. "
                + f"Instead {text_pos} was passed")

        plt.text(left_side, 0.9, f'Mean = {mean_fwhm} {fwhm_units}', 
            fontsize=14, transform=ax.transAxes)
        plt.text(left_side, 0.8, f'1-Sigma = {stdv_fwhm} {fwhm_units}', 
            fontsize=14, transform=ax.transAxes)

        plt.xlabel(f'FWHM ({axis_units})')
        plt.ylabel('Pixels') 
        plt.title(self.title('FWHM Histogram'))


class Leakage(Experiment):
    '''
    A class containing important experiment parameters with methods to supply
    data analysis functions for noise data.

    Public attributes:
        datapath: str
            A path to the noise data.
        detector: str
            The detector ID.
        temp: str
            The temperature in degrees Celsius.
        pos: int
            The detector position.
        voltages: 1D array-like of numbers:
            The bias voltages in Volts.
        etc: str
            Other important information to append to created files's names.
    '''
    def __init__(self, datapath, detector, temps, cp_voltages, n_voltages,
        pos, etc=''):
        '''
        Initialized an instance of the 'Noise' class.

        Arguments:
            datapath: str
                A path to the noise data.
            detector: str
                The detector ID.
            pos: int
                The detector position.
            temps: set of numbers
                The temperatures in degrees Celsius.
            cp_voltages: set or array-like of numbers:
                The bias voltages in volts used for testing in 
                charge-pump mode.
            n_voltages: set or array-like of numbers:
                The bias voltages in volts used for testing in 
                normal mode.

        Keyword arguments:
            etc: str
                Other important information to append to created files's names.
        '''
        # Remove any unit symbols from temperature
        temp = str(temp)
        numericize = str.maketrans('', '', string.ascii_letters)
        temp = temp.translate(numericize)

        # Convert temperatures and voltages to sets to avoid repeats
        temps = set(temps)
        cp_voltages = set(cp_voltages)
        n_voltages = set(n_voltages)

        self.datapath = datapath
        self.detector = detector
        self.temps = temps
        self.cp_voltages = cp_voltages
        self.n_voltages = n_voltages
        self.all_voltages = cp_voltages | n_voltages
        self.pos = int(pos)
        self.etc = etc

    #
    # Small helper method: 'title'.
    #

    # TODO: There's a lot to do for this method. Probably pass voltage as
    # an argument and stuff. Or not? idk.
    def title(self, plot):
        '''
        Returns a plot title based on the instance's attributes and the 
        type of plot. 'plot' is a str indicating the type of plot.
        '''
        temp = r'$' + self.temp + r'^{\circ}$C'

        title = f'Noise {self.detector} {plot} ({temp})'

        if self.etc:
            title += f' -- {self.etc}'

        return title


    #
    # Heavy-lifting data analysis method: 'leakage'
    #

    def leakage(self, save=True, save_dir='', ext='.txt'):
        '''
        TODO
        '''
        # This dict will form a pandas DataFrame. Initialized as a temporary
        # private attribute for easy access by the '_process_leakage_data'
        # helper method.
        self._table = {
            'mode'    : [], # Can be 'CP' or 'N' (charge-pump or normal)
            'voltage' : [], # The bias voltage in volts
            'temp'    : [], # The temperature in Celsius
            'mean'    : [], # The mean leakage current across the pixels
            'std'     : [], # The corresponding standard deviation
            'outliers': []  # Number of outlier pixels
        }

        # Sets 'filename' to the last directory in 'self.datapath'.
        filename = os.path.basename(self.datapath)

        # 'start' and 'end' define the indices of the pixels at the given 
        # detector position are. Temporary private attributes for easy
        # access by the '_process_leakage_data' helper method.
        self._start = -1024 * (1 + self.pos)
        self._end = start + 1024

        # Iterate through temperatures
        for temp in self.temps:
            # First, construct maps 'cp_zero' and 'n_zero' of the leakage 
            # current at bias voltage of zero as a control.
            cp_data = asciio.read(f'{self.datapath}/{filename}_{temp}.C0V.txt')
            n_data = asciio.read(f'{self.datapath}/{filename}_{temp}.N0V.txt')
            cp_zero = np.zeros((32, 32))
            n_zero = np.zeros((32, 32))

            for i in range(start, end):
                # Pixel coordinates in charge pump mode
                cp_col = cp_data.field('col4')[i]
                cp_row = cp_data.field('col5')[i]
                # Pixel coordinates in normal mode
                n_col = n_data.field('col4')[i]
                n_row = n_data.field('col5')[i]
                # Leakage at this pixel in each mode.
                cp_zero[cp_row, cp_col] = cp_data.field('col6')[i]
                n_zero[n_row, n_col] = n_data.field('col6')[i]

            # Iterating though non-zero bias voltages
            for voltage in self.all_voltages:
                # If we tested this voltage using charge-pump mode,
                # construct a map of the leakage current appropriately.
                if voltage in self.cp_voltages:
                    data = asciio.read(
                        f'{self.datapath}/{filename}_{temp}.C{voltage}V.txt')
                    self._process_leakage_data(data, 'CP', temp, voltage)
                           
                # If we tested this voltage using normal mode,
                # construct a map of the leakage current appropriately.
                if voltage in self.n_voltages:
                    data = asciio.read(
                        f'{self.datapath}/{filename}_{temp}.N{voltage}V.txt')
                    self._process_leakage_data(data, 'N', temp, voltage)

        self.leakage_stats = pd.DataFrame(self._table)

        # Deleting temporary private attributes.
        del self._table, self._start, self._end

        return self.leakage_stats


    def _process_leakage_data(self, data, mode, temp, voltage):
        '''
        Helper function for 'leakage'.
        '''
        leak_map = np.zeros((32, 32))

        if mode == 'CP':
            div = 3000:
        elif mode == 'N':
            div = 150

        # Generating a leakage current map at the current voltage,
        # realtive to what we had at 0V.
        for i in range(start, end):
            col = data.field('col4')[i]
            row = data.field('col5')[i]
            leak_map[row, col] = (data.field('col6')[i] 
                - cp_zero[row, col]) * (1.7e3) / 3000

        masked_map = np.ma.masked_where(leak_map > 100, leak_map)
        # 'outliers' in the number of pixels whose leakage 
        # currents are 5 standard deviations from the mean.
        outliers = np.sum(np.absolute(
            leak_map - np.mean(leak_map)
            )) > 5 * np.std(leak_map)
        # Record the data
        table['mode'].append(mode)
        table['voltage'].append(voltage)
        table['temp'].append(temp)
        table['mean'].append(np.mean(masked_map))
        table['std'].append(np.std(masked_map))
        table['outliers'].append(outliers)

        return masked_map


class GammaFlood(Experiment):
    '''
    A class containing important experiment parameters with methods to supply
    data analysis functions for gamma flood data.

    Public attributes:
        datapath: str
            Path to gamma flood data. Should be a FITS file. Used to access
            data and to construct new file names.
        detector: str
            The detector ID.
        source: str
            The X-ray source. Should correspond to the 'source' attribute 
            of a 'Line' object. A dict of instantiated Line objects can be 
            accessed by 'gamma.Line.lines'
        voltage: str
            Bias voltage in Volts
        temp: str
            Temperature of the detector in degrees Celsius
        etc: str
            Any other important information to include
        count_map: 2D numpy.ndarray
            A 32 x 32 array of floats. Each entry represents the number of
            counts read by the detector pixel at the corresponding index.
            (initialized to None)
        gain: 2D numpy.ndarray
            A 32 x 32 array of floats. Each entry represents its  
            respective pixel's gain, where channels * gain = energy.
            (initialized to None)
        spectrum: 2D numpy.ndarray
            This array represents a histogram wrt the energy of an event.
            spectrum[0] is a 1D array of counts in each bin, and  
            spectrum[1] is a 1D array of the middle enegies of each bin in 
            keV. E.g., if the ith bin counted events between 2 keV and 4 
            keV, then the value of spectrum[1, i] is 3. If None, defaults
            to the value stored in self.spectrum.
            (initialized to None)
    '''
    def __init__(self, datapath, detector, source, voltage, temp, etc=''):
        '''
        Initializes an instance of the 'GammaFlood' class.

        Arguments:
            datapath: str
                Path to gamma flood data. Should be a FITS file. Used to access
                data and to construct new file names.
            detector: str
                The detector ID.
            source: str
                The X-ray source. Should correspond to the 'source' attribute 
                of a 'Line' object. A dict of instantiated Line objects can be 
                accessed by 'gamma.Line.lines'
            voltage: str
                Bias voltage in Volts
            temp: str
                Temperature of the detector in degrees Celsius

        Keyword Arguments:
            etc: str
                Any other important information to include
        '''
        # Check that the source corresponds to a Line object.
        if source not in Line.lines:
            raise KeyError(f'''
                There is no Line object corresponing to the source 
                {self.source}. Print 'gamma.Line.lines.keys() for a list of 
                valid 'source' values, or call 'help(Line)' to see how to 
                define a new Line object.''')

        voltage = str(voltage)
        temp = str(temp)

        # Remove any unit symbols from voltage and temperature
        numericize = str.maketrans('', '', string.ascii_letters)
        voltage = voltage.translate(numericize)
        temp = temp.translate(numericize)

        # Initialize data-based attributes to 'None'
        self.count_map = None
        self.gain = None
        self.spectrum = None

        # Set user-supplied attributes
        self.datapath = datapath
        self.detector = detector
        self.source = source
        self.voltage = voltage
        self.temp = temp
        self.etc = etc


    # Defining 'Line' instances for Am241 and Co57. Co57's 'chan_low' and 
    # 'chan_high' attributes have not been tested.
    am = Line('Am241', 59.54, chan_low=3000, chan_high=6000)
    co = Line('Co57', 122.06, chan_low=5000, chan_high=8000)


    #
    # Small helper methods: 'line' and 'title'.
    #

    def line(self):
        '''Returns the 'Line' instance to which 'self.source' corresponds.'''
        try:
            return Line.lines[self.source]
        except KeyError:
            raise KeyError(f'''
                There is no Line object corresponing to the source 
                {self.source}. Call 'help(Line)' to see how to define a new
                Line object.''')


    def title(self, plot):
        '''
        Returns a plot title based on the instance's attributes and the 
        type of plot. 'plot' is a str indicating the type of plot.
        '''
        temp = r'$' + self.temp + r'^{\circ}$C'
        voltage = r'$' + self.voltage + r'$ V'

        title = r'$\gamma$ Flood '\
        + f'{self.detector} {self.source} {plot} ({voltage}, {temp})'

        if self.etc:
            title += f' -- {self.etc}'

        return title

    #
    # Heavy-lifting data analysis methods: 'count_map', 'quick_gain',
    # and 'get_spectrum'.
    #

    def count_map(self, save=True, ext='.txt', save_dir=''):
        '''
        Generates event count data for each pixel for raw gamma flood data.

        Keyword Arguments:
            save: bool 
                If True, saves count_map as an ascii file.
                (default: True)
            save_dir: str
                The directory to which the file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the file is saved to
                the directory 'figures/H100/pixels'.
                (default: '')
            ext: str
                The file name extension for the count_map file. 
                (default: '.txt')

        Return:
            count_map: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents the number of
                counts read by the detector pixel at the corresponding index.
        '''
        # Generating the save path, if needed.
        if save:
            save_path = self.construct_path(ext=ext, save_dir=save_dir, 
                description='count_data')

        # Get data from gamma flood FITS file
        with fits.open(self.datapath) as file:
            data = file[1].data

        # The masks ('mask', 'PHmask', 'STIMmask', and 'TOTmask') below show 
        # 'True' or '1' for valid/desired data points, and 'False' or '0' 
        # otherwise. Note that this is the reverse of numpy's convention for 
        # masking arrays.

        # 'start' and 'end' denote the indices between which 'data['TEMP']'
        # takes on a resonable value. start is the first index with a 
        # temperature greater than -20 C, and end is the last such index.
        mask = data['TEMP'] > -20
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        del mask

        # Masking out non-positive pulse heights
        PHmask = 0 < np.array(data['PH'][start:end])
        # Masking out artificially stimulated events
        STIMmask = np.array(data['STIM'][start:end]) == 0
        # Combining the above masks
        TOTmask = np.multiply(PHmask, STIMmask)

        # Generate the count_map from event data
        count_map = np.zeros((32, 32))

        for i in range(32):
            RAWXmask = np.array(data['RAWX'][start:end]) == i
            for j in range(32):
                RAWYmask = np.array(data['RAWY'][start:end]) == j
                count_map[i, j] = np.sum(np.multiply(
                    TOTmask, np.multiply(RAWXmask, RAWYmask)))

        # Saves the 'count_map' array as an ascii file.
        if save:
            np.savetxt(save_path, count_map)

        # Storing count data in our 'GammaFlood' instance
        self.count_map = count_map

        return count_map


    def quick_gain(self, line=None, fit_low=100, fit_high=200, 
        save_plot=True, plot_dir='', plot_ext='.pdf', 
        save_data=True, data_dir='', data_ext='.txt'):
        '''
        Generates gain correction data from the raw gamma flood event data.
        Currently, the fitting done might fail for sources other than Am241.

        Keyword Arguments:
            line: an instance of Line
                The attributes of 'line' will provide information for fitting.
                If None, defaults to the value referenced by self.line().
                (default: None)
            fit_low: int
                Channels this far below the centroid won't be considered in 
                fitting a gaussian to the spectral peak. Should be smaller 
                than 'fit_high' due to thick low-energy tails.
            fit_high: int
                Channels this far above the centroid won't be considered in 
                fitting a gaussian to the spectral peak.
            save_plot: bool
                If true, plots and energy spectrum for each pixel and saves
                the figure.
                (default: True)
            plot_dir: str
                The directory to which the plot file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the file is saved to
                the directory 'figures/H100/pixels'.
                (default: '')
            plot_ext: str
                The file name extension for the plot file.
                (default: '.pdf')  
            save_data: bool 
                If True, saves gain data as a .txt file.
                (default: True)
            data_dir: str
                The directory to which the gain file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the file is saved to
                the directory 'figures/H100/pixels'.
                (default: '')
            data_ext: str
                The file name extension for the gain file. 
                (default: '.txt')

        Return:
            gain: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents its  
                respective pixel's gain, where channels * gain = energy.
        '''

        if save_data:
            data_path = self.construct_path(ext=data_ext, 
                description='gain_data', save_dir=data_dir)

        if save_plot:
            plot_path = self.construct_path(description='gain', ext=plot_ext, 
                save_dir=plot_dir)

        # If no line is passed, take it from the GammaFlood instance.
        if line == None:
            line = self.line()

        # Get data from gamma flood FITS file
        with fits.open(self.datapath) as file:
            data = file[1].data

        mask = data['TEMP'] > -20
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        del mask

        maxchannel = 10000
        bins = np.arange(1, maxchannel)
        gain = np.zeros((32, 32))

        # Iterating through pixels
        for x in range(32):
            RAWXmask = data.field('RAWX')[start:end] == x
            for y in range(32):
                RAWYmask = data.field('RAWY')[start:end] == y
                # Getting peak height in 'channels' for all events for the 
                # current pixel.
                channel = data.field('PH')[start:end][np.nonzero(
                    np.multiply(RAWXmask, RAWYmask))]

                # If there were events at this pixel, fit the strongest peak
                # in the channel spectrum with a Gaussian.
                if len(channel):
                    # 'spectrum' contains counts at each channel
                    spectrum, edges = np.histogram(channel, bins=bins, 
                        range=(0, maxchannel))
                    # 'centroid' is the channel with the most counts in the 
                    # interval between 'line.chan_low' and 'line.chan_high'.
                    centroid = np.argmax(spectrum[line.chan_low:line.chan_high]
                       ) + line.chan_low
                    # Excluding funky tails for the fitting process.
                    fit_channels = np.arange(
                        centroid - fit_low, centroid + fit_high)
                    g_init = models.Gaussian1D(amplitude=spectrum[centroid], 
                        mean=centroid, stddev=75)
                    fit_g = fitting.LevMarLSQFitter()
                    g = fit_g(g_init, fit_channels, spectrum[fit_channels])

                    # If we can determine the covariance matrix (which implies
                    # that the fit succeeded), then calculate this pixel's gain
                    if fit_g.fit_info['param_cov'] is not None:
                        gain[y, x] = line.energy / g.mean
                        # Plot each pixel's spectrum
                        if save_plot:
                            plt.figure()
                            sigma_err = np.diag(fit_g.fit_info['param_cov'])[2]
                            fwhm_err = 2 * np.sqrt(2 * np.log(2)) * sigma_err
                            mean_err = np.diag(fit_g.fit_info['param_cov'])[1]
                            frac_err = np.sqrt(np.square(fwhm_err) 
                                + np.square(g.fwhm * mean_err / g.mean))\
                            / g.mean
                            str_err = str(int(round(
                                frac_err * line.energy * 1000)))
                            str_fwhm = str(int(round(
                                    line.energy * 1000 * g.fwhm / g.mean, 0)))
                            plt.text(
                                maxchannel * 3 / 5, spectrum[centroid] * 3 / 5,
                                r'$\mathrm{FWHM}=$' + str_fwhm + r'$\pm$' 
                                + str_err + ' eV', fontsize=13
                            )

                            plt.hist(
                                np.multiply(channel, gain[y, x]), 
                                bins=np.multiply(bins, gain[y, x]),
                                range=(0, maxchannel * gain[y, x]), 
                                histtype='stepfilled'
                            )

                            plt.plot(
                                np.multiply(fit_channels, gain[y, x]), 
                                g(fit_channels), label='Gaussian fit'
                            )

                            plt.ylabel('Counts')
                            plt.xlabel('Energy')
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(f'{plot_path}_x{x}_y{y}{plot_ext}')
                            plt.close()

        del RAWXmask, channel, data

        # Interpolate gain for pixels where fit was unsuccessful. Do it twice 
        # in case the first pass had insufficient data to interpolate 
        # some pixels.
        for _ in range(2):
            newgain = np.zeros((34, 34))
            # Note that newgain's indices will be shifted over one from 'gain'.
            newgain[1:33, 1:33] = gain
            # 'empty' contains indices at which the fit was unsuccessful
            empty = np.transpose(np.nonzero(gain == 0.0))
            # Iterating through pixels with failed fitting.
            for x in empty:
                # 'empty_grid' is the 3x3 array of gain values around the  
                # pixel for which the fitting failed.
                empty_grid = newgain[x[0]:x[0]+3, x[1]:x[1]+3]
                # If there are any nonzero values in 'empty_grid', set the  
                # pixel's gain to their mean.
                if np.count_nonzero(empty_grid):
                    gain[x[0], x[1]] =\
                        np.sum(empty_grid) / np.count_nonzero(empty_grid)

        # Save gain data to an ascii file.
        if save_data:
            np.savetxt(data_path, gain)

        # Storing gain data in our 'GammaFlood' instance
        self.gain = gain

        return gain


    def get_spectrum(self, gain=None, line=None, bins=10000, 
        energy_range=(0.01, 120), save=True, ext='.txt', save_dir=''):
        '''
        Applies gain correction to get energy data, and then bins the events
        by energy to obtain a spectrum.

        Keyword Arguments:
            gain: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents its  
                respective pixel's gain, where channels * gain = energy. If 
                None, defaults to the array in 'self.gain'.
                (default: None)
            line: an instance of Line
                The attributes of 'line' will provide information for fitting.
                If None, defaults to the value referenced by self.line().
                (default: None)
            bins: int
                Number of energy bins
            energy_range: tuple of numbers
                The bins will be made between these energies
            save:
                If True, 'spectrum' will be saved as an ascii file. Parameters 
                relevant to this saving are below
            save_dir: str
                The directory to which the  file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the file is saved to
                the directory 'figures/H100/pixels'.
                (default: '')
            ext: str
                The file name extension for the count_map file. 
                (default: '.txt')

        Return:
            spectrum: 2D numpy.ndarray
                This array represents a histogram wrt the energy of an event.
                spectrum[0] is a 1D array of counts in each bin, and  
                spectrum[1] is a 1D array of the middle enegies of each bin in 
                keV. E.g., if the ith bin counted events between 2 keV and 4 
                keV, then the value of spectrum[1, i] is 3.
        '''
        # Generating the save path, if needed.
        if save:
            save_path = self.construct_path(ext=ext, save_dir=save_dir, 
                description='spectrum')

        # If no gain is passed, take it from the GammaFlood instance.
        if gain is None:
            gain = self.gain
        # If no line is passed, take it from the GammaFlood instance.
        if line is None:
            line = self.line()

        # Adding a buffer of zeros around the 'gain' array. (Note that the
        # indices will now be shifted over by one.)
        gain_buffed = np.zeros((34, 34))
        gain_buffed[1:33, 1:33] = gain
        gain = gain_buffed

        # Get data from gamma flood FITS file
        with fits.open(self.datapath) as file:
            data = file[1].data

        # 'start' and 'end' denote the indices between which 'data['TEMP']'
        # takes on a resonable value. start is the first index with a 
        # temperature greater than -20 C, and end is the last such index.
        temp_mask = data['TEMP'] > -20
        start = np.argmax(temp_mask)
        end = len(temp_mask) - np.argmax(temp_mask[::-1])
        del temp_mask

        # PH_COM is a list of length 9 corresponding to the charge in pixels 
        # surrounding the event.
        #
        # PH_COM -> gain correct -> sum positive elements in the 3x3 array -> 
        # event in energy units

        # 'energies' is a list of event energies in keV.
        energies = []
        # iterating through pixels
        for row in range(32):
            row_mask = data['RAWY'] == row
            for col in range(32):
                col_mask = data['RAWX'] == col
                # Getting indices ('inds') and PH_COM values ('pulses') of 
                # all events at current pixel.
                inds = np.nonzero(np.multiply(row_mask, col_mask))
                pulses = data.field('PH_COM')[inds]
                # The gain for the 3x3 grid around this pixel
                gain_grid = gain[row:row + 3, col:col + 3]
                # iterating through the PH_COM values for this pixel
                for pulse in pulses:
                    # Append the sum of positive energies in the 
                    # pulse grid to 'energies'
                    pulse_grid = pulse.reshape(3, 3)
                    mask = (pulse_grid > 0).astype(int)
                    energies.append(np.sum(np.multiply(
                        np.multiply(mask, pulse_grid), gain_grid)))

        del data

        # Binning by energy
        counts, edges = np.histogram(energies, bins=bins, range=energy_range)
        del energies

        # Getting the midpoint of the edges of each bin.
        midpoints = (edges[:-1] + edges[1:]) / 2

        # Consolidating 'counts' and 'midpoints' into a 2D array 'spectrum'.
        spectrum = np.empty((2, counts.size))
        spectrum[0, :] = counts
        spectrum[1, :] = midpoints

        if save:
            np.savetxt(save_path, spectrum)

        self.spectrum = spectrum

        return spectrum

    #
    # Plotting methods with light data analysis: 'plot_spectrum' and
    # 'count_hist'. Also 'pixel_map' inherited from 'Experiment'.
    #

    def plot_spectrum(self, spectrum=None, line=None, fit_low=80, fit_high=150,
        title=None, save=True, ext='.pdf', save_dir=''):
        '''
        Fits and plots the spectrum returned from 'get_spectrum'. To show the 
        plot with an interactive interface, call 'plt.show()' right after 
        calling this function.

        Keyword Arguments:
            spectrum: 2D numpy.ndarray
                This array represents a histogram wrt the energy of an event.
                spectrum[0] is a 1D array of counts in each bin, and  
                spectrum[1] is a 1D array of the middle enegies of each bin in 
                keV. E.g., if the ith bin counted events between 2 keV and 4 
                keV, then the value of spectrum[1, i] is 3. If None, defaults
                to the value stored in self.spectrum.
                (default: None)
            line: an instance of Line
                The attributes of 'line' will provide information for fitting.
                If None, defaults to the value referenced by self.line().
                (default: None)
            fit_low: int
                Channels this far below the centroid won't be considered in 
                fitting a gaussian to the spectral peak. Should be smaller 
                than 'fit_high' due to thick low-energy tails.
            fit_high: int
                Channels this far above the centroid won't be considered in 
                fitting a gaussian to the spectral peak.
            title: str
                Figure title. If None, defaults to a title constructed by the 
                'Experiment' class's 'title' method.
                (default: None)
            save:
                If True, 'spectrum' will be saved as an ascii file. Parameters 
                relevant to this saving are below
            save_dir: str
                The directory to which the  file will be saved. If left
                unspecified, the file will be saved to the current directory.
                If the string passed to 'save_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                save_dir == 'figures/{}/pixels', then the file is saved to
                the directory 'figures/H100/pixels'.
                (default: '')
            ext: str
                The file name extension for the count_map file. 
                (default: '.pdf')

        '''
        # Constructing a save path, if needed
        if save:
            save_path = self.construct_path(ext=ext, save_dir=save_dir, 
                description='energy_spectrum')

        # If no spectrum is supplied take it from the instance.
        if spectrum is None:
            spectrum = self.spectrum
        # If no line is passed, take it from the GammaFlood instance.
        if line is None:
            line = self.line()
        # If no title is passed, construct one
        if title is None:
            title = self.title('Spectrum')

        maxchannel = 10000

        # 'centroid' is the bin with the most counts
        centroid = np.argmax(spectrum[0, 1000:]) + 1000
        # Fit in an asymetrical domain about the centroid to avoid 
        # low energy tails.
        fit_channels = np.arange(centroid - fit_low, centroid + fit_high)
        # Do the actual fitting.
        g_init = models.Gaussian1D(amplitude=spectrum[0, centroid], 
            mean=centroid, stddev=75)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, fit_channels, spectrum[0, fit_channels])

        sigma_err = np.diag(fit_g.fit_info['param_cov'])[2]
        fwhm_err = 2 * np.sqrt(2 * np.log(2)) * sigma_err
        mean_err = np.diag(fit_g.fit_info['param_cov'])[1]
        frac_err = np.sqrt(
            np.square(fwhm_err) 
            + np.square(g.fwhm * mean_err / g.mean)
        ) / g.mean

        # Displaying the FWHM on the spectrum plot, with error.
        display_fwhm = str(int(round(line.energy * 1000 * g.fwhm / g.mean, 0)))
        display_err  = str(int(round(frac_err * line.energy * 1000)))

        plt.text(70, spectrum[0, centroid] * 3 / 5, r'$\mathrm{FWHM}=$' 
            +  display_fwhm + r'$\pm$' + display_err + ' eV', 
            fontsize=13)

        plt.plot(spectrum[1], spectrum[0], label=line.latex)
        plt.plot(spectrum[1, fit_channels], g(fit_channels), 
            label = 'Gaussian fit')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()

        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(save_path)


# If this file is run as a script, the code below will run a complete pipeline
# for an experiment's data analysis with default parameter values.

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze detector data.')
    parser.add_argument('experiment', metavar='A', type=str,
        help="""Determines which experiment is being analyzed. Can take on
        the values 'gamma', 'noise', or 'leakage'.""")

    experiment = parser.parse_args().experiment.lower()

    # Setting the mpl backend to be compatible with the SRL server (or
    # something - I just found this here:
    # https://github.com/matplotlib/matplotlib/issues/3466)
    # Didn't want to put this else where because it screws stuff up if you 
    # want to plot stuff on your own machine.
    plt.switch_backend('agg')

    # Run complete gamma flood data analysis.
    if experiment == 'gamma' or experiment == 'gammaflood':

        datapath = input('Enter the path to the gamma flood data: ')
        while not os.path.exists(datapath):
            datapath = input("That path doesn't exist. " + 
                "Enter another path to the gamma flood data: ")

        source = input('Enter the name of the source used (Am241 or Co57): ')
        detector = input('Enter the detector ID: ')
        voltage = input('Enter the voltage in Volts (no unit symbol): ')
        temp = input('Enter the temperature in Celsius (no unit symbol): ')
        save_dir = input('Enter a directory to save outputs to: ')

        gamma = GammaFlood(datapath, detector, source, voltage, temp)

        pixel_dir = input('Enter a directory to save pixel spectra to: ')

        # Processing data
        print('Calculating count data...')
        count_map = gamma.count_map(save_dir=save_dir)

        print('Calculating gain data...')
        gain = gamma.quick_gain(plot_dir=pixel_dir, data_dir=save_dir)

        print('Calculating the energy spectrum...')
        gamma.get_spectrum(save_dir=save_dir)

        # Plotting
        print('Plotting...')

        gamma.plot_spectrum(save_dir=save_dir)
        gamma.count_hist(save_dir=save_dir)
        gamma.pixel_map(count_map, 'Counts', save_dir=save_dir)
        gamma.pixel_map(gain, 'Gain', save_dir=save_dir)

        print('Done!')

    # Run complete noise data analysis.
    elif experiment == 'noise':

        # Requesting paths to noise and gain data
        datapath = input('Enter the path to the noise data: ')
        while not os.path.exists(datapath):
            datapath = input("That path doesn't exist. " + 
                "Enter another path to the noise data: ")

        gainpath = input('Enter the path to the gain data, or leave blank ' + 
            'if there is no gain data: ')
    # Request a different input if a non-existent path (other than an
    # empty string) was given for 'gainpath'.
        while not os.path.exists(gainpath) and gainpath:
            datapath = input("That path doesn't exist. " + 
                "Enter another path to the noise data: ")
        
        gain = None
        if gainpath:
            gain = np.loadtxt(gainpath)

        # Requesting experiment information and where to save outputs
        detector = input('Enter the detector ID: ')
        pos = input('Enter the detector positon: ')
        voltage = input('Enter the voltage in Volts (no unit symbol): ')
        temp = input('Enter the temperature in Celsius (no unit symbol): ')
        save_dir = input('Enter a directory to save outputs to: ')

        noise = Noise(datapath, detector, voltage, temp, pos, gain=gain)

        pixel_dir = input('Enter a directory to save pixel spectra to: ')

        # Processing data
        print('Calculating fwhm and count data...')
        noise.noise_map(data_dir=save_dir, plot_dir=pixel_dir)

        print('Done!')
