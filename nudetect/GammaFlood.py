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


class GammaFlood(Experiment):
    '''
    A class containing important experiment parameters with methods to supply
    data analysis functions for gamma flood data.

    Public attributes:
        raw_data_path: str
            Path to gamma flood data. Should be a FITS file. Used to access
            data and to construct new file names.
        detector: str
            The detector ID.
        source: a 'nudetect.Source' instance
            The X-ray source. This supplies documentation of the source and
            information about its spectral lines and fitting them.
        voltage: str
            Bias voltage in Volts
        temp: str
            Temperature of the detector in degrees Celsius
        etc: str
            Any other important information to include
        save_dir: str
            A default directory to save file outputs to from this instance's 
            methods. Method arguments let one choose a subdirectory of this 
            path, or override it altogether.

            If the string passed to 'save_dir' has an empty pair of curly 
            braces '{}', they will be replaced by the detector ID 
            'self.detector'. For example, if self.detector == 'H100' and 
            save_dir == 'figures/{}/pixels', then the directory that 
            'save_path' points to is 'figures/H100/pixels'.
            (default: '')

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
    def __init__(self, raw_data_path, detector, source, voltage, temp, 
        data_dir='', plot_dir='', save_dir='', etc=''):

        '''
        Initializes an instance of the 'GammaFlood' class.

        Arguments:
            raw_data_path: str
                Path to gamma flood data. Should be a FITS file. Used to access
                data and to construct new file names.
            detector: str
                The detector ID.
            source: a 'nudetect.Source' instance
                The X-ray source. This supplies documentation of the source and
                information about its spectral lines and fitting them.
            voltage: str
                Bias voltage in Volts
            temp: str
                Temperature of the detector in degrees Celsius


        Keyword Arguments:
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
                Any other important information to include
        '''
        if not isinstance(source, Source):
            raise TypeError("'source' must be 'nudetect.Source' instance.")

        voltage = str(voltage)
        temp = str(temp)

        # Remove any unit symbols from voltage and temperature
        voltage = voltage.translate(self.numericize)
        temp = temp.translate(self.numericize)

        # Initialize data-based attributes to 'None'
        self.count_map = None
        self.gain = None
        self.gain_dict = {}
        self.spectrum = None

        # Set user-supplied attributes
        self.raw_data_path = raw_data_path
        self.detector = detector
        self.source = source
        self.voltage = voltage
        self.temp = temp
        self.etc = etc

        self._set_save_dir(save_dir)
        self._set_save_dir(plot_dir, save_type='plot')
        self._set_save_dir(data_dir, save_type='data')


    def load_raw_data(self):
        '''Loads raw data from FITS file into attributes of this instance.'''
        self.raw_data_1d, self.raw_data_2d = fits_to_df(self.raw_data_path,
            colnames={'RAWX', 'RAWY', 'PH', 'PH_COM', 'STIM'})


    #
    # Heavy-lifting data analysis methods: 'gen_count_map', 'gen_quick_gain',
    # and 'gen_spectrum'.
    #

    def gen_count_map(self, mask_PH=True, mask_STIM=True, 
        mask_sigma_below=None, mask_sigma_above=None, 
        save_data=True, data_ext='.txt', data_dir='', data_subdir=''):
        '''
        Generates event count data for each pixel for raw gamma flood data.

        Keyword Arguments:
            mask_PH: bool
                If True, non-positive pulse heights will not be counted 
                as counts.
                (default: True)
            mask_STIM: bool
                If True, stimulated events will no be counted as counts.
                (default: True)
            mask_sigma_above: int or float
                If a pixel has counts this many standard deviations above
                the mean, it will be masked in the output. If None, no 
                pixels will be masked on this basis.
                (default: None)
            mask_sigma_below: int or float
                If a pixel has counts this many standard deviations below
                the mean, it will be masked in the output. If None, no 
                pixels will be masked on this basis.
                (default: None)
            save_data: bool 
                If True, saves count_map as an ascii file.
                (default: True)
            data_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'data_dir' attribute. If an empty string,
                will default to the 'data_dir' attribute.
                If the string passed to 'data_dir' has an empty pair of curly 
                braces '{}', they will be replaced by the detector ID 
                'self.detector'. For example, if self.detector == 'H100' and 
                data_dir == 'figures/{}/pixels', then then the directory to 
                which the data is saved is 'figures/H100/pixels'.
                (default: '')
            data_subdir: str
                A path to a sub-directory of 'data_dir' to which the file will
                be saved. Empty curly braces '{}' are formatted the same way
                as in 'data_dir'. 
                (default: '')
            data_ext: str
                The file name extension for the count_map file. 
                (default: '.txt')

        Return:
            count_map: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents the number of
                counts read by the detector pixel at the corresponding index.
        '''
        # Generating the save path, if needed.
        if save_data:
            save_path = self.construct_path('data', ext=data_ext, 
                save_dir=data_dir, description='count_data', 
                subdir=data_subdir)


        # Masking out non-positive pulse heights and/or artificially 
        # stimulated events, if requested
        mask = pd.Series(np.ones_like(self.raw_data_1d.loc[:, 'STIM']))

        if mask_STIM:
            mask &= (self.raw_data_1d.loc[:, 'STIM'] == 0)
        if mask_PH:
            mask &= (self.raw_data_1d.loc[:, 'PH'] > 0)

        # Generate the count_map from event data
        count_map = np.zeros(self._det_shape, dtype='uint32')

        for col in self._col_iter:
            col_mask = self.raw_data_1d.loc[:, 'RAWX'] == col
            for row in self._row_iter:
                row_mask = self.raw_data_1d.loc[:, 'RAWY'] == row
                maprow = row - self._start_row
                mapcol = col - self._start_col
                count_map[maprow, mapcol] = (mask & col_mask & row_mask).values.sum()

        # Masking pixels that were turned off, before calculating
        # the rest of the masks (otherwise they'll skew mean and stddev)
        count_map = np.ma.masked_values(count_map, 0.0)

        # Masking pixels whose counts are too many standard deviations
        # away from mean.
        if mask_sigma_above is not None:
            mask_value = np.mean(count_map)\
                + (np.std(count_map) * mask_sigma_above)
            count_map = np.ma.masked_greater(count_map, mask_value)

        if mask_sigma_below is not None:
            mask_value = np.mean(count_map)\
                - (np.std(count_map) * mask_sigma_below)
            count_map = np.ma.masked_less(count_map, mask_value)

        # Saves the 'count_map' array as an ascii file.
        if save_data:
            np.savetxt(save_path, count_map)

        count_map = np.ma.masked_values(count_map, 0.0)

        # Storing count data in our 'GammaFlood' instance
        self.count_map = count_map

        return count_map


    def gen_quick_gain(self, energy=None, chan_range=None, gain_estimate=0.014,
        search_width=3000, fit_below=100, fit_above=200, interpolations=2,
        save_plot=True, plot_dir='', plot_subdir='', plot_ext='.pdf', 
        save_data=True, data_dir='', data_subdir='', data_ext='.txt',
	misc_mask=1, etc=''):
        '''
        Generates gain correction data from the raw gamma flood event data.
        Currently, the fitting done might fail for sources other than Am241.

        Keyword Arguments:
            energy: int
                The approximate energy in keV of the line being fit. If None,
                a the default value can be found in the 'default_energy'
                attribute of the 'Source' instance being used, or in the dict
                'Source.default_energies'.
                (default: None)
            chan_range: Tuple(int, int)
                Allows the user to manually specify the channels in between
                which to look for the specral line. If None, it is calculated
                using the 'Source.chan_range' method.
                (default: None)
            gain_estimate: float
                An estimate of the gain for the detector. Used to estimate the
                location of the spectral line in units of channels.
                (energy in keV) / gain = (energy in channels)
                (defautl: 0.014)
            width: int
                The width of the channel interval in which to search for the 
                spectral line.
                (default: 3000)
            fit_below: int
                Channels this far below the centroid won't be considered in 
                fitting a gaussian to the spectral peak. Should be smaller 
                than 'fit_above' due to thick low-energy tails.
            fit_above: int
                Channels this far above the centroid won't be considered in 
                fitting a gaussian to the spectral peak.
            interpolations: int
                The number of times to attempt interpolating gain for pixels
                whose spectra couldn't be fit to a Gaussian.
                (default: 2)
            save_plot: bool
                If true, plots and energy spectrum for each pixel and saves
                the figure.
                (default: True)
            plot_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'plot_dir' attribute. If an empty string,
                will default to the attribute 'plot_dir'.
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
                If True, saves gain data as a .txt file.
                (default: True)
            data_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'data_dir' attribute. If an empty string,
                will default to the attribute 'data_dir'.
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
                The file name extension for the gain file. 
                (default: '.txt')
            misc_mask: 1D numpy.array-like
                An additional mask to be applied to the raw pulse height data.
                (default: 1)
            etc: str
                Other important information (e.g. masks used) that will be
                appended to data and plot filenames.
                (default: '')

        Return:
            gain: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents its  
                respective pixel's gain, where channels * gain = energy.
        '''

        if save_data:
            data_path = self.construct_path('data', ext=data_ext, 
                description='gain', save_dir=data_dir, subdir=data_subdir,
                etc=etc)

        if save_plot:
            plot_path = self.construct_path('plot', description='gain', 
                ext=plot_ext,  save_dir=plot_dir, subdir=plot_subdir,
                etc=etc)

        # Setting parameters of this emission line to help with fitting it
        # later. 'energy' is the line's energy in keV. 'chan_low' and 
        # 'chan_high' indicates the range of channels in which we should
        # expect to find the emission line.
        energy = self.source.line(energy)
        chan_low, chan_high = self.source.chan_range(energy, gain_estimate,
            width=search_width)

        maxchannel = 10000
        bins = np.arange(1, maxchannel)
        gain = np.zeros(self._det_shape)

        # Iterating through pixels
        for col in self._col_iter:
            col_mask = self.raw_data_1d.loc[:, 'RAWX'] == col
            for row in self._row_iter:
                row_mask = self.raw_data_1d.loc[:, 'RAWY'] == row

                # Getting pulse height in channels for all events for the 
                # current pixel. We store this in 'channel' as a numpy.ndarray,
                # since we don't need the index of the original DataFrame, and
                # np.histogram should be faster on an ndarray than a DataFrame.
                channel = self.raw_data_1d.loc[
                    (col_mask) & (row_mask) & (misc_mask), 'PH'].values

                # If there were events at this pixel, fit the strongest peak
                # in the channel spectrum with a Gaussian.
                if len(channel):
                    # 'spectrum' contains counts at each channel
                    spectrum, edges = np.histogram(channel, bins=bins, 
                        range=(0, maxchannel))
                    # 'centroid' is the channel with the most counts in the 
                    # interval between 'chan_low' and 'chan_high'.
                    centroid = np.argmax(spectrum[chan_low:chan_high]
                       ) + chan_low
                    # Excluding funky tails for the fitting process.
                    fit_channels = np.arange(
                        centroid - fit_below, centroid + fit_above)
                    g_init = models.Gaussian1D(amplitude=spectrum[centroid], 
                        mean=centroid, stddev=75)
                    fit_g = fitting.LevMarLSQFitter()
                    g = fit_g(g_init, fit_channels, spectrum[fit_channels])

                    # If we can determine the covariance matrix (which implies
                    # that the fit succeeded), then calculate this pixel's gain
                    if fit_g.fit_info['param_cov'] is not None:
                        maprow = row - self._start_row
                        mapcol = col - self._start_col
                        gain[maprow, mapcol] = energy / g.mean
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
                                frac_err * energy * 1000)))
                            str_fwhm = str(int(round(
                                    energy * 1000 * g.fwhm / g.mean, 0)))
                            plt.text(
                                maxchannel * 3 / 5, spectrum[centroid] * 3 / 5,
                                r'$\mathrm{FWHM}=$' + str_fwhm + r'$\pm$' 
                                + str_err + ' eV', fontsize=13)

                            plt.hist(
                                np.multiply(channel, gain[maprow, mapcol]), 
                                bins=np.multiply(bins, gain[maprow, mapcol]),
                                range=(0, maxchannel * gain[maprow, mapcol]), 
                                histtype='stepfilled')

                            plt.plot(
                                fit_channels * gain[maprow, mapcol], 
                                g(fit_channels), label='Gaussian fit')

                            plt.ylabel('Counts')
                            plt.xlabel('Energy')
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(f'{plot_path}_x{col}_y{row}{plot_ext}')
                            plt.close()

        del col_mask, row_mask, channel

        # Interpolate gain for pixels where fit was unsuccessful. Do it
        # multiple times if specified.
        for _ in range(interpolations):
            newgain = np.zeros(self._det_shape_buff)
            # Note that newgain's indices will be shifted over one from 'gain'.
            newgain[1:self._num_rows + 1,
                    1:self._num_cols + 1] = gain
            # 'empty' contains indices at which the fit was unsuccessful
            empty = np.transpose(np.nonzero(gain == 0.0))
            # Iterating through pixels with failed fitting.
            for x in empty:
                # 'empty_grid' is the 3 x 3 array of gain values around the  
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

        gain = np.ma.masked_values(gain, 0.0)
        self.gain = gain
        self.gain_dict[int(round(energy, 0))] = gain

        return gain


    def gen_spectrum(self, gain=None, bins=10000, energy_range=(0.01, 120), 
        save_data=True, data_ext='.txt', data_dir='', data_subdir='',
        misc_mask=1, etc=''):
        '''
        Applies gain correction to get energy data, and then bins the events
        by energy to obtain a spectrum.

        Keyword Arguments:
            gain: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents its  
                respective pixel's gain, where channels * gain = energy. If 
                None, defaults to the array in 'self.gain'.
                (default: None)
            bins: int
                Number of energy bins
                (default: 10000)
            energy_range: tuple of numbers
                The bins will be made between these energies
                (default: (0.01, 120))
            save_data:
                If True, 'spectrum' will be saved as an ascii file. Parameters
                relevant to this saving are below
            data_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'data_dir' attribute. If an empty string,
                will default to the attribute 'data_dir'.
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
                The file name extension for the count_map file. 
                (default: '.txt')
            misc_mask: 1D numpy.array-like
                An additional mask to be applied to the raw pulse height data.
                (default: 1)
            etc: str
                Other important information (e.g. masks used) that will be
                appended to data and plot filenames.
                (default: '')

        Return:
            spectrum: 2D numpy.ndarray
                This array represents a histogram wrt the energy of an event.
                spectrum[0] is a 1D array of counts in each bin, and  
                spectrum[1] is a 1D array of the middle enegies of each bin in
                keV. E.g., if the ith bin counted events between 2 keV and 4 
                keV, then the value of spectrum[1, i] is 3.
        '''
        # Generating the save path, if needed.
        if save_data:
            save_path = self.construct_path('data', ext=data_ext, 
                save_dir=data_dir, subdir=data_subdir, description='spectrum',
                etc=etc)

        # If no gain is passed, take it from the GammaFlood instance.
        if gain is None:
            gain = self.gain

        # Adding a buffer of zeros around the 'gain' array. (Note that the
        # indices will now be shifted over by one.)
        gain_buffed = np.zeros(self._det_shape_buff)
        gain_buffed[1:self._num_rows + 1,
                    1:self._num_cols + 1] = gain
        gain = gain_buffed

        # PH_COM is a list of length 9 corresponding to the charge in pixels 
        # surrounding the event.
        #
        # PH_COM -> gain correct -> sum positive elements in the 3x3 array -> 
        # event in energy units

        ph_com = self.raw_data_2d['PH_COM']

        # 'energies' is a list of event energies in keV.
        energies = []
        energy_map = [[[]
            for col in range(self._num_cols)]
            for row in range(self._num_rows)]

        # iterating through pixels in the selected region
        for row in self._row_iter:
            row_mask = self.raw_data_1d.loc[:, 'RAWY'] == row
            for col in self._col_iter:
                col_mask = self.raw_data_1d.loc[:, 'RAWX'] == col

                maprow = row - self._start_row
                mapcol = col - self._start_col
                # Getting PH_COM values ('pulses') of all events at current 
                # pixel and storing as an ndarray in 'pulses'.
                pulses = ph_com.loc[(row_mask) & (col_mask) & (misc_mask)].values
                # The gain for the 3 x 3 grid around this pixel
                gain_grid = gain[maprow:maprow + 3, mapcol:mapcol + 3]
                # iterating through the PH_COM values for this pixel
                for pulse in pulses:
                    # Append the sum of positive energies in the 
                    # pulse grid to 'energies'
                    pulse_grid = pulse.reshape((3, 3))
                    mask = (pulse_grid > 0).astype(int)
                    energies.append(np.sum(np.multiply(
                        np.multiply(mask, pulse_grid), gain_grid)))
                    energy_map[row][col].append(
                        (mask * pulse_grid * gain_grid).sum())

        # Binning by energy
        energies = [e for row in energy_map for col in row for e in col]
        counts, edges = np.histogram(energies, bins=bins, range=energy_range)
        del energies

        # Getting the midpoint of the edges of each bin, representing an energy
        # in keV.
        midpoints = (edges[:-1] + edges[1:]) / 2

        # Consolidating 'counts' and 'midpoints' into a 2D array 'spectrum'.
        spectrum = np.empty((2, counts.size))
        spectrum[0, :] = counts
        spectrum[1, :] = midpoints

        if save_data:
            np.savetxt(save_path, spectrum)

        self.spectrum = spectrum
        self._energy_map = energy_map

        return spectrum


    #
    # Plotting method with light data analysis: 'plot_spectrum'.
    #

    def plot_spectrum(self, energy=None, spectrum=None, fit_below=80, 
        fit_above=150, title='', save_plot=True, plot_ext='.pdf', plot_dir='',
        plot_subdir='', etc=''):
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
            fit_below: int
                Channels this far below the centroid won't be considered in 
                fitting a gaussian to the spectral peak. Should be smaller 
                than 'fit_above' due to thick low-energy tails.
            fit_above: int
                Channels this far above the centroid won't be considered in 
                fitting a gaussian to the spectral peak.
            title: str
                The figure title. If 'auto', a title is generated using the
                'title' method. If an empty string is passed, no title
                is shown.
                (default: '')
            save:
                If True, 'spectrum' will be saved as an ascii file. Parameters
                relevant to this saving are below
            data_dir: str
                The directory to which the file will be saved, overriding any
                path specified in the 'data_dir' attribute. If an empty string,
                will default to the attribute 'data_dir'.
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
            ext: str
                The file name extension for the count_map file. 
                (default: '.pdf')
            etc: str
                Other important information (e.g. masks used) that will be
                appended to data and plot filenames.
                (default: '')

        '''
        # Constructing a save path, if needed
        if save_plot:
            save_path = self.construct_path('plot', ext=plot_ext, 
                save_dir=plot_dir, subdir=plot_subdir, 
                description='energy_spectrum', etc=etc)

        # If no spectrum is supplied take it from the instance.
        if spectrum is None:
            spectrum = self.spectrum
        # If no title is passed, construct one
        if title == 'auto':
            title = self.title('Spectrum')

        maxchannel = 10000

        # 'energy' is the precise energy of the line being fit.
        energy = self.source.line(energy)
        # 'energy_low' and 'energy_high' represent the interval, in keV, in 
        # which we expect the emission line to be.
        energy_low, energy_high = energy - 2, energy + 2
        # 'bool_spectrum' is a boolean array with 'True' at ea. index of 
        # 'spectrum' whose energy is in the above interval.
        bool_spectrum = np.logical_and(spectrum[1] > energy_low, 
                                       spectrum[1] < energy_high)
        start = np.argmax(bool_spectrum)
        end = len(bool_spectrum) - np.argmax(bool_spectrum[::-1])
        del bool_spectrum, energy_low, energy_high
        # 'centroid' is the index of the bin with the most counts in 
        # the above interval.
        centroid = np.argmax(spectrum[0, start:end]) + start
        # Fit in an asymetrical domain about the centroid to avoid 
        # low energy tails.
        fit_channels = np.arange(centroid - fit_below, centroid + fit_above)
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
        display_fwhm = str(int(round(energy * 1000 * g.fwhm / g.mean, 0)))
        display_err  = str(int(round(frac_err * energy * 1000)))

        plt.text(70, spectrum[0, centroid] * 3 / 5, 
            r'$\mathrm{FWHM}=' + display_fwhm + r'\pm' + display_err + r'$ eV',
            fontsize=13)

        plt.plot(spectrum[1], spectrum[0], label=self.source.latex)
        plt.plot(spectrum[1, fit_channels], g(fit_channels), 
            label = 'Gaussian fit')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()

        plt.title(title)
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_path)
