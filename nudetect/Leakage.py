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

# Data analysis packages
import numpy as np
import pandas as pd
import astropy.io.ascii as asciio

# Plotting packages
import matplotlib.pyplot as plt

# Internal imports
from .util import to_set


class Leakage(Experiment):
    '''
    A class containing important experiment parameters with methods to supply
    data analysis functions for leakage current data.

    Public attributes:
        raw_data_path: str
            A path to a directory containing ascii files of leakage data.
        detector: str
            The detector ID.
        temps: set of numbers
            The set of temperatures at which leakage current was tested.
        cp_voltages: set of numbers
            The bias voltages in Volts at which leakage current was tested
            using charge-pump mode.
            (default: {100, 200, 300, 400, 500, 600})
        n_voltages: set of numbers
            The bias voltages in Volts at which leakage current was tested
            using normal mode.
            (default: {300, 400, 500, 600})
        all_voltages: set of numbers
            All bias voltages at which leakage current was tested (could have
            been in normal mode, charge-pump mode, or both). Generated from 
            'cp_voltages' and 'n_voltages'.
        num_trials: int
            The number of trials/measurements of leakage current done given
            this raw_data_path (at different combinations of mode, temperature, and
            bias voltage). Calculated from 'cp_voltages', 'n_voltages', and 
            'temps'.
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

        stats: pandas.DataFrame
            A DataFrame with 1 row for each combination of parameters. The
            columns are described as follows: 
                'mode'    : Can be 'CP' or 'N' (charge-pump or normal)
                'voltage' : The bias voltage in Volts
                'temp'    : The temperature in Celsius
                'mean'    : The mean leakage current across the pixels
                'stddev'  : The corresponding standard deviation
                'outliers': Number of outlier pixels
        maps: 3D numpy.ndarray
            An array of shape (n, 32, 32), where 'n' is the value held by
            the 'num_trials' attribute, which indicates the number of 
            combinations of mode, voltage, and temperature. Slicing like
            'maps[n]' gives a 32 x 32 pixel map of leakage current.

    '''
    def __init__(self, raw_data_path, detector, temps, 
        cp_voltages={100, 200, 300, 400, 500, 600}, 
        n_voltages={300, 400, 500, 600},
        pos=0, data_dir='', plot_dir='', save_dir='', etc=''):
        '''
        Initialize an instance of the 'Leakage' class.

        Arguments:
            raw_data_path: str
                A path to a directory containing ascii files of leakage data.
            detector: str
                The detector ID.
            temps: set of numbers
                The set of temperatures at which leakage current was tested.

        Keyword arguments:
            cp_voltages: set of numbers
                The bias voltages in Volts at which leakage current was tested
                using charge-pump mode.
                (default: {100, 200, 300, 400, 500, 600})
            n_voltages: set of numbers
                The bias voltages in Volts at which leakage current was tested
                using normal mode.
                (default: {300, 400, 500, 600})
            pos: int
                The detector position.
                (default: 0)
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
        # Convert temperatures and voltages to sets to avoid repeats
        temps = to_set(temps)
        cp_voltages = to_set(cp_voltages)
        n_voltages = to_set(n_voltages)

        self.raw_data_path = raw_data_path
        self.detector = detector
        self.temps = temps
        self.cp_voltages = cp_voltages
        self.n_voltages = n_voltages
        self.all_voltages = cp_voltages | n_voltages
        self.num_trials = (len(cp_voltages) + len(n_voltages)) * len(temps)
        self.pos = int(pos)
        self.etc = etc

        self._set_save_dir(save_dir)
        self._set_save_dir(plot_dir, save_type='plot')
        self._set_save_dir(data_dir, save_type='data')

    #
    # Small helper and wrapper methods: 'title' and 'slice_stats'
    #

    def title(self, plot, conditions=None):
        '''
        Returns a plot title based on the instance's attributes and the 
        type of plot. 'plot' is a str indicating the type of plot.
        '''
        # Formatting the temperature and voltage conditions in the title,
        # if specified.
        if conditions is not None:
            mode, temp, voltage = conditions

            temp = r'$' + str(temp) + r'^{\circ}$C'
            conditions = f'({temp}'

            voltage = r'$' + str(voltage) + r'$V'
            conditions += f', {voltage}'

            conditions += f', {mode})'

        title = f'Leakage {plot} {self.detector} {conditions}'.strip()

        if self.etc:
            title += f' -- {self.etc}'

        return title


    def slice_stats(self, mode=None, temp=None, voltage=None):
        '''
        A wrapper around the '.loc' method of pandas. This returns
        row(s) of the 'stats' DataFrame containing the given mode(s), 
        temperature(s), and voltage(s). If 'None' (the default value) is
        passed to any of the arguments, the DataFrame won't be sliced
        with respect to the arguments respective value.

        For example, setting mode='CP', temp={-5, 0, 5}, and leaving
        voltage=None will slice out all rows with charge-pump mode, a
        temperature of -5, 0, or 5 degrees Celsius, and any voltage.

        For more advanced indexing options, the the pandas documentation:
            https://pandas.pydata.org/pandas-docs/stable/indexing.html
        The section on 'Boolean Indexing' is particularly helpful.

        Keyword Arguments:
            mode: str or set of str
                String values can be can be 'CP' or 'N'. Indicates whether the 
                desired measurement was done in charge-pump or normal mode.
                (default: None)
            temp: int or set of ints
                Indicates temperature in degrees Celsius at which the desired 
                measurement was done.
                (default: None)
            voltage: int or set of ints
                The bias voltage in Volts at which the desired measurement
                was done.
                (default: None)

        Return: pandas.DataFrame
            A slice of the 'stats' attribute's DataFrame, as at the beginning 
            of this method's docstring.
        '''
        # Aliasing the 'stats' attribute
        df = self.stats

        # Formatting inputs
        if temp is not None: temp = to_set(temp)
        if voltage is not None: voltage = to_set(voltage)
        if mode is not None:
            mode = to_set(mode)
            # Ensuring 'mode' contains uppercase strings only
            for m in mode:
                mode.remove(m)
                mode.add(m.upper())

        # Creating a Series full of True with the same shape as one
        # column from 'self.stats'.
        true_df = pd.Series(np.ones(df.shape[0]), dtype=bool)

        # If 'None' was supplied for mode, temp, or voltage, set its
        # respective boolean Series to 'true_df', so its respective
        # value is ignored when slicing the 'stats' DataFrame. Otherwise,
        # generate the boolean Series
        if mode is None: 
            bool_mode = true_df
        else: 
            bool_mode = df.loc[:, 'mode'].isin(mode)

        if temp is None: 
            bool_temp = true_df
        else: 
            bool_temp = df.loc[:, 'temp'].isin(temp)

        if voltage is None: 
            bool_voltages = true_df
        else: 
            bool_voltages = df.loc[:, 'voltage'].isin(voltage)

        # Generating a boolean DataFrame
        bool_df = (bool_temp) & (bool_mode) & (bool_voltages)

        return df.loc[bool_df]


    def slice_maps(self, mode=None, temp=None, voltage=None):
        '''
        If 'None' (the default value) is
        passed to any of the arguments, the DataFrame won't be sliced
        with respect to the arguments respective value.

        For example, setting mode='CP', temp={-5, 0, 5}, and leaving
        voltage=None will slice out all rows with charge-pump mode, a
        temperature of -5, 0, or 5 degrees Celsius, and any voltage.

        For more advanced indexing options, the the pandas documentation:
            https://pandas.pydata.org/pandas-docs/stable/indexing.html
        The section on 'Boolean Indexing' is particularly helpful.

        Keyword Arguments:
            mode: str or set of str
                String values can be can be 'CP' or 'N'. Indicates whether the 
                desired measurement was done in charge-pump or normal mode.
                (default: None)
            temp: int or set of ints
                Indicates temperature in degrees Celsius at which the desired 
                measurement was done.
                (default: None)
            voltage: int or set of ints
                The bias voltage in Volts at which the desired measurement
                was done.
                (default: None)

        Return: 2D or 3D numpy.ndarray
            A slice of the 'stats' attribute's DataFrame, as at the beginning 
            of this method's docstring.
        '''
        idx = self.slice_stats(mode, temp, voltage).index
        return self.maps[idx]


    #
    # Heavy-lifting data analysis method: 'gen_leakage_maps'
    #

    def gen_leak_maps(self, save_data=True, data_dir='', data_subdir='', 
        data_ext='.csv'):
        '''
        For each combination of mode (charge-pump or normal), voltage, and 
        temperature, formats leakage current data into 32 x 32 pixel maps and 
        calculates mean, standard deviation, and number of outlier pixels.

        The indices in the 'stats' pandas.DataFrame and the 'maps' 3D
        numpy.ndarray correspond to each other. I.e., stats[i] contains the
        mean, stddev, outliers, and experimental conditions for the leakage
        map in maps[i].

        Keyword Arguments:
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
                The file name extension for the file containing the 'stats'
                return value. The file will be a CSV no matter the extension.
                The 'leak_maps' return file is also saved to a numpy binary
                file, so its extension cannot be changed.
                (default: '.csv')

        Return: Tuple(pandas.DataFrame, numpy.ndarray)
            stats: pandas.DataFrame
                A data frame with 1 row for each combination of parameters. The
                columns are described as follows: 
                    'mode'    : Can be 'CP' or 'N' (charge-pump or normal)
                    'voltage' : The bias voltage in Volts
                    'temp'    : The temperature in Celsius
                    'mean'    : The mean leakage current across the pixels (pA)
                    'stddev'  : The corresponding standard deviation (pA)
                    'outliers': Number of outlier pixels
            maps: 3D numpy.ndarray
                An array of shape (n, 32, 32), where 'n' is the value held by
                the 'num_trials' attribute, which indicates the number of 
                combinations of mode, voltage, and temperature. Slicing like
                'maps[n]' gives a 32 x 32 pixel map of leakage current.
        '''
        # Generating a save path, if necessary
        if save_data:
            stats_path = self.construct_path('data', description='leak_stats', 
                ext=data_ext, save_dir=data_dir, subdir=data_subdir)
            maps_path = self.construct_path('data', description='leak_maps',
                ext='.npy', save_dir=data_dir, subdir=data_subdir)

        self.stats = pd.DataFrame(np.zeros((self.num_trials, 6)),
            columns=['mode', 'temp', 'voltage', 'mean', 'stddev', 'outliers'])

        # This array will store leakage maps for each combination of 
        # mode, voltage, and temperature.
        self.maps = np.empty((
            self.num_trials, self._num_rows, self._num_cols))

        # Sets 'filename' to the last directory in 'self.raw_data_path'.
        filename = os.path.basename(self.raw_data_path)

        # 'start' and 'end' define the indices of the pixels at the given 
        # detector position are.
        start = -1024 * (1 + self.pos)
        end = start + 1024

        idx = 0 # for populating 'leak_maps' and 'stats'.

        # Iterate through temperatures
        for temp in self.temps:
            # First, construct maps 'cp_zero' and 'n_zero' of the leakage 
            # current at bias voltage of zero as a control.
            n_zero = np.empty(self._det_shape)
            cp_zero = np.empty(self._det_shape)

            # TODO:
            # I recommend reading this in with pandas, like the rest of this
            # module does, but I don't have time for that rn, and this part
            # of the code runs pretty fast anyway, though you could probably
            # get it around twice as fast using pandas.
            cp_zero_data = asciio.read(
                f'{self.raw_data_path}/{filename}_{temp}C.C0V.txt')
            n_zero_data = asciio.read(
                f'{self.raw_data_path}/{filename}_{temp}C.N0V.txt')
            
            for pix in range(start, end): # Iterating through pixels
                # Pixel coordinates in charge pump mode
                cp_col = cp_zero_data.field('col4')[pix]
                cp_row = cp_zero_data.field('col5')[pix]

                # Pixel coordinates in normal mode
                n_col = n_zero_data.field('col4')[pix]
                n_row = n_zero_data.field('col5')[pix]

                # Leakage at this pixel in each mode.
                cp_zero[cp_row, cp_col] = cp_zero_data.field('col6')[pix]
                n_zero[n_row, n_col] = n_zero_data.field('col6')[pix]

            # Iterating though non-zero bias voltages
            for voltage in self.all_voltages:
                # 'modes' keeps record of with which mode(s) the current 
                # voltage was tested.
                modes = set()
                if voltage in self.cp_voltages:
                    modes.add('CP')
                if voltage in self.n_voltages:
                    modes.add('N')

                for mode in modes:
                    leak_map = np.zeros(self._det_shape)

                    # Set a conversion constant between raw readout and 
                    # current in pA based on the mode. 
                    if mode == 'CP':
                        conversion = 1.7e3 / 3000
                    elif mode == 'N':
                        conversion = 1.7e3 / 150

                    # Read in the data file for the current voltage and 
                    # temperature in CP mode.
                    data = asciio.read(f'{self.raw_data_path}/{filename}_'
                        + f'{temp}C.{mode[0]}{voltage}V.txt')

                    # Generating a leakage current map at the current voltage,
                    # realtive to what we had at 0V.
                    for pix in range(start, end): # iterating through pixels
                        col = data.field('col4')[pix]
                        row = data.field('col5')[pix]
                        leak_map[row, col] = (data.field('col6')[pix] 
                            - cp_zero[row, col]) * conversion

                    del data

                    leak_map = np.ma.masked_where(leak_map > 100, leak_map)

                    mean = np.mean(leak_map)
                    stddev = np.std(leak_map)
                    # 'outliers' in the number of pixels whose leakage 
                    # currents are 5 standard deviations from the mean.
                    outliers = np.sum(np.absolute(leak_map - mean)
                        > 5 * stddev)

                    # Record the data

                    # Populate a row of the stats DataFrame with the
                    # corresponding parameters and measurements for this trial
                    row = [mode, temp, voltage, mean, stddev, outliers]
                    self.stats.loc[idx] = row
                    # Populate a layer of the leak_maps array with the leakage 
                    # leakage current map for the same parameters at the same 
                    # index as above.
                    self.maps[idx] = leak_map

                    idx += 1

        # Saving data
        if save_data:
            # Leakage statistics go to a CSV file. Since the index is trivial
            # and inferred by pd.read_csv, we omit it in the save file.
            self.stats.to_csv(stats_path, index=False)
            # The amalgam of leakage maps go to a .npy file (numpy binary file
            # - can't do ascii because it's a 3D array).
            np.save(maps_path, self.maps)

        return self.stats, self.maps


    # 
    # Plotting methods: 'plot_leak_maps', 'plot_leak_hists',  
    # 'plot_line_current', and 'plot_line_outliers'.
    #


    def plot_leak_maps(self, mode=None, temp=None, voltage=None, 
        cmap_name='inferno', cb_label='', vmin=None, vmax=None, title='', 
        save_plot=True, plot_ext='.pdf', plot_dir='', plot_subdir=''):
        '''
        Plots a pixel histogram of leakage current at the designated 
        combinations of mode, temperature, and leakage for this experiment. 
        How these combinations are made is specified in the docstring for the 
        'Leakage.slice_stats' method. 'plot_leak_maps' is essentially a 
        wrapper around the 'Experiment.plot_pixel_map' method.

        Keyword Arguments:
            mode: str or set of str
                String values can be can be 'CP' or 'N'. Indicates whether the 
                desired measurement was done in charge-pump or normal mode.
                (default: None)
            temp: int or set of ints
                Indicates temperature in degrees Celsius at which the desired 
                measurement was done.
                (default: None)
            voltage: int or set of ints
                The bias voltage in Volts at which the desired measurement
                was done.
                (default: None)
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
        '''
        inds = self.slice_stats(mode, temp, voltage).index

        for i in inds:
            row = self.stats.loc[i]
            leak_map = self.maps[i]

            mode = row.at['mode']
            temp = int(row.at['temp'])
            voltage = int(row.at['voltage'])

            conditions = (mode, temp, voltage)
            etc = f'{mode}_{temp}C_{voltage}V'

            if title  == 'auto':
                title = self.title('Map', conditions)

            self.plot_pixel_map('Leakage', leak_map, cmap_name=cmap_name, 
                cb_label=cb_label, vmin=vmin, vmax=vmax, title=title, 
                save_plot=save_plot, plot_ext=plot_ext, plot_dir=plot_dir, 
                plot_subdir=plot_subdir, etc=etc)

            plt.close()


    def plot_leak_hists(self, mode=None, temp=None, voltage=None, 
        bins=70, hist_range=None, title='', text_pos='right', save_plot=True, 
        plot_dir='', plot_subdir='', plot_ext='.pdf', **kwargs):
        '''
        Plots a pixel histogram of leakage current at the designated 
        combinations of mode, temperature, and leakage for this experiment. 
        How these combinations are made is specified in the docstring for the 
        'Leakage.slice_stats' method. 'plot_leak_hists' is essentially a 
        wrapper around the 'Experiment.plot_pixel_hist' method.

        Keyword Arguments:
            mode: str or set of str
                String values can be can be 'CP' or 'N'. Indicates whether the 
                desired measurement was done in charge-pump or normal mode.
                (default: None)
            temp: int or set of ints
                Indicates temperature in degrees Celsius at which the desired 
                measurement was done.
                (default: None)
            voltage: int or set of ints
                The bias voltage in Volts at which the desired measurement
                was done.
                (default: None)
            bins: int
                The number of bins in which to histogram the data. Passed 
                directly to plt.hist.
                (default: 50)
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
        '''
        inds = self.slice_stats(mode, temp, voltage).index

        for i in inds:
            row = self.stats.loc[i]
            leak_map = self.maps[i]

            mode = row.at['mode']
            temp = int(row.at['temp'])
            voltage = int(row.at['voltage'])

            conditions = (mode, temp, voltage)
            etc = f'{mode}_{temp}C_{voltage}V'

            if title  == 'auto':
                title = self.title('Histogram', conditions)

            self.plot_pixel_hist('Leakage', leak_map, bins=bins, 
                hist_range=hist_range, title=title, text_pos=text_pos, 
                save_plot=save_plot, plot_dir=plot_dir, 
                plot_subdir=plot_subdir, plot_ext=plot_ext, etc=etc, **kwargs)

            plt.close()


    def plot_line_current(self, title='', mode='CP', save_plot=True, 
        plot_dir='', plot_subdir='', plot_ext='.pdf', etc=''):
        '''
        Plots mean leakage current versus bias voltage as a line plot, with a 
        line for each temperature. Done for only one mode. Error bars included
        and represent the standard deviation of leakage current across pixels.

        Keyword Arguments:
            title: str
                The figure title. If 'auto', a title is generated using the
                'title' method. If an empty string is passed, no title
                is shown.
                (default: '')
            mode: str
                The mode that is plotted.
                (default: 'CP')
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
                Additional information about the plot
                (default: '')
        '''
        if save_plot:
            description = 'leakage_voltage_line'
            save_path = self.construct_path('plot', ext=plot_ext, 
                description=description, save_dir=plot_dir, subdir=plot_subdir,
                etc=etc)

        stats = self.stats

        plt.figure()

        for temp in self.temps:
            bool_df = (stats.loc[:, 'mode'] == mode) &\
                (stats.loc[:, 'temp'] == temp)
            rows = stats.loc[bool_df]
            temp_label = r'$T = {}^\circ C$'.format(temp)
            plt.errorbar(rows['voltage'], rows['mean'], yerr=rows['stddev'],
                label=temp_label)

        plt.legend()
        plt.xlabel('Bias Voltage (V)')
        plt.ylabel('Mean Leakage Current (pA)')

        if save_plot:
            plt.savefig(save_path)


    def plot_line_outliers(self, title='', mode='CP', save_plot=True, 
        plot_dir='', plot_subdir='', plot_ext='.pdf', etc=''):
        '''
        Plots number of outlier pixels (with leakage > 5-sigma from mean)
        versus bias voltage as a line plot, with a line for each temperature. 
        Done for only one mode.

        Keyword Arguments:
            title: str
                The figure title. If 'auto', a title is generated using the
                'title' method. If an empty string is passed, no title
                is shown.
                (default: '')
            mode: str
                The mode that is plotted.
                (default: 'CP')
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
                Additional information about the plot
                (default: '')
        '''
        if save_plot:
            description = 'outliers_voltage_line'
            save_path = self.construct_path('plot', ext=plot_ext, 
                description=description, save_dir=plot_dir, subdir=plot_subdir,
                etc=etc)

        stats = self.stats

        plt.figure()

        for temp in self.temps:
            bool_df = (stats.loc[:, 'mode'] == mode) & (stats.loc[:, 'temp'] == temp)
            rows = stats.loc[bool_df]
            temp_label = r'$T = {}^\circ C$'.format(temp)
            plt.plot(rows['voltage'], rows['outliers'], label=temp_label)

        plt.legend()
        plt.xlabel('Bias Voltage (V)')
        plt.ylabel(r'Number of Outlier Pixels ($> 5 \sigma$)')

        if save_plot:
            plt.savefig(save_path)
