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
import datetime

# Data analysis packages
import pandas as pd

# Internal imports
from .util import to_set, check_positive, check_channel, check_isotope_format


##
## Functions and a class for managing radioisotope data.
##

# These functions are primarily helper functions.

def parse_name(isotope):
    '''
    Returns a 2-tuple of the form (atomic symbol, mass number)
    corresponding to the 'isotope' attribute.
    '''
    sym, num = '', ''
    for char in isotope:
        if char in string.ascii_letters:
            sym += char
        elif char in string.digits:
            num += char

    return sym, num


def lara_to_df(filepath, energy_threshold=None):
    '''
    Reads the emission line data from the 'lara' ascii file for a nucleide
    into a pandas DataFrame. Such files can be found using the nuclear
    data table of the Laboratoire National Henri Becquirel, found at this 
    link: http://www.lnhb.fr/nuclear-data/nuclear-data-table/

    Argument:
        filepath: str
            The path to the 'lara' file.

    Return: pandas.DataFrame
        A DataFrame with emission line data and the following column names:
            'Energy (keV)'
            'Ener. unc. (keV)'
            'Intensity (%)'
            'Int. unc. (%)'
            'Type':
                indicates the type of X-ray using Seigenbach notation,
                or indicates that the line is a gamma ray (I think).
    '''
    # 'header' is the 0-indexed line number where the column headers are.
    # For these 'lara' files, the column headers are preceded by a long
    # line of '-' characters.
    header = 0
    with open(filepath, 'r') as lara_file:
        while True:
            # Increment 'header' here since the header line is actually 
            # the line after the line of '-'.
            header += 1
            line = lara_file.readline()
            # Stop incrementing 'header' once we find the '-' line.
            if '-' * 10 in line:
                break
            if not line:
                raise EOFError("The 'lara_to_df' method looks for at least"
                    " 10 '-' characters together in a line to indicate"
                    " the location of the header row. Such a sequence"
                    " was not found.")

    # We set skipfooter=1 below because the lara files have a long line of
    # '=' characters at the end of the emission line data.
    df = pd.read_table(filepath, sep=' ; ', header=header, skipfooter=1,
        engine='python')
    if energy_threshold is not None:
        # Make a boolean DataFrame that will cutoff values above an 
        # energy threshold, e.g., 140 keV, from the output DataFrame.
        df_bool = df.loc[:, 'Energy (keV)'] < energy_threshold
    else:
        # A lazy way to make a boolean DataFrame of all True.
        df_bool = df.loc[:, 'Energy (keV)'] > 0
    # Below we omit the columns 'Origin', 'Lvl. start', and 'Lvl. end', 
    # which aren't really relevant to this module.
    return df.loc[df_bool, 'Energy (keV)':'Type']


# These functions are for handling the CSV file containing information about
# the detector lab's X-ray sources

def slice_source_df(CIT_number=None, alias=None):
    '''
    A helper method that returns the index of the row of 'source_df'
    containing the CIT number specified, or the alias specified if 
    a CIT number is not passed.

    Keyword Arguments:
        CIT_number: int
            CIT number of the source.
            (default: None)
        alias: str
            A short string uniquely identifying the source used. Nice for
            switching between sources without memorizing CIT numbers. For 
            example, the alias for an Am241 source whose container filters 
            out some low energy radiation might be 'Am filtered'.
            (default: None)

    Return:
        idx: int
            The index of the row of 'source_df' containing the CIT number
            or alias specified.
        series: pandas.Series
            The series representing the row of 'source_df' containing the
            CIT number or alias specified.
    '''
    # Alias 'Source.source_df' for readability
    df = Source.source_df

    # Get the row refered to by the CIT_number or alias
    if CIT_number is not None:
        row = df.loc[df.loc[:, 'CIT number'] == CIT_number]
    elif alias is not None:
        row = df.loc[df.loc[:, 'alias'] == alias]
    else:
        raise ValueError("Must supply at least one of "
            "'CIT_number' or 'alias'")

    # Get the index of the row if the source_df.
    idx = row.index[0]
    series = row.squeeze()
    if len(row.index) > 1:
        raise RuntimeError("There is more than one of "
            "the CIT number or alias supplied in 'source_df'. "
            "It is ambiguous which row is being referenced.")

    return idx, series


def set_default_source(CIT_number=None, alias=None):
    '''
    Sets the source specified by 'CIT_number' or 'alias' to the default
    source to use when initializing a source object of a given isotope.
    At least one of 'CIT_number' or 'alias' must be specified.

    Keyword Arguments:
        CIT_number: int
            CIT number of the source.
            (default: None)
        alias: str
            A short string uniquely identifying the source used. Nice for
            switching between sources without memorizing CIT numbers. For 
            example, the alias for an Am241 source whose container filters 
            out some low energy radiation might be 'Am filtered'.
            (default: None)
    '''
    idx = slice_source_df(CIT_number, alias)[0]

    # Aliasing for brevity
    df = Source.source_df

    # Set the 'default_source' status of all other isotopes of the 
    # type referenced by 'CIT_number' or 'alias' to False.
    isotope = df.loc[idx, 'isotope']
    isotope_df = df.loc[df.loc[:, 'isotope'] == isotope]
    for i in isotope_df.index:
        Source.source_df.loc[i, 'default source'] = False

    # Set the 'default_source' status of the specified source to 'True'.
    Source.source_df.loc[idx, 'default source'] = True

    Source.source_df.to_csv(Source.source_csv_path, index=False)


def modify_source_info(info, CIT_number=None, alias=None, append=True):
    '''
    By default, appends the string 'info' to the field of the same name
    for this source in the 'xray_sources.csv' file. At least one of
    'CIT_number' or 'alias' must be specified.

    Argument:
        info: str
            The string to add to the 'info' column for this source in 
            the CSV file.

    Keyword Arguments:
        CIT_number: int
            CIT number of the source.
            (default: None)
        alias: str
            A short string uniquely identifying the source used. Nice for
            switching between sources without memorizing CIT numbers. For 
            example, the alias for an Am241 source whose container filters 
            out some low energy radiation might be 'Am filtered'.
            (default: None)
        append: bool
            If True, 'info' is appended to whatever is already written 
            for this source in the CSV. If False, 'info' will overwrite
            whatever was already there.
            (default: True)
    '''
    idx = slice_source_df(CIT_number, alias)[0]

    if append:
        Source.source_df.loc[idx, 'info'] += info
    else:
        Source.source_df.loc[idx, 'info'] = info

    Source.source_df.to_csv(Source.source_csv_path, index=False)


def set_source_alias(alias, CIT_number):
    '''
    Sets the alias of the source with the CIT number 'CIT_number' in the
    CSV file to 'alias'.
    '''
    idx = slice_source_df(CIT_number)[0]
    Source.source_df.loc[idx, 'alias'] = alias
    Source.source_df.to_csv(Source.source_csv_path, index=False)


class Source:
    '''
    A class whose instances each represent an X-ray source used by the lab.
    The methods return useful information/calculations about each source.

    Public Class Attributes:
        all_isotopes: set of str
            All sources that this class is fully configured to deal with.
        all_singlets: dict (keys: str, values: sets of floats)
            Contains the energies in keV of specral lines for fitting for each
            X-ray source. The keys are strings representing the source in the
            form '{element symbol}{mass number}'. The values are sets of  
            representing energies of this source's emission lines that are 
            floats intense and isolated enough to be fit.
        default_singlets: dict (keys: str, values: floats)
            Inicates which energy to supply to methods if none is specified.
            The keys are strings representing the source in the form 
            '[element symbol][mass number]'. The values are floats 
            representing energies of this source's emission lines.
        all_doublets: dict (keys: str, values: dicts of tuples of floats
            All doublet emission lines that might be fit with superimposed 
            Gaussians. The key-value pairs in the inner dicts take the form
            (energy1, energy2): (intensity1, intensity2). The intensities 
            are recorded to facilitate fitting a doublet with a single
            Gaussian centered at the intensity-weighted average of the 
            two energies.
        line_data_dir: str
            The directory in which 'lara'-style ascii files of emission line
            data are stored.
        source_csv_path: str
            The directory that contains the CSV file of X-ray sources used by 
            the detector test lab.
        source_df: pandas.DataFrame
            A DataFrame loaded from the CSV file 'xray_sources.csv' with 
            documentation of the X-ray sources used for detector tests.
            This attribute can be modified with the functions 
            'set_default_source' or 'modify_source_info', or the instance 
            methods 'add_source', 'set_default_source', or 
            'modify_source_info'. Colums are described below:
                isotope: 
                    The isotope name in form [symbol][mass number].
                alias: 
                    A short string uniquely identifying the source.
                reference activity (mCi): 
                    The activity measured at the reference date.
                reference date:
                    The date at which the reference activity was measured.
                default source:
                    Boolean indicating whether this source is the default
                    to load for its isotope. Specifically used by the
                    'source_from_csv' function.
                info:
                    Any additional information about the source.

    Public Instance Attributes:
        isotope: str
            The isotope this instance represents, in the form 
            '[element symbol][mass number]'. For example, 'Am241'.
        sym: str
            The isotopes elemental symbol. E.g., 'Am'.
        num: str
            The isotope's mass number. E.g., '241'.
        latex: str
            A string to be formatted by LaTeX into the isotope name with
            mass number superscipted. E.g., '{}^{241}Am'
        default_energy: float
            The energy of the specral line instance methods will use by 
            defualt if none is specified by the user.
        energies: set of floats
            The energies of all strong singlet lines for this isotope.
        doublets: dict of tuples of floats
            All doublet emission lines for this isotope that might be fit   
            with superimposed Gaussians. The key-value pairs in the dict take 
            the form (energy1, energy2): (intensity1, intensity2). The  
            intensities are recorded to faciliate fitting a doublet with a 
            single Gaussian centered at the intensity-weighted average of the 
            energies.
        line_data: pandas.DataFrame
            A DataFrame containing information about all emission lines 
            for the isotope for energies below 140 keV. The columns are 
            described below:
                'Energy (keV)'     : Emission line energy
                'Ener. unc. (keV)' : Energy uncertainty
                'Intensity (%)'    : Relative intensity in units of
                                     photons per 100 disintegrations
                'Int. unc. (%)'    : Intensity uncertainty
                'Type'             : In Siegbahn notation (e.g., XKa2), or 'g'
                                     if the emissin is a gamma ray (I think)
        CIT_number: int
            The CIT number of the source
            (default: None)
        ref_activity: float
            The activity, in mCi, measured at the reference date 'ref_date'
            (default: None)
        ref_date: str
            The date in the format'[day]-[month]-[year]' at which the 
            reference activity 'ref_activity' was measured.
            (default: None)
        alias: str
            A short string uniquely identifying the source used. Nice for
            switching between sources without memorizing CIT numbers. For 
            example, the alias for an Am241 source whose container filters 
            out some low energy radiation might be 'Am filtered'
            (default: None)
        info: str
            Any additional information about this source.
            (default: None)
    '''
    # The set of all sources that this class is fully configured to deal with.
    all_isotopes = {'Am241', 'Co57', 'Eu155', 'Fe55', 'Ba133', 'Cs137'}

    # All singlet emission lines (> 1 keV away from other lines) that might
    # be used for fitting a spectrum. Energies in keV.
    all_singlets = {
        'Am241': {13.8520, 21.1600, 26.3446, 59.5409},
        'Co57' : {14.41295, 122.06065, 136.47356},
        'Eu155': {6.73255, 60.00860, 86.54790, 105.30830},
        'Fe55' : {0.63850, 6.51280, 125.94900},
        'Ba133': {4.67355, 53.16220, 79.61420, 80.99790},
        'Cs137': {4.8815, 661.657}
    }

    # Default emission lines to fit if none is specified. Energies in keV.
    default_singlets = {
        'Am241': 59.5409,
        'Co57' : 122.06065,
        'Eu155': 86.54790,
        'Fe55' : 6.51280,
        'Ba133': 80.99790,
        'Cs137': 661.657
    }


    # All doublet emission lines that might be fit with two Gaussians. 
    all_doublets = {
        'Am241': {
            (15.8760, 16.9600): (0.384000, 18.580000)
        },
        'Co57' : {
            (6.39091, 6.40391): (17.12, 33.50)
        },
        'Eu155': {
            (42.30930, 42.99670): (6.70000, 12.05000),
            (86.05910, 86.54790): (0.15400, 30.70000)
        },
        'Fe55' : {
            (5.88765, 5.89875): (8.45, 16.57)
        },
        'Ba133': {
            (30.62540, 30.97310): (33.80, 62.40),
            (35.05300, 35.90030): (18.24, 4.45)
        },
        'Cs137': {
            (31.8174, 32.1939): (1.950, 3.590),
            (36.4457, 37.3317): (1.005, 0.266)
        }
    }

    pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # The directory in which 'lara'-style ascii files of emission line
    # data are stored.
    line_data_dir = os.path.join(pwd, 'isotope_data')

    # The directory that contains the CSV file of X-ray sources used by the 
    # detector test lab.
    source_csv_path = os.path.join(pwd, 'data', 'xray_sources.csv')

    # A DataFrame containing the sources used by the detector test lab.
    source_df = pd.read_csv(source_csv_path, 
        true_values=['True', 'TRUE', 'true'], 
        false_values=['False', 'FALSE', 'false'])


    def __init__(self, isotope, CIT_number=None, ref_activity=None, 
        ref_date=None, alias=None, info=None):
        '''
        Arguments:
            isotope: str
                The isotope name, in the form '{element symbol}{mass number}'.
                For example, 'Am241'.

        Keyword Arugments:
            CIT_number: int
                The CIT number of the source
                (default: None)
            ref_activity: float
                The activity, in mCi, measured at the reference date 'ref_date'
                (default: None)
            ref_date: str
                The date in the format'[day]-[month]-[year]' at which the 
                reference activity 'ref_activity' was measured.
                (default: None)
            alias: str
                A short string uniquely identifying the source used. Nice for
                switching between sources without memorizing CIT numbers. For 
                example, the alias for an Am241 source whose container filters
                out some low energy radiation might be 'Am filtered'
                (default: None)
            info: str
                Any additional information about this source.
                (default: None)
        '''
        isotope = check_isotope_format(isotope)

        # Parsing the atomic symbol and mass number out from 'isotope'.
        sym, num = parse_name(isotope) # atomic symbol, mass number
        self.sym = sym
        self.num = num

        # Generating a LaTeX string of the isotope name
        self.latex = r'${}^{' + self.num + r'}$' + self.sym # e.g., {}^{241}Am

        self.isotope = isotope
        self.alias = alias
        self.info = str(info)
        self.ref_activity = ref_activity
        self.ref_date = ref_date
        self.CIT_number = CIT_number
        self.default_energy = self.default_singlets[isotope]
        self.energies = self.all_singlets[isotope]
        self.doublets = self.all_doublets[isotope]

        self.line_data = lara_to_df(f'{self.line_data_dir}/{sym}-{num}'
            '.lara.txt', energy_threshold=150)


    @classmethod
    def from_csv(self, isotope=None, CIT_number=None, alias=None):
        '''
        Initializes a 'Source' object from the CSV file logging X-ray sources
        pointed to by the 'source_csv_path' class attribute. 

        If the 'CIT_number' is specified, the source with that CIT number will
        be used, no matter what other parameters were passed. 

        If 'alias' is specified but not 'CIT_number', the source with that
        alias will be used, regardless of the value of 'isotope'. 

        If only the 'isotope' kwarg is specified, the default source of that
        isotope will be used, if a default exists. 

        Keyword Arguments:
            isotope: str
                The isotope name, in the form '{element symbol}{mass number}'.
                For example, 'Am241'.
                (default: None)
            CIT_number: int
                The CIT number of the source
                (default: None)
            alias: str
                A short string uniquely identifying the source used. Nice for
                switching between sources without memorizing CIT numbers. For 
                example, the alias for an Am241 source whose container filters
                out some low energy radiation might be 'Am filtered'
                (default: None)
        
        Return: A 'Source' instance
        '''
        if CIT_number is not None or alias is not None:
            series = slice_source_df(CIT_number, alias)[1]
        elif isotope is not None:
            df_bool = (self.source_df.loc[:, 'isotope'] == isotope) &\
                (self.source_df.loc[:, 'default source'] == True)
            series = self.source_df.loc[df_bool].squeeze()

        arg_dict = {}

        for i in series.index:
            if pd.isna(series.loc[i]):
                arg_dict[i] = None
            else:
                arg_dict[i] = series[i]

        return Source(arg_dict['isotope'], arg_dict['CIT number'], 
            arg_dict['reference activity (mCi)'], arg_dict['reference date'], 
            arg_dict['alias'], arg_dict['info'])


    #
    # Methods supplying useful information about this 'Source' instance. 
    #

    def line(self, energy=None):
        '''
        Given the approximate energy of this source's desired spectral line, 
        will return the most accurate available value of that energy. Valid
        accurate energies can be found in a 'Source' instance's 'energies'
        attriute, or in the 'all_energies' class attribute of 'Source'.

        Keyword Arguments:
            energy: number
                The approximate energy of the spectral line in keV. Will
                work as long as rounding both 'energy' and the actual energy
                (as recorded in the 'all_energies' class attribute) to the 
                ones place yeild the same number. If None, will default to the 
                value stored in the 'default_energy' attribute.
                (default: None)

        Return: float
            accurate_energy: float
                The maximally accurate value of the line's energy in keV.
        '''
        if energy is not None:
            for accurate_energy in self.energies:
                # If supplied an energy close to an accurate energy, 
                # return the accurate energy.
                if round(energy, 0) == round(accurate_energy, 0):
                    return accurate_energy
            # If supplied an energy without a match, throw an exception.
            raise ValueError("Couldn't find anything in the 'energies' "
                + f"attribute close enough to {energy} keV.")
        # If no energy was specified, return 'default_energy'
        return self.default_energy


    def chan_range(self, energy=None, gain_estimate=0.014,
        lower_bound=100, upper_bound=9900, width=3000):
        '''
        Estimates the range, in channels, in which a spectral line at the given
        energy might be found in a channel spectrum. Good for gain correction.

        Keyword Arguments:
            energy: number
                The approximate energy of the spectral line in keV. Will
                work as long as rounding both 'energy' and the actual energy
                (as recorded in the 'all_energies' class attribute) to the 
                ones place yeild the same number. If None, will default to the 
                value stored in the 'default_energy' attribute.
                (default: None)
            gain_estimate: float
                An estimate of the gain for the detector. Used to estimate the
                recorded response of the line in channels. 
                (energy in keV) / gain = (energy in channels)
                (defautl: 0.014)
            lower_bound: int
                If either element of 'chan_range' is calculated to be below
                'lower_bound', it will instead be set equal to 'lower_bound'.
                (default: 100)
            upper_bound: int
                If either element of 'chan_range' is calculated to be above
                'upper_bound', it will instead be set equal to 'upper_bound'.
                (default: 9900)
            width: int
                The width of the interval specified by 'chan_range'.
                (default: 3000)

        Return: Tuple(int, int)
            chan_low: int
                Indicates that methods using this information for fitting
                should not look for this line lower than 'chan_low' channels.
            chan_high: int
                Indicates that methods using this information for fitting
                should not look for this line higher than 'chan_high' channels.
        '''
        # Getting an accurate energy (or the default energy).
        energy = self.line(energy)

        # Checking mainly for value errors that would screw up the chan_range.
        check_positive(energy=energy, gain_estimate=gain_estimate)
        check_channel(lower_bound=lower_bound, upper_bound=upper_bound, 
            width=width)

        # Calculating a preliminary channel range.
        chan_central = energy / gain_estimate
        chan_low = chan_central - (width / 2)
        chan_high = chan_central + (width / 2)

        # Correcting the range incase it is not within the bounds.
        if chan_low  < lower_bound: chan_low  = lower_bound
        if chan_high < lower_bound: chan_high = lower_bound
        if chan_high > upper_bound: chan_high = upper_bound
        if chan_low  > upper_bound: chan_low  = upper_bound

        return int(chan_low), int(chan_high)

    # TODO
    def estimate_activity(self, date=None):
        '''
        Estimates the current activity of the source (in mCi) based on its  
        half life and the reference activity (ref_activity) and corresponding 
        reference date (ref_date).

        Keyword Arguments:
            date: str or datetime.datetime or datetime.date or pandas.Timestamp
                The date (year-month-day, if string) at which to calculate the
                activity. If None, uses the current date.
        '''
        pass


    #
    # Methods for handling more general emission line data for all isotopes.
    #

    @classmethod
    def load_line_data(self, isotopes=all_isotopes, data_dir=line_data_dir):
        '''
        Loads emission line data for all isotopes in 'isotopes' from 
        'lara'-style ascii files in the directory 'data_dir'. Returns the data 
        as a pandas.DataFrame.

        Keyword Arguments:
            isotopes: set of str or str or array-like of str
                The set of isotopes for which to get data.
                (default: Source.all_sources)
            data_dir:
                The directory containing the 'lara'-style ascii files of 
                emission line data.
                (default: Source.line_data_dir)

        Return: 
            line_data: dict of pandas.DataFrame
        '''
        isotopes = to_set(isotopes)
        line_data = {}
        for isotope in isotopes:
            sym, num = parse_name(isotope)
            line_data[isotope] = lara_to_df(f'{data_dir}/{sym}-{num}.lara.txt')
        return line_data


    @classmethod
    def print_line_data(self, isotopes=all_isotopes, data_dir=line_data_dir,
        energy_threshold=150, columns=['Energy (keV)', 'Intensity (%)'],
        intensity_threshold=0.05):

        data = Source.load_line_data(isotopes=isotopes, data_dir=data_dir)
        for isotope in isotopes:
            df = data[isotope]
            df_bool = (df.loc[:, 'Energy (keV)'] < energy_threshold) &\
                (df.loc[:, 'Intensity (%)'] > intensity_threshold)
            print(isotope)
            print(df.loc[df_bool, columns]) 
            print('')


    # 
    # Methods for managing 'source_df' and the corresponding CSV file.
    #

    def add_source(self):
        '''
        Adds information about the current source instance to the DataFrame 
        'source_df' and to the corresponding CSV file 'xray_sources.csv'.
        '''
        if self.alias and self.alias in self.source_df.loc[:, 'alias']:
            raise ValueError(f"The alias {alias} already exists in the "
                "record. Please enter a different one.")
        if self.CIT_number in self.source_df.loc[:, 'CIT number']:
            raise ValueError(f"The CIT number {CIT_number} already exists "
                "in the record. Seems like this source has already been "
                "recorded here.")
        if self.CIT_number is None and (self.alias is None or not self.alias):
            raise ValueError("To add this source to the file, at least one\n"
                "of the attributes 'CIT_number' or 'alias' must contain a "
                "valid value.")

        idx = self.source_df.index[-1] + 1

        # The False value entered below makes the entry for 'default source'
        # for this source in the CSV False to start with.
        self.source_df.loc[idx] = \
            [self.isotope, self.alias, self.CIT_number, self.ref_activity, 
            self.ref_date, False, self.info]

        self.source_df.to_csv(self.source_csv_path, index=False)


    def slice_source_df(self):
        '''
        A helper method that returns the index of the row of 'source_df'
        representing the source instance this method was called upon.

        Return:
            idx: int
                The index of the row of 'source_df' containing the CIT number
                or alias specified.

        '''
        return slice_source_df(self.CIT_number, self.alias)


    def set_default_source(self):
        '''
        Sets the source represented by this 'Source' instance to the default
        source to use when initializing a source object of a given isotope
        from the 'xray_sources.csv' file.
        '''
        set_default_source(self.CIT_number, self.alias)


    def modify_source_info(self, info, append=True):
        '''
        By default, appends the string 'info' to the field of the same name
        for this source in the 'xray_sources.csv' file.

        Argument:
            info: str
                The string to add to the 'info' column for this source in 
                the CSV file.

        Keyword Argument:
            append: bool
                If True, 'info' is appended to whatever is already written 
                for this source in the CSV. If False, 'info' will overwrite
                whatever was already there.
        '''
        modify_source_info(info, self.CIT_number, self.alias, append)


    def set_source_alias(self, alias):
        '''Sets the alias of this source in the CSV file to 'alias'.'''
        set_source_alias(alias, self.CIT_number)
