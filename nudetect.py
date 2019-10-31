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



##
## Functions for checking and correcting values and types.
##

def to_set(x):
    '''
    Returns 'x' converted to a set. If 'x' is a string or scalar, then
    a set containing 'x' as its only element is returned. If an iterable
    other than a string is passed, it is converted to a set via the 
    built-in 'set()' function.
    '''
    try:
        # If 'x' is a string, return a set with 'x' as the only element.
        if type(x) == str:
            return {x}
        # If 'x' is an iterable other than a string, convert normally.
        return set(x)
    # If 'x' is a scalar (other than a string), return a set with 'x'
    # as its only element.
    except TypeError:
        return {x}


def check_positive(**kwargs):
    '''Raises a ValueError if a parameter is non-positive.'''
    for name in kwargs:
        x = kwargs[name]
        if x <= 0:
            raise ValueError(f"'{name}' must be positive. Instead got {x}")


def check_non_negative(**kwargs):
    '''Raises a ValueError if a parameter is negative.'''
    for name in kwargs:
        x = kwargs[name]
        if x < 0:
            raise ValueError(f"'{name}' must be non-negative. Instead got {x}")


def check_channel(**kwargs):
    '''Checks that values representing channels are correctly formatted.'''
    for name in kwargs:
        x = kwargs[name]
        if x < 0 or x > 10000:
            raise ValueError(f"Should have 0 <= {name} <= 10000. "
                + f"Instead got {name} == {x}.")


def check_isotope_format(isotope):
    '''
    Checks the format of the string representing an isotope name.

    Argument:
        isotope: str
            Isotope name

    Return: str
        The isotope name with the first letter capitalized, if needed.
    '''
    sym, num = parse_name(isotope) # atomic symbol, mass number
    if f'{sym}{num}' != isotope:
        raise ValueError("'isotope' should be supplied in the"
            " form [symbol][number].")
    if isotope[0] not in string.ascii_uppercase:
        return isotope[0].upper() + isotope[1:]
    return isotope


def get_mean_stdv(values, precision, value_label):
    '''
    Helper function for returning mean and stdv of some values, with
    some exception handling.
    '''
    try:
        if precision == 0:
            m = int(round(np.mean(values)), precision)
        else:
            m = round(np.mean(values), precision)
    except TypeError:
        m = np.mean(values)

    try:
        if precision == 0:
            s = int(round(np.std(values)), precision)
        else:
            s = round(np.std(values), precision)
    except TypeError:
        s = np.std(values)

    return m, s


##
## Miscellaneous helper functions that the user may also find useful.
##

def fits_to_df(filepath, colnames, pos=None, temp_threshold=-20,
    swap_byte_order=True):
    '''
    Loads and slices out good data from a FITS file of detector test data.
    '''

    #
    # Trim out any columns and rows of the table that we don't need. Doing
    # this early reduces the RAM used later on and makes things faster.
    #

    # Call this up here so that if an exception is raised, it's before
    # the time-expensive part of this function.
    if colnames != 'all':
        colnames = to_set(colnames)

    original_colnames = colnames

    # Add in temperature and position data, since we'll need them for trimming
    # out unnecessary rows of the table later.
    if temp_threshold is not None:
        colnames.add('TEMP')
    if pos is not None:
        colnames.add('DET_ID')

    # Get data from FITS file (this line takes a long time to run)
    table = Table.read(filepath)

    all_names = set(table.colnames)

    # If 'colnames' was not specified, assign it to the set of 
    # all column names in the table.
    if colnames is None:
        colnames = all_names

    # Remove any columns not in 'colnames'. Doing this early lowers
    # RAM usage, which may be vital with big tables.
    excluded_colnames = all_names - colnames
    table.remove_columns(excluded_colnames)

    col_length = table['TEMP'].shape[0]

    mask = np.ones(col_length, dtype=bool)

    # 'start' and 'end' denote the indices between which 'table['TEMP']'
    # takes on a resonable value. start is the first index with a 
    # temperature greater than -20 C, and end is the last such index.
    if temp_threshold is not None:
        mask *= table['TEMP'] > temp_threshold
    if pos is not None:
        mask *= table['DET_ID'] == pos

    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])
    del mask

    table.remove_rows(slice(0, start))
    table.remove_rows(slice(end, col_length))

    # Remove 'TEMP' and 'DET_ID' columns if they weren't requested.
    table.remove_columns(colnames - original_colnames)


    #
    # Convert our data table to a pandas.DataFrame. DataFrames should be faster
    # than astropy Tables for indexing/scalar value access.
    #

    # These lists will divide columns by their dimensionality. One dimensional
    # columns (i.e, a column of scalars) can be converted directly to a pandas
    # DataFrame via the 'to_pandas' method of astropy.table.Table. 
    one_dim_col_names = []
    two_dim_col_names = []

    for colname in table.colnames:
        if len(table[colname].shape) == 1:
            one_dim_col_names.append(colname)
        else:
            two_dim_col_names.append(colname)

    # If any one dimensional columns were requested, convert them all to a
    # single pandas.DataFrame called 'one_dim_df'.
    if one_dim_col_names:
        one_dim_df = table[one_dim_col_names].to_pandas()
        table.remove_columns(one_dim_col_names)
    else:
        one_dim_df = None

    # If any two dimensional columns were requested, convert them all to a
    # dict of pandas.DataFrames, one DataFrame for
    if two_dim_col_names:
        dict_of_cols = table[two_dim_col_names].columns
        two_dim_dfs = {}
        for colname, col in dict_of_cols.items():
            if swap_byte_order:
                two_dim_dfs[colname] = pd.DataFrame(
                    table[colname].data.byteswap().newbyteorder())
            else:
                two_dim_dfs[colname] = pd.DataFrame(table[colname].data)
            table.remove_column(colname)
    else:
        two_dim_dfs = None

    return one_dim_df, two_dim_dfs


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

    pwd = os.path.dirname(os.path.abspath(__file__))

    # The directory in which 'lara'-style ascii files of emission line
    # data are stored.
    line_data_dir = os.path.join(pwd, 'isotope_data')

    # The directory that contains the CSV file of X-ray sources used by the 
    # detector test lab.
    source_csv_path = os.path.join(pwd, 'xray_sources.csv')

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

        If the 'CIT_number' is specified, the source with that CIT number will be used, no matter what other parameters were passed. 

        If 'alias' is specified but not 'CIT_number', the source with that alias will be used, regardless of the value of 'isotope'. 

        If only the 'isotope' kwarg is specified, the default source of that isotope will be used, if a default exists. 

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


##
## Classes for analyzing data from detector tests.
##

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
	misc_mask=1):
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

        Return:
            gain: 2D numpy.ndarray
                A 32 x 32 array of floats. Each entry represents its  
                respective pixel's gain, where channels * gain = energy.
        '''

        if save_data:
            data_path = self.construct_path('data', ext=data_ext, 
                description='gain', save_dir=data_dir, subdir=data_subdir)

        if save_plot:
            plot_path = self.construct_path('plot', description='gain', 
                ext=plot_ext,  save_dir=plot_dir, subdir=plot_subdir)

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
        save_data=True, data_ext='.txt', data_dir='', data_subdir='', misc_mask=1):
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
                save_dir=data_dir, subdir=data_subdir, description='spectrum')

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
        plot_subdir=''):
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

        '''
        # Constructing a save path, if needed
        if save_plot:
            save_path = self.construct_path('plot', ext=plot_ext, 
                save_dir=plot_dir, subdir=plot_subdir, 
                description='energy_spectrum')

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
