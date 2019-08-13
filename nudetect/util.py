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
import string

# Data analysis packages
import numpy as np
import pandas as pd
from astropy.table import Table


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
