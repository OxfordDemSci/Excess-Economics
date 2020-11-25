# !/usr/local/bin/python
"""Parsing and preprocessing files for excess economic losses."""
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm


def make_heatmap_df(heatmap_forc):
    """Make the heatmap to be visualised from the forecasts."""
    regions_list = ['neast', 'nwest', 'ykhu', 'emids',
                    'wmids', 'east', 'lon', 'seast',
                    'swest', 'wal', 'scot']
    ages = ['16_17', '18_24', '25_34', '35_49', '50_64', '65']
    heatmap_df = pd.DataFrame(index=ages, columns=regions_list)
    for age in ages:
        for reg in regions_list:
            hold = age + '_empratio_' + reg + '_p'
            heatmap_df.at[age, reg] = heatmap_forc[hold+'_f'].sum() -\
                heatmap_forc[hold].sum()
    heatmap_df = heatmap_df.rename(columns={'neast': 'N. East'})
    heatmap_df = heatmap_df.rename(columns={'nwest': 'N. West'})
    heatmap_df = heatmap_df.rename(columns={'ykhu': 'YKHU'})
    heatmap_df = heatmap_df.rename(columns={'emids': 'E. Mids'})
    heatmap_df = heatmap_df.rename(columns={'wmids': 'W. Mids'})
    heatmap_df = heatmap_df.rename(columns={'seast': 'S. East'})
    heatmap_df = heatmap_df.rename(columns={'east': 'East'})
    heatmap_df = heatmap_df.rename(columns={'lon': 'Lon'})
    heatmap_df = heatmap_df.rename(columns={'swest': 'S. West'})
    heatmap_df = heatmap_df.rename(columns={'wal': 'Wales'})
    heatmap_df = heatmap_df.rename(columns={'scot': 'Scot'})
    return heatmap_df.astype(float)


def make_heatmap_vars(df):
    """Make the variables relative to the heatmap."""
    heatmap_vars = pd.DataFrame(index=df.index)
    regions_list = ['neast', 'nwest', 'ykhu', 'emids',
                    'wmids', 'east', 'lon', 'seast',
                    'swest', 'wal', 'scot']
    ages = ['16_17', '18_24', '25_34', '35_49', '50_64', '65']
    for age in ages:
        for reg in regions_list:
            temp = pd.DataFrame(df[age+'_emp_'+reg+'_p'] /
                                df[age+'_allpop_'+reg+'_p'],
                                index=df.index, columns=[age + '_empratio_' +
                                                         reg + '_p'])
            heatmap_vars = pd.merge(heatmap_vars, temp, how='left',
                                    left_index=True, right_index=True)
    return heatmap_vars


def make_allreg_df(df_in):
    """Make all regions dataframe."""
    regions_list = ['neast', 'nwest', 'ykhu', 'emids',
                    'wmids', 'east', 'lon', 'seast',
                    'swest', 'wal', 'scot']
    allreg_df = pd.DataFrame(index=regions_list,
                             columns=['m_diff', 'm_ci',
                                      'w_diff', 'w_ci'])
    hold = 'allage_empratio_'
    for reg in regions_list:
        for sex in ['m', 'w']:
            allreg_df.loc[reg, sex+'_diff'] = df_in[hold + reg + '_'+sex + '_f'].sum()-\
                                              df_in[hold + reg + '_' + sex].sum()
            allreg_df.loc[reg, sex+'_ci'] = df_in[hold + reg + '_'+sex + '_f'].sum()-\
                                              df_in[hold + reg + '_'+sex + '_ci_left'].sum()
    allreg_df.index = allreg_df.index.str.replace('neast', 'N. East')
    allreg_df.index = allreg_df.index.str.replace('nwest', 'N. West')
    allreg_df.index = allreg_df.index.str.replace('ykhu', 'YKHU')
    allreg_df.index = allreg_df.index.str.replace('emids', 'E. Mids')
    allreg_df.index = allreg_df.index.str.replace('wmids', 'W. Mids')
    allreg_df.index = allreg_df.index.str.replace('seast', 'S. East')
    allreg_df.index = allreg_df.index.str.replace('east', 'East')
    allreg_df.index = allreg_df.index.str.replace('lon', 'Lon')
    allreg_df.index = allreg_df.index.str.replace('swest', 'S. West')
    allreg_df.index = allreg_df.index.str.replace('wal', 'Wales')
    allreg_df.index = allreg_df.index.str.replace('scot', 'Scot')
    return allreg_df


def make_reg_forc(df_input):
    """Make regional forecasts."""
    regions_list = ['neast', 'nwest', 'ykhu', 'emids',
                    'wmids', 'east', 'lon', 'seast',
                    'swest', 'wal', 'scot']
    allreg = pd.DataFrame(index=df_input.index)
    for reg in regions_list:
        for sex in ['_m', '_w']:
            temp = df_input['allage_emp_' + reg + sex] / df_input['allage_allpop_' + reg + sex]
            allreg = pd.merge(allreg, pd.DataFrame(temp, index=df_input.index,
                                                   columns=['allage_empratio_' + reg + sex]),
                              how='left', left_index=True, right_index=True)
    return allreg


def make_age_df(age_forc):
    """Make the age df for plotting."""
    ages = ['16_17', '18_24', '25_34', '35_49', '50_64', '65']
    age_df = pd.DataFrame(index=ages, columns=['obs_m', 'forc_m',
                                               'obs_w', 'forc_w',
                                               'lci_m', 'rci_m',
                                               'lci_w', 'rci_w'])
    hold = '_empratio_allreg'
    for sex in ['_m', '_w']:
        for age in ages:
            age_df.at[age, 'obs' + sex] = age_forc[age + hold + sex].sum()
            age_df.at[age, 'forc' + sex] = age_forc[age + hold +  sex + '_f'].sum()
            age_df.at[age, 'lci' + sex] = age_forc[age + hold + sex + '_ci_left'].sum()
            age_df.at[age, 'rci' + sex] = age_forc[age + hold + sex + '_ci_right'].sum()
        age_df['ci'+sex] = age_df['rci' + sex] - age_df['forc' + sex]
    return age_df


def make_empage_ratios(df):
    """Make age ratios of employment for the pyramids."""
    age_ratio = pd.DataFrame(index=df.index)
    for sex in ['_m', '_w', '_p']:
        for age in ['16_17', '18_24', '25_34', '35_49', '50_64', '65', 'allage']:
            emp = pd.DataFrame(df[age + '_emp_' + 'allreg' + sex] /
                               df[age + '_allpop_' + 'allreg' + sex],
                               columns=[age+'_empratio_' + 'allreg' + sex],
                               index=df.index)
            age_ratio = pd.merge(age_ratio, emp, how='left',
                                 left_index=True, right_index=True)
    return age_ratio


def make_seasonal(df, freq):
    """Seasonally adjust the series."""
    if freq == 'quarterly':
        period = 4
    elif freq == 'monthly':
        period = 12
    for col in df.columns:
        decompose = sm.tsa.seasonal_decompose(df[col], period=period)
        df[col+'_s'] = decompose.observed - decompose.seasonal
    return df


def make_total_columns(input_df):
    """Make pop-level columns for later calcs."""
    data_type = ['emp', 'unemp', 'inact']
    regions_list = ['neast', 'nwest', 'ykhu', 'emids',
                    'wmids', 'east', 'lon', 'seast',
                    'swest', 'wal', 'scot']
    sex_list = ['p', 'm', 'w']
    ages = ['16_17', '18_24', '25_34', '35_49', '50_64', '65']

    for sex in sex_list:  # make a column for ages
        for region in regions_list:
            allage = 'allage_emp_' + region + '_' + sex
            input_df[allage] = 0
            for age in ages:
                specific = age + '_emp_' + region + '_' + sex + '_s'
                input_df[allage] = input_df[allage] + input_df[specific]

    for sex in sex_list:  # make a column for all regions
        for age in ages:
            allreg = age + '_emp_allreg_' + sex
            input_df[allreg] = 0
            for region in regions_list:
                specific = age + '_emp_' + region + '_' + sex + '_s'
                input_df[allreg] = input_df[allreg] + input_df[specific]

    for sex in sex_list:  # make a column for all types/regions
        allpop = 'allage_emp_allreg_' + sex
        input_df[allpop] = 0
        for region in regions_list:
            for age in ages:
                specific = age + '_emp_' + region + '_' + sex + '_s'
                input_df[allpop] = input_df[allpop] + input_df[specific]

    for sex in sex_list:  # make a column for all types
        for age in ages:
            for region in regions_list:
                allpop = age + '_allpop_' + region + '_' + sex
                input_df[allpop] = 0
                specific = age + '_emp_' + region + '_' + sex + '_s'
                specific_r = age + '_emp_r_' + region + '_' + sex + '_s'
                input_df[allpop] = input_df[allpop] + ((100/input_df[specific_r])*input_df[specific])

    for sex in sex_list:  # make a column for all types/regions
        for age in ages:
            allpop = age + '_allpop_' + 'allreg_' + sex
            input_df[allpop] = 0
            for region in regions_list:
                specific = age + '_emp_' + region + '_' + sex + '_s'
                specific_r = age + '_emp_r_' + region + '_' + sex + '_s'
                input_df[allpop] = input_df[allpop] + ((100/input_df[specific_r])*input_df[specific])

    for sex in sex_list:  # for all types/regions/ages (i.e. total uk wide)
        allpop = 'allage' + '_allpop_' + 'allreg_' + sex
        input_df[allpop] = 0
        for age in ages:
            for region in regions_list:
                specific = age + '_emp_' + region + '_' + sex + '_s'
                specific_r = age + '_emp_r_' + region + '_' + sex + '_s'
                input_df[allpop] = input_df[allpop] + ((100/input_df[specific_r])*input_df[specific])

    for sex in sex_list:  # for all types/regions/ages (i.e. total uk wide)
        for region in regions_list:
            allpop = 'allage' + '_allpop_' + region + '_' + sex
            input_df[allpop] = 0
            for age in ages:
                for dtype in data_type:
                    specific = age + '_emp_' + region + '_' + sex + '_s'
                    specific_r = age + '_emp_r_' + region + '_' + sex + '_s'
                    input_df[allpop] = input_df[allpop] + ((100/input_df[specific_r])*input_df[specific])
    return input_df


def make_interp(input_df):
    """Interpolate the few missing columns. Could print counts of mising."""
    for col in input_df.columns:
        input_df[col] = input_df[col].astype(str)
        input_df[col] = input_df[col].replace('*', np.nan)
        input_df[col] = input_df[col].astype(float)
        input_df[col] = input_df[col].interpolate()
    return input_df


def make_clean(df, var, reg, sex, freq):
    """Clean the sheets which are getting fed in."""
    df = df[['Unnamed: 0', 'Unnamed: 3', 'Unnamed: 4',
             'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8',
             'Unnamed: 9', 'Unnamed: 12', 'Unnamed: 13',
             'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17',
             'Unnamed: 18']]
    df = df.rename({'Unnamed: 0': 'Time',
                    'Unnamed: 3': '16_17_' + var + '_' + reg + '_' + sex,
                    'Unnamed: 4': '18_24_' + var + '_' + reg + '_' + sex,
                    'Unnamed: 6': '25_34_' + var + '_' + reg + '_' + sex,
                    'Unnamed: 7': '35_49_' + var + '_' + reg + '_' + sex,
                    'Unnamed: 8': '50_64_' + var + '_' + reg + '_' + sex,
                    'Unnamed: 9': '65_' + var + '_' + reg + '_' + sex,
                    'Unnamed: 12': '16_17_' + var + '_r_' + reg + '_' + sex,
                    'Unnamed: 13': '18_24_' + var + '_r_' + reg + '_' + sex,
                    'Unnamed: 15': '25_34_' + var + '_r_' + reg + '_' + sex,
                    'Unnamed: 16': '35_49_' + var + '_r_' + reg + '_' + sex,
                    'Unnamed: 17': '50_64_' + var + '_r_' + reg + '_' + sex,
                    'Unnamed: 18': '65_' + var + '_r_' + reg + '_' + sex},
                   axis=1)
    if freq == 'quarterly':
        searchfor = ['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov']
        df = df[df['Time'].str.contains('|'.join(searchfor))]
    df = df.set_index('Time')
    return df


def make_dataset(data_path, data_list, freq):
    """Parse raw excel files."""
    regions_list = ['neast', 'nwest', 'ykhu', 'emids',
                    'wmids', 'east', 'lon', 'seast',
                    'swest', 'wal', 'scot']
    sex_list = ['p', 'm', 'w']
    merge = pd.DataFrame()
    for filename, vartype in data_list.items():
        for reg in regions_list:
            for sex in sex_list:
                raw = pd.read_excel(os.path.join(data_path, filename),
                                    sheet_name=reg + '_' + sex,
                                    skiprows=9)[0:230]
                if len(merge) == 0:
                    merge = make_clean(raw, vartype, reg, sex, freq)
                else:
                    merge = pd.merge(merge, make_clean(raw, vartype,
                                                       reg, sex, freq),
                                     left_index=True, right_index=True,
                                     how='outer')
                    if 'Time_y' in merge.columns:
                        merge = merge.drop('Time_y', 1)
                        merge = merge.rename({'Time_x': 'Time'}, axis=1)
    return merge
