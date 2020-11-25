"""Calculating Excess Economic Loss.

Author: Charlie
Date: 12/10/2020
"""
from parser import (make_dataset, make_interp,
                    make_total_columns, make_seasonal,
                    make_empage_ratios,
                    make_age_df, make_allreg_df,
                    make_reg_forc, make_heatmap_vars,
                    make_heatmap_df)
from make_forecasts import gen_forecast
from make_visuals import plot_func
import os
import warnings
import re

warnings.filterwarnings("ignore")


def main():
    """Master function."""
    data_path = os.path.join(os.getcwd(), '..', 'data')
    fig_path = os.path.join(os.getcwd(), '..', 'figures')
    data_list = {'regionalemploymentbyagenovember2020.xls': 'emp'}
    freq = 'monthly'
    df = make_dataset(data_path, data_list, freq)
    df = make_interp(df)
    df = make_seasonal(df, freq)
    df = make_total_columns(df)
    age_ratios = make_empage_ratios(df)
    age_forecasts = gen_forecast(age_ratios, freq)*100
    allreg = make_reg_forc(df)
    allreg_forecasts = gen_forecast(allreg, freq)*100
    heatmap_vars = make_heatmap_vars(df)
    heatmap_forc = gen_forecast(heatmap_vars, freq)*100
    for back in range(1, 6):
        if back == 1:
            filename = re.sub(r'\W+', '',
                              age_forecasts[-back:].index[0]) + '.csv'
            age_df = make_age_df(age_forecasts[-back:])
            allreg_df = make_allreg_df(allreg_forecasts[-back:])
            heatmap_df = make_heatmap_df(heatmap_forc[-back:])
            age_df.to_csv(os.path.join(data_path, 'age_df.csv'))
            age_forecasts.to_csv(os.path.join(data_path, 'age_forecasts.csv'))
            heatmap_df.to_csv(os.path.join(data_path, 'heatmap_df.csv'))
            allreg_df.to_csv(os.path.join(data_path, 'allreg_df.csv'))
        else:
            filename = re.sub(r'\W+', '',
                              age_forecasts[-back:-back+1].index[0]) + '.csv'
            age_df = make_age_df(age_forecasts[-back:-back+1])
            allreg_df = make_allreg_df(allreg_forecasts[-back:-back+1])
            heatmap_df = make_heatmap_df(heatmap_forc[-back:-back+1])
        plot_func(age_forecasts, allreg_df,
                  age_df, heatmap_df, fig_path, filename)


if __name__ == '__main__':
    main()
