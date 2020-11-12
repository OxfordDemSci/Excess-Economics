"""Generate Autoarima forecasts."""
import pmdarima as pm
import pandas as pd
from pmdarima.arima import ndiffs


def gen_forecast(df, freq, n_diffs=0, alpha=0.1):
    """Generate forecasts."""
    if freq == 'quarterly':
        maxp = 4
        maxq = 4
        start = 75
    elif freq == 'monthly':
        maxp = 12
        maxq = 12
        start = 223
    for col in df.columns:
        if ('allage_empratio_allreg' in col):
            determ = 't'
        else:
            determ = 'ct'
        auto = pm.auto_arima(df[col][:start], d=n_diffs,
                             seasonal=True, stepwise=True, disp=0,
                             trace=False, suppress_warnings=True,
                             error_action="ignore",
                             max_p=maxp, max_q=maxq, max_order=None,
                             quiet=True, start_p=1, start_q=1, trend=determ)
        forc_insamp, ci_insamp = auto.predict_in_sample(return_conf_int=True,
                                                        dynamic=False)
        forc_outsamp, ci_outsamp = auto.predict(n_periods=len(df)-start,
                                                return_conf_int=True,
                                                alpha=alpha)
        df.loc[:start, col+'_f'] = forc_insamp
        df.loc[:start, col+'_ci_left'] = pd.DataFrame(ci_insamp)[0].to_list()
        df.loc[:start, col+'_ci_right'] = pd.DataFrame(ci_insamp)[1].to_list()
        df.loc[start:, col+'_f'] = forc_outsamp
        df.loc[start:, col+'_ci_left'] = pd.DataFrame(ci_outsamp)[0].to_list()
        df.loc[start:, col+'_ci_right'] = pd.DataFrame(ci_outsamp)[1].to_list()
    return df
