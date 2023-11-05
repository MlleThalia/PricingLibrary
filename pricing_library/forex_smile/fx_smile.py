
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline


class ForexSmile:

    @classmethod
    def read_data(cls, data_path):
        data  = pd.read_csv(data_path, sep=';')
        columns = data.columns
        for col in columns:
            if col.startswith("Unnamed"):
                del data[col]
        data = data.rename(columns={ 'Strike' : 'strike', 'IV' : 'iv' })
        return data
    
    @classmethod
    def log_moneyness(cls, data: pd.DataFrame, F):
        data['log_moneyness']  = np.log(F/data['strike'])
        return data
    
    @classmethod
    def delta(cls, data: pd.DataFrame, T, t=0):
        data['d1']=(1/(data['iv']*np.sqrt(T-t)))*data['log_moneyness']+(1/2)*data['iv']*np.sqrt(T-t)
        data['delta'] = norm.cdf(data['d1'])
        return data
    
    @classmethod
    def search_delta_iv(cls, data, percentage):
        percentage = percentage/100
        delta_data = data[["delta", "iv"]]
        delta_data = delta_data.iloc[::-1]
        cs = CubicSpline(delta_data["delta"], delta_data['iv'])
        iv = cs(percentage)
        return iv
    
    @classmethod
    def butterfly_risk_metric(cls, data):
        iv_25 = cls.search_delta_iv(data, 25)
        iv_50 = cls.search_delta_iv(data, 50)
        iv_75 = cls.search_delta_iv(data, 75)
        butterfly_metric = ((iv_25+iv_75)/2)-iv_50
        return butterfly_metric
    
    @classmethod
    def risk_reversal_risk_metric(cls, data):
        iv_25 = cls.search_delta_iv(data, 25)
        iv_75 = cls.search_delta_iv(data, 75)
        risk_reversal_metric = iv_25-iv_75
        return risk_reversal_metric
