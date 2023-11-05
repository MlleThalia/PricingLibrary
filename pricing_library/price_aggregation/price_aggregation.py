import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

class PriceAggregation:

    @classmethod
    def read_data(cls, data_path):
        data  = pd.read_csv(data_path, sep=';')
        columns = data.columns
        for col in columns:
            if col.startswith("Unnamed"):
                del data[col]
        data = data.rename(columns={ 'Price' : 'price', 'Volume' : 'volume' })
        return data
    
    @classmethod
    def vwap(cls, data: pd.DataFrame):
        data['price_volume']=data['price']*data['volume']
        aggregate_price = sum(data['price_volume'])/sum(data['volume'])
        return aggregate_price
    
    @classmethod
    def linear_interpolation(cls, x, x0, x1, y0, y1):
        y= ((x1-x)*y0 + (x-x0)*y1)/(x1-x0)
        return y
    
    @classmethod
    def vwm(cls, data: pd.DataFrame):
        data = data.sort_values('price')
        new_index = np.arange(len(data))
        data.index = new_index
        data['cum_volume']= (np.cumsum(data['volume']))/(sum(data['volume']))
        for i in data.index:
            if data['cum_volume'][i]>=0.5:
                break
        aggregate_price = cls.linear_interpolation(0.5, data['cum_volume'][i-1], data['cum_volume'][i], data['price'][i-1], data['price'][i])
        return aggregate_price