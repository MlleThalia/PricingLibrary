
import pandas as pd
import numpy as np
from scipy.stats import norm

from option_pricing.options_pricing import CallVanillaOption


class DeltaHedging:

    @classmethod
    def read_data(cls, data_path):
        data  = pd.read_csv(data_path, sep=';')
        columns = data.columns
        for col in columns:
            if col.startswith("Unnamed"):
                del data[col]
        data = data.rename(columns={ 'Date' : 't', 'Spot' : 'St' })
        return data
    
    @classmethod
    def vector_variation(cls, vector):
        vector_lenght = len(vector)
        variation = [vector[i+1]-vector[i] for i in range(vector_lenght-1)]
        variation = [0]+variation
        return variation

    @classmethod
    def call_vanilla_option_delta_hedging(cls, data: pd.DataFrame, sigma, r, q, T, K):
        S0 = data['St'][0]
        call_price = CallVanillaOption.black_schole_price(0, sigma, r, q, S0, T, K)
        data['d1']=(1/(sigma*np.sqrt(T-data['t'])))*np.log((data['St']*np.exp((r-q)*(T-data['t'])))/K)+(1/2)*sigma*np.sqrt(T-data['t'])
        data['delta'] = np.exp(-q*(T-data['t']))*norm.cdf(data['d1'])
        data['montant_actif'] = data['delta']*data['St']
        data['delta_St'] = cls.vector_variation(data['St'])
        data['delta_t'] = cls.vector_variation(data['t'])
        montant_portefeuille = [call_price]
        for i in range(len(data)-1):
            delta_portefeuille = data['delta'][i]*(data['delta_St'][i+1]-data['St'][i]*(np.exp(r*data['delta_t'][i+1])-1))
            portofolio_new_price = montant_portefeuille[i]*np.exp(r*data['delta_t'][i+1])+delta_portefeuille
            montant_portefeuille.append(portofolio_new_price)
        data['montant_portefeuille'] = montant_portefeuille
        data['montant_cash'] = data['montant_portefeuille']-data['montant_actif']
        erreur_couverture = data['montant_portefeuille'].iloc[-1]-max(data['St'].iloc[-1]-K, 0)
        return erreur_couverture