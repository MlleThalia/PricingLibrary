import numpy as np
from trinomial.utilities import *

"""Trinomial price calculus"""

def trinomial_price(n,u,d,S0):
    price=np.zeros((n+1,2*n+1))
    for i in range(n+1): 
        coeff=count(i,2*i+1)
        for j in range(2*i+1):
            price[i,j]=(u**coeff[j,0])*(d**coeff[j,2])*S0
    return price