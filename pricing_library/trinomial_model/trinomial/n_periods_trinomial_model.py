
import numpy as np
from trinomial.utilities import *
from trinomial.n_periods_trinomial_price import trinomial_price


"""n periods trinomial model"""

def n_periods_trinomial_model(n,sigma,r,S0,T,K,type):
    """Call: type==0, Put: type==1"""
    h=T/n
    u=np.exp(sigma*np.sqrt(2*h))
    d=np.exp(-sigma*np.sqrt(2*h))
    R=np.exp(r*h)
    payoff=np.zeros((n+1,2*n+1))
    delta=np.zeros((n,2*n+1))
    prix=trinomial_price(n,u,d,S0)
    price=np.zeros((n+1,1))
    if R<=d or R>=u: 
        print("Error: Arbitrage")
    else:
        a=np.exp((r*h)/2)
        b=np.exp(sigma*np.sqrt(h/2))
        c=np.exp(-sigma*np.sqrt(h/2))
        qu=((a-c)/(b-c))**2
        qd=((b-a)/(b-c))**2
        qm=1-qu-qd
        for i in range(n+1):
            if type==0: 
                for j in range(2*i+1):
                    somme=0
                    coeff=count(n-i,2*(n-i)+1)
                    for k in range(2*(n-i)+1):
                        somme+=(call((u**coeff[k,0])*(d**coeff[k,2])*prix[i,j],K)*trinome(coeff[k,0],coeff[k,1],coeff[k,2],n-i)*(qu**coeff[k,0])*(qd**coeff[k,2])*(qm**coeff[k,1]))/(R**(n-i))
                    payoff[i,j]=somme
            else:
                for j in range(2*i+1):
                    somme=0
                    coeff=count(n-i,2*(n-i)+1)
                    for k in range(2*(n-i)+1):
                        somme+=(put((u**coeff[k,0])*(d**coeff[k,2])*prix[i,j],K)*trinome(coeff[k,0],coeff[k,1],coeff[k,2],n-i)*(qu**coeff[k,0])*(qd**coeff[k,2])*(qm**coeff[k,1]))/(R**(n-i))
                    payoff[i,j]=somme
            
        for i in range(n+1): 
            somme=0
            coeff=count(i,2*i+1)
            for j in range(2*i+1):
                somme+=payoff[i,j]*(qu**coeff[j,0])*(qd**coeff[j,2])*(qm**coeff[j,1])
            price[i]=somme
        
        for i in range(n):
            for j in range(2*i+1):
                delta[i,j]=(payoff[i+1,j+2]-payoff[i+1,j])/((u-d)*prix[i,j])
        return (payoff,price,delta)