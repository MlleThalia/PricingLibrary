
from random import gauss
import numpy as np
from scipy.integrate import trapezoid


"""Here we have utilities used for n period binomial period"""

#region call and put payoff

def call(S,K):
    return max(S-K,0)

def put(S,K):
    return max(K-S,0)

#endregion

#region factorial

def factorial(x):
    if x<0:
        print("Negative number")
    elif x==0 or x==1:
        return 1
    else: 
        return x*factorial(x-1)
    
#endregion

#region combinaisons calculus

def combinaison(k,n):
    return factorial(n)/(factorial(k)*factorial(n-k))

#endregion


"""Geometric Brownian"""

def geometric_brownian(r,sigma,S0,t):
    return S0*np.exp(r*t-((sigma**2)/2)*t+sigma*gauss(0,np.sqrt(t)))

"""Tree price for n periods binomial model"""

def n_period_price(n,u,d,S0):
    prix=np.zeros((n+1,n+1))
    for i in range(n+1):   
        for j in range(n+1):
            if i>=j:
                prix[i,j]=(u**j)*(d**(i-j))*S0
    return prix

#CRR model

def crr_model(n,sigma, r, S0, T, K, type):
    """Call: type==0, Put: type==1"""
    h=T/n
    u=np.exp(sigma*np.sqrt(h))
    d=np.exp(-sigma*np.sqrt(h))
    R=np.exp(r*h)
    payoff=np.zeros((n+1,n+1))
    delta=np.zeros((n,n))
    prix=n_period_price(n,u,d,S0)
    price=np.zeros((n+1,1))
    if R<=d or R>=u: 
        print("Error: Arbitrage")
    else:
        q=(R-d)/(u-d)
        for i in range(n+1):
            if type==0: 
                for j in range(i+1):
                    somme=0
                    
                    for k in range(n+1-i):
                        somme+=(call((u**k)*(d**(n-i-k))*prix[i,j],K)*combinaison(k,n-i)*(q**k)*((1-q)**(n-i-k)))/(R**(n-i))
                    payoff[i,j]=somme
            else:
                for j in range(i+1):
                    somme=0
                    for k in range(n+1-i):
                        somme+=(put((u**k)*(d**(n-i-k))*prix[i,j],K)*combinaison(k,n-i)*(q**k)*((1-q)**(n-i-k)))/(R**(n-i))
                    payoff[i,j]=somme
        for i in range(n+1): 
            somme=0
            for j in range(n+1):
                if i>=j:
                    somme+=payoff[i,j]*(q**j)*((1-q)**(i-j))
            price[i]=somme
        for i in range(n):
            for j in range(n):
                if i>=j:
                    delta[i,j]=(payoff[i+1,j+1]-payoff[i+1,j])/((u-d)*prix[i,j])
        return (payoff,price,delta)

"""Draw binomial tree with the prices"""

def n_periods_binomial_tree(pay_off, delta, axe, title):
    n=len(pay_off)
    m=len(delta)
    liste1=list()
    liste2=list()
    for i in range(n):
        for j in range(n-1,-1,-1):
            if i>=j:
                liste1.append(pay_off[i,j])
    for i in range(m):
        for j in range(m-1,-1,-1):
            if i>=j:
                liste2.append(delta[i,j])
    for i in range(n-1):
        x=[1,0,1]
        for j in range(i):
            x.append(0)
            x.append(1)
        x=np.array(x)+i
        y=np.arange(-(i+1),i+2)[::-1]
        axe.plot(x,y,'bo-')     
    w=0
    x=0
    for i in range(n):
        for j in range(i,-i-1,-2):
            if i!=n-1:
                axe.text(i,j,(round(liste1[w],2),round(liste2[x],2)))
            else:
                axe.text(i,j,round(liste1[w],2))
            w=w+1
            x=x+1
    axe.set_title(title)

"""One period binomial model"""

def one_period_binomial_model(u,d,r,S0,K,type):
    """Call: type==0, Put: type==1"""
    R=1+r
    if R<=d or R>=u: 
        print("Error: Arbitrage")
    else:
        q=(R-d)/(u-d)
        if type==0: 
            Cu=call(u*S0,K)
            Cd=call(d*S0,K)
        else:
            Cu=put(u*S0,K)
            Cd=put(d*S0,K)
        return ((q*Cu+(1-q)*Cd)/R,(Cu-Cd)/((u-d)*S0))
    
"""Call-put Parity"""

def call_put_parity(call, S0, K, r, T, t=0):
    put=call-S0+K*np.exp(-r*(T-t))
    return put

"""Gaussian Laplace Formula"""

def gaussian_laplace_formula(m, sigma, t):
    expected = np.exp(m*t + 0.5*sigma**2*t**2)
    return expected

"""Log-forward moneyness"""

def log_forward_moneyness(S0, K, r, T, t=0):
    log_moneyness = np.log((S0*np.exp(r*(T-t)))/K)
    return log_moneyness

"""Initial volatility to put in newton-raphson algorithm"""

def newton_raphson_initial_volatility(r, S0, K, T, t=0):
    log_moneyness = np.log((S0*np.exp(r*(T-t)))/K)
    initial_sigma = np.sqrt((2/(T-t))*np.abs(log_moneyness))
    return initial_sigma

"""Determinist volatility"""
def determinist_volatility(x, y):
    sigma = np.sqrt(np.sum((y**2)*(x[1:]-x[:-1])/(x[-1]-x[0])))
    return sigma

"""Spot volatility"""
def spot_vol(vol_vector):
    return np.sqrt(np.sum(np.power(vol_vector, 2)))

"""correlation"""
def correlation(vol_1, vol_2):
    vol_spot_1 = spot_vol(vol_1)
    vol_spot_2 = spot_vol(vol_2)
    correlation = np.sum((vol_1/vol_spot_1)*(vol_2/vol_spot_2))
    return correlation