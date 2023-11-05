
from scipy.stats import norm
import numpy as np
from option_pricing.utilities import *

from abc import ABC
from abc import abstractmethod

class Option(ABC):

    @abstractmethod
    def black_schole_price(cls, t,sigma, r, q, S0, T, K):
        pass

    @classmethod
    def dlm_price(cls, t,sigma, a, r, q, S0, T, K):
        S0 = S0+a
        K = K+a
        price = cls.black_schole_price(t,sigma, r, q, S0, T, K)
        return price

class VanillaOption(Option, ABC):

    @abstractmethod
    def black_schole_grecs(cls, t,sigma, r, q, S0, T, K):
        pass

    @classmethod
    def dlm_grecs(cls, t,sigma, a, r, q, S0, T, K):
        S0 = S0+a
        K = K+a
        delta,gamma,vega,theta,rho = cls.black_schole_grecs(t,sigma, r, q, S0, T, K)
        return delta,gamma,vega,theta,rho
    
    @classmethod
    def black_schole_newton_raphson_implied_volatility(cls, t, sigma, r, q, S0, T, K, call_market_price, iter):
        log_moneyness = np.log((S0*np.exp(r*(T-t)))/K)
        sigma = np.sqrt((2/(T-t))*np.abs(log_moneyness))
        for _ in range(iter):
            call_bs_price = cls.black_schole_price(t, sigma, r, q, S0, T, K)
            grecs_bs = cls.black_schole_grecs(t,sigma, r, q, S0, T, K)
            vega = grecs_bs[2]
            new_sigma = sigma - ((call_bs_price- call_market_price)/ (vega))
            sigma = new_sigma
        return sigma
    
    @classmethod
    def dlm_newton_raphson_implied_volatility(cls, t,sigma, a, r, q, S0, T, K, call_market_price, iter):
        log_moneyness = np.log((S0*np.exp(r*(T-t)))/K)
        sigma = np.sqrt((2/(T-t))*np.abs(log_moneyness))
        for _ in range(iter):
            call_bs_price = cls.dlm_price(t,sigma, a, r, q, S0, T, K)
            grecs_bs = cls.dlm_grecs(t,sigma, a, r, q, S0, T, K)
            vega = grecs_bs[2]
            new_sigma = sigma - ((call_bs_price- call_market_price)/ (vega))
            sigma = new_sigma
        return sigma
    
    @abstractmethod
    def binomial_price(cls, n, u, d, r, S0, K):
        pass

    @abstractmethod
    def binomial_delta(cls, n, u, d, r, S0, K):
        pass

class PowerOption(Option, ABC):
    pass

class BinaryOption(Option, ABC):
    pass

class CallVanillaOption(VanillaOption):

    @classmethod
    def black_schole_price(cls, t,sigma, r, q, S0, T, K):
        St=geometric_brownian(r,sigma,S0,t)
        d1=(1/(sigma*np.sqrt(T-t)))*np.log((St*np.exp((r-q)*(T-t)))/K)+(1/2)*sigma*np.sqrt(T-t)
        d0=d1-(sigma*np.sqrt(T-t))
        call=(St*np.exp(-q*(T-t)))*norm.cdf(d1)-np.exp(-r*(T-t))*K*norm.cdf(d0)
        return call
    
    @classmethod
    def black_schole_grecs(cls,t,sigma, r, q, S0, T, K):
        St=geometric_brownian(r,sigma,S0,t)
        d1=(1/(sigma*np.sqrt(T-t)))*np.log((St*np.exp((r-q)*(T-t)))/K)+(1/2)*sigma*np.sqrt(T-t)
        d0=d1-(sigma*np.sqrt(T-t))
        delta=norm.cdf(d1)
        gamma=norm.pdf(d1)/((St*np.exp(-q*(T-t)))*sigma*np.sqrt(T-t))
        vega=(St*np.exp(-q*(T-t)))*np.sqrt(T-t)*norm.pdf(d1)
        theta=(St*np.exp(-q*(T-t)))*norm.pdf(d1)*sigma/(2*np.sqrt(T-t))-r*np.exp(-r*(T-t))*K*norm.cdf(d0)
        rho=K*(T-t)*np.exp(-r*(T-t))*norm.cdf(d0)
        return delta,gamma,vega,theta,rho   
    
    @classmethod
    def binomial_price(cls, n, u, d, r, S0, K):
        R=1+r
        payoff=np.zeros((n+1,n+1))
        price=np.zeros((n+1,1))
        prix=n_period_price(n,u,d,S0)
        if R<=d or R>=u: 
            print("Error: Arbitrage")
        else:
            q=(R-d)/(u-d)
            for i in range(n+1):
                for j in range(n+1):
                    somme=0
                    if i>=j:
                        for k in range(n+1-i):
                            somme+=(call((u**k)*(d**(n-i-k))*prix[i,j],K)*combinaison(k,n-i)*(q**k)*((1-q)**(n-i-k)))/(R**(n-i))
                        payoff[i,j]=somme  
            for i in range(n+1): 
                somme=0
                for j in range(n+1):
                    if i>=j:
                        somme+=payoff[i,j]*(q**j)*((1-q)**(i-j))
                price[i]=somme
        
        return payoff

    @classmethod
    def binomial_delta(cls, n, u, d, r, S0, K):
        delta=np.zeros((n,n))
        Delta=np.zeros((n,1))
        payoff=cls.binomial_price(n,u,d,r,S0,K)[0]
        price=n_period_price(n,u,d,S0)
        R=1+r
        q=(R-d)/(u-d)
        for i in range(n):
            for j in range(n):
                if i>=j:
                    delta[i,j]=(payoff[i+1,j+1]-payoff[i+1,j])/((u-d)*price[i,j])
        for i in range(n): 
            somme=0
            for j in range(n):
                if i>=j:
                    somme+=delta[i,j]*(q**j)*((1-q)**(i-j))
            Delta[i]=somme
        return (delta,Delta)


class PutVanillaOption(VanillaOption):

    @classmethod
    def black_schole_price(cls, t,sigma, r, q, S0, T, K):
        St=geometric_brownian(r,sigma,S0,t)
        d1=(1/(sigma*np.sqrt(T-t)))*np.log((St*np.exp((r-q)*(T-t)))/K)+(1/2)*sigma*np.sqrt(T-t)
        d0=d1-(sigma*np.sqrt(T-t))
        put=np.exp(-r*(T-t))*K*norm.cdf(-d0)-(St*np.exp(-q*(T-t)))*norm.cdf(-d1)
        return put
    
    @classmethod
    def black_schole_grecs(cls,t,sigma, r, q, S0, T, K):
        St=geometric_brownian(r,sigma,S0,t)
        d1=(1/(sigma*np.sqrt(T-t)))*np.log((St*np.exp((r-q)*(T-t)))/K)+(1/2)*sigma*np.sqrt(T-t)
        d0=d1-(sigma*np.sqrt(T-t))
        delta=norm.cdf(d1)-1
        gamma=norm.pdf(d1)/((St*np.exp(-q*(T-t)))*sigma*np.sqrt(T-t))
        vega=(St*np.exp(-q*(T-t)))*np.sqrt(T-t)*norm.pdf(d1)
        theta=(St*np.exp(-q*(T-t)))*norm.pdf(d1)*sigma/(2*np.sqrt(T-t))+r*np.exp(-r*(T-t))*K*norm.cdf(-d0)
        rho=-K*(T-t)*np.exp(-r*(T-t))*norm.cdf(-d0)
        return delta,gamma,vega,theta,rho
    
    @classmethod
    def binomial_price(cls, n, u, d, r, S0, K):
        R=1+r
        payoff=np.zeros((n+1,n+1))
        price=np.zeros((n+1,1))
        prix=n_period_price(n,u,d,S0)
        if R<=d or R>=u: 
            print("Error: Arbitrage")
        else:
            q=(R-d)/(u-d)
            for i in range(n+1):
                for j in range(n+1):
                    somme=0
                    if i>=j:
                        for k in range(n+1-i):
                            somme+=(put((u**k)*(d**(n-i-k))*prix[i,j],K)*combinaison(k,n-i)*(q**k)*((1-q)**(n-i-k)))/(R**(n-i))
                        payoff[i,j]=somme 
            for i in range(n+1): 
                somme=0
                for j in range(n+1):
                    if i>=j:
                        somme+=payoff[i,j]*(q**j)*((1-q)**(i-j))
                price[i]=somme
            
            return payoff

    @classmethod
    def binomial_delta(cls, n, u, d, r, S0, K):
        delta=np.zeros((n,n))
        Delta=np.zeros((n,1))
        payoff=cls.binomial_price(n,u,d,r,S0,K)[0]
        price=n_period_price(n,u,d,S0)
        R=1+r
        q=(R-d)/(u-d)
        for i in range(n):
            for j in range(n):
                if i>=j:
                    delta[i,j]=(payoff[i+1,j+1]-payoff[i+1,j])/((u-d)*price[i,j])
        for i in range(n): 
            somme=0
            for j in range(n):
                if i>=j:
                    somme+=delta[i,j]*(q**j)*((1-q)**(i-j))
            Delta[i]=somme
        return (delta,Delta)

class CallBinaryOption(BinaryOption):

    @classmethod
    def black_schole_price(cls, t,sigma, r, q, S0, T, K):
        St=geometric_brownian(r,sigma,S0,t)
        d1=(1/(sigma*np.sqrt(T-t)))*np.log((St*np.exp((r-q)*(T-t)))/K)+(1/2)*sigma*np.sqrt(T-t)
        d0=d1-(sigma*np.sqrt(T))
        call=np.exp(-r*(T-t))*norm.cdf(d0)
        return call

class PutBinaryOption(BinaryOption):

    @classmethod
    def black_schole_price(cls, t,sigma, r, q, S0, T, K):
        St=geometric_brownian(r,sigma,S0,t)
        d1=(1/(sigma*np.sqrt(T-t)))*np.log((St*np.exp((r-q)*(T-t)))/K)+(1/2)*sigma*np.sqrt(T-t)
        d0=d1-(sigma*np.sqrt(T))
        put=np.exp(-r*(T-t))*norm.cdf(-d0)
        return put

class CallPowerOption(BinaryOption):

    @classmethod
    def black_schole_price(cls, t, n,sigma, r, q, S0, T, K):
        St=geometric_brownian(r,sigma,S0,t)
        S = np.power(S0, n)*np.exp((n-1)*(r+(n/2)*sigma**2)*T-n*q*T)
        d0=(1/(sigma*np.sqrt(T-t)))*np.log((St*np.exp((r-q)*(T-t)))/np.power(K, 1/n))-(1/2)*sigma*np.sqrt(T-t)
        d1=d0+(sigma*np.sqrt(T)*n)
        call=S*norm.cdf(d1)-np.exp(-r*(T-t))*K*norm.cdf(d0)
        return call

class PutPowerOption(BinaryOption):
    
    @classmethod
    def black_schole_price(cls, t, n,sigma, r, q, S0, T, K):
        St=geometric_brownian(r,sigma,S0,t)
        S = np.power(S0, n)*np.exp((n-1)*(r+(n/2)*sigma**2)*T-n*q*T)
        d0=(1/(sigma*np.sqrt(T-t)))*np.log((St*np.exp((r-q)*(T-t)))/np.power(K, 1/n))-(1/2)*sigma*np.sqrt(T-t)
        d1=d0+(sigma*np.sqrt(T)*n)
        put=np.exp(-r*(T-t))*K*norm.cdf(-d0)-S*norm.cdf(-d1)
        return put
    

class CallAmericaOption:  
    
    @classmethod
    def binomial_price(cls, n, u, d, r, S0, K):
        R=1+r
        payoff=np.zeros((n+1,n+1))
        prix=n_period_price(n,u,d,S0)
        if R<=d or R>=u: 
            print("Error: Arbitrage")
        else:
            q=(R-d)/(u-d)
            for i in range(n+1): 
                payoff[n, i] = call(prix[n, i], K)
            for i in range(n-1, -1, -1):
                for j in range(n):
                    if i>=j:
                        prix_n=(q*payoff[i+1,j+1]+(1-q)*payoff[i+1,j])/R
                        payoff[i,j]=max(call(prix[i,j],K), prix_n)
            return payoff
    
class PutAmericaOption:
    
    @classmethod
    def binomial_price(cls, n, u, d, r, S0, K):
        R=1+r
        payoff=np.zeros((n+1,n+1))
        prix=n_period_price(n,u,d,S0)
        if R<=d or R>=u: 
            print("Error: Arbitrage")
        else:
            q=(R-d)/(u-d)
            for i in range(n+1): 
                payoff[n, i] = put(prix[n, i], K)
            for i in range(n-1, -1, -1):
                for j in range(n):
                    if i>=j:
                        prix_n=(q*payoff[i+1,j+1]+(1-q)*payoff[i+1,j])/R
                        payoff[i,j]=max(put(prix[i,j],K), prix_n)
            return payoff