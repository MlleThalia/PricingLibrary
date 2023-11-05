import numpy as np

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

#region Possible combinaison of Nu, Nm et Nd

def count(i,j):
    w=0
    N=np.zeros((j,3))
    for l in range(j):
        for k in range(j):
            for m in range(j):
                if l+k+m==i:      
                    N[w,:]=[l,k,m] 
                    if k==i:
                        return N
                    N[j-1-w,:]=[m,k,l]
                    w+=1
#endregion

#region Newton trinome formula

def trinome(i,j,k,n):
    return(factorial(n))/(factorial(i)*factorial(j)*factorial(k))

#endregion