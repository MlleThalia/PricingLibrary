
import numpy as np

"""Draw binomial tree"""

def n_periods_trinomial_tree(payoff, delta, axe, title):
    n=payoff.shape[0]
    m=delta.shape[0]
    #Ecrivons data sous forme de liste dépendant de i
    liste1=list()
    liste2=list()
    for i in range(n):
        for j in range(2*i,-1,-1):
            liste1.append(payoff[i,j])
    for i in range(m):
        for j in range(2*i,-1,-1):
            liste2.append(delta[i,j])

    for i in range(n-1):
        x=[1,0,1]
        for j in range(i):
            x.append(0)
            x.append(1)
        x=np.array(x)+i
        y=np.arange(-(i+1),i+2)[::-1]

        z=np.array([i]*(n-i))
        w=np.array([-i]*(n-i))
        v=np.arange(i,n)
        
        axe.plot(x,y,'bo-')
        axe.plot(v,z,'bo-')   
        axe.plot(v,w,'bo-') 

    for i in range(n-2):
        x=[2,1,2]
        for j in range(i):
            x.append(1)
            x.append(2)
        x=np.array(x)+i
        y=np.arange(-(i+1),i+2)[::-1]
        axe.plot(x,y,'bo-')
    w=0
    x=0
    for i in range(n):
        for j in range(i,-i-1,-1):
            if i!=n-1:
                axe.text(i,j,(round(liste1[w],2),round(liste2[x],2)))
            else:
                axe.text(i,j,round(liste1[w],2))
            w=w+1
            x=x+1
    axe.set_title(title)