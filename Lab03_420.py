

import numpy as np
import matplotlib.pyplot as plt

def f(N, M):
    y = np.log10(N)
    return a - (b * M)

def multi_regress(y,Z):
    """
    Perform multiple linear regression.
    Parameters
    ----------
    y: float
        The number of events
    Z:  float
        Magnitude of events
    Returns
    -------
    a: 1D array with shape (p, 1)
        The regression coefficients
    residual_t: 1D array or column vector with shape (n, 1)
        The total residual errors
    residual_r: 1D array or column vector with shape (n, 1)
        The residual errors due to regression
    r_squared: float
        The coefficient of determination, r^2
    """
    
    

    # Perform multiple linear regression
    a = np.linalg.inv(Z.T @ Z) @ Z.T @ y
    ymodel = np.dot(Z, a)
    y_hat = np.average(y)
    residual_t = y - y_hat
    residual_r = y - ymodel
    sst = residual_t.T @ residual_t
    ssr = residual_r.T @ residual_r
    r_squared = (sst - ssr) / sst
    return a, residual_t, residual_r, r_squared





# Read the txt file
data = np.loadtxt("C:\\Users\\Chioma Jesus\\Downloads\\M_data.txt")
t_data = data[:,0]
M_data = data[:,1]

#different time windows
t_break=np.array([0,34,46,72,97,120])

#Display the image and include the time windows
plt.figure(figsize=(8,11))
plt.subplot(2,1,1)
plt.plot(t_data,M_data,'ko',markersize=1)
for tb in t_break:
    plt.plot([tb,tb],[-1.5,1.5],'--r')
plt.xlabel('time (hr)')
plt.ylabel('Magnitude of events')
M= np.linspace(-0.5,1.0,10)
N= np.zeros(M.shape)
#N= np.zeros_like(M)

#print(M_data>=-0.25)
#count the number of occurrence of each magnitude of event
for k,Mk in enumerate(M):
    N[k]=np.sum(np.where(M_data>=Mk,1,0))

y = np.log10(N)
Z =np.vstack([np.ones_like(M),-M]).T
#b = a[1]
a, residual_t, residual_r, r_squared = multi_regress(y,Z)
b = a[1]
print(a, residual_t, residual_r, r_squared)

#plot the total magnitude of events against the number of occurrence and calculate a and b
plt.subplot(2,1,2)
plt.semilogy(M,N,'ko')
plt.semilogy(M,10**(Z@a),'r--')
plt.title(f'G-R model: a={a[0]:0.4f}, b={a[1]:0.4f}, r_sq = {r_squared:0.4f}')
plt.xlabel('Magnitude , M')
plt.ylabel('Num. of events, N>=M')

#plot the total magnitude of events against the number of occurrence within the selected time windows and calculate: a and b
plt.figure(figsize=(11,11))
for j in range(len(t_break)-1):
    for k,Mk in enumerate(M):
        N[k]=np.sum(np.where((M_data>=Mk) & (t_data>= t_break[j]) & (t_data<= t_break[j+1]),1,0))
    y = np.log10(N)
    Z =np.vstack([np.ones_like(M),-M]).T

    a, residual_t, residual_r, r_squared = multi_regress(y,Z)
    plt.subplot(3,2,j+1)
    plt.semilogy(M,N,'ko')
    plt.semilogy(M,10**(Z@a),'r--')
    plt.title(f'a={a[0]:0.4f}, b={a[1]:0.4f}, r_sq = {r_squared:0.4f}')
    plt.ylim([1,1e4])
    plt.text(-0.5,5,f'{t_break[j]} <= t <= {t_break[j+1]}')
   
    plt.ylabel('Num. of events, N>=M')
plt.xlabel('Magnitude , M')
plt.show()