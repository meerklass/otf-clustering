import numpy as np
import matplotlib.pyplot as plt
import matplotlib as pl 
import matplotlib
import treecorr
from astropy import units as u
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from scipy.optimize import curve_fit
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
import corner

## Reading the data 
'''
this should not be a hardcoded file name, needs modification
'''
data_corr = np.load("/users/aishrila/clustering")
lines = data_corr.readlines()
header = lines[1]

header = header.strip()
headnames = header[1:]
headnames = headnames.split()
fmtlist=10*['<f8']
mydtype=zip(headnames,fmtlist)
Data_corr = np.loadtxt("w_theta.dat",skiprows=2,dtype=mydtype)
bin_centre = Data_corr["meanr"] # in arcmin 

theta_deg = (bin_centre)   # This is x value 
w_theta = np.abs(Data_corr["xi"])  # This is y value, we are taking all positive values  
w_theta_err = Data_corr["sigma_xi"] # this is error in y value 

#'data' stores (x,y) data along with the errors on y
data = np.ndarray(shape = (len(theta_deg), 3))
inf_data = np.ndarray(shape = (len(theta_deg), 3))

data[:,0]= theta_deg
data[:,1] = w_theta
data[:,2] = w_theta_err


def gen_model(A, gama, x):
    '''
    The model we want to fit (simple power law) of the form Ax^y, where x is the angular seperation
    and should y has a theoretical value of -0.8.

    A: amplitude of fitted function 
    gamma: power law index
    x: angular separation
    '''
    y = A*x**(-gama)
    return y


def log_likelihood(A, gama):
    '''
    Calculate the log likelihood for for the simple power law model
    '''
    lnL = -0.5*np.sum(((data[:,1] - gen_model(A, gama, data[:,0]))**2)/data[:,2]**2)
#    print(lnL)
    return lnL

#number of samples
nsample = 10**5

#burn in sample
nburn=int(nsample/10.0)

#Choose initial point to start the chain
#theta0 is in radian units
A0 = 10**-100
gama0 = 10**-100

#Parameters of proposal distribution
w_A = 0.0001
w_gama = 0.01
cov_matrix = np.ndarray(shape = (2,2))
cov_matrix[0,0] = w_A**2; cov_matrix[0,1] = 0.0; cov_matrix[1,0] = 0.0; cov_matrix[1,1] = w_gama**2

#Seed for random number generation while sampling
seed1 = 1012345
np.random.seed(seed = seed1)

Ai = A0   # Initial point 
gamai = gama0  # Initial point

#array to store total sample
total_sample = np.ndarray(shape = (nsample, 3))

#array to store accepted sample
acptd_sample = np.ndarray(shape = (nsample, 3))

#array to store logarithm of likelihood
lnL = np.ndarray(shape = (nsample, 1))*0.0
lnL[0] = log_likelihood(Ai, gamai)


n_accept=0
for i in range(1, nsample,1):

#gaussian proposal distribution, zero correlation between two chains.
    A_star, gama_star = np.random.multivariate_normal(mean = np.asarray([Ai,gamai]), cov = cov_matrix, size = None)
    temp_arr1 = np.asarray([i, A_star, gama_star])
    total_sample[i,:] = temp_arr1 

    lnL[i] = log_likelihood(A_star, gama_star)
    lnL_ratio = lnL[i] - lnL[i-1]
    
    if (lnL_ratio > 0.0) or (lnL_ratio > np.log(np.random.random_sample(size = None))):#accept proposed point
        Ai = A_star
        gamai = gama_star
        acptd_sample[i,:] = np.asarray([i, Ai, gamai])
        n_accept += 1
    else:
        Ai = Ai
        gamai = gamai
        #Currant (not proposed) point is re-added to the accepted sample.
        acptd_sample[i,:] = np.asarray([i, Ai, gamai])
        lnL[i] = lnL[i-1]
        print "acceptance ratio:", n_accept/(1.0*i)

        
#Compute and print the summary of sampled distribution.

A_mean = np.mean(acptd_sample[nburn:,1]); gama_mean = np.mean(acptd_sample[nburn:,2])

print "A_mean, A_std_dev:", np.mean(acptd_sample[nburn:,1]), np.sqrt(np.var(acptd_sample[nburn:,1]))
print "gama_mean, gama_std_dev:", np.mean(acptd_sample[nburn:,2]), np.sqrt(np.var(acptd_sample[nburn:,2]))
#print "covariance:", np.cov(acptd_sample[nburn:,1:3], y = None, rowvar=0, bias=0, ddof=None)


inf_data[:,1] = gen_model(A_mean, gama_mean, data[:,0])
np.savetxt("inferred_data.dat", inf_data)

np.savetxt("total_proposed_sample.dat", total_sample)
np.savetxt("accepted_sample.dat", acptd_sample)



### Corner plot ###
samp = np.ndarray(shape = ((len(acptd_sample)-nburn), 2))

samp[:,0] = np.log10(acptd_sample[nburn:,1]) 
samp[:,1] = acptd_sample[nburn:,2]

samples = samp
sample = acptd_sample[:,[1,2]]
fig = corner.corner(samples,labels=["$log(A)$",  "$\gamma$"],truths=[np.log10(A_mean), gama_mean],truth_color='crimson',bins=40,smooth=True,show_titles=True,title_kwargs={"fontsize": 12})
plt.show()



plt.figure(figsize=(8,6), dpi=100)
plt.subplot(2,1,1)
plot1 = plt.plot(acptd_sample[:,0], acptd_sample[:,1])
plt.xlabel(r"A", fontsize = 15)
plt.subplot(2,1,2)
plot2 = plt.plot(acptd_sample[:,0], acptd_sample[:,2])
plt.xlabel(r"$\gamma$", fontsize = 15)
#plt.savefig("chain.pdf")
plt.show()

plt.figure(figsize=(8,6), dpi=100)
plot3 = plt.scatter(acptd_sample[nburn:,1], acptd_sample[nburn:,2], c = lnL[nburn:,0])
plt.xlabel(r"A", fontsize = 15)

plt.ylabel(r"$\gamma$", fontsize = 15)
#plt.savefig("chain_scatterplot.pdf")
plt.show()

plt.figure(figsize=(8,6), dpi=100)
plot4 = plt.hist2d(acptd_sample[nburn:,1], acptd_sample[nburn:,2], bins=40, normed=True)
plt.xlabel(r"$\theta$ [radian]", fontsize = 15)
plt.ylabel(r"$c$", fontsize = 15)
#plt.savefig("twoD_histogram.pdf")
plt.show()

plt.figure(figsize=(8,6), dpi=100)
plt.subplot(2,1,1)
plot5 = plt.hist(acptd_sample[nburn:,1], bins = 20, normed=True)
plt.xlabel(r"$\theta$ [radian]", fontsize = 15)
plt.subplot(2,1,2)
plot6 = plt.hist(acptd_sample[nburn:,2], bins = 20, normed=True)
plt.xlabel(r"$c$", fontsize = 15)
#plt.savefig("oneD_histogram.pdf")
plt.show()

fig = plt.figure(figsize=(6,5), dpi=100)
plt.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt='o',markersize=4,color='red',capsize=5,capthick=0.5,linewidth=1.5,mfc='red',mec='red',ecolor='red',label="Lockman Hole 325 MHz") 
plt.plot(data[:,0], inf_data[:,1],color='black',label='rw_${fit}($')
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\theta^{\circ}$")
plt.ylabel(r'$\omega (\theta)$')
plt.legend()
plt.tight_layout()
plt.show()















