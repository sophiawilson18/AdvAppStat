# coding: utf-8

#Author: Sophia Wilson


import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib as mpl
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
import sys                                             # Modules to see files and folders in directories
from scipy import stats
from scipy.stats import binom, poisson, norm           # Functions from SciPy Stats...
import math
import itertools
from matplotlib.patches import ConnectionPatch
from tqdm.notebook import tqdm
import seaborn as sns
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance


sys.path.append('External_Functions')
from ExternalFunctions import Chi2Regression, UnbinnedLH, BinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax 

plt.rcParams['font.size'] = 14

# ======================================
# GENERAL
# ======================================


def weightedmean(x,sigma):
    mean = sum(x/sigma**2)/sum(sigma**-2)
    uncertainty = np.sqrt(1/sum(sigma**-2))
    return mean, uncertainty


def compare(mu1,mu2,sigma1,sigma2):
    'Two sided z-test'
    
    dmu = np.abs(mu1-mu2) 
    dsigma = np.sqrt(sigma1**2+sigma2**2)
    nsigma = dmu/dsigma
    
    p = norm.cdf(0, dmu, dsigma)*2
    return dmu, dsigma, nsigma, p 


def ztest(mu1,mu2,sigma1,sigma2,side):
    'z/t-test onesided or twosided'
    
    dmu = np.abs(mu1-mu2) 
    dsigma = np.sqrt(sigma1**2+sigma2**2)
    nsigma = dmu/dsigma
    
    p = norm.cdf(0, dmu, dsigma)*side
    return nsigma, p 

def ztest_corr(mu1,mu2,sigma1,sigma2, cov, side):
    'z/t-test onesided or twosided for correlated measurements'
    
    dmu = np.abs(mu1-mu2) 
    dsigma = np.sqrt(sigma1**2+sigma2**2-2*cov) #where cov is the covariance of mu1 and mu2
    nsigma = dmu/dsigma
    
    p = norm.cdf(0, dmu, dsigma)*side
    return nsigma, p 

def ztest_cauchy(mu1,mu2,sigma1,sigma2,side):
    'z/t-test onesided or twosided for numbers drawn from a Cauchy distribution'
    
    dmu = np.abs(mu1-mu2) 
    dsigma = sigma1**2+sigma2
    nsigma = dmu/dsigma
    
    p = norm.cdf(0, dmu, dsigma)*side
    return nsigma, p 


def chi_square(y_exp, y_obs, y_err):
    'Chi square'
    return np.sum((y_exp-y_obs)**2/y_err**2)

def chis_quare_and_prob(x_exp, x_obs, x_obs_sigma, N_par):
    'Chi square value and probability'
    chi2_value = sum((x_exp-x_obs)**2/x_obs_sigma**2)
    Ndof_value = len(x_exp)-N_par
    chi2_prob = stats.chi2.sf(chi2_value, Ndof_value)
    
    return chi2_value, chi2_prob


def variance(data, bias = False):
    'Biased and unbiased variance'
    N = len(data)
    if bias: return sum((data-np.average(data))**2) / N   # biased
    else: return sum((data-np.average(data))**2) / (N-1)  # unbiased



def std_of_std(x):
    'Uncertainty on the standard deviation'
    return x / np.sqrt(2*(len(x)-1))



def n_sigma(N, p_global = 0.01):
    'z-value determined from global probability and trial factor'
    p_local = 1-(1-p_global)**(1/N)
    p_local = 1-np.exp(np.log(1-p_global)/N)
    print(p_local)
    z = abs(stats.norm.ppf(p_local, 0, 1))
    return z


def chauvenet_criteria(x, z, verbose=False):
    'Removing data points using Chauvenet Criteria'
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    
    mask = ([abs(x-mu) <z*sigma])[0]
    x_keep = x[mask]
    
    if verbose==True:
        print('Index', np.where(~mask)[0],
              '\nValue', x[~mask])
              
    if len(x_keep)==len(x):
        return x_keep
    
    return chauvenet_criteria(x_keep, z, verbose)


def likelihood(data, pdf, pars):
    'Likelihood given data, pdf and parameters'
    return np.prod(pdf(data, *pars))

def llh(data, pdf, pars):
    'Negative log likelikehood given data, pdf and parameters'
    return - np.sum(np.log(pdf(data, *pars)))


def find_closest(data, value):
    'Function for finding the index of the element in array that is closets to the value'
    data_left = data[:np.argmin(data)]
    data_right = data[np.argmin(data):]
    diff_left = abs(np.array(data_left)-value)
    diff_right = abs(np.array(data_right)-value)
    idx_left = np.argmin(diff_left)
    idx_right = np.argmin(diff_right)
    return idx_left, np.argmin(data) + idx_right


def confidence_level(data, sigma = 1):
    
    if sigma == 1: frac = 0.6827 / 2
    if sigma == 2: frac = 0.9545 / 2
    if sigma == 3: frac = 0.9973 / 2
    
    n = len(data)
    data_mean = np.mean(data)
    data_sorted = sorted(data)
    
    # index for lower and higher limit
    low_lim = int(n * (0.5 - frac) + 0.5)
    high_lim = int(n - low_lim + 0.5)
    
    # confidence levels
    data_low_sigma = data_mean - data_sorted[low_lim]
    data_high_sigma = - (data_mean - data_sorted[high_lim])
    
    return data_mean, data_low_sigma, data_high_sigma

def loglikelihood(x, mu, sigma):
    'POSITIVE log likelihood'
    return np.sum(np.log(likelihood(x, mu, sigma)), axis=1)


def rasterscan(x_samples, a_true, b_true, a_range, b_range, a_name, b_name, ax):
    # 2D grid with all possible combinations of mu-sigma pairs
    grid = np.array(list(itertools.product(a_range,b_range)))
    
    # breaking down the a-b values in the 2D grid into separate arrays and expanding their axes to enable
    #arithmetic operations with x_samples.
    a_from_grid = np.expand_dims(grid[:,0],axis=1)
    b_from_grid = np.expand_dims(grid[:,1],axis=1)
    
    # evaluating the log_likelihood for all points in grid
    scanned_llh = loglikelihood(x_samples, a_from_grid, b_from_grid)

    #after the scan, we find the index in the flat scanned_llh array that gives us the maximum likelihood
    ind_max_llh = np.argmax(scanned_llh)

    # PLOT RESULT    
    # raster scan results 
    sc = ax.scatter(a_from_grid, b_from_grid, c = -2*(scanned_llh-max(scanned_llh)), marker = 's', s = 100, cmap = 'viridis')
    # c = -2*(scanned_llh+res.fun)

    # the true value
    ax.plot(a_true, b_true, marker='*', color='red', markersize=15, ls='none', label = 'True value')

    #the maximum of the raster scan LLH
    ax.plot(a_from_grid[ind_max_llh], b_from_grid[ind_max_llh], marker = '*', ls='none',
            color = 'black',markersize = 15, label = 'Maximum LLH from the scan')
    
    contours = ax.contour(a_from_grid.reshape(len(a_range), len(b_range)),
                          b_from_grid.reshape(len(a_range), len(b_range)),
                          -2*(scanned_llh-max(scanned_llh)).reshape(len(a_range), len(b_range)),
                          [2.3,4.5,6], colors=['white', 'grey', 'black']) # 2.3, 4.5 and 6 are C.L. for distributions
                                                                          # with two parameters

    ax.legend(loc = (0.1,0.8), labelcolor='w') 
    ax.set_xlabel(a_name)
    ax.set_ylabel(b_name)

    cb = fig.colorbar(sc,label= r'$-2 \Delta$ LLH')
    #put_ticks(fig,ax)
    
    return [a_from_grid[ind_max_llh], b_from_grid[ind_max_llh]]
    
    
def integrate(x, y, x_low_lim, x_high_lim):
    'Integrates pdf from low limit to high limit'
    idx_low_lim, idx_high_lim = np.searchsorted(x, [x_low_lim, x_high_lim])
    integral = np.trapz(y[idx_low_lim:idx_high_lim+1], x[idx_low_lim:idx_high_lim+1])
    return integral


# ======================================
# PROBABILITY DENSITY / MASS FUNCTIONS
# ======================================


def binomial_pmf(x, n, p):
    """Biominal distribution """
    return binom.pmf(x, n, p)

def poisson_pmf(x, lamb):
    """Poisson distribution"""
    return poisson.pmf(x, lamb)

def gaussian_pdf(x, mu, sigma):
    """Gaussian distribution"""
    return norm.pdf(x, mu, sigma)

def gaussian_unit(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x - mu)**2 / sigma**2)

def gauss_extended(x, N, mu, sigma) :
    """Extended (non-normalized) Gaussian"""
    return N * gaussian_pdf(x, mu, sigma)

def poisson_extended(x, N, lamb):
    """Extended (non-normalized) Poisson distribution"""
    return N * poisson.pmf(x, lamb)

def doublegauss(x, N1, N2, mu, sigma1, sigma2):
    'Doubble Gaussian distribution'
    return N1 * gaussian_pdf(x, mu, sigma1) +  N2 * gaussian_pdf(x, mu, sigma2)

def onegauss(x, N, mu, sigma, c) :
    """Extended (non-normalized) Gaussian"""
    return N * gaussian_pdf(x, mu, sigma) + c

def twogauss(x, N1, N2, mu1, mu2, sigma1, sigma2, c):
    'Doubble Gaussian distribution'
    return N1 * gaussian_pdf(x, mu1, sigma1) +  N2 * gaussian_pdf(x, mu2, sigma2) + c

def threegauss(x, N1, N2, N3, mu1, mu2, mu3, sigma1, sigma2, sigma3, c):
    'Doubble Gaussian distribution'
    return N1 * gaussian_pdf(x, mu1, sigma1) +  N2 * gaussian_pdf(x, mu2, sigma2) + N3 * gaussian_pdf(x, mu3, sigma3) + c





# ======================================
# CHI SQUARE FIT
# ======================================


def llh_fit(data, likelihood, startparms, parmsnames, parmslimits=None, step=False, alpha_range=None, beta_range=None,
           verbose=False):
    steps_taken = []
    ''' Unbinned loglikelihood fit - ADVANCED APPSTAT '''
    
    def negloglikelihood(parms):
        '''Minuit needs a negative log likelihood function that only has the parameters as input'''
        steps_taken.append(parms) # appends the parameters for each step in fitting
        neg_llh = - np.sum(np.log(likelihood(data, *parms)))
        return neg_llh
    
    minuit_ullh = Minuit(negloglikelihood, startparms, name = parmsnames)
    
    if parmslimits:
        for i in range(len(parmslimits)):
            minuit_ullh.limits[i] = parmslimits[i]

    minuit_ullh.errordef = 0.5     # Value for likelihood fits
    minuit_ullh.migrad()           # Perform the actual fit
    
    if verbose: print(minuit_ullh.migrad())
    
    if (not minuit_ullh.fmin.is_valid):
        print("  WARNING: The Unbinned Likelihood fit DID NOT converge!!! ")
    
    
    par = minuit_ullh.values[:]
    par_err = minuit_ullh.errors[:] 
    par_name = minuit_ullh.parameters[:]
    negLLH_val = minuit_ullh.fval  
    Ndof_value = len(data) - minuit_ullh.nfit
    
    if step: 
        
        def find_nearest(a, a0):
            idx = np.abs(a - a0).argmin()
            return idx
    
        alpha_step = []
        beta_step = []
    
        for step in range(len(steps_taken)):
            alpha_step.append(alpha_range[find_nearest(alpha_range, steps_taken[step][0])])
            beta_step.append(beta_range[find_nearest(beta_range, steps_taken[step][1])])
        
        return par, par_err, alpha_step, beta_step
 

    else: 
        return par, par_err, negLLH_val
    
def llh_fit_text(par, par_err, negLLH_val, par_name, ax, d_xy):
    
    d = {'Neg. llh-value': negLLH_val
        #'Ndata':    len(x),
         #'Chi2':     chi2_value,
         #'Ndof':     Ndof_value,
         #'Prob':     chi2_prob,
        }
        
    for i in range(len(par)):
        d[f'{par_name[i]}'] = [par[i], par_err[i]]

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(*d_xy, text, ax, fontsize=16)
    
    
def chi2_hist(data, pdf, pars, N_bins):
    counts, bin_edges = np.histogram(data, bins=N_bins);
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    binwidth = bin_edges[1] - bin_edges[0]

    # Poisson errors on the count in each bin
    yerr = np.sqrt(counts)

    # We remove any bins, which don't have any counts in them:
    x = bin_centers[counts>0]
    y = counts[counts>0]
    yerr = yerr[counts>0]

    # Calculate the expected values (N*binwidth unnormalises it so it fits the histogram)
    y_exp = pdf(x, *pars) * len(data) * binwidth 

    # Degrees of freedom
    N_ddof = len(x) - len(pars) #n_point - n_parameters, notice n_points is the no. of bin centers

    # Calc chi2 and p value
    chi2 = np.sum( (y-y_exp)**2 / y_exp ) # y_exp = y_err^2
    chi2_reduced = chi2 / N_ddof
    prob = stats.chi2.sf(chi2, df=N_ddof) 
    print(f'The reduced Chi2 is: {chi2_reduced:.4f} with a p-value of: {prob:.4f}')
    
    
    
    return chi2_reduced, prob, binwidth

def chi2_hist_discrete(data, pdf, pars, N_bins):
    counts, bin_edges = np.histogram(data, bins=np.arange(-0.5, N_bins,1))
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    binwidth = bin_edges[1] - bin_edges[0]

    # Poisson errors on the count in each bin
    s_counts = np.sqrt(counts)

    # We remove any bins, which don't have any counts in them:
    x3 = bin_centers[counts>0]
    y3 = counts[counts>0]
    sy3 = s_counts[counts>0]

    # ---- POISSON -----
    # Calculate the expected values (N*binwidth unnormalises it so it fits the histogram)
    f_plot = pdf(x3, *pars)
    y_exp_p = f_plot * len(data) * binwidth

    # Degrees of freedom
    N_ddof = len(x3) - len(pars) #n_point - n_parameters, notice n_points is the no. of bin centers

    # Calc chi2 and p value
    chi2 = np.sum( (y3-y_exp_p)**2 / y_exp_p )
    chi2_reduced = chi2 / N_ddof
    prob = stats.chi2.sf(chi2, df=N_ddof)
    print(f'The reduced Chi2 is: {chi2_reduced:.4f} with a p-value of: {prob:.4f}')
    
    return chi2_reduced, prob, binwidth


def chisquarefit(x, y, ysigma, fitfunction, startparameters, ax, plot=False, xlabel='x', ylabel='y', funclabel='Chi2 fit model', d_xy=[0.05, 0.30]):
    'Chi square fit for (X,Y)-data' 
    
    'Chi-square fit'
    chi2fit = Chi2Regression(fitfunction, x, y, ysigma)
    minuit_chi2 = Minuit(chi2fit, *startparameters)
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad()
    
    if (not minuit_chi2.fmin.is_valid):
        print("  WARNING: The chi-square fit DID NOT converge!!! ")
    
    'Parameters and uncertainties'
    par = minuit_chi2.values[:]   
    par_err = minuit_chi2.errors[:]
    par_name = minuit_chi2.parameters[:]
    
    'Chi-square value, number of degress of freedom and probability'
    chi2_value = minuit_chi2.fval 
    Ndof_value = len(x)-len(par)
    chi2_prob = stats.chi2.sf(chi2_value, Ndof_value)
    
    'Plotting'
    if plot==True:
        x_axis = np.linspace(min(x), max(x), 1000000)
        
        d = {'Ndata':    len(x),
             'Chi2':     chi2_value,
             'Ndof':     Ndof_value,
             'Prob':     chi2_prob,
            }
        
        for i in range(len(par)):
            d[f'{par_name[i]}'] = [par[i], par_err[i]]
            
        ax.plot(x, y, 'k.', zorder=1)
        ax.plot(x_axis, fitfunction(x_axis, *minuit_chi2.values[:]), label=funclabel,zorder=3) 
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.errorbar(x, y, ysigma, fmt='ro', ecolor='k', label='Data', elinewidth=2, capsize=2, capthick=1,zorder=2)
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(*d_xy, text, ax, fontsize=16)
        ax.legend(fontsize=14)
    
    return chi2_value, Ndof_value, chi2_prob, par, par_err


def chisquarefit_histogram(X, fitfunction, startparameters, xmin, xmax, N_bins, ax, plot=False, verbose = False, xlabel='x', histlabel='Histogram', funclabel='Fitted function', d_xy=[0.65, 0.35], color='lightskyblue'):
   
    '''CHI SQUARE FIT FOR HISTROGRAMS'''
    
    'Histrogram'
    counts, bin_edges = np.histogram(X, bins=N_bins, range=(xmin, xmax)) #, density=True)
    
    'Define x, y, sy. Makes sure all bins are nonzero'
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])
    
    'Chi-square fit'
    chi2fit = Chi2Regression(fitfunction, x, y, sy)
    minuit_chi2 = Minuit(chi2fit, *startparameters)
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad()   
    
    if (not minuit_chi2.fmin.is_valid):
        print("  WARNING: The chi-square fit DID NOT converge!!! ")
    
    'Parameters and uncertainties'
    par = minuit_chi2.values[:]
    par_err = minuit_chi2.errors[:] 
    par_name = minuit_chi2.parameters[:]
    chi2_value = minuit_chi2.fval           
    N_NotEmptyBin = np.sum(y > 0)
    Ndof_value = N_NotEmptyBin - minuit_chi2.nfit
    Prob_value = stats.chi2.sf(chi2_value, Ndof_value) 
    
    'Printing Chi2 value, Ndof, prob, value and errors of fit parameters'
    if verbose == True:
        print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")
        
        for name in minuit_chi2.parameters:
            value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
            print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")

    'Plot histogram and fit'
    if plot == True:
        x_axis = np.linspace(xmin, xmax, 1000)
        
        
        hist_trans = ax.hist(X, bins=N_bins, range=(xmin, xmax), color=color, label=histlabel) #, density=True)
        #ax.errorbar(x, y, yerr=sy, fmt='.k', elinewidth=1, capsize=1, capthick=1, label = 'Counts with Possion errors')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.plot(x_axis, fitfunction(x_axis,*minuit_chi2.values), label=funclabel, color='black', linewidth=2)
        
            
        d = { 'CHI2':    chi2_value,
             'Entries':  len(X), 
             'Mean':     np.mean(X),
             'Std':      np.std(X),
             'Chi2':     chi2_value,
             'ndf':      Ndof_value,
             'Prob':     Prob_value,
            }
        
        for i in range(len(par)):
            d[f'{par_name[i]}'] = [par[i], par_err[i]]
            
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(*d_xy, text, ax, fontsize=14)
        ax.legend(loc='best', fontsize=14)
        
    
    return chi2_value, Ndof_value, Prob_value, par, par_err


def ullhfit(X, fitfunction, plotfunction, startpar, xmin, xmax, N_bins, ax, plot=False, verbose=False, xlabel='x', 
              funclabel='Fitted function (ULLH)', d_xy=[0.65, 0.35]): #histlabel='Histogram', 
    
    '''UNBINNED LIKELIHOOD FIT'''
    
    ullhfit = UnbinnedLH(fitfunction, X, bound=(xmin, xmax), extended=True)
    minuit_ullh = Minuit(ullhfit, *startpar)
    minuit_ullh.errordef = 0.5     # Value for likelihood fits
    minuit_ullh.migrad()           # Perform the actual fit
    
    if (not minuit_ullh.fmin.is_valid):
        print("  WARNING: The Unbinned Likelihood fit DID NOT converge!!! ")
        
    'Parameters and uncertainties'
    par = minuit_ullh.values[:]
    par_err = minuit_ullh.errors[:] 
    par_name = minuit_ullh.parameters[:]
    negativlogp_value = minuit_ullh.fval  
    Ndof_value = len(X) - minuit_ullh.nfit
    
    
    'Printing -log(P), Ndof, prob, value and errors of fit parameters'
    if verbose == True:
        print(f"-log(P) value: {negativlogp_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")
        
        for name in minuit_chi2.parameters:
            value, error = minuit_ullh.values[name], minuit_ullh.errors[name]
            print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")
            
    'Plot histogram and fit'
    if plot == True:
        x_axis = np.linspace(xmin, xmax, 1000)
        
        hist_trans = ax.hist(X, bins=N_bins, range=(xmin,xmax), color= 'lightskyblue') #, label=histlabel
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        #ax.plot(x_axis, fitfunction(x_axis,*par), label=funclabel)
        ax.plot(x_axis, plotfunction(x_axis, *par), '-', label='ULLH-fit model result', color='orange', linewidth=2.5) 
        
        
            
        d = {f'{funclabel}':  '---', 
             'Entries':  len(X), 
             '-log(P)':  negativlogp_value,
             'ndf':      Ndof_value,
            }
        
        for i in range(len(par)):
            d[f'{par_name[i]}'] = [par[i], par_err[i]]
            
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(*d_xy, text, ax, fontsize=14)
        ax.legend(loc='best', fontsize=14)
        
        counts, bin_edges = np.histogram(X, bins=N_bins, range=(xmin, xmax))
        x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
        y = counts[counts>0]
        sy = np.sqrt(counts[counts>0])
        ax.errorbar(x, y, yerr=sy, fmt='.k', elinewidth=1, capsize=1, capthick=1, label = 'Counts with Possion errors')
        
    
    return negativlogp_value, Ndof_value, par, par_err
  

    
def chisquarefit_histogram_bp(X, fitfunction, startparameters, xmin, xmax, N_bins, N_trials, ax, plot=False, verbose = False, xlabel='x', histlabel='Histogram', funclabel='Fitted function', d_xy=[0.65, 0.35]):

    '''CHI SQUARE FITTING BINOMIAL AND POISSON FOR HISTROGRAMS'''
    
    'Histrogram'
    counts, bin_edges = np.histogram(X, bins=N_bins, range=(xmin, xmax))
    
    'Define x, y, sy. Makes sure all bins are nonzero'
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])
    
    'Chi-square fit'
    chi2fit = Chi2Regression(fitfunction, x, y, sy)
    minuit_chi2 = Minuit(chi2fit, *startparameters)
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad() 
    
    if (not minuit_chi2.fmin.is_valid):
        print("  WARNING: The chi-square fit DID NOT converge!!! ")
    
    'Parameters and uncertainties'
    par = minuit_chi2.values[:]
    par_err = minuit_chi2.errors[:] 
    par_name = minuit_chi2.parameters[:]
    chi2_value = minuit_chi2.fval           
    N_NotEmptyBin = np.sum(y > 0)
    Ndof_value = N_NotEmptyBin - minuit_chi2.nfit
    Prob_value = stats.chi2.sf(chi2_value, Ndof_value) 
    
    'Printing Chi2 value, Ndof, prob, value and errors of fit parameters'
    if verbose == True:
        print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")
        
        for name in minuit_chi2.parameters:
            value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
            print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")
            
            
    'Plot histogram and fit'
    if plot == True:
        xaxis = np.linspace(xmin-0.5001, xmax, 100000) 
        yaxis = fitfunction(np.floor(xaxis+0.5), *par)
        
        #fig, ax = plt.subplots(figsize=(12, 6))
        hist = ax.hist(X, bins=N_bins, range=(xmin, xmax), histtype='bar', color= 'lightskyblue', label=histlabel)
        ax.errorbar(x, y, yerr=sy, fmt='.k', elinewidth=1, capsize=1, capthick=1, label = 'Counts with Possion errors')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.plot(xaxis, yaxis, '-', label=funclabel, lw=2, color='red')
            
        d = {#f'{funclabel}':  '---',
             'Entries':  len(X), 
             'Chi2':     chi2_value,
             'ndf':      Ndof_value,
             'Prob':     Prob_value,
            }
        
        for i in range(len(par)):
            d[f'{par_name[i]}'] = [par[i], par_err[i]]
            
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(*d_xy, text, ax, fontsize=14)
        ax.legend(loc='best', fontsize=14)
        
    
    return chi2_value, Ndof_value, Prob_value, par, par_err



# ======================================
# MONTE CARLO; ACCEPT AND REJECT 
# ======================================

def gen_MC(xrange, likelihood, parms, n_samples, pmf = False):
    '''Accept and reject MC - ADVANCED APPSTAT
    Input: range of x-values, likelihood function, list of parameters and number of samples
    Output: accepted and rejected x and pdf values, pdf curve and efficiency '''

    pdf_curve = likelihood(xrange, *parms)
    max_value = np.max(pdf_curve) # upper boundary for the accept-reject box
    min_value = 0

    n_accept = 0
    n_reject = 0
    x_accept, x_reject, pdf_accept, pdf_reject = [], [], [], []
    
    while n_accept < n_samples:
        # generating data 
        x_sample = np.random.uniform(min(xrange), max(xrange))   # random x sample
        
        if pmf: x_sample = int(x_sample + 0.5)
        
        pdf_sample = np.random.uniform(min_value, max_value)     # random y/pdf sample
        evaluated_pdfs_at_x = likelihood(x_sample, *parms)       # evaluate pdf at x sample given likelihood
        
        # accept-reject test 
        if (pdf_sample <= evaluated_pdfs_at_x):     
            x_accept = np.append(x_accept, x_sample)
            pdf_accept = np.append(pdf_accept, pdf_sample)
            n_accept += 1
        else:
            x_reject = np.append(x_reject, x_sample)
            pdf_reject = np.append(pdf_reject, pdf_sample)        
            n_reject += 1

    eff = n_samples/(n_samples+n_reject)
    
    return x_accept, x_reject, pdf_accept, pdf_reject, pdf_curve, eff



def acceptandreject(func, xmin, xmax, N_points): #N_bins
    'Random number with fixed seed'
    r = np.random
    
    'Generate random numbers within the fixed box'
    xaxis = np.linspace(xmin,xmax,N_points) 
    X_rnd = r.uniform(xmin, xmax, size=N_points)                       #random x values  
    Y_rnd = r.uniform(min(func(xaxis)),max(func(xaxis)),size=N_points) #random y values 
    
    'Accept and reject'
    fX = func(X_rnd)                    #fit function used on random x values
    fX_accepted = fX > Y_rnd            #condition for accept / reject
    X_accepted = X_rnd[fX_accepted]     #accepted X values
    Y_accepted = Y_rnd[fX_accepted]
    eff = len(X_accepted) / len(X_rnd)  #efficiency 
    return X_accepted, Y_accepted, eff

def acceptandrejectwpar(func, par, xmin, xmax, N_points, N_bins):
    'Random number with fixed seed'
    r = np.random
    
    'Generate random numbers within the fixed box'
    xaxis = np.linspace(xmin,xmax,N_points) 
    X_rnd = r.uniform(xmin, xmax, size=N_points)                       #random x values  
    Y_rnd = r.uniform(min(func(xaxis, *par)),max(func(xaxis, *par)),size=N_points) #random y values 
    
    'Accept and reject'
    fX = func(X_rnd, *par)                    #fit function used on random x values
    fX_accepted = fX > Y_rnd            #condition for accept / reject
    X_accepted = X_rnd[fX_accepted]     #accepted X values
    eff = len(X_accepted) / len(X_rnd)  #efficiency 
    return X_accepted, eff


# ======================================
# MONTE CARLO; COMBINATION OF METHODS 
# ======================================

def smart_func(x, tau, const=4):
    return const*np.exp(-x/tau)

#def smart_func(x, tau, const=1):
#    return const*x**3*np.sin(np.pi*x)+0.5

def test_smart_box(func, tau, xmin, xmax, const=1):
    xaxis = np.linspace(xmin,xmax,1000)
    plt.plot(xaxis, func(xaxis), label='fx')
    plt.plot(xaxis, smart_func(xaxis, tau, const), label='smart')
    plt.legend()


def smart_box(func, funcsmart, tau, ax, const=1, N=1000, plot = False, xlabel='x'):
    #x = np.linspace(0,20,100000)
    x = np.linspace(0,1,10000)
    N_accept = 0  
    N_reject = 0 
    x_accept = np.zeros(N)  
    
    xx = []
    yy = []
    
    while N_accept < N:
        r1 = np.random.random()
        x1 = -np.log(r1)*tau
        y1 = np.random.random()*funcsmart(x1, tau, const)
        if (y1 < func(x1)):        
            x_accept[N_accept] = x1
            N_accept += 1
            xx.append(x1)
            yy.append(y1)
        if (y1 > func(x1)):        
            N_reject+=1
            
    eff = N/(N+N_reject)  #N_accept = N 

    if plot == True:
        
        d = {'Entries': len(x_accept),
             'Efficiency': eff,
             'Mean': x_accept.mean(),
             'Std': x_accept.std(ddof=1),
             'Median': np.median(x_accept)
             }
        

        ax.scatter(xx,yy,color='dodgerblue', label='Generated values', s=3)
        xaxis = np.linspace(min(x),max(x_accept),len(x_accept))
        ax.plot(xaxis, func(xaxis), color='black', label = 'PDF',  linewidth=2)
        ax.plot(xaxis, funcsmart(xaxis, tau, const), color='red', label = 'Box function',  linewidth=2)
        ax.set_ylabel('PDF value', fontsize=14)
        #ax.vlines(np.median(x_accept),0,func(np.median(x_accept)), color = 'black', label='Median', 
        #           linestyle='dashed', linewidth=2)
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.13, 0.95, text, ax, fontsize=14)
        ax.legend(fontsize=14)
        
        
    
    return x_accept, eff
    








# ======================================
# COMPARE TWO HISTOGRAMS E.G. TRANSFORMATION AND ACCEPTANDREJECT 
# ======================================
def comparehistograms(hist1, hist2, verbose=False):
    'Terms in the chi2 sum'
    terms_in_sum = (hist_trans[0]-hist_accept[0])**2/(hist_trans[0]+hist_accept[0])
    
    'Chi2, Ndof and prob'
    chi2_value = sum(terms_in_sum)
    Ndof_value = len(terms_in_sum)
    Prob_value = stats.chi2.sf(chi_square,Ndof_value)
    
    'Printing Chi2 value, Ndof and prob value'
    if verbose == True:
        print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}   Prob(Chi2,Ndof) = {Prob_value:5.3f}")
    
    return chi2_value, Prob_value






# ======================================
# PLOTTING 
# ======================================
def plot_hist(X, xmin, xmax, N_bins, ax, xlabel='x', ylabel='PDF (nomalized freuqency)', histlabel='Histogram', d_xy = [0.80, 0.20], color= 'lightskyblue'):
    hist = ax.hist(X, bins=N_bins, range=(xmin, xmax), color=color, label=histlabel, density = True)
    
    d = {'Entries': len(X),
         'Mean': np.mean(X),
         'Std': np.std(X)
         }
    
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    #add_text_to_ax(*d_xy, text, ax, fontsize=14)
    
    counts, bin_edges = hist[0], hist[1]
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])
    #ax.errorbar(x, y, yerr=sy, fmt='.k', elinewidth=1, capsize=1, capthick=1, label = 'Possion errors')
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc='best', fontsize=14)
    


def plot_histandfunc(X, func, xmin, xmax, N_bins, ax, xlabel='x'):
    '''Plot of histogram with fit with no parameters'''
    
    hist = ax.hist(X, bins=N_bins, range=(xmin, xmax), color= 'lightskyblue', label='Histogram', density=True)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Frequency density", fontsize=14)

    x_axis = np.linspace(xmin, xmax, 1000)
    y_axis = func(x_axis)
    ax.plot(x_axis, y_axis, 'r-', label='Plotted function')

    d = {'Entries': len(X),
         'Mean': X.mean(),
         'Std': X.std(ddof=1),
         }
    
    counts, bin_edges = hist[0], hist[1]
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])
    #ax.errorbar(x, y, yerr=sy, fmt='.k', elinewidth=1, capsize=1, capthick=1, label = 'Counts with Possion errors')
    

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.05, 0.95, text, ax, fontsize=14)
    ax.legend(loc='best')
    
    
def plot_histandfunc_scatter(X,Y, eff, func, xmin, xmax, N_bins, ax, xlabel='x', d_xy=[0.13, 0.95]):
    '''Plot of histogram with fit with no parameters'''
    
    x_axis = np.linspace(xmin, xmax, 1000)
    y_axis = func(x_axis)
    ax.plot(x_axis, y_axis, 'r-', label='Plotted function')


    
    d = {'Entries': len(X),
        'Efficiency': eff,
        'Mean': X.mean(),
        'Std': X.std(ddof=1),
        }
        

    ax.scatter(X,Y,color='dodgerblue', label='Generated values', s=3)
    xaxis = np.linspace(min(X),max(X),len(X))
    #ax.plot(xaxis, func(xaxis), color='black', label = 'PDF',  linewidth=2)
    #ax.plot(xaxis, funcsmart(xaxis, tau, const), color='red', label = 'Box function',  linewidth=2)
    #ax.set_ylabel('PDF value', fontsize=14)
    
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(*d_xy, text, ax, fontsize=14)
    ax.legend(fontsize=14)
        
    
    
def plot_histandfuncwpar(X, func, par, xmin, xmax, N_bins, ax, xlabel='x'):
    '''Plot of histogram with fit with parameters'''
    hist = ax.hist(X, bins=N_bins, range=(xmin, xmax), color= 'lightskyblue', label='Histogram', density=True)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Frequency density", fontsize=14)
    
    x_axis = np.linspace(xmin, xmax, 1000)
    y_axis = func(x_axis, *par)
    ax.plot(x_axis, y_axis, 'r-', label='Plotted function')

    d = {'Entries': len(X),
         'Mean': np.mean(X.mean),
         'Std': np.std(X.std),
         }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.05, 0.95, text, ax, fontsize=14)

    counts, bin_edges = hist[0], hist[1]
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])
    #ax.errorbar(x, y, yerr=sy, fmt='.k', elinewidth=1, capsize=1, capthick=1, label = 'Counts with Possion errors')
    ax.legend(loc='best')
    
    
    
# ======================================  
# Correlation, Fischer discriminant and error rates (alpha and beta)
# ======================================    
    
def get_covariance_offdiag(X, Y):
    'Covariance'
    return np.cov(X, Y, ddof=1)[0, 1]


def calc_separation(x, y):
    'Seperation between data'
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    d = np.abs((mean_x - mean_y)) / np.sqrt(std_x**2 + std_y**2)
    
    return d
   
    
def correlation(x,y):
    'Correlation between two data sets'
    V_xy = 1/len(x) * sum((x - np.mean(x))*(y-np.mean(y)))
    rho_xy = V_xy / (np.std(x)*np.std(y))
    return rho_xy     

def correlation_plot(X, Y, N_bins, fig, ax, xlabel='X', ylabel='Y', d_xy=[0.05, 0.36]):
    'Plot 2d hist of two data sets'
    
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LogNorm

    counts, xedges, yedges, im = ax.hist2d(X, Y, bins=N_bins, cmap='coolwarm'); #, norm=LogNorm()
    divider = make_axes_locatable(ax)
    fig.colorbar(im, ax=ax)

    d = {'Entries': len(X),
         'Mean ' + xlabel : X.mean(),
         'Mean ' + ylabel : Y.mean(),
         'Std  ' + xlabel : X.std(ddof=1),
         'Std  ' + ylabel : Y.std(ddof=1),
         'Correlation' : np.cov(X,Y)[1,0]/(np.std(X)*np.std(Y))
        }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(*d_xy, text, ax, fontsize=14)
    
    

def Fisher_discriminant(A, B, N_bins, ax, hist1label='A', hist2label='B', d_xy=[-34, 200]):
    from numpy.linalg import inv
    'Fischer discriminant + plot'
    mu_A = np.mean(A, axis = 0)
    mu_B = np.mean(B, axis = 0)
    cov_A = np.cov(A.T)
    cov_B = np.cov(B.T)
    cov_sum = cov_A + cov_B
    cov_sum_inv = inv(cov_sum)
    wf = cov_sum_inv @ (mu_A - mu_B)
    fisher_data_A = A[:]@wf 
    fisher_data_B = B[:]@wf
    delta_fischer = calc_separation(fisher_data_A, fisher_data_B)
    xmin = min( min(fisher_data_A), min(fisher_data_B))
    xmax = max( max(fisher_data_A), max(fisher_data_B))

    bindwidth = abs(xmin - xmax)/N_bins *2
    xmin -= bindwidth
    xmax += bindwidth

    ax.hist(fisher_data_A, N_bins, (xmin, xmax),  histtype='step', color='Pink', lw=2, label=hist1label)
    ax.hist(fisher_data_B, N_bins, (xmin, xmax),  histtype='step', color='Lightskyblue', lw=2, label=hist2label)
    ax.set(xlim=(xmin, xmax), xlabel='Fisher-discriminant')
    ax.legend(fontsize=14)
    ax.text(*d_xy, fr'$\Delta_{{fisher}} = {delta_fischer:.3f}$', fontsize=14)
    
    return fisher_data_A, fisher_data_B, wf, delta_fischer




def errorrates(fisher_data_A, fisher_data_B, thres):
    'Error rates alpha and beta'
    if np.mean(fisher_data_A) > np.mean(fisher_data_B):
        cutof_A = len(fisher_data_A[fisher_data_A<=thres])
        alpha = cutof_A / len(fisher_data_A)
        cutof_B = len(fisher_data_B[fisher_data_B>=thres])
        beta = cutof_B / len(fisher_data_B)
        
    else:
        cutof_A = len(fisher_data_A[fisher_data_A>=thres])
        alpha = cutof_A / len(fisher_data_A)
        cutof_B = len(fisher_data_B[fisher_data_B<=thres])
        beta = cutof_B / len(fisher_data_B)
    return alpha, beta

    

    
    
# ======================================  
# P-VALUES
# ======================================
def ax_text(x, ax, posx, posy, color='k'):
    
    d = {'Entries': len(x), 
         'Mean': np.mean(x),
         'Std': np.std(x),
        }
    
    add_text_to_ax(posx, posy, nice_string_output(d), ax, fontsize=14, color=color)
    return None

def mean_std_sdom(x):
    std = np.std(x, ddof=1)
    return np.mean(x), std, std / np.sqrt(len(x))


def plot_pvalues(func1, func2, ax, xmin=0, xmax=20, N_exp = 1000, N_bins = 100):
    all_p_mean = np.zeros(N_exp)
    all_p_chi2 = np.zeros(N_exp)
    all_p_ks   = np.zeros(N_exp)
    verbose = True
    
    for iexp in range(N_exp):
        
        # Generate data:
        x_A_array = func1()
        x_B_array = func2()
    
        # Test if there is a difference in the mean:
        # ------------------------------------------
        # Calculate mean and error on mean:
        mean_A, width_A, sdom_A = mean_std_sdom(x_A_array) 
        mean_B, width_B, sdom_B = mean_std_sdom(x_B_array) 

        # Consider the difference between means in terms of the uncertainty:
        d_mean = mean_A - mean_B
        z_mean = d_mean/np.sqrt(sdom_A**2+sdom_B**2)

        # Turn a number of sigmas into a probability (i.e. p-value):
        p_mean = 1.0 - stats.norm.cdf(z_mean,loc=0,scale=1)  
        all_p_mean[iexp] = p_mean
    
    
        # Test if there is a difference with the chi2:
        # --------------------------------------------
        # Chi2 Test (where data must be binned first):
        [bins_A,edges_A] = np.histogram(x_A_array, bins=N_bins, range=(xmin,xmax))
        [bins_B,edges_B] = np.histogram(x_B_array, bins=N_bins, range=(xmin,xmax))
        centres_common = edges_A[1:] + (edges_A[:-1]-edges_A[1:])/2      # Same for A and B
        mask  = (bins_A + bins_B)!=0           # Mask empty bins to avoid dividing through 0
        chi2  = np.sum(((bins_A[mask] - bins_B[mask]) / np.sqrt(bins_A[mask]+bins_B[mask]))**2)
        n_dof = len(bins_A[mask])            # There are no parameters as it is not a fit!
        p_chi2= stats.chi2.sf(chi2,n_dof)
        all_p_chi2[iexp] = p_chi2

    
        # Test if there is a difference with the Kolmogorov-Smirnov test on arrays (i.e. unbinned):
        # -----------------------------------------------------------------------------------------
        p_ks = stats.ks_2samp(x_A_array, x_B_array)[1]           # Fortunately, the K-S test is implemented in stats!
        all_p_ks[iexp] = p_ks

        '''
        # Print the results for the first 10 experiments
        if (verbose and iexp < 10) :
            print(f"{iexp:4d}:  p_mean: {p_mean:7.5f}   p_chi2: {p_chi2:7.5f}   p_ks: {p_ks:7.5f}")

        '''
        # In case one wants to plot the distribution for visual inspection:
    if (N_exp > 1):
        
        ax[0].hist(x_A_array, N_bins, (xmin, xmax), histtype='step', label='A', color='blue')
        ax[0].set(title='Histograms of A and B', xlabel='A / B', ylabel='Frequency')        
        ax_text(x_A_array, ax[0], 0.04, 0.85, 'blue')

        ax[0].hist(x_B_array, N_bins, (xmin, xmax), histtype='step', label='B', color='red')
        ax_text(x_B_array, ax[0], 0.04, 0.65, 'red')
        
        ax[0].legend()
    
        ax[1].hist(all_p_mean, bins=50, range=(0, 1), histtype='step')
        ax[1].set(title='Probability mu', xlabel='p-value', ylabel='Frequency', xlim=(0, 1))
        ax_text(all_p_mean, ax[1], 0.04, 0.25)
    

        ax[2].hist(all_p_chi2, bins=50, range=(0, 1), histtype='step')
        ax[2].set(title='Probability chi2', xlabel='p-value', ylabel='Frequency', xlim=(0, 1))
        ax_text(all_p_chi2, ax[2], 0.04, 0.25)
    
        ax[3].hist(all_p_ks, bins=50, range=(0, 1), histtype='step')
        ax[3].set(title='Probability Kolmogorov Smirnov', xlabel='p-value', ylabel='Frequency', xlim=(0, 1))
        ax_text(all_p_ks, ax[3], 0.04, 0.25)

        
    return print('P-values')
    
    
    
    
    


# ======================================  
# ERROR PROBABGATION 
# ======================================
def error_prop(f,variables,values,uncertainties, cov = None, ftype = 'Sympy', verbose  = False):
    from sympy.tensor.array import derive_by_array
    from numpy import identity, array, dot, matmul
    from latex2sympy2 import latex2sympy
    from sympy import sqrt, Symbol, latex
    from sympy.abc import sigma
    
    if ftype == 'LaTeX':
        f = latex2sympy(f)


    if type(cov) == type(None):
        cov = np.diag(np.array(uncertainties)**2)

    

    subs_dict = dict(zip(variables, values))
    gradient = derive_by_array(f,variables).subs(subs_dict)
    
    VECTOR = array([element.evalf() for element in gradient])


    if verbose:
        from sympy.printing.latex import print_latex
        print('           -- python --         ')
        print(derive_by_array(f,variables))

        print('\n         -- LaTeX  --         ')
        print_latex(derive_by_array(f,variables)) 

        print('\n         -- variables  --         ')
        print(variables)


        print('\n         -- value  --         ')
        print(f.subs(subs_dict).evalf())


        print('\n         -- Forsøg på at sammensætte  --         ')
        F = 0


        for i in range(len(variables)):
            var  = str(variables[i])
            term = derive_by_array(f,variables)[i]
            F += term**2 * Symbol('sigma_' + var)**2

        print('\n         -- Function Value  --         ')
        
        print(latex(sqrt(F)))

    return float(dot(VECTOR  , matmul(cov , VECTOR))**0.5)



### ROC CURVE

# Calculate ROC curve from two histograms (hist1 is signal, hist2 is background):
def calc_ROC(hist1, hist2):

    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges):
        # kat: they have to be equal otherwise we cannot compare the bins of the differet data
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers):  # i = [1,2,3,...], x = [x_center[0], x_center[1,...]]
            
            # The cut mask
            cut = (x_centers < x) # True/false array where it determines if the values in x_centers
                                  # are lower than x
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / (TP + FN)                    # True positive rate
            
            #TP is the sum of all values higher than x. TN is the sum of all values lower than x.
            # TPR is the ratio, that gives the points on the ROC-curve. 
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / (FP + TN)                     # False positive rate            
            
        AUC = np.trapz(TPR[::-1], FPR[::-1])
        return FPR, TPR, AUC
    
    else:
        return('Signal and Background histograms have different bins and ranges')





######### Mollweide projection, random isotropic data, KS test ######### 

def mollweide_projection(azi, zen, ax, title='Visualization of Data on Mollweide Projection'):
    # Background settings
    ax.set_facecolor('gainsboro')
    ax.grid(color='white')
    
    # Meridian - zenith angle from 0rad to pi (0deg to 180 deg) -pi/2 so instead goes from -pi to pi
    meridian = np.stack([np.zeros(100), np.linspace(0,np.pi,100) - np.pi/2], axis=1)
    
    # Equator from 0 rad to 2pi - pi
    equator = np.stack([np.linspace(0,2*np.pi,100) - np.pi, np.zeros(100)], axis=1)
    
    # Display it
    ax.plot(equator[:,0], equator[:,1], lw=1, color='w')
    ax.plot(meridian[:,0], meridian[:,1], lw=1, color='w')

    # Plot the data
    ax.scatter(azi-np.pi, -zen+np.pi/2, marker='*', color='k', s=50, zorder=2, label='Data')

    # Title
    ax.set_title(title)
    
    
def sample_isotropic_data(seed, N_points):
    np.random.seed(seed)
    MC_azi = np.random.uniform(low=0, high=2*np.pi, size=N_points)
    MC_cos_zen = np.random.uniform(low=-1, high=1, size=N_points)
    MC_zen = np.arccos(MC_cos_zen)
    return MC_azi, MC_zen



def two_point_func(azi_arr, zen_arr):
    'Counts the pairs of events that are within a angular distance ϕ'
    
    # Define number of points
    N_tot = len(azi_arr)
    
    # Convert each point into cartesian coordinates (get unit vectors: r=1)
    r = 1
    Nx_arr = r * np.cos(azi_arr) * np.sin(zen_arr)
    Ny_arr = r * np.sin(azi_arr) * np.sin(zen_arr)
    Nz_arr = r * np.cos(zen_arr)
    
    #Defining the range of cos phi's (x-axis)
    N_bins = 50
    cos_phi_bins = np.linspace(-1, 1, N_bins)

    # Array to store angular distance pairs, i.e. all cos_phi_ij
    cos_phi_ij_arr = []
    
    # Loop over N_tot (outer sum)
    for i in range(0,N_tot):
        # Loop over i-1 (inner sum)
        for j in range(0,i):
            
            # Compute the angular distance cos(phi_ij) as the dot product of N_i dot N_j
            cos_phi_ij = Nx_arr[i]*Nx_arr[j] + Ny_arr[i]*Ny_arr[j] + Nz_arr[i]*Nz_arr[j]
            
            # Store it
            cos_phi_ij_arr.append(cos_phi_ij)

    
    # Emmpty array to store two point auto correlation function (y-values)
    two_point_arr = []
    
    # Loop over our cos phi bins (x-values)
    for i in range(N_bins):
        
        # =1 for x>= 0, =0 for x<0
        heaviside = np.heaviside(cos_phi_ij_arr - cos_phi_bins[i], 0)
        two_point = ( 2/(N_tot*(N_tot-1)) ) * np.sum( heaviside )
        
        # Store it
        two_point_arr.append(two_point)
    
    # Calculate the isotropic prediction
    'This cumulative two point auto-correlation function for the data can'
    'then be compared to the predicted function for perfect isotropic data'
    iso_pred = (1/2)*(1-cos_phi_bins)
    
    return cos_phi_bins, two_point_arr, iso_pred


def KS_test(dataA, dataB):
    return np.max( np.abs(dataA-dataB) )



def KS_plot(x, y, y_pred, fig, ax, label='Cumulative auto-correlation of data'):
    for i in range(2):
        ax[i].grid(color='grey', alpha=0.3)
    
    # Plot the two functions
    ax[0].plot(x, y, color='k', label=label)
    ax[0].plot(x, y_pred, color='r', label='Predicted cumulative auto-correlation for isotropic data')
    ax[0].legend(prop={'size':15})

    # Plot the residuals in subplot below
    resi = y_pred-y
    ax[1].plot(x, resi, color='k', label='Residual')
    ax[1].hlines(0, x[0], x[-1], linestyle='dashed', color='r')
    ax[1].legend(prop={'size':15})

    # ---------- Plot zoom -------------
    # Create extra axis
    ax1 = fig.add_axes([0.18, 0.35, 0.27, 0.25]) # add_axes([x0, y0, width, height])
    #ax1.set_facecolor('gainsboro')
    ax1.grid(color='grey', alpha=0.3)

    # Plot again
    ax1.plot(x, y, color='k')
    ax1.plot(x, y_pred ,color='r')
    

    # Get supremum index
    index = np.argmax(abs(resi))
    KS_H0 = KS_test(y, y_pred)
    #KS_MC_H0 = aas.KS_test(MC_y, MC_y_pred)

    # Adjust limits
    if resi[index] > 0: # y_pred is highest at supremum
        ymin, ymax = y[index] - resi[index], y_pred[index] + resi[index]

    if resi[index] < 0: # y is highest at supremum
        ymin, ymax = y_pred[index] + resi[index], y[index] - resi[index]

    xmin, xmax = x[index] - 0.05 * x[index], x[index] + 0.05 * x[index]

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    # Mark supremum
    supremum = ConnectionPatch(xyA=(x[index], y_pred[index]), xyB=(x[index], y[index]), coordsA=ax1.transData, 
                               arrowstyle='<->', color='k')
    fig.add_artist(supremum)
    ax1.set_title(f'KS = {KS_H0:.4f}', color='k', fontsize=14)

    # Add zoom lines
    con1 = ConnectionPatch(xyA=(xmin, ymin), coordsA=ax[0].transData, xyB=(xmax, ymin), coordsB=ax1.transData, alpha=0.5)
    con2 = ConnectionPatch(xyA=(xmax, ymax), coordsA=ax[0].transData, xyB=(xmax,ymax), coordsB=ax1.transData, alpha=0.5)

    sq1 = ConnectionPatch(xyA=(xmin, ymin), xyB=(xmax, ymin), coordsA=ax[0].transData, alpha=0.5)
    sq2 = ConnectionPatch(xyA=(xmin, ymax), xyB=(xmax, ymax), coordsA=ax[0].transData, alpha=0.5)
    sq3 = ConnectionPatch(xyA=(xmin, ymin), xyB=(xmin, ymax), coordsA=ax[0].transData, alpha=0.5)
    sq4 = ConnectionPatch(xyA=(xmax, ymin), xyB=(xmax, ymax), coordsA=ax[0].transData, alpha=0.5)

    fig.add_artist(con1)
    fig.add_artist(con2)
    fig.add_artist(sq1)
    fig.add_artist(sq2)
    fig.add_artist(sq3)
    fig.add_artist(sq4)
    # ----------------------------------

    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].tick_params(axis='both', which='minor', labelsize=15)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[1].tick_params(axis='both', which='minor', labelsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=12)

    ax[1].set_xlabel(r'$cos(\phi)$', fontsize=20)
    ax[0].set_ylabel(r'$\mathcal{C}(cos(\phi))$', fontsize=20)

    ax[0].set_title('KS test of cumulative two-point auto-correlation function')

    #if SaveFig:
    #    plt.tight_layout()
    #    plt.savefig('Plots/2a_3.pdf')

    plt.show()


def sample_KS(N_samples, N_points):
    
    # Array to store KS val in
    KS_arr = []
    
    # Loop over samples
    for i in tqdm(range(N_samples)):
    
        # Produce sample of isotropic data
        MC_azim, MC_zeni = sample_isotropic_data(i*100, N_points)
        
        # Get two point correlation func
        x, y, y_pred = two_point_func(MC_azim, MC_zeni)
        
        # Get KS val
        KS_val = np.max( np.abs(y_pred-y) )
        KS_arr.append(KS_val)
        
    return KS_arr

def compute_p_val(crit_val, sample):
    """
    crit_val = the critical val to compute p from
    sample = the data
    """
    
    return np.sum(sample >= crit_val) / len(sample)






######### Decision trees ######### 

def explore_data(train, train_class, test, test_class, label):
    N_1_train = int(sum(train_class))
    N_0_train = int(len(train_class)-N_1_train)
    SN_ratio_train = N_1_train/N_0_train

    N_1_test = int(sum(test_class))
    N_0_test = int(len(test_class)-N_1_test)
    SN_ratio_test = N_1_test/N_0_test

    print(f'Training data \nTotal = %s \n%s == 1: %s \n%s == 0: %s \nSN-ratio = %s' \
          %(len(train), label, N_1_train, label, N_0_train, round(SN_ratio_train, 3)))
    print('\nTesting data \nTotal = %s \n%s == 1: %s \n%s == 0: %s \nSN-ratio = %s' \
          %(len(test), label, N_1_test, label, N_0_test, round(SN_ratio_test, 3)))
    
def twoclass_output_for_cm(twoclass_output):
    twoclass_output_cm = twoclass_output.copy()
    twoclass_output_cm[twoclass_output_cm<0]=0 # noise
    twoclass_output_cm[twoclass_output_cm>0]=1 # signal
    return twoclass_output_cm


def clf_result(twoclass_output, twoclass_output_cm, t_class, label, axes):
    xrange = (-1, 1)
    hist0 = axes[0].hist(twoclass_output[t_class==0], bins=100, range=xrange, label='%s == 0' %label, histtype='step', linewidth=2, color='red')
    hist1 = axes[0].hist(twoclass_output[t_class==1], bins=100, range=xrange, label='%s == 1' %label, histtype='step', linewidth=2, color='black')
    axes[0].set_xlabel('BDT score')
    axes[0].legend()
    #axes[0].set_xlim(-0.6, 0.6)
    axes[0].vlines(0, 0, 900, color='grey', ls='dashed')
    axes[0].set_title(f'%s data result' %label)

    matrix_conf = confusion_matrix(t_class, twoclass_output_cm)
    sns_plot = sns.heatmap(
        matrix_conf,
        annot=True,
        cmap='GnBu',
        xticklabels=['%s == 0' %label, '%s == 1' %label],
        yticklabels=['%s == 0' %label, '%s == 1' %label],
        fmt="d",
        annot_kws={"fontsize": 14}
    )
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Predicted')
    axes[1].set_title("%s data confusion matrix" %label)
    
    tn, fp, fn, tp = matrix_conf.ravel()
    
    return hist0, hist1, [tn, fp, fn, tp]



def feature_importance_mdi(clf, features, ax, print_table=False):
    # Based on mean decrease in impurity (MDI)
    importances_mdi = clf.feature_importances_
    std_mdi = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    clf_importances_mdi = pd.Series(importances_mdi, index=features)
 
    # Plot
    clf_importances_mdi.plot.bar(yerr=std_mdi, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    
    # Rank
    rank_mdi = clf_importances_mdi.sort_values(ascending=False)
    N = np.arange(1, len(features)+1)
    
    if print_table:
        print("\\begin{table}")
        print("\\small")
        print("\\centering")
        print("\t \\begin{tabular}{c|cc}")
        print("\t \t \\hline")
        print("\t \t \\hline")
        print("\t \t {0:8s} & {1:11s} & {2:5s}  \\\\".format('Rank', 'Feature', 'MDI score'))
        print("\t \t \\hline")

        mdi_rank.keys()[0]

        for n, feature_mdi, score_mdi, sig_mid in zip(N, mdi_rank.keys(), mdi_rank.iloc, std_mdi):
            print(f"\t \t {n}  & \t {feature_mdi} & \t {score_mdi:2.3f} \pm {sig_mid:2.3f}    \\\\")


        print("\t \t \\hline")
        print("\t \\end{tabular}")
        print("\\end{table}")


def clf_roc(hist0_train, hist1_train, hist0_test, hist1_test, ax):
    
    FPR_train, TPR_train, AUC_train = calc_ROC(hist1_train, hist0_train) # first histogram has to be signal
    ax.plot(FPR_train, TPR_train, label = f'Train data')
    ax.text(0.5, 0.7, r'$AUC_{train}$  = %s' %round(AUC_train,3))
    
    FPR_test, TPR_test, AUC_test = calc_ROC(hist1_test, hist0_test) # first histogram has to be signal
    ax.plot(FPR_test, TPR_test, label = f'Train data')
    ax.text(0.5, 0.5, r'$AUC_{test}$  = %s' %round(AUC_test,3))
    
    ax.set(title='ROC curves')
    ax.legend()
    
    
def feature_importance(clf, features, test, test_class, axes, print_table=False):
    # Based on mean decrease in impurity (MDI)
    importances_mdi = clf.feature_importances_
    std_mdi = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    clf_importances_mdi = pd.Series(importances_mdi, index=features)
    
    # Based on feature permutation
    importances_fp = permutation_importance(clf, test, test_class, n_repeats=10, random_state=42, n_jobs=2)
    std_fp = importances_fp.importances_std
    clf_importances_fp = pd.Series(importances_fp.importances_mean, index=features)

    clf_importances_mdi.plot.bar(yerr=std_mdi, ax=axes[0])
    axes[0].set_title("Feature importances using MDI")
    axes[0].set_ylabel("Mean decrease in impurity")

    clf_importances_fp.plot.bar(yerr=std_fp, ax=axes[1])
    axes[1].set_title("Feature importances using permutation on full model")
    axes[1].set_ylabel("Mean accuracy decrease")
    
    # Ranks
    rank_mdi = clf_importances_mdi.sort_values(ascending=False)
    rank_fp = clf_importances_fp.sort_values(ascending=False)
    N = np.arange(1, len(features)+1)
    
    if print_table:
        print("\\begin{table}")
        print("\\small")
        print("\\centering")
        print("\t \\begin{tabular}{c|cc|cc}")
        print("\t \t \\hline")
        print("\t \t \\hline")
        print("\t \t {0:8s} & {1:11s} & {2:5s} & {3:5s} & {4:5s} \\\\".format('Rank', 'Feature', 'MDI score', 'Feature', 'FP score'))
        print("\t \t \\hline")

        mdi_rank.keys()[0]
        mdi_rank.iloc[0]

        for n, feature_mdi, score_mdi, sig_mid, feature_fp, score_fp, sig_fp in zip(N, mdi_rank.keys(), mdi_rank.iloc, std_mdi, rank_fp.keys(), rank_fp.iloc, std_fp):
            print(f"\t \t {n}  & \t {feature_mdi} & \t {score_mdi:2.3f} \pm {sig_mid:2.3f}  & \t {feature_fp}  & \t {score_fp:2.3f} \pm {sig_fp:2.3f}    \\\\")


        print("\t \t \\hline")
        print("\t \\end{tabular}")
        print("\\end{table}")

def TPR(tp, fn):
    'True positive rate'
    return tp/(tp+fn)

def FPR(fp, tn):
    'True positive rate'
    return fp/(fp+tn)
