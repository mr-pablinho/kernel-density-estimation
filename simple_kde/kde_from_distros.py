# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:48:26 2021
Kernel density estimation with samples from different distributions
@author: PMR
"""


# %% Import libraries

import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

# %% Create distribution 

nSamples = 1000
np.random.seed(123)
distro_names = ['Cauchy', 
                'Double Gamma',
                'Double Weibull',
                'Generalized extreme value',
                'Hyperbolic Secant',
                'Laplace',
                'Log-Gamma',
                'Logistic',
                'Normal',
                'Power Normal (Box-Cox)',
                'Non-central Student-t']

# create temporal samples
vector = [cp.Cauchy(scale=1, shift=0).sample(nSamples), 
          cp.DoubleGamma(shape=1, scale=1, shift=0).sample(nSamples),
          cp.DoubleWeibull(shape=1, scale=1, shift=0).sample(nSamples),
          cp.GeneralizedExtreme(shape=0, scale=1, shift=0).sample(nSamples),
          cp.HyperbolicSecant(scale=1, shift=0).sample(nSamples),
          cp.Laplace(mu=0, sigma=1).sample(nSamples),
          cp.LogGamma(shape=1, scale=1, shift=0).sample(nSamples),
          cp.Logistic(skew=1, shift=0, scale=1).sample(nSamples),
          cp.Normal(mu=0, sigma=1).sample(nSamples),
          cp.PowerNormal(shape=1, mu=0, sigma=1).sample(nSamples),
          cp.StudentT(df=1, mu=0, sigma=1).sample(nSamples)
          ]


for i in range(len(vector)):
    
    # load vector
    input_vector = vector[i]
    
    # create density function
    kde = stats.gaussian_kde(input_vector)
    
    start = int(np.min(input_vector)) - 1
    stop = int(np.max(input_vector)) + 1
    steps = np.abs(start) + np.abs(stop)
    
    x_o = np.linspace(start, stop, nSamples)
    y_o = kde.pdf(x_o)
    
    # plot results
    plt.figure(str(i))
    plt.title(distro_names[i] + ' distribution')
    plt.hist(input_vector, bins=30, density=True, alpha=.5, color='black', label='histogram')
    plt.fill_between(x_o, 0, y_o, alpha=.3, color='red', label='estimated density')
    plt.ylabel('Normalized density')
    plt.xlabel('Parameter value')
    plt.xlim(start, stop)
    plt.legend()


